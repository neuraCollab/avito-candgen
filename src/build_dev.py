"""Шаг 1. Локальная валидация, повторяющая устройство бенчмарка.

Наблюдения (подробнее в README):
  * все 2452 текста запросов бенчмарка уникальны, и только ~37% из них встречаются в train.
    Если взять случайный *уникальный текст* train, он повторяется в train в ~40% случаев —
    т.е. бенчмарк похож на выборку «один запрос на уникальный текст»;
  * объявления корпуса почти не пересекаются с train (~5-10%), снимок признаков тот же;
  * корпус собран «вокруг запросов»: число объявлений в локации почти пропорционально
    числу запросов бенчмарка в ней (~60-77 объявлений на запрос, корреляция 0.93),
    и эти объявления тематически близки к запросам (похоже на выдачу поиска + шум).

Первая версия валидации (случайные тексты, ответы 15k запросов в одном корпусе) давала
сильно завышенную оценку: в городах, где у бенчмарка мало объявлений, вставленные
ответы почти не имели конкурентов, и ранкер учился «угадывать по локации».

Поэтому dev строится так:
  1. N_FOLDS фолдов; в каждом для каждой локации s берём столько запросов из train,
     сколько запросов бенчмарка в s (одна случайная группа на уникальный текст,
     тексты между фолдами не повторяются);
  2. из train убираем строки этих групп и строки с их объявлениями -> «fit»
     (на fit считаются все статистики и модели первого уровня);
  3. корпус фолда k = корпус бенчмарка + объявления-ответы фолда k.
     Плотность «ответов» на объявление — как в бенчмарке, фон — те же 189k объявлений.
Фолд 0 — валидация, фолды 1..N-1 — обучение ранкера.

Режим --twins («двойники»). Корпус бенчмарка содержит «выдачу» каждого запроса:
десятки похожих объявлений в его локации. У случайного dev-запроса таких конкурентов
нет, поэтому ранжирование на dev получается легче, чем на бенчмарке (out-of-fold
Recall@50 падает с 0.915 до ~0.83, когда dev-запрос похож на запрос бенчмарка из той же
локации). В режиме twins для каждого запроса бенчмарка b в локации s берём N_FOLDS
самых похожих на него (по TF-IDF слов и символьных n-грамм) запросов train из той же
локации — по одному в каждый фолд. Тогда правильный ответ dev-запроса конкурирует с
реальной «выдачей» b, как и в самом бенчмарке. Если похожих нет (сходство < TWIN_MIN_SIM),
добираем взвешенной случайной выборкой, как в обычном режиме.
"""
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import CACHE, DATA, ITEM_COLS, QUERY_KEY, fix_item_dtypes, load_items_bench, load_queries_bench

SEED = 42
N_FOLDS = 5
TWIN_MIN_SIM = 0.25
P_UNSEEN = 0.9  # подобрано так, чтобы доля «невиденных» текстов в dev была как в бенчмарке (~63%)


def text_vectors(texts):
    """TF-IDF по стемам слов + символьным n-граммам (как в query_knn.py), L2-нормированные."""
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as l2norm

    from common import normalize, tokenize
    clean = [" ".join(normalize(t).split()) for t in texts]
    a = TfidfVectorizer(analyzer=tokenize, sublinear_tf=True).fit_transform(clean)
    b = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2).fit_transform(clean)
    return l2norm(sp.hstack([a * 0.7, b * 0.3]).tocsr())


def main():
    rng = np.random.default_rng(SEED)
    tr = pd.read_parquet(DATA / "train.parquet", columns=QUERY_KEY + ["item_id"])
    tr["search_infm_params_text"] = tr["search_infm_params_text"].fillna("")
    tr["gid"] = tr.groupby(QUERY_KEY, sort=False).ngroup()
    bq = load_queries_bench()
    need = bq.search_location_id.value_counts()  # сколько запросов нужно в каждой локации

    # Бенчмарк ~ «равномерно по уникальным текстам». Совместное распределение тогда
    #   P(текст, s) = 1/T * n(текст, s) / n(текст),
    # и внутри локации s текст надо выбирать с весом n(текст, s) / n(текст):
    # редкий текст получает вес ~1, «маникюр», который ищут во всех городах, — почти 0.
    g = tr[["gid", "search_query", "search_location_id"]].drop_duplicates("gid")
    n_ts = g.groupby(["search_query", "search_location_id"]).size().rename("n_ts").reset_index()
    n_ts["w"] = n_ts.n_ts / n_ts.groupby("search_query").n_ts.transform("sum")
    # взвешенная выборка без возвращения (Efraimidis-Spirakis): ключ u^(1/w), берём top-n
    n_ts["key"] = rng.random(len(n_ts)) ** (1.0 / n_ts.w)
    n_ts = n_ts.sort_values("key", ascending=False)
    # одна случайная группа на (локация, текст)
    g = g.sample(frac=1.0, random_state=SEED).drop_duplicates(["search_location_id", "search_query"])
    gid_of = dict(zip(zip(g.search_location_id, g.search_query), g.gid))
    by_loc = {s: d.search_query.tolist() for s, d in n_ts.groupby("search_location_id", sort=False)}

    used_texts: set = set()
    fold_of_gid = {}
    if "--twins" not in sys.argv:
        for s in rng.permutation(need.index.to_numpy()):
            n = need[s] * N_FOLDS
            take = [t for t in by_loc.get(s, []) if t not in used_texts][:n]
            used_texts.update(take)
            for j, t in enumerate(take):
                fold_of_gid[gid_of[(s, t)]] = j % N_FOLDS
    else:
        # тексты бенчмарка и train в одном векторном пространстве
        train_texts = g.search_query.unique()
        V = text_vectors(list(train_texts) + bq.search_query.tolist())
        row_of = {t: i for i, t in enumerate(train_texts)}
        n_groups = g.groupby("search_query").size().to_dict()
        Vb = V[len(train_texts):]
        sims_taken = []
        for j in rng.permutation(len(bq)):
            s = bq.search_location_id.iat[j]
            cands = [t for t in by_loc.get(s, []) if t not in used_texts]
            if not cands:
                continue
            sim = (V[[row_of[t] for t in cands]] @ Vb[j].T).toarray().ravel()
            order = [k for k in np.argsort(-sim) if sim[k] >= TWIN_MIN_SIM]
            # Похожие тексты чаще «популярные», а в бенчмарке 63% текстов в train не встречаются.
            # Поэтому каждый слот с вероятностью P_UNSEEN берём из текстов, которые есть в train
            # ровно в одной группе (после удаления из fit они станут «невиденными»).
            uniq = [k for k in order if n_groups[cands[k]] == 1]
            multi = [k for k in order if n_groups[cands[k]] > 1]
            take = []
            for _ in range(N_FOLDS):
                first, second = (uniq, multi) if rng.random() < P_UNSEEN else (multi, uniq)
                src = first if first else second
                if not src:
                    break
                k = src.pop(0)
                take.append(cands[k])
                sims_taken.append(float(sim[k]))
            # не хватило похожих — добираем взвешенной случайной выборкой (by_loc уже перемешан)
            take += [t for t in cands if t not in take][: N_FOLDS - len(take)]
            used_texts.update(take)
            for k, t in zip(rng.permutation(N_FOLDS), take):
                fold_of_gid[gid_of[(s, t)]] = int(k)
        print(f"twins: {len(sims_taken)} похожих (сходство {np.mean(sims_taken):.2f}), "
              f"остальные {len(fold_of_gid) - len(sims_taken)} — случайные")

    tr["fold"] = tr.gid.map(fold_of_gid)
    is_dev = tr.fold.notna()
    dev_items = set(tr.loc[is_dev, "item_id"])
    fit_mask = ~is_dev & ~tr.item_id.isin(dev_items)
    np.save(CACHE / "fit_rows.npy", np.flatnonzero(fit_mask.values))

    sub = tr[is_dev]
    qs = sub.drop_duplicates("gid")[QUERY_KEY + ["gid", "fold"]].copy()
    qs["fold"] = qs["fold"].astype(int)
    qs["query_id"] = "f" + qs["fold"].astype(str) + "_" + qs["gid"].astype(str)
    truth = sub.groupby("gid")["item_id"].agg(lambda s: " ".join(sorted(set(s))))
    qs["truth"] = qs["gid"].map(truth)
    qs = qs.drop(columns="gid").reset_index(drop=True)
    qs.to_parquet(CACHE / "dev_queries.parquet")

    # признаки объявлений-ответов берём из train (снимок тот же, что в корпусе бенчмарка)
    pf = pq.read_table(DATA / "train.parquet", columns=ITEM_COLS).to_pandas()
    pos_items = fix_item_dtypes(pf[pf.item_id.isin(dev_items)].drop_duplicates("item_id"))
    del pf
    bench_items = load_items_bench(ITEM_COLS)
    bench_items.to_parquet(CACHE / "corpus_bench.parquet")
    bench_ids = set(bench_items.item_id)
    for k in range(N_FOLDS):
        ids = set(" ".join(qs.loc[qs.fold == k, "truth"]).split()) - bench_ids
        corpus = pd.concat([bench_items, pos_items[pos_items.item_id.isin(ids)][ITEM_COLS]], ignore_index=True)
        corpus.to_parquet(CACHE / f"corpus_f{k}.parquet")
        print(f"fold {k}: {int((qs.fold == k).sum())} queries, corpus {len(corpus)} (+{len(ids)} positives)")

    # sanity: похож ли dev на бенчмарк
    fit = tr[fit_mask]
    v = qs[qs.fold == 0]
    print(f"fit rows: {len(fit)} / {len(tr)}")
    print("text seen in fit:", round(v.search_query.isin(set(fit.search_query)).mean(), 3), "(bench 0.370)")
    fit_ql = set(zip(fit.search_query, fit.search_location_id))
    print("(text,loc) seen in fit:", round(np.mean([k in fit_ql for k in zip(v.search_query, v.search_location_id)]), 3), "(bench 0.065)")
    print("params empty:", round((v.search_infm_params_text == "").mean(), 3), "(bench 0.631)")
    print("relevant per query:", round(v.truth.str.split().str.len().mean(), 3))


if __name__ == "__main__":
    main()
