"""Шаг 4. Широкий пул кандидатов (~400 на запрос) + признаки для ранкера.

Для каждого запроса считаем по ВСЕМУ корпусу несколько «скоринговых векторов»:
  * BM25 слов запроса по заголовку / параметрам / описанию;
  * BM25 «расширения» запроса — слов из заголовков объявлений, которые выбирали
    по похожим запросам в train (query_knn.py);
  * P(microcat | запрос) — априорная вероятность подкатегории (query_knn.py);
  * локационный вес: доля переходов «локация поиска -> локация объявления» в train.

Пул = объединение нескольких top-N из разных источников (разные источники ловят
разные промахи: лексический BM25, семантическое расширение, запросы без фильтра
по локации для онлайн/междугородних услуг). Затем для каждой пары
(запрос, кандидат) собираем признаки, а LightGBM (ranker.py) выбирает итоговые 50.
"""
from __future__ import annotations

import bisect
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2norm

import text_index
from common import CACHE, normalize, tokenize
from fitdata import load_fit, load_mode
from query_knn import QueryKNN

from dense import DenseIndex, available

# Эмбеддинги bi-encoder'ов (biencoder.py). Основной — дообученный rubert-tiny2 (входит в общий скор);
# дополнительные (по умолчанию rubert-mini-frida без дообучения и дообученная) дают свои источники
# кандидатов и свои признаки. Набор задаётся переменной AVITO_EXTRA_DENSE (через запятую).
USE_DENSE = available("tiny2ft")
EXTRA_DENSE = [e for e in os.environ.get("AVITO_EXTRA_DENSE", "frida0,fridaft").split(",") if e]

# размеры источников пула
N_COMBO = 300     # основной: BM25 + расширение + prior подкатегории, внутри «своих» локаций
N_BASE = 100      # чистый BM25 по словам запроса, внутри локаций
N_EXP = 50        # только расширение запроса (синонимы), внутри локаций
N_GLOBAL = 50     # основной скор без фильтра по локации (онлайн-услуги, межгород)
N_DENSE = 100     # ближайшие по эмбеддингу bi-encoder'а, внутри локаций
DENSE_W = 20.0    # вес косинуса эмбеддингов в основном скоре (BM25 ~ 10..60, косинус ~ 0.3..0.9)
LOC_MIN_SHARE = 0.001

# ключи фильтров в search_infm_params_text (найдены при разведочном анализе)
_FILTER_KEYS = [
    "Тип услуги автосервиса", "Вид услуги", "Тип услуги", "Предмет или специальность",
    "Кто оказывает услуги", "Чем вы занимаетесь", "Онлайн-запись", "Поиск по слотам",
    "Срочная услуга (мультистатус)", "Участие в пилоте CPL", "Сортировка для URL",
    "Рейтинг пользователя 4 звезды и выше",
]
_FILTER_RE = re.compile("(" + "|".join(re.escape(k) for k in _FILTER_KEYS) + ")")
# фильтры, значение которых должно встречаться в параметрах объявления
_MATCHABLE = {"Тип услуги автосервиса", "Вид услуги", "Тип услуги", "Предмет или специальность",
              "Кто оказывает услуги", "Чем вы занимаетесь"}


def parse_filters(s: str) -> list[tuple[str, str]]:
    """'Тип услуги Сантехника Вид услуги Ремонт и отделка' ->
    [('Тип услуги', 'Сантехника'), ('Вид услуги', 'Ремонт и отделка')]"""
    parts = _FILTER_RE.split(s or "")
    out = []
    for i in range(1, len(parts), 2):
        out.append((parts[i], parts[i + 1].strip() if i + 1 < len(parts) else ""))
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class Retriever:
    def __init__(self, mode: str):
        t0 = time.time()
        self.mode = mode
        self.queries, self.corpus = load_mode(mode)
        self.idx = text_index.load(mode)
        assert (self.idx["item_id"] == self.corpus["item_id"].to_numpy()).all()
        self.vocab = self.idx["vocab"]
        self.N = len(self.corpus)
        fit = load_fit()

        # --- kNN по запросам train: prior подкатегории + расширение запроса
        self.knn = QueryKNN().fit(fit, self.vocab)
        mc_pos = {m: i for i, m in enumerate(self.knn.mc_codes)}
        self.item_mc_idx = np.array([mc_pos.get(m, -1) for m in self.corpus.item_microcat_id], dtype=np.int64)

        # --- локации: доля переходов search_loc -> item_loc в train
        loc_codes, self.item_loc_code = np.unique(self.corpus.item_location_id.to_numpy(), return_inverse=True)
        self.loc_index = {l: i for i, l in enumerate(loc_codes)}
        T = fit.groupby(["search_location_id", "item_location_id"]).size().rename("n").reset_index()
        T["share"] = T.n / T.groupby("search_location_id").n.transform("sum")
        T = T[T.item_location_id.isin(self.loc_index)]
        self.loc_share = {
            s: dict(zip(g.item_location_id.map(self.loc_index), g.share))
            for s, g in T.groupby("search_location_id")
        }
        # «центр» локации поиска: медиана координат выбранных в ней объявлений
        # (сначала по объявлениям той же локации, иначе — по всем выбранным)
        same = fit[fit.search_location_id == fit.item_location_id]
        c1 = same.groupby("search_location_id")[["item_latitude", "item_longitude"]].median()
        c2 = fit.groupby("search_location_id")[["item_latitude", "item_longitude"]].median()
        c3 = self.corpus.groupby("item_location_id")[["item_latitude", "item_longitude"]].median()
        self.loc_center = pd.concat([c1, c2[~c2.index.isin(c1.index)], c3[~c3.index.isin(c1.index.union(c2.index))]])

        # популярность подкатегории в локации поиска (сколько раз её выбирали)
        self.mc_loc_cnt = fit.groupby(["search_location_id", "item_microcat_id"]).size().to_dict()
        self.mc_cnt = fit.item_microcat_id.value_counts().to_dict()

        # --- символьные n-граммы заголовков: нечёткое совпадение с запросом (опечатки)
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=2,
                                        max_features=200_000)
        self.title_char = l2norm(self.char_vec.fit_transform(self.corpus.item_title_raw.map(normalize)))

        # --- статические признаки объявлений
        c = self.corpus
        self.params_text = c.item_infm_params_text.to_numpy()
        self.item_static = pd.DataFrame({
            "it_rating": c.item_rating.to_numpy(),
            "it_reviews_log": np.log1p(c.item_rating_reviews_count.fillna(0).to_numpy()),
            "it_reviews_nan": c.item_rating_reviews_count.isna().to_numpy().astype(np.int8),
            "it_price_log": np.log1p(c.item_price.clip(lower=0).fillna(0).to_numpy()),
            "it_phone_hidden": c.item_is_phone_hidden.to_numpy().astype(np.int8),
            "it_msg_forbidden": c.item_is_message_forbidden.to_numpy().astype(np.int8),
            "it_cat_services": (c.item_category_id.to_numpy() == 114).astype(np.int8),
            "it_mc_pop_log": np.log1p(c.item_microcat_id.map(self.mc_cnt).fillna(0).to_numpy()),
            "it_title_len": np.diff(self.idx["title"].tocsr().indptr),
            "it_desc_len": np.diff(self.idx["desc"].tocsr().indptr),
        })
        # префиксный поиск по словарю: пользователи часто не дописывают последнее слово
        # («прокат квад» -> «квадроцикл»), автодополнение обрывает запрос на середине
        self.sorted_vocab = sorted(self.vocab)
        self.term_by_str = self.vocab
        self.title_df = np.diff(self.idx["title"].indptr)
        # стемы заголовков по порядку — для совпадения словосочетаний (биграмм) запроса
        self.title_bigrams = [set(zip(t, t[1:])) for t in map(tokenize, self.corpus.item_title_raw)]
        self.dense = DenseIndex(self.corpus.item_id.to_numpy()) if USE_DENSE else None
        missing = [e for e in EXTRA_DENSE if not available(e)]
        assert not missing, f"не посчитаны эмбеддинги {missing}: python biencoder.py embed <encoder>"
        self.extra_dense = {e: DenseIndex(self.corpus.item_id.to_numpy(), e) for e in EXTRA_DENSE}
        # бинарные «есть ли слово в поле» — для доли покрытых слов запроса
        self.present = {f: self.idx[f].astype(bool).astype(np.float32).tocsc() for f in ("title", "params", "desc")}
        print(f"Retriever[{mode}] ready in {time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------------ utils
    def _score(self, W: sp.csc_matrix, terms, weights=None) -> np.ndarray:
        """Сумма столбцов W по словам (с весами) = BM25-скор по всему корпусу."""
        if len(terms) == 0:
            return np.zeros(self.N, np.float32)
        idx_parts, val_parts = [], []
        for k, t in enumerate(terms):
            a, b = W.indptr[t], W.indptr[t + 1]
            idx_parts.append(W.indices[a:b])
            v = W.data[a:b]
            val_parts.append(v if weights is None else v * weights[k])
        return np.bincount(np.concatenate(idx_parts), weights=np.concatenate(val_parts),
                           minlength=self.N).astype(np.float32)

    def _loc_vector(self, s_loc) -> np.ndarray:
        share = np.zeros(len(self.loc_index), np.float32)
        for li, sh in self.loc_share.get(s_loc, {}).items():
            share[li] = sh
        if s_loc in self.loc_index:
            # своя локация всегда разрешена, даже если в train её не было
            share[self.loc_index[s_loc]] = max(share[self.loc_index[s_loc]], 0.5)
        return share[self.item_loc_code]

    def _prefix_terms(self, tok: str, max_terms: int = 10) -> list[int]:
        """Стемы словаря, начинающиеся с tok и встречающиеся в заголовках (самые частые)."""
        out = []
        j = bisect.bisect_left(self.sorted_vocab, tok)
        while j < len(self.sorted_vocab) and self.sorted_vocab[j].startswith(tok):
            t = self.term_by_str[self.sorted_vocab[j]]
            if self.sorted_vocab[j] != tok and self.title_df[t] >= 2:
                out.append(t)
            j += 1
        out.sort(key=lambda t: -self.title_df[t])
        return out[:max_terms]

    @staticmethod
    def _top(score: np.ndarray, n: int) -> np.ndarray:
        n = min(n, len(score) - 1)
        top = np.argpartition(-score, n)[:n]
        return top[score[top] > -1e8]

    # -------------------------------------------------------------- main loop
    def run(self, queries: pd.DataFrame) -> pd.DataFrame:
        t0 = time.time()
        P, H, top_sim = self.knn.query(queries.search_query.tolist())
        q_char = l2norm(self.char_vec.transform(queries.search_query.map(normalize)))
        out = []
        for i, r in enumerate(queries.itertuples(index=False)):
            out.append(self._one(i, r, P, H, top_sim, q_char))
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(queries)} queries, {time.time() - t0:.0f}s", flush=True)
        return pd.concat(out, ignore_index=True)

    def _one(self, i, r, P, H, top_sim, q_char) -> pd.DataFrame:
        qtoks = list(dict.fromkeys(tokenize(r.search_query)))
        terms = [self.vocab[t] for t in qtoks if t in self.vocab]

        # скоры по словам запроса
        bt = self._score(self.idx["title"], terms)
        bp = self._score(self.idx["params"], terms)
        bd = self._score(self.idx["desc"], terms)
        # префиксное расширение последнего слова, если оно редкое в заголовках (вероятно, не дописано):
        # берём максимум BM25 по вариантам, чтобы много вариантов не «перевесили» остальные слова
        pref = np.zeros(self.N, np.float32)
        raw = re.findall(r"[a-zа-я0-9]+", normalize(r.search_query))
        if raw and len(raw[-1]) >= 3 and not raw[-1].isdigit():
            last = raw[-1]
            t_last = self.vocab.get(qtoks[-1]) if qtoks else None
            if t_last is None or self.title_df[t_last] < 5:
                for t in self._prefix_terms(last):
                    pref = np.maximum(pref, self._score(self.idx["title"], [t]) + 0.3 * self._score(self.idx["desc"], [t]))
        base = bt + 0.5 * bp + 0.3 * bd + 0.8 * pref

        # расширение запроса: 15 самых частых стемов заголовков у похожих запросов
        h = H[i]
        order = np.argsort(-h.data)[:15]
        et, ew = h.indices[order], h.data[order]
        ew = ew / ew.max() if len(ew) else ew
        ext = self._score(self.idx["title"], et, ew)
        exd = self._score(self.idx["desc"], et, ew)
        exp = ext + 0.3 * exd

        # prior подкатегории
        pm = P[i].toarray().ravel()
        pmc = np.where(self.item_mc_idx >= 0, pm[np.maximum(self.item_mc_idx, 0)], 0.0).astype(np.float32)

        # косинус эмбеддингов bi-encoder'а (семантика: синонимы, перефразировки)
        ds = self.dense.scores(r.query_id) if self.dense is not None else np.zeros(self.N, np.float32)

        combo = base + exp + np.log(pmc + 0.01) + DENSE_W * ds
        locv = self._loc_vector(r.search_location_id)
        inloc = locv > LOC_MIN_SHARE
        combo_l = np.where(inloc, combo, -1e9)

        sources = [
            self._top(combo_l, N_COMBO),
            self._top(np.where(inloc & (base > 0), base, -1e9), N_BASE),
            self._top(np.where(inloc & (exp > 0), exp, -1e9), N_EXP),
            self._top(combo, N_GLOBAL),
        ]
        if self.dense is not None:
            sources.append(self._top(np.where(inloc, ds, -1e9), N_DENSE))
        extra = {e: idx.scores(r.query_id) for e, idx in self.extra_dense.items()}
        for dse in extra.values():
            sources.append(self._top(np.where(inloc, dse, -1e9), N_DENSE))
        cand = np.unique(np.concatenate(sources))

        # ---- признаки пары (запрос, кандидат)
        n_terms = max(len(terms), 1)
        cov = {f: self._score(self.present[f], terms)[cand] / n_terms for f in ("title", "params", "desc")}
        f = pd.DataFrame({
            "bm25_title": bt[cand], "bm25_params": bp[cand], "bm25_desc": bd[cand], "bm25_base": base[cand],
            "exp_title": ext[cand], "exp_desc": exd[cand], "exp": exp[cand],
            "combo": combo[cand], "pmc": pmc[cand], "prefix": pref[cand],
            "cov_title": cov["title"], "cov_params": cov["params"], "cov_desc": cov["desc"],
            "cov_any": np.maximum(np.maximum(cov["title"], cov["params"]), cov["desc"]),
            "loc_share": locv[cand], "in_loc": inloc[cand].astype(np.int8),
        })
        f["same_loc"] = (self.corpus.item_location_id.to_numpy()[cand] == r.search_location_id).astype(np.int8)
        f["char_sim_title"] = np.asarray((self.title_char[cand] @ q_char[i].T).todense()).ravel()
        # «точность» по заголовку: какая доля слов заголовка взята из запроса (короткий точный
        # заголовок «Ремонт кофемашин» лучше длинного SEO-заголовка с теми же словами)
        f["title_prec"] = cov["title"] * n_terms / np.maximum(self.item_static.it_title_len.to_numpy()[cand], 1)
        # словосочетания: доля пар соседних слов запроса, которые стоят рядом и в заголовке
        qbig = set(zip(qtoks, qtoks[1:]))
        f["title_bigram"] = (np.array([len(qbig & self.title_bigrams[j]) for j in cand], np.float32) / len(qbig)
                             if qbig else np.nan)
        if self.dense is not None:
            f["dense"] = ds[cand]
            ds_l = np.where(inloc, ds, -1e9)
            f["rank_dense"] = _rank_desc(ds_l, ds_l[cand])
            f["dense_rel"] = ds[cand] - (ds[inloc].max() if inloc.any() else ds.max())
        for e, dse in extra.items():
            f[f"dense_{e}"] = dse[cand]
            dl = np.where(inloc, dse, -1e9)
            f[f"rank_dense_{e}"] = _rank_desc(dl, dl[cand])
            f[f"dense_rel_{e}"] = dse[cand] - (dse[inloc].max() if inloc.any() else dse.max())

        # ранги и нормировки внутри «своих» локаций: насколько кандидат лучше соседей
        for col, vec in (("combo", combo_l), ("bm25_base", np.where(inloc, base, -1e9)),
                         ("exp", np.where(inloc, exp, -1e9))):
            f[f"rank_{col}"] = _rank_desc(vec, vec[cand])
        for col, vec in (("bm25_title", bt), ("bm25_base", base), ("exp", exp), ("combo", combo)):
            m = vec[inloc].max() if inloc.any() else vec.max()
            f[f"{col}_rel"] = f[col] / (abs(m) + 1e-6)

        # подкатегория: ранг в prior-распределении и доля в top-30 основного скора (PRF)
        mc = self.item_mc_idx[cand]
        f["pmc_rank"] = (pm[None, :] > pmc[cand][:, None]).sum(axis=1)
        top30 = self._top(combo_l, 30)
        prf = np.bincount(self.item_mc_idx[top30][self.item_mc_idx[top30] >= 0], minlength=len(pm)) / max(len(top30), 1)
        f["mc_prf"] = np.where(mc >= 0, prf[np.maximum(mc, 0)], 0)
        mcid = self.corpus.item_microcat_id.to_numpy()[cand]
        f["mc_loc_pop_log"] = np.log1p([self.mc_loc_cnt.get((r.search_location_id, m), 0) for m in mcid])

        # гео-расстояние до центра локации поиска
        if r.search_location_id in self.loc_center.index:
            clat, clon = self.loc_center.loc[r.search_location_id]
            f["dist_km"] = haversine_km(clat, clon, self.corpus.item_latitude.to_numpy()[cand],
                                        self.corpus.item_longitude.to_numpy()[cand])
        else:
            f["dist_km"] = np.nan

        # фильтры поиска: совпадают ли значения «Вид услуги ...», «Тип услуги ...» с параметрами
        flt = [(k, v) for k, v in parse_filters(r.search_infm_params_text) if k in _MATCHABLE and v]
        ptxt = self.params_text[cand]
        if flt:
            hits = np.zeros(len(cand), np.float32)
            for k, v in flt:
                needle = f"{k} {v}"
                hits += np.fromiter((needle in p for p in ptxt), dtype=np.float32, count=len(cand))
            f["flt_n"] = len(flt)
            f["flt_frac"] = hits / len(flt)
        else:
            f["flt_n"] = 0
            f["flt_frac"] = np.nan
        rating_flt = "Рейтинг пользователя 4 звезды и выше" in r.search_infm_params_text
        f["rating_flt_ok"] = np.where(rating_flt, (self.corpus.item_rating.to_numpy()[cand] >= 4).astype(np.float32), np.nan)

        # признаки запроса (одинаковы для всех кандидатов — помогают дереву калибровать остальные)
        f["q_nterms"] = len(qtoks)
        f["q_nterms_known"] = len(terms)
        f["q_knn_sim"] = top_sim[i]
        f["q_pool_inloc"] = np.log1p(inloc.sum())
        f["q_pmc_max"] = pm.max() if len(pm) else 0

        f = pd.concat([f, self.item_static.iloc[cand].reset_index(drop=True)], axis=1)
        f.insert(0, "item_idx", cand.astype(np.int32))
        f.insert(0, "query_id", r.query_id)
        return f


def _rank_desc(vec: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Ранг каждого значения v среди всего вектора vec (0 = лучший)."""
    s = np.sort(vec)
    return (len(s) - np.searchsorted(s, v, side="right")).astype(np.int32)


def main(mode: str, limit: int | None = None):
    R = Retriever(mode)
    q = R.queries.reset_index(drop=True)
    if limit:
        q = q.head(limit)
    feats = R.run(q)
    feats["item_id"] = R.corpus.item_id.to_numpy()[feats.item_idx.to_numpy()]
    if "truth" in q.columns:
        truth = {qid: set(t.split()) for qid, t in zip(q.query_id, q.truth)}
        feats["label"] = [int(iid in truth[qid]) for qid, iid in zip(feats.query_id, feats.item_id)]
        # полнота пула: какая доля правильных ответов вообще попала в кандидаты
        got = feats[feats.label == 1].groupby("query_id").size()
        rec = np.mean([got.get(qid, 0) / len(t) for qid, t in truth.items()])
        print(f"pool recall = {rec:.4f}, avg pool size = {len(feats) / len(q):.0f}")
    feats.to_parquet(CACHE / f"feats_{mode}.parquet")
    print("saved", mode, feats.shape)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
