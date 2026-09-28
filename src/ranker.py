"""Шаг 5. LightGBM-ранкер: выбирает 50 лучших из ~350 кандидатов.

Данные — dev-фолды (build_dev.py): для каждого запроса пул кандидатов с признаками
(candidates.py) и метка «выбрал ли пользователь это объявление».
  * валидация: обучаемся на f1..f4, меряем Recall@50 на f0 — по ПОЛНОМУ списку
    правильных ответов (ответы, не попавшие в пул, честно штрафуют метрику);
  * финальная модель: обучаемся на всех фолдах с числом деревьев, подобранным на валидации.
Целевая функция lambdarank: важен порядок внутри запроса, а не абсолютная вероятность.
"""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import CACHE, recall_at_k

NON_FEATURES = {"query_id", "item_idx", "item_id", "label"}
FOLDS = ["f0", "f1", "f2", "f3", "f4"]

PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[50],
    # метрике важно только попасть в top-50 (порядок внутри не важен),
    # поэтому lambdarank фокусируем на верхних ~60 позициях
    lambdarank_truncation_level=60,
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    num_threads=8,
)


def load(names) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(CACHE / f"feats_{n}.parquet") for n in names], ignore_index=True)
    df = df.sort_values("query_id", kind="stable").reset_index(drop=True)
    fcols = [c for c in df.columns if c not in NON_FEATURES]
    df[fcols] = df[fcols].astype(np.float32)
    return df


def groups(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("query_id", sort=False).size().to_numpy()


def only_with_pos(df: pd.DataFrame) -> pd.DataFrame:
    """Запросы без правильного ответа в пуле не несут сигнала для ранжирования."""
    return df[df.groupby("query_id")["label"].transform("max") > 0].reset_index(drop=True)


def top50(df: pd.DataFrame, score: np.ndarray, k: int = 50) -> dict:
    d = df[["query_id", "item_id"]].copy()
    d["s"] = score
    d = d.sort_values(["query_id", "s"], ascending=[True, False])
    return d.groupby("query_id", sort=False)["item_id"].apply(lambda s: list(s[:k])).to_dict()


def truth_of(fold: str) -> dict:
    q = pd.read_parquet(CACHE / "dev_queries.parquet")
    q = q[q.fold == int(fold[1:])]
    return {qid: set(t.split()) for qid, t in zip(q.query_id, q.truth)}


def validate(val_fold="f0", drop=()):
    tr = only_with_pos(load([f for f in FOLDS if f != val_fold]))
    va = load([val_fold])
    fcols = [c for c in tr.columns if c not in NON_FEATURES and c not in drop]
    dtr = lgb.Dataset(tr[fcols], tr["label"], group=groups(tr))
    dva = lgb.Dataset(va[fcols], va["label"], group=groups(va))
    model = lgb.train(PARAMS, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
    truth = truth_of(val_fold)
    pred = top50(va, model.predict(va[fcols], num_threads=8))
    pool = va[va.label == 1].groupby("query_id").size()
    pool_rec = np.mean([pool.get(q, 0) / len(t) for q, t in truth.items()])
    print(f"[{val_fold}] Recall@50 = {recall_at_k(pred, truth):.4f}   (потолок пула {pool_rec:.4f}, "
          f"best_iter {model.best_iteration})")
    imp = pd.Series(model.feature_importance("gain"), index=fcols).sort_values(ascending=False)
    print((imp / imp.sum()).round(4).head(20).to_string())
    return model.best_iteration


N_BAG = 5  # усредняем ранкеры с разными seed: на f0 это +0.1 и меньше разброс


def final(n_rounds: int, n_bag: int = N_BAG):
    tr = only_with_pos(load(FOLDS))
    fcols = [c for c in tr.columns if c not in NON_FEATURES]
    dtr = lgb.Dataset(tr[fcols], tr["label"], group=groups(tr), free_raw_data=False)
    for old in CACHE.glob("ranker_final*.txt"):
        old.unlink()
    for seed in range(n_bag):
        p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
        model = lgb.train(p, dtr, n_rounds)
        # LightGBM (C++) на Windows не открывает пути с кириллицей — пишем модель строкой из Python
        (CACHE / f"ranker_final_{seed}.txt").write_text(model.model_to_string(), encoding="utf-8")
    print(f"final ranker: {n_bag} x {n_rounds} trees on {tr.query_id.nunique()} queries")


def load_final() -> list[lgb.Booster]:
    files = sorted(CACHE.glob("ranker_final*.txt"))
    assert files, "нет финального ранкера: python ranker.py --final"
    return [lgb.Booster(model_str=f.read_text(encoding="utf-8")) for f in files]


def predict_final(models: list[lgb.Booster], feats: pd.DataFrame) -> np.ndarray:
    """Средний скор бэггинга ранкеров."""
    fc = models[0].feature_name()
    X = feats[fc].astype(np.float32)
    return np.mean([m.predict(X, num_threads=8) for m in models], axis=0)


if __name__ == "__main__":
    best = validate()
    if "--final" in sys.argv:
        final(int(best * 1.1) + 1)  # данных на 25% больше -> чуть больше деревьев
