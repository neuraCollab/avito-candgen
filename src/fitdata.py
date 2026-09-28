"""Загрузка «fit»-части train и корпусов/запросов для dev и bench режимов.

Режим fK:    запросы dev-фолда K (см. build_dev.py), корпус = бенчмарк + их ответы.
Режим bench: запросы = benchmark_queries, корпус = benchmark_items.
В обоих режимах статистики считаются по одной и той же fit-части train,
поэтому признаки ранкера имеют одинаковое распределение на обучении и на сабмите.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CACHE, DATA, ITEM_COLS, QUERY_KEY, fix_item_dtypes, load_queries_bench


def load_fit(with_text=False) -> pd.DataFrame:
    cols = QUERY_KEY + ["item_id", "item_title_raw", "item_microcat_id", "item_location_id",
                        "item_latitude", "item_longitude", "item_infm_params_text"]
    if with_text:
        cols += ["item_description_raw"]
    tr = pd.read_parquet(DATA / "train.parquet", columns=cols)
    tr = tr.iloc[np.load(CACHE / "fit_rows.npy")].reset_index(drop=True)
    tr["search_infm_params_text"] = tr["search_infm_params_text"].fillna("")
    return fix_item_dtypes(tr)


def load_mode(mode: str):
    """mode: 'f0'..'f4' (dev-фолды, см. build_dev.py) или 'bench'.
    -> (queries, corpus). У dev-запросов есть колонка truth (правильные item_id через пробел)."""
    if mode.startswith("f"):
        q = pd.read_parquet(CACHE / "dev_queries.parquet")
        q = q[q.fold == int(mode[1:])]
    else:
        q = load_queries_bench()
    corpus = pd.read_parquet(CACHE / f"corpus_{mode}.parquet",
                             columns=[c for c in ITEM_COLS if c != "item_description_raw"])
    return q.reset_index(drop=True), fix_item_dtypes(corpus)
