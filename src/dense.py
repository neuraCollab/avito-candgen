"""Доступ к предпосчитанным эмбеддингам bi-encoder'ов (см. biencoder.py) в порядке корпуса."""
from __future__ import annotations

import numpy as np
import pandas as pd

from biencoder import emb_files


def available(encoder: str) -> bool:
    return all(p.exists() for p in emb_files(encoder))


class DenseIndex:
    def __init__(self, item_ids: np.ndarray, encoder: str = "tiny2ft"):
        item_file, query_file = emb_files(encoder)
        emb = pd.read_parquet(item_file)
        pos = pd.Series(np.arange(len(emb)), index=emb.item_id.to_numpy())
        rows = pos.reindex(item_ids).to_numpy()
        assert not np.isnan(rows).any(), f"[{encoder}] не для всех объявлений корпуса посчитаны эмбеддинги"
        E = np.stack(emb.emb.to_numpy())
        self.E = np.ascontiguousarray(E[rows.astype(np.int64)], dtype=np.float32)  # [N x d], L2-нормированы
        q = pd.read_parquet(query_file)
        self.q = dict(zip(q.query_id, q.emb))

    def scores(self, query_id: str) -> np.ndarray:
        """Косинусная близость запроса ко всем объявлениям корпуса."""
        return self.E @ np.asarray(self.q[query_id], dtype=np.float32)
