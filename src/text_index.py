"""Шаг 2. Разреженный текстовый индекс корпуса для BM25.

Для каждого поля объявления (заголовок, параметры, описание) строим матрицу
«документ x стем» с BM25-весами. Тогда BM25-скор запроса по всему корпусу —
это сумма нескольких столбцов матрицы (по одному на слово запроса), что
считается за миллисекунды без внешних поисковых движков.

BM25 (Robertson & Zaragoza) реализован вручную на scipy.sparse — формула стандартная:
  w(t, d) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
  idf(t)  = ln(1 + (N - df + 0.5) / (df + 0.5))
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from common import CACHE, tokenize

FIELDS = {
    # поле -> (колонка, k1, b)
    "title": ("item_title_raw", 1.2, 0.5),   # короткие заголовки: слабая нормализация длины
    "params": ("item_infm_params_text", 1.2, 0.75),
    "desc": ("item_description_raw", 1.2, 0.75),
}


class Vocab:
    """Общий словарь стемов для всех полей и запросов."""

    def __init__(self):
        self.idx: dict[str, int] = {}

    def ids(self, toks, grow=True):
        out = []
        for t in toks:
            i = self.idx.get(t)
            if i is None:
                if not grow:
                    continue
                i = self.idx[t] = len(self.idx)
            out.append(i)
        return out


def tf_matrix(texts, vocab: Vocab, max_tokens=None) -> sp.csr_matrix:
    """Список текстов -> CSR-матрица частот стемов."""
    indptr, indices = [0], []
    for s in texts:
        toks = tokenize(s)
        if max_tokens:
            toks = toks[:max_tokens]
        indices.extend(vocab.ids(toks))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float32)
    m = sp.csr_matrix((data, np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int64)),
                      shape=(len(texts), 0 if not vocab.idx else len(vocab.idx)))
    m.sum_duplicates()  # одинаковые (doc, term) -> частота
    return m


def bm25_weights(tf: sp.csr_matrix, k1: float, b: float) -> tuple[sp.csc_matrix, np.ndarray]:
    """TF-матрица -> матрица BM25-весов (CSC: быстрые срезы по столбцам-словам) и idf."""
    tf = tf.tocsr().astype(np.float32)
    n = tf.shape[0]
    dl = np.asarray(tf.sum(axis=1)).ravel()
    avgdl = max(dl.mean(), 1e-9)
    df = np.bincount(tf.indices, minlength=tf.shape[1])
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    rows = np.repeat(np.arange(n), np.diff(tf.indptr))
    norm = k1 * (1 - b + b * dl[rows] / avgdl)
    w = tf.data * (k1 + 1) / (tf.data + norm) * idf[tf.indices]
    out = sp.csr_matrix((w.astype(np.float32), tf.indices, tf.indptr), shape=tf.shape)
    return out.tocsc(), idf


def build(corpus_name: str):
    t0 = time.time()
    corpus = pd.read_parquet(CACHE / f"corpus_{corpus_name}.parquet",
                             columns=["item_id"] + [c for c, _, _ in FIELDS.values()])
    vocab = Vocab()
    tfs = {}
    for f, (col, _, _) in FIELDS.items():
        # описание обрезаем до 400 токенов: хвосты длинных описаний — в основном шум/SEO
        tfs[f] = tf_matrix(corpus[col].fillna("").tolist(), vocab, max_tokens=400 if f == "desc" else None)
        print(f"  tokenized {f}: {time.time() - t0:.0f}s, vocab={len(vocab.idx)}", flush=True)
    V = len(vocab.idx)
    index = {"item_id": corpus["item_id"].to_numpy(), "vocab": vocab.idx}
    for f, (_, k1, b) in FIELDS.items():
        tf = tfs[f]
        tf.resize((tf.shape[0], V))
        W, idf = bm25_weights(tf, k1, b)
        index[f] = W
        index[f + "_idf"] = idf
    with open(CACHE / f"index_{corpus_name}.pkl", "wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"index {corpus_name}: {len(corpus)} docs, vocab {V}, {time.time() - t0:.0f}s")


def load(corpus_name: str) -> dict:
    # pickle безопасен здесь: файл — локальный кеш, который создаёт build() выше
    with open(CACHE / f"index_{corpus_name}.pkl", "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    for name in sys.argv[1:] or ["dev", "bench"]:
        build(name)
