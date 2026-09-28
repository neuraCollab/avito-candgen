"""Шаг 3. «Память» train: похожие запросы из истории и что по ним выбирали.

Бенчмарк состоит в основном из редких запросов, а объявления в нём новые, поэтому
запоминать пары (запрос, item_id) бесполезно. Но train хорошо обобщается на уровне
*смысла*: по похожим запросам пользователи выбирали объявления определённых
подкатегорий (microcat) и с определёнными словами в заголовке.

Модель — kNN по текстам запросов:
  * каждый уникальный текст запроса из fit — «документ» с векторами
      - TF-IDF по стемам слов + TF-IDF по символьным n-граммам (устойчивость к
        опечаткам и словообразованию: «видеодомофонов» ~ «домофон»);
  * для нового запроса находим top-K соседей по косинусу и агрегируем:
      - распределение подкатегорий объявлений, выбранных по соседям -> P(microcat | q);
      - стемы заголовков выбранных объявлений -> расширение запроса
        (так «трамбовочная машина» получает слово «виброплита»).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2norm

from common import normalize, tokenize


def _clean(s: str) -> str:
    return " ".join(normalize(s).split())


class QueryKNN:
    def __init__(self, k: int = 30, power: float = 2.0):
        self.k = k
        self.power = power  # sim**power: ближайшие соседи весят заметно больше

    def fit(self, fit_df: pd.DataFrame, vocab: dict):
        """fit_df: строки train (fit), vocab: словарь стемов текстового индекса корпуса."""
        df = fit_df[["search_query", "item_microcat_id", "item_title_raw"]].copy()
        df["qn"] = df["search_query"].map(_clean)
        texts = df["qn"].unique()
        self.texts = texts
        tid = pd.Series(np.arange(len(texts)), index=texts)
        df["tid"] = df["qn"].map(tid).values

        # векторизаторы запросов: слова (стемы) + символьные n-граммы
        self.wv = TfidfVectorizer(analyzer=tokenize, sublinear_tf=True, min_df=1)
        self.cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2,
                                  max_features=300_000)
        self.X = self._vec(texts, fit=True)

        # распределение microcat по каждому тексту (доли выбранных объявлений)
        self.mc_codes, mc_idx = np.unique(df["item_microcat_id"].values, return_inverse=True)
        M = sp.csr_matrix((np.ones(len(df), np.float32), (df["tid"].values, mc_idx)),
                          shape=(len(texts), len(self.mc_codes)))
        self.M = l2norm(M, norm="l1")
        # насколько «уверенно» текст задан: число строк (для взвешивания соседей)
        self.cnt = np.asarray(M.sum(axis=1)).ravel()

        # мешок стемов заголовков выбранных объявлений, в терминах словаря корпуса
        rows, cols = [], []
        for t, title in zip(df["tid"].values, df["item_title_raw"].values):
            for s in set(tokenize(title)):
                j = vocab.get(s)
                if j is not None:
                    rows.append(t)
                    cols.append(j)
        H = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(texts), len(vocab)))
        self.H = l2norm(H, norm="l1")  # доля объявлений с данным словом в заголовке
        return self

    def _vec(self, texts, fit=False):
        if fit:
            a = self.wv.fit_transform(texts)
            b = self.cv.fit_transform(texts)
        else:
            a = self.wv.transform(texts)
            b = self.cv.transform(texts)
        # слова важнее символьных n-грамм, но n-граммы спасают незнакомые словоформы
        return l2norm(sp.hstack([a * 0.7, b * 0.3]).tocsr())

    def query(self, queries: list[str], batch: int = 512):
        """-> (P_mc: [n x n_mc] csr, H_exp: [n x V] csr, top_sim: [n])."""
        qn = [_clean(s) for s in queries]
        Q = self._vec(qn)
        P_out, H_out, top_sim = [], [], []
        for i in range(0, len(qn), batch):
            S = (Q[i : i + batch] @ self.X.T).toarray()
            k = min(self.k, S.shape[1] - 1)
            nb = np.argpartition(-S, k, axis=1)[:, :k]
            sims = np.take_along_axis(S, nb, axis=1)
            top_sim.append(sims.max(axis=1))
            w = np.clip(sims, 0, None) ** self.power
            # сосед с одной строкой — шумный; сглаживаем вес по числу наблюдений
            w = w * (self.cnt[nb] / (self.cnt[nb] + 1.0))
            w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
            r = np.repeat(np.arange(nb.shape[0]), k)
            Wm = sp.csr_matrix((w.ravel(), (r, nb.ravel())), shape=(nb.shape[0], self.X.shape[0]))
            P_out.append(Wm @ self.M)
            H_out.append(Wm @ self.H)
        return sp.vstack(P_out).tocsr(), sp.vstack(H_out).tocsr(), np.concatenate(top_sim)
