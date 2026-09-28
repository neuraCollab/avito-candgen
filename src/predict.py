"""Шаг 6. Предсказание для бенчмарка и запись answer.csv с проверкой формата."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from common import CACHE, ROOT, load_items_bench, load_queries_bench
from ranker import NON_FEATURES, load_final, predict_final, top50

K = 50


def main():
    models = load_final()
    feats = pd.read_parquet(CACHE / "feats_bench.parquet")
    assert set(models[0].feature_name()) <= set(feats.columns) - NON_FEATURES
    pred = top50(feats, predict_final(models, feats), k=K)

    q = load_queries_bench()
    corpus_ids = set(load_items_bench(["item_id"]).item_id)
    rows = []
    for qid in q.query_id:
        items = list(dict.fromkeys(pred.get(qid, [])))[:K]  # без повторов, не больше 50
        assert items, f"пустой ответ для {qid}"
        assert all(i in corpus_ids for i in items)
        rows.append((qid, " ".join(items)))
    ans = pd.DataFrame(rows, columns=["query_id", "answer"])

    # проверки из условия
    assert len(ans) == len(q) == ans.query_id.nunique()
    assert set(ans.query_id) == set(q.query_id)
    assert ans.query_id.str.len().eq(16).all()
    lens = ans.answer.str.split().str.len()
    assert lens.le(K).all()
    assert ans.answer.str.split().map(lambda x: all(len(i) == 16 and i == i.lower() for i in x)).all()

    out = Path(os.environ.get("AVITO_ANSWER", ROOT / "answer.csv"))
    ans.to_csv(out, index=False, encoding="utf-8")
    print(f"saved {out}: {len(ans)} rows, items per query: min {lens.min()}, mean {lens.mean():.1f}")


if __name__ == "__main__":
    main()
