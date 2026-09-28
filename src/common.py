"""Общие утилиты: пути, загрузка данных, нормализация текста, метрика.

Всё, что используется в нескольких скриптах пайплайна, собрано здесь,
чтобы шаги (построение валидации, индексация, генерация кандидатов, ранкер)
работали с одинаковой предобработкой.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import snowballstemmer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
# Все промежуточные артефакты (фолды, индексы, признаки, дообученная модель) зависят от того,
# как построена валидация, поэтому варианты живут в разных каталогах: AVITO_CACHE=cache_twins
CACHE = ROOT / os.environ.get("AVITO_CACHE", "cache")
CACHE.mkdir(exist_ok=True)

# Колонки, которые однозначно задают «поисковый запрос» в train.
# В train нет query_id, поэтому запросом считаем уникальную комбинацию признаков поиска.
QUERY_KEY = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

ITEM_COLS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_category_id",
    "item_microcat_id",
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_location_id",
    "item_latitude",
    "item_longitude",
    "item_is_phone_hidden",
    "item_is_message_forbidden",
]


def fix_item_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """decimal128 из parquet -> float64 (иначе медленная арифметика в pandas)."""
    for c in ("item_price", "item_latitude", "item_longitude"):
        if c in df.columns:
            df[c] = df[c].astype("float64")
    for c in ("item_title_raw", "item_description_raw", "item_infm_params_text"):
        if c in df.columns:
            df[c] = df[c].fillna("")
    return df


def load_queries_bench() -> pd.DataFrame:
    q = pd.read_parquet(DATA / "benchmark_queries.parquet")
    q["search_infm_params_text"] = q["search_infm_params_text"].fillna("")
    return q


def load_items_bench(columns=None) -> pd.DataFrame:
    it = pd.read_parquet(DATA / "benchmark_items.parquet", columns=columns)
    return fix_item_dtypes(it)


# ---------------------------------------------------------------- текст ----
# Snowball-стеммер (open-source, пакет snowballstemmer) для русского и английского:
# приводит словоформы к общей основе: «телевизоров» -> «телевизор», «монтажа» -> «монтаж».
_stem_ru = snowballstemmer.stemmer("russian")
_stem_en = snowballstemmer.stemmer("english")
_TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
_CYR_RE = re.compile(r"[а-я]")


@lru_cache(maxsize=2_000_000)
def stem(tok: str) -> str:
    """Стемминг одного токена с кешем: словарь корпуса ограничен, кеш даёт x10 ускорение."""
    if _CYR_RE.search(tok):
        return _stem_ru.stemWord(tok)
    if tok.isdigit():
        return tok
    return _stem_en.stemWord(tok)


def normalize(text: str) -> str:
    """Нижний регистр + ё->е: «ё» в объявлениях и запросах пишут непоследовательно."""
    return text.lower().replace("ё", "е")


def tokenize(text: str) -> list[str]:
    """Текст -> список стемов (без стоп-слов: их вес и так подавит IDF в BM25)."""
    return [stem(t) for t in _TOKEN_RE.findall(normalize(text))]


# ---------------------------------------------------------------- метрика ---
def recall_at_k(pred: dict, truth: dict, k: int = 50) -> float:
    """Recall@k как в условии: доля релевантных в top-k, усреднённая по запросам.

    pred:  query_id -> список item_id (порядок важен только для отсечения по k)
    truth: query_id -> множество релевантных item_id
    """
    scores = []
    for qid, rel in truth.items():
        top = set(pred.get(qid, [])[:k])
        scores.append(len(top & rel) / len(rel))
    return float(np.mean(scores))
