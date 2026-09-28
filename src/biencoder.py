"""Шаг 3b. Нейросетевой семантический поиск: bi-encoder'ы.

BM25 и kNN по запросам не видят синонимов, которых нет в train:
«мини кран» -> «автовышка», «трамбовочная машина» -> «виброплита».
Используются две open-source модели (обе работают локально на CPU):

  * cointegrated/rubert-tiny2 (29M параметров, 3 слоя) — ДООБУЧАЕТСЯ как bi-encoder на парах
    (текст запроса, текст выбранного объявления) из fit-части train -> энкодер «tiny2ft»;
  * sergeyzh/rubert-mini-frida (32M, 7 слоёв, дистиллят FRIDA) — сильная русская модель
    эмбеддингов; уже без дообучения ранжирует почти как дообученный tiny2 -> энкодер «frida0».
    Её общие знания дополняют «доменные» знания tiny2ft. Она же, дообученная так же, как tiny2,
    -> «fridaft». В финальном решении используются все три энкодера.

Дообучение:
  * loss — InfoNCE с in-batch негативами (MultipleNegativesRanking): для каждого
    запроса в батче правильное объявление должно быть ближе остальных 127;
    симметричный вариант (запрос->объявление и объявление->запрос);
  * в батче нет повторов текстов запросов и объявлений (иначе ложные негативы);
  * на один текст запроса берём не больше MAX_PER_TEXT пар: бенчмарк — это «хвост»,
    и модель не должна переучиться на «маникюр» и «массаж».

Модели обучаются только на fit (без dev-запросов и их объявлений), поэтому их
скоры на dev-фолдах честные, и те же модели используются для бенчмарка.

    python biencoder.py train tiny2|frida
    python biencoder.py embed tiny2ft|frida0|fridaft [f0 f1 ... bench]
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from common import CACHE, ROOT
from fitdata import load_fit

# Базовые модели: где лежат, как получать вектор текста, какие префиксы ждёт модель
BASES = {
    "tiny2": dict(path=ROOT / "models" / "rubert-tiny2", pool="cls", qp="", dp="", lr=1e-4),
    "frida": dict(path=ROOT / "models" / "rubert-mini-frida", pool="mean",
                  qp="search_query: ", dp="search_document: ", lr=5e-5),
}
# Энкодеры, эмбеддинги которых идут в признаки: (базовая модель, дообучена ли на fit)
ENCODERS = {
    "tiny2ft": ("tiny2", True),
    "frida0": ("frida", False),
    "fridaft": ("frida", True),
}
MAX_PER_TEXT = 30
BATCH = 128
EPOCHS = 1
Q_LEN, D_LEN = 24, 128
SCALE = 20.0  # 1 / temperature

torch.set_num_threads(8)


def tuned_path(base: str):
    # дообученная модель зависит от fit-части, поэтому лежит в кеше варианта
    return CACHE / f"{BASES[base]['path'].name}-avito"


def emb_files(encoder: str):
    """Файлы с эмбеддингами объявлений и запросов (у основного энкодера — исторические имена)."""
    if encoder == "tiny2ft":
        return CACHE / "item_emb.parquet", CACHE / "query_emb.parquet"
    return CACHE / f"item_emb_{encoder}.parquet", CACHE / f"query_emb_{encoder}.parquet"


def item_text(title: str, params: str, desc: str) -> str:
    """Текст объявления для модели: заголовок + начало параметров (вид/тип услуги) + начало описания."""
    params = params.split(" Место оказания услуг")[0][:150]  # адрес модели не нужен — локацию учитываем отдельно
    return f"{title} | {params} | {desc[:300]}"


def query_text(query: str, params: str) -> str:
    return f"{query} | {params}" if params else query


def pool_hidden(out, attention_mask, pool: str) -> torch.Tensor:
    """Вектор текста: CLS-токен (rubert-tiny2) или среднее по токенам (mini-frida), L2-нормированный."""
    h = out.last_hidden_state
    if pool == "mean":
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        h = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
    else:
        h = h[:, 0]
    return F.normalize(h, dim=-1)


def encode(model, tok, texts, max_len, pool, batch=256) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i : i + batch], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            out.append(pool_hidden(model(**enc), enc["attention_mask"], pool).numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, model.config.hidden_size), np.float32)


def make_batches(pairs: pd.DataFrame, rng) -> list[np.ndarray]:
    """Жадно раскладываем перемешанные пары по батчам без повторов запросов/объявлений."""
    order = rng.permutation(len(pairs))
    qs, its = pairs["qt"].to_numpy(), pairs["item_id"].to_numpy()
    batches, cur, seen_q, seen_i = [], [], set(), set()
    for j in order:
        if qs[j] in seen_q or its[j] in seen_i:
            continue
        cur.append(j)
        seen_q.add(qs[j])
        seen_i.add(its[j])
        if len(cur) == BATCH:
            batches.append(np.array(cur))
            cur, seen_q, seen_i = [], set(), set()
    return batches  # пропущенные из-за повторов пары и неполный батч отбрасываем — это <1% пар


def train(base: str = "tiny2"):
    cfg = BASES[base]
    rng = np.random.default_rng(0)
    torch.manual_seed(0)  # dropout при обучении — для воспроизводимости
    fit = load_fit(with_text=True)
    fit["qt"] = [query_text(q, p) for q, p in zip(fit.search_query, fit.search_infm_params_text)]
    pairs = fit.drop_duplicates(["qt", "item_id"])
    pairs = pairs.sample(frac=1.0, random_state=0).groupby("search_query").head(MAX_PER_TEXT)
    pairs = pairs.reset_index(drop=True)
    pairs["dt"] = [item_text(t, p, d) for t, p, d in
                   zip(pairs.item_title_raw, pairs.item_infm_params_text, pairs.item_description_raw)]
    print(f"[{base}] pairs: {len(pairs)} (unique query texts {pairs.search_query.nunique()})", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["path"])
    model = AutoModel.from_pretrained(cfg["path"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)

    all_batches = [b for _ in range(EPOCHS) for b in make_batches(pairs, rng)]
    total = len(all_batches)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / total))))
    print(f"steps: {total}", flush=True)

    qt = np.array([cfg["qp"] + t for t in pairs["qt"]], dtype=object)
    dt = np.array([cfg["dp"] + t for t in pairs["dt"]], dtype=object)
    model.train()
    t0, run = time.time(), 0.0
    for step, b in enumerate(all_batches):
        qe = tok(list(qt[b]), padding=True, truncation=True, max_length=Q_LEN, return_tensors="pt")
        de = tok(list(dt[b]), padding=True, truncation=True, max_length=D_LEN, return_tensors="pt")
        q = pool_hidden(model(**qe), qe["attention_mask"], cfg["pool"])
        d = pool_hidden(model(**de), de["attention_mask"], cfg["pool"])
        logits = SCALE * q @ d.T
        labels = torch.arange(len(b))
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 100 == 0:
            el = time.time() - t0
            print(f"step {step}/{total} loss {run:.3f}  {el:.0f}s, eta {el / (step + 1) * (total - step):.0f}s", flush=True)

    out = tuned_path(base)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print("saved", out)


def embed_corpora(encoder: str, modes):
    """Эмбеддинги объявлений всех корпусов (кешируются по item_id) и всех запросов."""
    base, tuned = ENCODERS[encoder]
    cfg = BASES[base]
    path = tuned_path(base) if tuned else cfg["path"]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path)
    item_file, query_file = emb_files(encoder)

    done = pd.read_parquet(item_file) if item_file.exists() else pd.DataFrame(columns=["item_id", "emb"])
    have = set(done.item_id)
    new_rows = []
    for mode in modes:
        c = pd.read_parquet(CACHE / f"corpus_{mode}.parquet",
                            columns=["item_id", "item_title_raw", "item_infm_params_text", "item_description_raw"])
        c = c[~c.item_id.isin(have)].drop_duplicates("item_id")
        if len(c):
            t0 = time.time()
            texts = [cfg["dp"] + item_text(t, p or "", d or "") for t, p, d in
                     zip(c.item_title_raw, c.item_infm_params_text, c.item_description_raw)]
            E = encode(model, tok, texts, D_LEN, cfg["pool"])
            new_rows.append(pd.DataFrame({"item_id": c.item_id.to_numpy(), "emb": list(E)}))
            have |= set(c.item_id)
            print(f"[{encoder}] {mode}: encoded {len(c)} items in {time.time() - t0:.0f}s", flush=True)
    if new_rows:
        done = pd.concat([done] + new_rows, ignore_index=True)
        done.to_parquet(item_file)

    # запросы: dev + бенчмарк
    from fitdata import load_mode
    qs = pd.concat([pd.read_parquet(CACHE / "dev_queries.parquet"), load_mode("bench")[0]], ignore_index=True)
    E = encode(model, tok, [cfg["qp"] + query_text(q, p) for q, p in zip(qs.search_query, qs.search_infm_params_text)],
               Q_LEN, cfg["pool"])
    pd.DataFrame({"query_id": qs.query_id.to_numpy(), "emb": list(E)}).to_parquet(query_file)
    print(f"[{encoder}] queries encoded: {len(qs)}")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2] if len(sys.argv) > 2 else "tiny2")
    else:
        embed_corpora(sys.argv[2], sys.argv[3:] or ["f0", "f1", "f2", "f3", "f4", "bench"])
