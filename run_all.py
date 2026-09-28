"""Полный воспроизводимый прогон решения: от parquet-файлов до answer.csv.

    python run_all.py            # воспроизвести отправленный answer.csv (дообученные веса из models/)
    python run_all.py --retrain  # всё с нуля, включая дообучение обоих bi-encoder'ов (~7 ч на CPU)
    python run_all.py --twins    # экспериментальный вариант валидации «двойники» (с дообучением)

Перед запуском (см. README):
  * data/train.parquet, data/benchmark_queries.parquet, data/benchmark_items.parquet;
  * models/ — python src/fetch_models.py: базовая sergeyzh/rubert-mini-frida и две дообученные нами
    модели (rubert-tiny2-avito, rubert-mini-frida-avito) из релиза репозитория.
    Для --retrain нужна ещё базовая cointegrated/rubert-tiny2 (python src/fetch_models.py --base).

Дообучение нейросетей на CPU не воспроизводится бит в бит (порядок операций в многопоточных
вычислениях), поэтому основной режим берёт веса, которыми был получен отправленный ответ.
Всё остальное (фолды, индексы, эмбеддинги, кандидаты, ранкер) пересчитывается с нуля.
Время основного режима: ~2 часа на 8-ядерном CPU. Внешних API не вызывает.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
FOLDS = ["f0", "f1", "f2", "f3", "f4"]
# энкодеры (см. src/biencoder.py): дообученный rubert-tiny2, rubert-mini-frida без дообучения и дообученная
ENCODERS = ["tiny2ft", "frida0", "fridaft"]
FINETUNED = ["rubert-tiny2-avito", "rubert-mini-frida-avito"]


def run(cache: str, *args):
    print(">>", " ".join(args), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", AVITO_CACHE=cache)
    subprocess.run([sys.executable, *args], cwd=SRC, check=True, env=env)


def use_shipped_weights(cache: str):
    """Кладём выложенные дообученные веса туда, где их ищет biencoder.py (каталог кеша)."""
    for name in FINETUNED:
        src, dst = ROOT / "models" / name, ROOT / cache / name
        if not (src / "model.safetensors").exists():
            sys.exit(f"нет {src}: python src/fetch_models.py (или запустите с --retrain)")
        if not dst.exists():
            shutil.copytree(src, dst)


def pipeline(cache: str, twins: bool, retrain: bool):
    run(cache, "build_dev.py", *(["--twins"] if twins else []))  # 1. dev-фолды, fit-часть train, корпуса
    run(cache, "text_index.py", *FOLDS, "bench")                  # 2. BM25-индексы
    if retrain:
        run(cache, "biencoder.py", "train", "tiny2")              # 3. дообучение rubert-tiny2 на fit (~1.5 ч)
        run(cache, "biencoder.py", "train", "frida")              #    дообучение rubert-mini-frida на fit (~3.5 ч)
    else:
        (ROOT / cache).mkdir(exist_ok=True)
        use_shipped_weights(cache)
    for enc in ENCODERS:
        run(cache, "biencoder.py", "embed", enc, *FOLDS, "bench") #    эмбеддинги объявлений и запросов
    for mode in FOLDS + ["bench"]:
        run(cache, "candidates.py", mode)                         # 4. пулы кандидатов + признаки
    run(cache, "ranker.py", "--final")                            # 5. валидация ранкера на f0 + финальная модель
    run(cache, "predict.py")                                      # 6. answer.csv


def main():
    twins = "--twins" in sys.argv
    # каталог артефактов можно переопределить (AVITO_CACHE=...), чтобы не трогать существующий
    cache = os.environ.get("AVITO_CACHE") or ("cache_twins" if twins else "cache")
    pipeline(cache, twins=twins, retrain=twins or "--retrain" in sys.argv)


if __name__ == "__main__":
    main()
