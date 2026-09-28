"""Загрузка моделей в models/ (один раз; дальше всё работает офлайн).

    python src/fetch_models.py         # models.zip из релиза репозитория: всё, что нужно для run_all.py
    python src/fetch_models.py --base  # + базовая cointegrated/rubert-tiny2 (нужна только для --retrain)

models.zip содержит:
  * rubert-mini-frida        — базовая sergeyzh/rubert-mini-frida (HuggingFace, MIT), ревизия FRIDA_REV;
  * rubert-tiny2-avito       — cointegrated/rubert-tiny2, дообученная на fit-части train (biencoder.py);
  * rubert-mini-frida-avito  — rubert-mini-frida, дообученная на fit-части train (biencoder.py).
Именно этими весами получен отправленный answer.csv.
"""
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RELEASE_URL = "https://github.com/neuraCollab/avito-candgen/releases/download/v1.0/models.zip"
NEEDED = ["rubert-mini-frida", "rubert-tiny2-avito", "rubert-mini-frida-avito"]
# ревизии HuggingFace, с которыми получен результат (модели на хабе могут обновиться)
TINY2_REV = "e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae"
FRIDA_REV = "32a82388a9f99c99eca74494731f7ee6e9a9fe3a"


def fetch_release():
    if all((MODELS / m / "model.safetensors").exists() for m in NEEDED):
        print("models/: всё уже на месте")
        return
    MODELS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "models.zip"
        print("скачиваю", RELEASE_URL)
        with urllib.request.urlopen(RELEASE_URL) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(MODELS)
    print("распаковано в", MODELS, ":", ", ".join(NEEDED))


def fetch_base_tiny2():
    from huggingface_hub import snapshot_download
    snapshot_download("cointegrated/rubert-tiny2", revision=TINY2_REV, local_dir=MODELS / "rubert-tiny2",
                      allow_patterns=["*.json", "*.safetensors", "*.txt"])
    print("models/rubert-tiny2: готово")


if __name__ == "__main__":
    fetch_release()
    if "--base" in sys.argv:
        fetch_base_tiny2()
