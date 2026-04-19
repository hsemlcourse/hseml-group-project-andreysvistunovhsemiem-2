"""Загрузка и первичное чтение датасета UCI Credit Card Default (Taiwan)."""
import shutil
from pathlib import Path

import pandas as pd

from src.config import settings


def download_data(force: bool = False) -> Path:
    """Скачивает датасет c Kaggle через kagglehub и кладёт CSV в data/raw/.

    Для работы требует установленный kagglehub и валидные Kaggle credentials
    (через ~/.kaggle/kaggle.json или переменные KAGGLE_USERNAME/KAGGLE_KEY).
    """
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    target = settings.raw_csv
    if target.exists() and not force:
        return target

    import kagglehub  # локальный импорт — не все сценарии требуют сеть

    path = Path(kagglehub.dataset_download(settings.kaggle_dataset))
    candidates = list(path.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"CSV не найден в {path}")
    shutil.copyfile(candidates[0], target)
    return target


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Читает исходный CSV и приводит имена колонок к единому виду."""
    path = path or settings.raw_csv
    df = pd.read_csv(path)
    rename_map = {
        "default.payment.next.month": settings.target_col,
        "default payment next month": settings.target_col,
        "PAY_0": "PAY_1",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


if __name__ == "__main__":
    p = download_data()
    print(f"Saved: {p}")
    print(load_raw(p).head())
