"""Конфигурация проекта на базе pydantic-settings.

Значения берутся из переменных окружения (или .env в корне проекта);
при отсутствии используются дефолты. Экспортируется singleton `settings`.
"""
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    random_state: int = Field(default=42, description="Seed для воспроизводимости")

    project_root: Path = Field(default=PROJECT_ROOT)
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    models_dir: Path = Field(default=PROJECT_ROOT / "models")
    report_dir: Path = Field(default=PROJECT_ROOT / "report")

    raw_csv_name: str = Field(default="UCI_Credit_Card.csv")
    target_col: str = Field(default="target")

    kaggle_dataset: str = Field(default="uciml/default-of-credit-card-clients-dataset")
    kaggle_username: str | None = Field(default=None)
    kaggle_key: str | None = Field(default=None)

    test_size: float = Field(default=0.2)
    cv_splits: int = Field(default=5)

    @computed_field  # type: ignore[misc]
    @property
    def raw_csv(self) -> Path:
        return self.raw_dir / self.raw_csv_name

    @computed_field  # type: ignore[misc]
    @property
    def images_dir(self) -> Path:
        return self.report_dir / "images"


settings = Settings()
