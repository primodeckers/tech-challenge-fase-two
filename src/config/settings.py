from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações do projeto, lidas de variáveis de ambiente ou .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Por padrão, registra os experimentos localmente em SQLite, de modo que o
    # pipeline roda sem depender do servidor e já suporta o Model Registry. Para
    # usar o servidor central, aponte para http://localhost:5000 via ambiente.
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "recomendador-ecommerce"

    raw_data_path: Path = Path("data/raw/events.csv")
    processed_data_dir: Path = Path("data/processed")
    model_output_dir: Path = Path("models")

    random_seed: int = 42


settings = Settings()
