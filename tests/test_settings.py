from src.config.settings import Settings


def test_settings_load_defaults():
    settings = Settings(_env_file=None)
    assert settings.mlflow_experiment_name == "recomendador-ecommerce"
    assert settings.random_seed == 42
