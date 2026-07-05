from src.config.settings import settings
from src.data.loader import load_data
from src.data.preprocess import preprocess_data
from src.features.encoding import IdEncoder
from src.models.factory import ModelFactory
from src.training.trainer import ModelTrainer


def treinar() -> None:
    """Executa o pipeline de treino: carrega, pré-processa e treina o modelo."""
    raw_data = load_data(settings.raw_data_path)
    processed_data = preprocess_data(raw_data)

    user_encoder = IdEncoder().fit(processed_data["user_id"])
    item_encoder = IdEncoder().fit(processed_data["item_id"])
    processed_data["user_idx"] = user_encoder.transform(processed_data["user_id"])
    processed_data["item_idx"] = item_encoder.transform(processed_data["item_id"])

    model = ModelFactory.create("mlp", n_users=user_encoder.n_unique, n_items=item_encoder.n_unique)
    ModelTrainer(model).train(processed_data)
    print("Treino concluído.")
