from experiment.training_pipeline import TrainingPipeline
from app.config import get_settings

_SETTINGS = get_settings()


def suspicious_domestic_transfer_model_training(
    train_data_path: str,
    test_data_path: str,
    target: str = "is_fraud",
):
    training_pipeline = TrainingPipeline(
        target=target,
        embedding_api_key=_SETTINGS.GEMINI_EMBEDDING_API_KEY,
        embedding_model=_SETTINGS.GEMINI_EMBEDDING_MODEL,
        threshold=_SETTINGS.SDTD_CLF_PREDICTION_THRESHOLD,
    )
    training_pipeline.training_pipeline(train_data_path, test_data_path)
