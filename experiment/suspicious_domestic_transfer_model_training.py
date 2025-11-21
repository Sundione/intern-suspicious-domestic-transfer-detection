import logging
from experiment.training_pipeline import TrainingPipeline
from adapter.embedding.base import BaseGoogleEmbedding
from domain.suspicious_domestic_transfer_detection.sdtd_helper import SDTDHelper
from app.config import get_settings

_SETTINGS = get_settings()


def suspicious_domestic_transfer_model_training(
    train_data_path: str,
    test_data_path: str,
    target: str = "is_fraud",
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    texts_embedding_model = BaseGoogleEmbedding(
        embedding_api_key=_SETTINGS.GENAI_API_KEY,
        embedding_model=_SETTINGS.GOOGLE_EMBEDDING_MODEL,
    )
    helper = SDTDHelper(logger=logger)
    training_pipeline = TrainingPipeline(
        target=target,
        texts_embedding_model=texts_embedding_model,
        helper=helper,
    )
    training_pipeline.training_pipeline(train_data_path, test_data_path)


"""
Training Pipeline Example
suspicious_domestic_transfer_model_training(
    "train_pipeline_test.csv", "train_pipeline_testset.csv"
)
"""
