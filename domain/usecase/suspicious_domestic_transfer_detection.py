import pandas as pd
import logging
from domain.suspicious_domestic_transfer_detection.suspicious_domestic_transfer_detector import (
    SuspiciousDomesticTransferDetector,
)
from adapter.embedding.base import BaseGoogleEmbedding
from domain.suspicious_domestic_transfer_detection.sdtd_helper import SDTDHelper
from app.config import get_settings


_SETTINGS = get_settings()


def apply_suspicious_domestic_transfer_detection(transaction: pd.DataFrame):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    texts_embedding_model = BaseGoogleEmbedding(
        embedding_api_key=_SETTINGS.GENAI_API_KEY,
        embedding_model=_SETTINGS.GOOGLE_EMBEDDING_MODEL,
    )
    helper = SDTDHelper(logger=logger)
    suspicious_domestic_transfer_detector = SuspiciousDomesticTransferDetector(
        texts_embedding_model=texts_embedding_model,
        helper=helper,
        prediction_threshold=_SETTINGS.SDTD_CLF_PREDICTION_THRESHOLD,
        logger=logger,
    )
    is_suspicious, suspicious_score = suspicious_domestic_transfer_detector.detect(
        transaction
    )
    return_data = {"is_suspicious": is_suspicious, "suspicious_score": suspicious_score}
    return return_data


"""
Input Example for Detection.
transaction = pd.read_csv(
    "predict_transaction.csv"
)
apply_suspicious_domestic_transfer_detection(transaction)
"""
