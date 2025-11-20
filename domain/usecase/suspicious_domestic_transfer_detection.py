import pandas as pd
from domain.suspicious_domestic_transfer_detection.suspicious_domestic_transfer_detector import (
    SuspiciousDomesticTransferDetector,
)
from app.config import get_settings


_SETTINGS = get_settings()


def apply_suspicious_domestic_transfer_detection(transaction: pd.DataFrame):
    suspicious_domestic_transfer_detector = SuspiciousDomesticTransferDetector(
        embedding_api_key=_SETTINGS.GEMINI_EMBEDDING_API_KEY,
        embedding_model=_SETTINGS.GEMINI_EMBEDDING_MODEL,
        prediction_threshold=_SETTINGS.SDTD_CLF_PREDICTION_THRESHOLD,
    )
    is_suspicious, suspicious_score = suspicious_domestic_transfer_detector.detect(
        transaction
    )
    return_data = {"is_suspicious": is_suspicious, "suspicious_score": suspicious_score}
    return return_data
