import pandas as pd
from domain.suspicious_domestic_transfer_detection.suspicious_domestic_transfer_detector import (
    SuspiciousDomesticTransferDetector,
)
from app.config import get_settings


_SETTINGS = get_settings()


def apply_suspicious_domestic_transfer_detection(transaction: pd.DataFrame):
    suspicious_domestic_transfer_detector = SuspiciousDomesticTransferDetector(
        prediction_threshold=_SETTINGS.SDTD_CLF_PREDICTION_THRESHOLD,
    )
    is_suspicious, suspicious_score = suspicious_domestic_transfer_detector.detect(
        transaction
    )
    return_data = {"is_suspicious": is_suspicious, "suspicious_score": suspicious_score}
    return return_data


"""
transaction = pd.read_csv(
    "D:/N/T2P/Tag29/intern-suspicious-domestic-transfer-detection/domain/usecase/test_profile_fraud.csv"
)

transaction = pd.read_csv(
    "D:/N/T2P/Tag29/intern-suspicious-domestic-transfer-detection/domain/usecase/test_profile_normal.csv"
)
apply_suspicious_domestic_transfer_detection(transaction)
"""
