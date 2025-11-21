import joblib
import os
import pandas as pd
import logging
from typing import Optional
from adapter.embedding.base import BaseGoogleEmbedding
from domain.suspicious_domestic_transfer_detection.sdtd_helper import SDTDHelper


class SuspiciousDomesticTransferDetector:
    SDTD_CLF_DIR = os.path.dirname(os.path.abspath(__file__))
    SDTD_CLF_MODEL_PATH = os.path.join(SDTD_CLF_DIR, "xgboost_model.joblib")

    def __init__(
        self,
        texts_embedding_model: BaseGoogleEmbedding,
        helper: SDTDHelper,
        prediction_threshold: float = 0.5,
        logger: Optional[logging.Logger] = None,
    ):
        self.texts_embedding_model = texts_embedding_model
        self.helper = helper
        self.prediction_threshold = prediction_threshold
        self.classification_model = self._load_classification_model()
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def _load_classification_model(self):
        classification_model = joblib.load(self.SDTD_CLF_MODEL_PATH)
        return classification_model

    def _embedding_data(self, data: pd.DataFrame):
        texts = data["texts"].tolist()
        texts_embeded = self.texts_embedding_model.embedding(
            input_texts=texts,
        )

        return pd.DataFrame(
            texts_embeded,
            columns=[f"clf_dim_{i}" for i in range(128)],
        )

    def detect(self, data):
        preprocessed_data = self.helper.preprocess_data(data)
        self.logger.info(f"Data preprocessed :\n{preprocessed_data}")

        embeded_data = self._embedding_data(preprocessed_data)
        self.logger.info(f"Data Embeded :\n{embeded_data}")

        model_input = embeded_data[
            self.classification_model.get_booster().feature_names
        ]
        self.logger.info(f"Model input :\n{model_input}")

        suspicious_score = float(
            self.classification_model.predict_proba(model_input)[:, 1][0]
        )
        self.logger.info(f"Prediction with threshold : {self.prediction_threshold}")
        self.logger.info(f"Prediction score calculated : {suspicious_score:.4f}")

        is_suspicious = suspicious_score >= self.prediction_threshold
        self.logger.info(f"Prediction is_suspicious : {is_suspicious}")
        self.logger.info(
            f"Prediction result = 'is_suspicious' : {is_suspicious}, 'suspicious_score': {suspicious_score:.4f}"
        )
        return is_suspicious, suspicious_score
