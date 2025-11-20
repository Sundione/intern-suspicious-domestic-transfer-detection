import joblib
import os
import pandas as pd
from typing import Optional
from adapter.embedding.base import BaseGoogleEmbedding
from experiment.suspicious_domestic_transfer_helper import SDTDHelper


class SuspiciousDomesticTransferDetector:
    SDTD_CLF_DIR = os.path.dirname(os.path.abspath(__file__))
    SDTD_CLF_MODEL_PATH = os.path.join(SDTD_CLF_DIR, "xgboost_model.joblib")
    HELPER = SDTDHelper()

    def __init__(
        self,
        embedding_api_key: str,
        embedding_model: Optional[str] = "gemini-embedding-001",
        prediction_threshold: float = 0.5,
    ):
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.prediction_threshold = prediction_threshold
        self.classification_model = self._load_classification_model()

    def _load_classification_model(self):
        classification_model = joblib.load(self.SDTD_CLF_MODEL_PATH)
        return classification_model

    def _embedding_data(self, data: pd.DataFrame):
        texts_embedding_model = BaseGoogleEmbedding(
            embedding_api_key=self.embedding_api_key,
            embedding_model=self.embedding_model,
        )

        texts = data["texts"].tolist()
        texts_embeded = texts_embedding_model.embedding(
            input_texts=texts,
        )

        return pd.DataFrame(
            texts_embeded,
            columns=[f"clf_dim_{i}" for i in range(128)],
        )

    def detect(self, data):
        preprocessed_data = self.HELPER.preprocess_data(data)
        print(f"Data preprocessed :\n{preprocessed_data}")

        embeded_data = self._embedding_data(preprocessed_data)
        print(f"Data Embeded :\n{embeded_data}")

        model_input = embeded_data[
            self.classification_model.get_booster().feature_names
        ]
        print(f"Model input :\n{model_input}")

        suspicious_score = float(
            self.classification_model.predict_proba(model_input)[:, 1][0]
        )
        print(f"Prediction with threshold : {self.prediction_threshold}")
        print(f"Prediction score calculated : {suspicious_score:.4f}")

        is_suspicious = suspicious_score >= self.prediction_threshold
        print(f"Prediction is_suspicious : {is_suspicious}")
        print(
            f"Prediction result = 'is_suspicious' : {is_suspicious}, 'suspicious_score': {suspicious_score:.4f}"
        )
        return is_suspicious, suspicious_score
