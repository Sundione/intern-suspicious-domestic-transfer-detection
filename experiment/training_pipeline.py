import os
import pandas as pd
import numpy as np
import joblib
from adapter.embedding.base import BaseGoogleEmbedding
from domain.suspicious_domestic_transfer_detection.sdtd_helper import SDTDHelper
from experiment.training_schema import TRAIN_SCHEMA
from xgboost import XGBClassifier


class TrainingPipeline:
    SDTD_CLF_DIR = os.path.dirname(os.path.abspath(__file__))
    SDTD_CLF_MODEL_PATH = os.path.join(SDTD_CLF_DIR, "xgboost_model.joblib")

    def __init__(
        self,
        texts_embedding_model: BaseGoogleEmbedding,
        helper: SDTDHelper,
        threshold: float = 0.5,
        target: str = "is_fraud",
    ):
        self.target = target
        self.texts_embedding_model = texts_embedding_model
        self.helper = helper
        self.threshold = threshold

    def _load_data(self, path: str = None):
        data = pd.read_csv(path)
        TRAIN_SCHEMA.validate(data)
        return data

    def _embedding_data(self, data: pd.DataFrame):
        # if embedding have limit, need to batch process
        texts = data["texts"].tolist()
        texts_embedding = self.texts_embedding_model.embedding(
            input_texts=texts,
        )

        texts_embeded = pd.DataFrame(
            texts_embedding,
            columns=[f"clf_dim_{i}" for i in range(128)],
            index=data.index,
        )

        if "profile_ref" in data.columns:
            id_col = "profile_ref"
        else:
            id_col = "profile_id"

        texts_embeded.insert(0, id_col, data[id_col].values)
        texts_embeded.insert(1, self.target, data[self.target].values)

        return texts_embeded

    def _training_model(self, data: pd.DataFrame):
        X = data.filter(regex="^clf_dim_")
        y = data[self.target]
        model = XGBClassifier()
        model.fit(X, y)

        return model

    def _calculate_metrics(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(np.unique(y_true)) < 2:
            raise ValueError("Test set has only 1 class.")

        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))

        print("Test set Evaluate")
        accuracy = (tp + tn) / (tn + fp + fn + tp) if (tn + fp + fn + tp) > 0 else 0.0
        print(f"Accuracy : {accuracy}")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        print(f"Precision : {precision}")
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"Recall : {recall}")
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        print(f"F1-score : {f1}")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        print(f"FPR : {fpr}")
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        print(f"FNR : {fnr}")

    def _evaluate_model(self, data: pd.DataFrame, model):
        X = data.filter(regex="^clf_dim_")
        y = data[self.target]
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)
        self._calculate_metrics(y, y_pred)

    def training_pipeline(self, train_data_path: str, test_data_path: str):
        train_data = self._load_data(train_data_path)
        print("Loaded Train Data")
        processed_train_data = self.helper.preprocess_data(
            train_data, split_domestic_iteration=True
        )
        print("Processed Train Data (with snapshots)")
        embeded_train_data = self._embedding_data(processed_train_data)
        print("Train Data Embeded\n")
        model = self._training_model(embeded_train_data)

        test_data = self._load_data(test_data_path)
        print("Loaded Test Data")
        processed_test_data = self.helper.preprocess_data(test_data)
        print("Processed Test Data")
        embeded_test_data = self._embedding_data(processed_test_data)
        print("Test Data Embeded\n")
        print(f"Test with threshold : {self.threshold}")
        self._evaluate_model(embeded_test_data, model)

        joblib.dump(model, self.SDTD_CLF_MODEL_PATH)
