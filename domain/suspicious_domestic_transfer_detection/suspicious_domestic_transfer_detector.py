import joblib
import os
import pandas as pd
from adapter.embedding.base import BaseGoogleEmbedding
from const import tx_type_const


class SuspiciousDomesticTransferDetector:
    SDTD_CLF_DIR = os.path.dirname(os.path.abspath(__file__))
    SDTD_CLF_MODEL_PATH = os.path.join(SDTD_CLF_DIR, "xgboost_model.joblib")

    def __init__(
        self,
        prediction_threshold: float = 0.5,
    ):
        self.prediction_threshold = prediction_threshold
        self.classification_model = self._load_classification_model()

    def _load_classification_model(self):
        classification_model = joblib.load(self.SDTD_CLF_MODEL_PATH)
        return classification_model

    def _transaction_to_text(self, row):
        datetime = pd.to_datetime(row["dtcreate"]).strftime("%d %b %Y %H:%M")
        amount = row["ncashdiff"]
        description = row["vdescription"]
        type = row["vcodetype"]
        sub_type = row["sub_type"]
        remain_balance = row["nendingbalance"]
        if type == tx_type_const.TX_TYPE_CH:
            if sub_type == tx_type_const.TRANS_TYPE_TOPUP:
                return f"On {datetime}, the user Top-up {amount} baht, balance {remain_balance} baht."
            else:
                return f"On {datetime}, the user received {amount} baht via {description}, balance {remain_balance} baht."
        elif (sub_type == tx_type_const.TRANS_TYPE_WITHDRAW) or (
            (sub_type == tx_type_const.TRANS_TYPE_QR_TRANSFER)
        ):
            return f"On {datetime}, the user made a domestic transfer {amount} baht, balance {remain_balance} baht."
        elif sub_type == tx_type_const.TRANS_TYPE_PAYMENT:
            return f"On {datetime}, the user made a payment {amount} baht with {description}, balance {remain_balance} baht."
        elif sub_type == tx_type_const.TRANS_TYPE_TRANSFER:
            return f"On {datetime}, the user made an international money transfer ({description}) {amount} baht, balance {remain_balance} baht."
        elif (
            sub_type == tx_type_const.TRANS_TYPE_INTERNAL_TRANSFER
            or sub_type == tx_type_const.TRANS_TYPE_RECEIVE_TRANSFER
        ):
            return f"On {datetime}, the user {description} {amount} baht, balance {remain_balance} baht."
        elif type == tx_type_const.TX_TYPE_CTU:
            return f"On {datetime}, the user made a cancel {amount} baht with {description}, balance {remain_balance} baht."
        else:
            return f"On {datetime}, the user performed a transaction {amount} baht ({description}), balance {remain_balance} baht."

    def _create_input_from_dataframe(self, data):
        input = ""
        data = data.sort_values(by="dtcreate")
        transaction_bullets = data.apply(self._transaction_to_text, axis=1)
        for bullet in transaction_bullets:
            input += bullet + "\n"
        return input

    def _preprocess_data(self, data: pd.DataFrame):
        data["dtcreate"] = pd.to_datetime(data["dtcreate"])
        data["register_date"] = pd.to_datetime(data["register_date"])
        data = data.sort_values(["profile_id", "dtcreate"])

        texts = self._create_input_from_dataframe(data)

        return texts

    def _embedding_data(self, data: str):
        # Need to change
        texts_embedding_model = BaseGoogleEmbedding(
            embedding_api_key="GEMINI_EMBEDDING_API_KEY"
        )

        texts = [data]
        texts_embeded = texts_embedding_model.embedding(
            input_texts=texts,
        )

        return pd.DataFrame(
            texts_embeded,
            columns=[f"clf_dim_{i}" for i in range(128)],
        )

    def detect(self, data):
        preprocessed_data = self._preprocess_data(data)
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
