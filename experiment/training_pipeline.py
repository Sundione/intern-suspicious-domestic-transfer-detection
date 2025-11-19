import os
import pandas as pd
import joblib
from adapter.embedding.base import BaseGoogleEmbedding
from experiment.training_schema import TRAIN_SCHEMA
from experiment import tx_type_const
from xgboost import XGBClassifier


class TrainingPipeline:
    SDTD_CLF_DIR = os.path.dirname(os.path.abspath(__file__))
    SDTD_CLF_MODEL_PATH = os.path.join(SDTD_CLF_DIR, "xgboost_model.joblib")

    def __init__(
        self,
        target: str = "is_fraud",
    ):
        self.target = target

    def _load_data(self, path: str = None):
        data = pd.read_csv(path)
        TRAIN_SCHEMA.validate(data)
        return data

    def _qr_withdraw_snapshot(
        self,
        transaction: pd.DataFrame,
        filter_col: str = "sub_type",
        filter_values=None,
        sort_col: str = "dtcreate",
    ):
        if filter_values is None:
            filter_values = ["QR_TRANSFER", "WITHDRAW"]

        if filter_col not in transaction.columns:
            raise ValueError(f"{filter_col} not found in transaction columns")

        transaction = transaction.copy()
        snapshots = []

        for profile_id, group in transaction.groupby("profile_id", sort=False):
            group = group.sort_values(sort_col).reset_index(drop=True)

            triggers = group[group[filter_col].isin(filter_values)].index

            if len(triggers) == 0:
                continue

            for i, idx in enumerate(triggers, start=1):
                snap = group.iloc[: idx + 1].copy()
                snap["target_trigger"] = i
                snap["profile_ref"] = (
                    snap["profile_id"].astype(str)
                    + "_"
                    + snap["target_trigger"].astype(str)
                )
                snapshots.append(snap)

        return pd.concat(snapshots, ignore_index=True)

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

    def _preprocess_data(self, data: pd.DataFrame, group_col: str = "profile_ref"):
        data["dtcreate"] = pd.to_datetime(data["dtcreate"])
        data["register_date"] = pd.to_datetime(data["register_date"])
        data = data.sort_values(["profile_id", "dtcreate"])

        train_txn_snapshot = self._qr_withdraw_snapshot(data)

        grouped_list = []
        for profile, grp in train_txn_snapshot.groupby(group_col):
            is_fraud = grp[self.target].max()
            texts = self._create_input_from_dataframe(grp)
            grouped_list.append(
                {group_col: profile, "texts": texts, self.target: is_fraud}
            )
        train_profile_txn = pd.DataFrame(grouped_list)

        return train_profile_txn

    def _embedding_data(self, data: pd.DataFrame):
        # Need to change
        texts_embedding_model = BaseGoogleEmbedding(
            embedding_api_key="GEMINI_EMBEDDING_API_KEY"
        )

        # if embedding have limit, need to batch process
        texts = data["texts"].tolist()
        texts_embeded = texts_embedding_model.embedding(
            input_texts=texts,
        )

        train_texts_embeded = pd.DataFrame(
            texts_embeded,
            columns=[f"clf_dim_{i}" for i in range(128)],
            index=data.index,
        )

        train_texts_embeded.insert(0, "profile_ref", data["profile_ref"].values)
        train_texts_embeded.insert(1, self.target, data[self.target].values)

        return train_texts_embeded

    def _training_model(self, data: pd.DataFrame):
        X = data.drop(columns=["profile_ref", self.target])
        y = data[self.target]
        model = XGBClassifier()
        model.fit(X, y)

        return model

    def training_pipeline(self, train_data_path: str):
        data = self._load_data(train_data_path)
        processed_data = self._preprocess_data(data)
        embeded_data = self._embedding_data(processed_data)
        model = self._training_model(embeded_data)
        joblib.dump(model, self.SDTD_CLF_MODEL_PATH)
