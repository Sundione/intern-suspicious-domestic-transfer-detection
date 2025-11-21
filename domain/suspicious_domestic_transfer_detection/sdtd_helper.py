import pandas as pd
from const import tx_type_const
from typing import List, Optional
import logging


class SDTDHelper:

    def __init__(self, logger: Optional[logging.Logger] = None):
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def _sub_type(
        self,
        transaction,
    ):
        if (transaction["vcodetype"] == tx_type_const.TX_TYPE_TU) and (
            transaction["vappcode"] in tx_type_const.PAYMENT_MERCHANT_APP_CODE
        ):
            return tx_type_const.TRANS_TYPE_PAYMENT
        elif (transaction["vcodetype"] == tx_type_const.TX_TYPE_TU) and (
            transaction["vappcode"] in tx_type_const.TRANSFER_MERCHANT_APP_CODE
        ):
            return tx_type_const.TRANS_TYPE_TRANSFER

        elif (transaction["vcodetype"] == tx_type_const.TX_TYPE_FA) or (
            (transaction["vcodetype"] == tx_type_const.TX_TYPE_TU)
            and (
                tx_type_const.FILTER_CHANNEL_PATTERN_INTERNAL_TRANSFER
                in transaction["vdescription"]
            )
        ):
            return tx_type_const.TRANS_TYPE_INTERNAL_TRANSFER
        elif (transaction["vcodetype"] == tx_type_const.TX_TYPE_CH) and (
            tx_type_const.FILTER_CHANNEL_PATTERN_TOPUP in transaction["vdescription"]
        ):
            return tx_type_const.TRANS_TYPE_TOPUP
        elif transaction["vcodetype"] == tx_type_const.TX_TYPE_FB or (
            (transaction["vcodetype"] == tx_type_const.TX_TYPE_CH)
            and (
                tx_type_const.FILTER_CHANNEL_PATTERN_RECEIVE_TRANSFER
                in transaction["vdescription"]
            )
        ):
            return tx_type_const.TRANS_TYPE_RECEIVE_TRANSFER
        elif (transaction["vcodetype"] == tx_type_const.TX_TYPE_TU) and (
            (
                tx_type_const.FILTER_CHANNEL_PATTERN_WITHDRAW
                in transaction["vdescription"]
            )
        ):
            return tx_type_const.TRANS_TYPE_WITHDRAW
        elif (transaction["vcodetype"] == tx_type_const.TX_TYPE_TU) and (
            transaction["vappcode"] in tx_type_const.QR_TRANSFER_MERCHANT_APP_CODE
        ):
            return tx_type_const.TRANS_TYPE_QR_TRANSFER
        elif (transaction["vcodetype"]) == tx_type_const.TX_TYPE_CTU:
            return tx_type_const.TRANS_TYPE_REVERSAL

    def _split_domestic_transfer_iteration(
        self,
        transaction: pd.DataFrame,
        filter_col: str = "sub_type",
        filter_values: List[str] = ["QR_TRANSFER", "WITHDRAW"],
        sort_col: str = "dtcreate",
    ):
        if filter_col not in transaction.columns:
            raise ValueError(f"{filter_col} not found in transaction columns")

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

    def preprocess_data(
        self,
        data: pd.DataFrame,
        split_domestic_iteration: bool = False,
        group_col: str = "profile_id",
        target="is_fraud",
    ):
        self.logger.info("Preprocess Data")
        data["sub_type"] = data.apply(self._sub_type, axis=1)
        self.logger.info("Sub-Type Mapped.")

        data["dtcreate"] = pd.to_datetime(data["dtcreate"])
        data = data.sort_values(["profile_id", "dtcreate"])

        if split_domestic_iteration:
            data = self._split_domestic_transfer_iteration(data)
            self.logger.info("Split Domestic Transfer Iteration.")
            group_col = "profile_ref"

        grouped_list = []
        for profile, grp in data.groupby(group_col):

            is_fraud = None
            if target in grp.columns:
                is_fraud = grp[target].max()

            texts = self._create_input_from_dataframe(grp)

            result_dict = {group_col: profile, "texts": texts}
            if is_fraud is not None:
                result_dict[target] = is_fraud

            grouped_list.append(result_dict)
        self.logger.info("Transform Transaction to Texts for Embedding.")

        return pd.DataFrame(grouped_list)
