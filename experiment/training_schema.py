import pandera as pa
from pandera.pandas import Column, DataFrameSchema

TRAIN_SCHEMA = DataFrameSchema(
    {
        "profile_id": Column(pa.String),
        "icardid": Column(pa.Int),
        "register_date": Column(pa.String),
        "dtcreate": Column(pa.String),
        "ncashdiff": Column(pa.Float),
        "ncashback": Column(pa.Float, nullable=True),
        "vcodetype": Column(pa.String),
        "vremark_en": Column(pa.String),
        "vdescription": Column(pa.String),
        "imerchantid": Column(pa.Int, nullable=True),
        "nendingbalance": Column(pa.Float),
        "vappcode": Column(pa.String, nullable=True),
        "sub_type": Column(pa.String, nullable=True),
        "is_fraud": Column(pa.Float),
    }
)
