import pandera as pa
from pandera.pandas import Column, DataFrameSchema

TRAIN_SCHEMA = DataFrameSchema(
    {
        "profile_id": Column(pa.String),
        "dtcreate": Column(pa.String),
        "ncashdiff": Column(pa.Float),
        "vcodetype": Column(pa.String),
        "vdescription": Column(pa.String),
        "nendingbalance": Column(pa.Float),
        "vappcode": Column(pa.String, nullable=True),
        "is_fraud": Column(pa.Float),
    }
)
