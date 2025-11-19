from typing import Callable, Optional

from pydantic import computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ENV: Optional[str] = "development"
    AI_LOGGING_SERVICE_CODE: Optional[str] = "FRAUD_SCAMMER_DETECTION"
    SDTD_CLF_PREDICTION_THRESHOLD: float = 0.5


def _configure_initial_settings() -> Callable[[], Settings]:
    settings = Settings()

    def fn() -> Settings:
        return settings

    return fn


get_settings = _configure_initial_settings()
