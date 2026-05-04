from pydantic_settings import BaseSettings


class Config(BaseSettings):
    openai_api_key: str = ""
    amount_tolerance: float = 0.01
    date_tolerance_days: int = 3
    fuzzy_narration_threshold: int = 85
    max_ai_calls: int = 500
    skip_ai: bool = False
    log_level: str = "INFO"

    model_config = {"env_file": ".env"}


config = Config()
