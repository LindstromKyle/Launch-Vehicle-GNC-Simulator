from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_title: str = "6DOF Launch Simulator"
    api_description: str = (
        "Configurable launch-to-orbit simulation for mission software testing"
    )
    environment: str = "dev"
    debug: bool = False
    simulator_executor_max_workers: int = 4
    monte_carlo_storage_dir: str | None = None

    model_config = SettingsConfigDict(
        env_prefix="SIM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
