from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_title: str = "6DOF Launch Simulator"
    api_description: str = (
        "Configurable launch-to-orbit simulation for mission software testing"
    )
    environment: str = "dev"
    debug: bool = False
    simulator_executor_max_workers: int = 4
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "launch_sim"
    db_user: str = "sim_user"
    db_password: str = "sim_password"

    model_config = SettingsConfigDict(
        env_prefix="SIM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def db_connection_string(self) -> str:
        return (
            f"host={self.db_host} "
            f"port={self.db_port} "
            f"dbname={self.db_name} "
            f"user={self.db_user} "
            f"password={self.db_password}"
        )
