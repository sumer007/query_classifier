from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    classifier_model_name: str
    classifier_model_path: str
    classifier_embeddings_path: str
    classifier_class_examples_path: str
    classifier_complaints_in_scope_path: str
    classifier_complaints_out_of_scope_path: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
