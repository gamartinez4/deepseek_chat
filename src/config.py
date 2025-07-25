"""Configuración centralizada de la aplicación."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
   
    ENDPOINT: str = "https://models.github.ai/inference"
    MODEL_NAME: str = "deepseek/DeepSeek-V3-0324"
    TOP_K: int = 3
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    INDEX_PATH: str = "store/faiss_index"
    REQUEST_TIMEOUT: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings() 