
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    embedding_model_name: str = 'all-MiniLM-L6-v2'
    embedding_dim: int = 384
    use_gpu: bool = False
    # Use Railway volume for persistent storage in production
    index_path: str = os.getenv("INDEX_PATH", "/app/data/indexes" if os.getenv("RAILWAY_ENVIRONMENT") else "./indexes")
    data_path: str = os.getenv("DATA_PATH", "/app/data" if os.getenv("RAILWAY_ENVIRONMENT") else "./data")

    class Config:
        env_file = ".env"

settings = Settings()
