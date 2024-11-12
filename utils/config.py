
from pydantic_settings import BaseSettings 
from pydantic import Field
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logger for this module
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    google_maps_api_key: str = Field(..., env='GOOGLE_MAPS_API_KEY')
    google_cloud_secret: str = Field(..., env='GOOGLE_CLOUD_SECRET')
    openai_api_key: str = Field(..., env='OPENAI_API_KEY')
    database_url: str = Field(..., env='DATABASE_URL')
    census_api_key: str = Field(..., env='CENSUS_API_KEY')
    api_ninjas_key: str = Field(..., env='API_NINJAS_KEY')

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = 'forbid'  # Ensures no extra fields are present

# Initialize settings
try:
    settings = Settings()
    logger.info("Configuration settings loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load settings: {e}")
    raise
