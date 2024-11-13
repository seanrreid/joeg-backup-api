# backend/utils/external_api.py

import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

API_KEY = "EXPLODING_TOPICS_API_KEY"  # Ideally, load this from environment variables

def fetch_trend_data(topic: str, region: Optional[str] = None) -> Dict[str, Any]:
   
    base_url = "https://api.explodingtopics.com/v1/topics"  # Hypothetical endpoint
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    params = {
        "query": topic,
        "region": region if region else "global"
    }

    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched trend data for topic: {topic}")
        return data
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while fetching trend data: {http_err}")
    except Exception as err:
        logger.error(f"An error occurred while fetching trend data: {err}")
    return {}
    


def get_economic_indicator(location: str) -> float:
    """
    Fetches economic indicators for a given location.
    Placeholder implementation; replace with actual API calls as needed.
    """
    economic_data = {
        'New York, NY': 0.8,
        'Los Angeles, CA': 0.75,
        'Chicago, IL': 0.7,
        'Houston, TX': 0.65,
        'San Francisco, CA': 0.85
        # Add more mappings as needed
    }
    return economic_data.get(location, 0.5)  # Default to 0.5



 
