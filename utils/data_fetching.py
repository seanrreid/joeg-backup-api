import requests
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

API_NINJAS_KEY = os.getenv('API_NINJAS_KEY')

if not API_NINJAS_KEY:
    raise ValueError("API_NINJAS_KEY environment variable is not set.")

# Function to fetch population data using the API Ninjas City API
def get_population_data(location):
    headers = {
        "X-Api-Key": API_NINJAS_KEY
    }

    city, state = map(str.strip, location.split(','))

    params = {
        "name": city,
        "country": "US",  # Assuming only US cities for now
        "limit": 1
    }

    url = "https://api.api-ninjas.com/v1/city"

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to get valid response from API Ninjas City API: {e}")
        raise

    if not data or "population" not in data[0]:
        logging.warning(f"No population data found for '{location}'.")
        return None

    try:
        population = int(data[0]["population"])
        logging.info(f"Population data for {location}: {population}")
        return population
    except ValueError as e:
        logging.error(f"Failed to parse population data: {e}")
        return None

# Other functions like `get_average_spending` and `get_industry_interest_rate` remain unchanged

def get_average_spending(industry):
    """
    Returns the average annual spending per customer for a given industry.
    This function currently uses placeholder values and should be replaced
    with actual data fetching logic.
    """
    # Placeholder values; replace with actual data fetching
    industry_spending = {
        'Fitness Center': 600,  # Average annual spending per customer
        'Coffee Shop': 400,
        # Add more industries
    }
    return industry_spending.get(industry, 500)  # Default value

def get_industry_interest_rate(industry):
    """
    Returns the proportion of the population interested in a given industry.
    This function currently uses placeholder values and should be replaced
    with actual data fetching logic.
    """
    # Placeholder values
    interest_rates = {
        'Fitness Center': 0.2,  # 20% of the population
        'Coffee Shop': 0.5,
        # Add more industries
    }
    return interest_rates.get(industry, 0.3)  # Default value
