# backend/utils/external_api.py
from pytrends.request import TrendReq
import os

pytrends = TrendReq(hl='en-US', tz=360)

def get_google_trends(business_idea, location):
    kw_list = [business_idea]
    pytrends.build_payload(kw_list, geo=location_to_geo(location), timeframe='today 12-m')
    trends = pytrends.interest_over_time()
    if trends.empty:
        return 0
    return trends[business_idea].mean()

def location_to_geo(location):
    # Simplified: Map location to country code or regional code for Google Trends
    location_mapping = {
        'New York, NY': 'US-NY',
        'Los Angeles, CA': 'US-CA',
        'Chicago, IL': 'US-IL',
        'Houston, TX': 'US-TX',
        'San Francisco, CA': 'US-CA'
        # Add more mappings as needed
    }
    return location_mapping.get(location, 'US')  # Default to US

# backend/utils/external_api.py (continued)
def get_economic_indicator(location):
    # Placeholder: In a real scenario, fetch from an API like World Bank or local government data
    economic_data = {
        'New York, NY': 0.8,
        'Los Angeles, CA': 0.75,
        'Chicago, IL': 0.7,
        'Houston, TX': 0.65,
        'San Francisco, CA': 0.85
        # Add more mappings as needed
    }
    return economic_data.get(location, 0.5)  # Default to 0.5
