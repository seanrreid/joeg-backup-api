# backend/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import openai # type: ignore
from rapidfuzz import process, fuzz
import json
import logging

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)


OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

print(f"OPENAI_API_KEY is set: {bool(OPENAI_API_KEY)}")
print(f"GOOGLE_MAPS_API_KEY is set: {bool(GOOGLE_MAPS_API_KEY)}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

openai.api_key = OPENAI_API_KEY

KNOWN_LOCATIONS = [
    "New York, NY",
    "Los Angeles, CA",
    "Chicago, IL",
    "Houston, TX",
    "Phoenix, AZ",
    "Philadelphia, PA",
    "San Antonio, TX",
    "San Diego, CA",
    "Dallas, TX",
    "San Jose, CA",
    "Greenville, SC",
    "Miami, FL",
    "Atlanta, GA",
    "Spartanburg, SC",
    "Charlotte, NC",
    "Anderson, SC",
    "Columbia, SC",
    "Charleston, SC",
    "Asheville, NC",
    "Easley, SC",
    "Hendersonville, NC",
    "Simpsonville, SC",
    "Greer, SC",
    "Seneca, SC",
    "Greenwood, SC",
    "Taylors, SC",
    "Mauldin, SC",
    "Travelers Rest, SC",
    "Piedmont, SC",
    "Fountain Inn, SC",
    "Laurens, SC",
    "Clinton, SC",
    "Berea, SC",
    "Williamston, SC",
    "Central, SC",
    "Liberty, SC",
    "Pickens, SC",
    "Six Mile, SC",
    "Clemson, SC",
    "Pendleton, SC",
    
]

KNOWN_BUSINESS_IDEAS = [
    "Accounting",
    "Advertising",
    "Agriculture",
    "Architectural",
    "Automotive",
    "Banking",
    "Beauty",
    "Biotechnology",
    "Business",
    "Construction",
    "Consulting",
    "Cosmetics",
    "Design",
    "Education",
    "Energy",
    "Engineering",
    "Entertainment",
    "Fashion",
    "Finance",
    "Food",
    "Health",
    "Hospitality",
    "Insurance",
    "Legal",
    "Manufacturing",
    "Marketing",
    "Media",
    "Medical",
    "Music",
    "Nonprofit",
    "Pharmaceutical",
    "Photography",
    "Real Estate",
    "Retail",
    "Sports",
    "Technology",
    "Telecommunications",
    "Transportation",
    "Travel",
    "Utilities",
    "Web Design",
    "Web Development",
    "Writing",
    "Coffee Shop",
    "Restaurant",
    "Bar",
    "Brewery",
    "Cafe",
    "Food Truck",
    "Bakery",
    "Catering",
    "Grocery Store",
    "Food Delivery",
    "Gym",
    "Yoga Studio",
    "Fitness Center",
    "Personal Training",
    "Physical Therapy",
    "Florist",
    "Landscaping",
    "Gardening",
    "Lawn Care",
    "Pool Cleaning",
    "Pest Control",
    "Cleaning",
    "Home Staging",
    "Interior Design",
    "Home Renovation",
    "Home Inspection",
    "Real Estate Agent",
    "Property Management",
    "Mortgage Broker",
    "Insurance Agent",
    "Financial Planner",
    "Accountant",
    "Lawyer",
    "Legal Services",
    "Consultant",
    "Marketing Agency",
    "Advertising Agency",
    "Public Relations",
    "Event Planning",
    "Graphic Design",
    "Web Design",
    "Web Development",
    "Photographer",
    "Videographer",
    "Musician",
    "Band",
    "DJ",
    "Music Producer",
    "Music Teacher",
    "Music Lessons",
    "Music Studio",
    "Logistics",
    "Delivery",
    "Moving",
    "Storage",
    "Courier",
    "Transportation",
    "Rideshare",
    "Taxi",
    "Car Service",
    "Limo Service",
    "Bus Service",
    "Train Service",
    "Airline",
    "Travel Agent",
    "Travel Agency",
    "Tour Guide",
    "Tour Operator",
    "Tourism",
    "Hotel",
    "Motel",
    "Bed and Breakfast",
    "Vacation Rental",
    "Hostel",
    "Resort",
    "Campground",
    "Ski Resort",
    "Theme Park",
    "Amusement Park",
    "Water Park",
    "Zoo",
    "Aquarium",
    "Museum",
    "Art Gallery",
    "Historical Site",
    "National Park",
    "State Park",
    "City Park",
    "Botanical Garden",
    "Nature Preserve",
    "Wildlife Sanctuary",
    "Nature Center",
    "Planetarium",
    "Observatory",
    "Science Museum",
    "Children's Museum",
    "Natural History Museum",
    "History Museum",
    "Art Museum",
    "Art Studio",
    "Art School",
    "Art Class",
    "Art Workshop",
    "Art Camp",
    "Art Gallery",
    "Art Exhibit",
    "Art Show",
    "Art Fair",
    "Art Festival",
    "Art Walk",
    "Art Tour",
    "Antiques",
    "Collectibles",
    "Vintage",
    "Thrift",
    "Secondhand",
    "Artisan",
    "Handmade",
    "Craft",
    "DIY",
    "Homemade",
    "Local",
    "Small Business",
    "Startup",
    "Entrepreneur",
    "Freelancer",
    "Independent Contractor",
    "Self-Employed",
    "Work From Home",
    "Home-Based Business",
    "Online Business",
    "E-Commerce",
    "Internet Business",
    "Digital Business",
    "Mobile Business",
    "Social Media",
    "Content Marketing",
    "Email Marketing",
    "Search Engine Optimization",
    "Pay-Per-Click Advertising",
    "Affiliate Marketing",
    "Influencer Marketing",
    "Video Marketing",
    "Podcast Marketing",
    "Asset Management",
    "Investment Management",
    "Wealth Management",
    "Financial Planning",
    "Retirement Planning",
    "Estate Planning",
    "Tax Planning",
    "Insurance Planning",
    "Risk Management",
    "Portfolio Management",
    "Ice Cream Shop",
    "Frozen Yogurt Shop",
    "Gelato Shop",
    "Sorbet Shop",
    "Popsicle Shop",
    "Ice Cream Truck",
    "Ice Cream Cart",
    "Ice Cream Stand",
    "Ice Cream Parlor",
    "Ice Cream Cafe",
    "Mexican Restaurant",
    "Italian Restaurant",
    "Chinese Restaurant",
    "Japanese Restaurant",
    "Thai Restaurant",
    "Indian Restaurant",
    "Korean Restaurant",
    "Vietnamese Restaurant",
    "Greek Restaurant",
    "Mediterranean Restaurant",
    "Middle Eastern Restaurant",
    "American Restaurant",
    "Burger Restaurant",
    "Pizza Restaurant",
    "Sandwich Restaurant",
    "Sub Restaurant",
    "Salad Restaurant",
    "Soup Restaurant",
    "Seafood Restaurant",
    "Steakhouse",
    "Barbecue Restaurant",
    "Breakfast Restaurant",
    "Brunch Restaurant",
    "Lunch Restaurant",
    "Dinner Restaurant",
    "Fine Dining Restaurant",
    "Casual Dining Restaurant",
    "Fast Food Restaurant",
    "Family Restaurant",
    "Chain Restaurant",
    "Franchise Restaurant",
    "Local Restaurant",
    "Independent Restaurant",
    "Mom and Pop Restaurant",
    "Hole-in-the-Wall Restaurant",
    "Dive Restaurant",
    "Cafe Restaurant",
    "Bistro Restaurant",
    "Brasserie Restaurant",
    "Gastropub Restaurant",
    "Cafe",
    "Coffee Shop",
    "Espresso Bar",
    "Tea House",
    "Bakery",
    "Pastry Shop",
    "Dessert Shop",
    "Ice Cream Shop",
    "Juice Bar",
    "Smoothie Bar",
    "Bubble Tea Shop",
    "Milkshake Bar",
    "Donut Shop",
    "Bagel Shop",
    "Sandwich Shop",
    "Salad Bar",
    "Soup Kitchen",
    "Food Truck",
    "Mobile Cafe",
    "Mobile Coffee Shop",
    "Mobile Bakery",
    "Mobile Dessert Shop",
    "Mobile Juice Bar",
    "Mobile Smoothie Bar",
    "Mobile Bubble Tea Shop",

]
def correct_input(user_input, known_list, threshold=80):
    match, score, _ = process.extractOne(
        user_input, known_list, scorer=fuzz.WRatio
    )
    if score >= threshold:
        return match
    else:
        return None

class EvaluationRequest(BaseModel):
    business_idea: str
    location: str

@app.post("/evaluate")
async def evaluate(request: EvaluationRequest):
    print(f"Received request - Business Idea: {request.business_idea}, Location: {request.location}")
    logging.info("Received request at /evaluate endpoint.")
    business_idea = request.business_idea
    location = request.location

    # Step 1: Fuzzy match and correct the location
    corrected_location = correct_input(location, KNOWN_LOCATIONS, threshold=80)
    print(f"Location correction: {location} -> {corrected_location}")
    
    logging.info(f"Corrected Location: {corrected_location}")
    if not corrected_location:
        raise HTTPException(
            status_code=400, 
            detail=f"Location '{location}' not recognized. Available locations: {', '.join(KNOWN_LOCATIONS[:5])}..."
        )

    # Step 2: Fuzzy match and correct the business idea
    corrected_business_idea = correct_input(business_idea, KNOWN_BUSINESS_IDEAS, threshold=80)
    print(f"Business idea correction: {business_idea} -> {corrected_business_idea}")
    
    if not corrected_business_idea:
        raise HTTPException(
            status_code=400,
            detail=f"Business idea '{business_idea}' not recognized. Available ideas: {', '.join(KNOWN_BUSINESS_IDEAS[:5])}..."
        )

    try:
        geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={corrected_location}&key={GOOGLE_MAPS_API_KEY}"
        print(f"Attempting geocoding for location: {corrected_location}")
        
        geocode_response = requests.get(geocode_url)
        geocode_data = geocode_response.json()
        
        print(f"Geocoding response status: {geocode_data['status']}")
        if geocode_data['status'] != 'OK':
            print(f"Geocoding error response: {geocode_data}")
            raise HTTPException(
                status_code=400,
                detail=f"Geocoding failed: {geocode_data['status']} - Please check if Google Maps API key is configured properly"
            )

        lat = geocode_data['results'][0]['geometry']['location']['lat']
        lng = geocode_data['results'][0]['geometry']['location']['lng']
        print(f"Successfully geocoded to: {lat}, {lng}")

    except requests.RequestException as e:
        print(f"Request error during geocoding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Geocoding request failed: {str(e)}")
    except KeyError as e:
        print(f"Unexpected geocoding response format: {geocode_data}")
        raise HTTPException(status_code=500, detail="Invalid response from geocoding service")
    except Exception as e:
        print(f"Unexpected error during geocoding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Geocoding error: {str(e)}")

    # Rest of your code remains the same...

    # Step 3: Geocode the corrected location to get latitude and longitude
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={corrected_location}&key={GOOGLE_MAPS_API_KEY}"
    geocode_response = requests.get(geocode_url)
    geocode_data = geocode_response.json()

    if geocode_data['status'] != 'OK':
        raise HTTPException(status_code=400, detail='Invalid location provided after correction.')

    lat = geocode_data['results'][0]['geometry']['location']['lat']
    lng = geocode_data['results'][0]['geometry']['location']['lng']

    # Step 4: Use Places API to find nearby competitors
    places_url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius=1500&type=establishment&keyword={corrected_business_idea}&key={GOOGLE_MAPS_API_KEY}"
    places_response = requests.get(places_url)
    places_data = places_response.json()

    competitors = []
    if places_data['status'] == 'OK':
        for place in places_data['results']:
            competitors.append({
                'name': place.get('name'),
                'rating': place.get('rating'),
                'user_ratings_total': place.get('user_ratings_total'),
                'vicinity': place.get('vicinity')
            })

    # Step 5: Prepare data for OpenAI
    if competitors:
        competitors_list = "\n".join([f"{c['name']} (Rating: {c['rating']}, Reviews: {c['user_ratings_total']}) - {c['vicinity']}" for c in competitors])
    else:
        competitors_list = "No competitors found in the vicinity."

    prompt = f"""
    Analyze the viability of the following business idea in {corrected_location}.

    Business Idea: {corrected_business_idea}

    Nearby Competitors:
    {competitors_list}

    Provide an assessment rating it as 'Great', 'Okay', or 'Bad' and explain the reasoning.
    """

    try:
        openai_response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=300,
            temperature=0.3
        )
        

        assessment = openai_response.choices[0].text.strip()
        finish_reason = openai_response.choices[0].get('finish_reason')
        logging.info(f"OpenAI finish reason: {finish_reason}")
        logging.info(f"Assessment: {assessment}")
        
        if finish_reason == 'length':
            logging.warning("OpenAI response was truncated due to max_tokens limit.")
        
        # Simple parsing to find 'Great', 'Okay', 'Bad'
        rating = 'Unknown'
        if 'Great' in assessment:
            rating = 'Great'
        elif 'Okay' in assessment:
            rating = 'Okay'
        elif 'Bad' in assessment:
            rating = 'Bad'

        response = {
            'rating': rating,
            'explanation': assessment,
            'competitors': competitors,
            'corrected_location': corrected_location,
            'corrected_business_idea': corrected_business_idea
        }
        logging.info("Successfully processed the request.")
        return response


    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
