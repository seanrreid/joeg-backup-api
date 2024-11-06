# backend/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import openai
from rapidfuzz import process, fuzz
import logging
from db import Base, engine, SessionLocal
from models import Location, BusinessIdea
from sqlalchemy.orm import Session

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

# Validate essential environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')

if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY is not set.")
    raise ValueError("OPENAI_API_KEY environment variable is missing.")

if not GOOGLE_MAPS_API_KEY:
    logging.error("GOOGLE_MAPS_API_KEY is not set.")
    raise ValueError("GOOGLE_MAPS_API_KEY environment variable is missing.")

if not DATABASE_URL:
    logging.error("DATABASE_URL is not set.")
    raise ValueError("DATABASE_URL environment variable is missing.")

openai.api_key = OPENAI_API_KEY

Base.metadata.create_all(bind=engine)

logging.info(f"OPENAI_API_KEY is set: {bool(OPENAI_API_KEY)}")
logging.info(f"GOOGLE_MAPS_API_KEY is set: {bool(GOOGLE_MAPS_API_KEY)}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Frontend development URL
        "http://127.0.0.1:3000",  # Another possible frontend URL
    ],  # Replace with actual URL
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def seed_database():
    db = SessionLocal()
    try:
        # Seed locations
        known_locations = ["New York, NY", "Los Angeles, CA", "Chicago, IL"]
        for loc in known_locations:
            if not db.query(Location).filter_by(name=loc).first():
                db.add(Location(name=loc))

        # Seed business ideas
        known_business_ideas = ["Coffee Shop", "Book Store", "Gym"]
        for idea in known_business_ideas:
            if not db.query(BusinessIdea).filter_by(name=idea).first():
                db.add(BusinessIdea(name=idea))

        db.commit()
        logging.info("Database seeding completed.")
    except Exception as e:
        logging.error(f"Error seeding database: {str(e)}", exc_info=True)
        db.rollback()
    finally:
        db.close()

def get_known_locations(db: Session):
    locations = db.query(Location.name).all()
    return [loc[0] for loc in locations]

def get_known_business_ideas(db: Session):
    ideas = db.query(BusinessIdea.name).all()
    return [idea[0] for idea in ideas]

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

@app.on_event("startup")
def startup_event():
    seed_database()

@app.post("/evaluate")
async def evaluate(request: EvaluationRequest, db: Session = Depends(get_db)):
    try:
        logging.info(f"Received request - Business Idea: {request.business_idea}, Location: {request.location}")

        KNOWN_LOCATIONS = get_known_locations(db)
        KNOWN_BUSINESS_IDEAS = get_known_business_ideas(db)

        business_idea = request.business_idea
        location = request.location

        # Step 1: Fuzzy match and correct the location
        corrected_location = correct_input(location, KNOWN_LOCATIONS, threshold=80)
        logging.info(f"Location correction: {location} -> {corrected_location}")

        if not corrected_location:
            raise HTTPException(
                status_code=400, 
                detail=f"Location '{location}' not recognized. Available locations: {', '.join(KNOWN_LOCATIONS[:5])}..."
            )

        # Step 2: Fuzzy match and correct the business idea
        corrected_business_idea = correct_input(business_idea, KNOWN_BUSINESS_IDEAS, threshold=80)
        logging.info(f"Business idea correction: {business_idea} -> {corrected_business_idea}")

        if not corrected_business_idea:
            raise HTTPException(
                status_code=400,
                detail=f"Business idea '{business_idea}' not recognized. Available ideas: {', '.join(KNOWN_BUSINESS_IDEAS[:5])}..."
            )

        # Step 3: Geocode the corrected location to get latitude and longitude
        geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={corrected_location}&key={GOOGLE_MAPS_API_KEY}"
        logging.info(f"Attempting geocoding for location: {corrected_location}")

        geocode_response = requests.get(geocode_url)
        geocode_data = geocode_response.json()

        if geocode_data['status'] != 'OK':
            logging.error(f"Geocoding failed: {geocode_data}")
            raise HTTPException(
                status_code=400,
                detail=f"Geocoding failed: {geocode_data['status']} - Please check if Google Maps API key is configured properly"
            )

        lat = geocode_data['results'][0]['geometry']['location']['lat']
        lng = geocode_data['results'][0]['geometry']['location']['lng']
        logging.info(f"Successfully geocoded to: {lat}, {lng}")

        # Step 4: Use Places API to find nearby competitors
        places_url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius=1500&type=establishment&keyword={corrected_business_idea}&key={GOOGLE_MAPS_API_KEY}"
        logging.info(f"Fetching competitors with URL: {places_url}")

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
        else:
            logging.warning(f"No competitors found or Places API error: {places_data['status']}")

        # Step 5: Prepare data for OpenAI
        if competitors:
            competitors_list = "\n".join([
                f"{c['name']} (Rating: {c['rating']}, Reviews: {c['user_ratings_total']}) - {c['vicinity']}" 
                for c in competitors
            ])
        else:
            competitors_list = "No competitors found in the vicinity."

        prompt = f"""
        Analyze the viability of the following business idea in {corrected_location}.

        Business Idea: {corrected_business_idea}

        Nearby Competitors:
        {competitors_list}

        Provide an assessment rating it as 'Great', 'Okay', or 'Bad' and explain the reasoning.
        """

        logging.info("Sending prompt to OpenAI for assessment.")

        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business viability analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )

        assessment = openai_response.choices[0].message['content'].strip()
        finish_reason = openai_response.choices[0].finish_reason
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
    except requests.RequestException as e:
        logging.error(f"Request error during API call: {str(e)}", exc_info=True)
        raise HTTPException(status_code=502, detail="Bad Gateway: External API request failed.")
    except KeyError as e:
        logging.error(f"Key error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error: Unexpected response format.")
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

class LocationCreate(BaseModel):
    name: str

class BusinessIdeaCreate(BaseModel):
    name: str

@app.post("/locations")
def create_location(location: LocationCreate, db: Session = Depends(get_db)):
    if db.query(Location).filter_by(name=location.name).first():
        raise HTTPException(status_code=400, detail="Location already exists.")
    new_location = Location(name=location.name)
    db.add(new_location)
    db.commit()
    db.refresh(new_location)
    return new_location

@app.post("/business-ideas")
def create_business_idea(business_idea: BusinessIdeaCreate, db: Session = Depends(get_db)):
    if db.query(BusinessIdea).filter_by(name=business_idea.name).first():
        raise HTTPException(status_code=400, detail="Business idea already exists.")
    new_idea = BusinessIdea(name=business_idea.name)
    db.add(new_idea)
    db.commit()
    db.refresh(new_idea)
    return new_idea


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
