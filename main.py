from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import logging
from db import Base, engine, SessionLocal
from models import Location, BusinessIdea, Evaluation
from sqlalchemy.orm import Session
from ml.model import model
from utils.external_api import get_google_trends, get_economic_indicator
from utils.data_fetching import get_population_data
from utils.financials import (
    estimate_annual_revenue,
    estimate_startup_costs,
    estimate_operational_expenses,
    generate_p_and_l_statement,
)
from utils.risks import (
    identify_potential_risks,
    assess_risks,
    suggest_mitigation_strategies,
)
import pandas as pd
from typing import List, Optional
from rapidfuzz import process, fuzz

# Load environment variables
load_dotenv()

# Validate essential environment variables
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
API_NINJAS_KEY = os.getenv("API_NINJAS_KEY")

print("GOOGLE_MAPS_API_KEY:", GOOGLE_MAPS_API_KEY)
print("DATABASE_URL:", DATABASE_URL)
print("API_NINJAS_API_KEY:", API_NINJAS_KEY)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("app.log")],
)

# Validate essential environment variables
if not GOOGLE_MAPS_API_KEY:
    logging.error("GOOGLE_MAPS_API_KEY is not set.")
    raise ValueError("GOOGLE_MAPS_API_KEY environment variable is missing.")

if not DATABASE_URL:
    logging.error("DATABASE_URL is not set.")
    raise ValueError("DATABASE_URL environment variable is missing.")

Base.metadata.create_all(bind=engine)

logging.info(f"GOOGLE_MAPS_API_KEY is set: {bool(GOOGLE_MAPS_API_KEY)}")
logging.info(f"DATABASE_URL is set: {bool(DATABASE_URL)}")
logging.info(f"API_NINJAS_API_KEY is set: {bool(API_NINJAS_KEY)}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    match, score, _ = process.extractOne(user_input, known_list, scorer=fuzz.WRatio)
    if score >= threshold:
        return match
    else:
        return None

# Define Pydantic models

class FinancialProjection(BaseModel):
    revenue: float
    cost_of_goods_sold: float
    gross_profit: float
    operational_expenses: float
    net_profit: float
    break_even_revenue: float

class Competitor(BaseModel):
    name: str
    rating: float
    user_ratings_total: int
    vicinity: str

class RiskAssessment(BaseModel):
    risk: str
    likelihood: int
    impact: int
    risk_score: int

class EvaluationRequest(BaseModel):
    business_idea: str
    location: str

class EvaluationResponse(BaseModel):
    rating: str
    explanation: str
    competitors: Optional[List[dict]] = []
    corrected_location: str
    corrected_business_idea: str
    new_location_added: bool
    new_business_idea_added: bool
    trend_score: float
    economic_indicator: float
    financial_projection: Optional[FinancialProjection] = None
    risks: Optional[List[RiskAssessment]] = []
    mitigation_strategies: Optional[List[str]] = []

class LocationCreate(BaseModel):
    name: str

class BusinessIdeaCreate(BaseModel):
    name: str

@app.on_event("startup")
def startup_event():
    seed_database()

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest, db: Session = Depends(get_db)):
    try:
        logging.info(
            f"Received request - Business Idea: {request.business_idea}, Location: {request.location}"
        )

        business_idea = request.business_idea
        location = request.location

        # Fetch known data from DB
        known_locations = get_known_locations(db)
        known_business_ideas = get_known_business_ideas(db)

        # Step 1: Fuzzy match and correct the location
        corrected_location = correct_input(location, known_locations, threshold=80)

        if not corrected_location:
            # Geocode the new location
            geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={location}&key={GOOGLE_MAPS_API_KEY}"
            logging.info(f"Attempting geocoding for new location: {location}")
            try:
                geocode_response = requests.get(geocode_url, timeout=10)
                geocode_data = geocode_response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"Geocoding API request failed: {e}")
                raise HTTPException(status_code=500, detail="Geocoding service unavailable.")

            if geocode_data.get('status') != 'OK':
                logging.error(f"Geocoding API error: {geocode_data.get('status')}")
                raise HTTPException(status_code=400, detail='Invalid location provided.')

            # Extract geocode
            lat = geocode_data['results'][0]['geometry']['location']['lat']
            lng = geocode_data['results'][0]['geometry']['location']['lng']
            geo_code = f"{lat},{lng}"

            # Add new location with geo_code
            new_location = Location(name=location, geo_code=geo_code)
            db.add(new_location)
            db.commit()
            corrected_location = location
            new_location_added = True
            logging.info(f"New location added: {location} with geo_code: {geo_code}")
        else:
            new_location_added = False
            logging.info(f"Location correction: {location} -> {corrected_location}")

        # Step 2: Fuzzy match and correct the business idea
        corrected_business_idea = correct_input(
            business_idea, known_business_ideas, threshold=80
        )
        if not corrected_business_idea:
            # Optionally add new business idea
            new_business_idea = BusinessIdea(name=business_idea)
            db.add(new_business_idea)
            db.commit()
            corrected_business_idea = business_idea
            new_business_idea_added = True
            logging.info(f"New business idea added: {business_idea}")
        else:
            new_business_idea_added = False
            logging.info(
                f"Business idea correction: {business_idea} -> {corrected_business_idea}"
            )

        # Step 3: Geocode the corrected location to get latitude and longitude
        geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={corrected_location}&key={GOOGLE_MAPS_API_KEY}"
        logging.info(f"Attempting geocoding for location: {corrected_location}")
        try:
            geocode_response = requests.get(geocode_url, timeout=10)
            geocode_data = geocode_response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Geocoding API request failed: {e}")
            raise HTTPException(status_code=500, detail="Geocoding service unavailable.")

        if geocode_data.get("status") != "OK":
            logging.error(f"Geocoding API error: {geocode_data.get('status')}")
            raise HTTPException(
                status_code=400, detail="Invalid location provided after correction."
            )

        lat = geocode_data["results"][0]["geometry"]["location"]["lat"]
        lng = geocode_data["results"][0]["geometry"]["location"]["lng"]
        logging.info(f"Successfully geocoded to: {lat}, {lng}")

        # Step 4: Use Places API to find nearby competitors
        places_url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius=1500&type=establishment&keyword={corrected_business_idea}&key={GOOGLE_MAPS_API_KEY}"
        logging.info(f"Fetching competitors with URL: {places_url}")
        try:
            places_response = requests.get(places_url, timeout=10)
            places_data = places_response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Places API request failed: {e}")
            raise HTTPException(status_code=500, detail="Places service unavailable.")

        competitors = []
        if places_data.get("status") == "OK":
            for place in places_data.get("results", []):
                competitors.append(
                    {
                        "name": place.get("name"),
                        "rating": place.get("rating"),
                        "user_ratings_total": place.get("user_ratings_total"),
                        "vicinity": place.get("vicinity"),
                    }
                )
        else:
            logging.warning(
                f"No competitors found or Places API error: {places_data.get('status')}"
            )

        # Step 5: Fetch external data for enhanced evaluation
        trend_score = get_google_trends(corrected_business_idea, corrected_location)
        economic_indicator = get_economic_indicator(corrected_location)

        # Step 6: Prepare features for the ML model
        features = {
            "business_idea": corrected_business_idea,
            "location": corrected_location,
            "competitors": len(competitors),
            "trend_score": trend_score,
            "economic_indicator": economic_indicator,
        }

        # Convert features to the format expected by the model
        df_features = pd.DataFrame([features])
        df_features_encoded = pd.get_dummies(
            df_features, columns=["business_idea", "location"], drop_first=True
        )

        # Ensure all model features are present
        model_features = model.feature_names_in_
        for feature in model_features:
            if feature not in df_features_encoded.columns:
                df_features_encoded[feature] = 0

        df_features_encoded = df_features_encoded[model_features]

        # Predict success using the ML model
        prediction = model.predict(df_features_encoded)[0]
        rating = "Great" if prediction == 1 else "Bad"

        # Generate explanation
        explanation = (
            f"The business idea '{corrected_business_idea}' in '{corrected_location}' "
            f"has a predicted success rating of '{rating}'. This is based on current market trends "
            f"and economic indicators."
        )

        # Step 7: Financial Projections
        annual_revenue = estimate_annual_revenue(
            corrected_business_idea, corrected_location, competitors
        )
        startup_costs = estimate_startup_costs(corrected_business_idea)
        operational_expenses = estimate_operational_expenses(
            corrected_business_idea, corrected_location
        )
        cost_of_goods_sold = annual_revenue * 0.3  # Assuming COGS is 30% of revenue
        p_and_l_statement = generate_p_and_l_statement(
            annual_revenue, cost_of_goods_sold, operational_expenses
        )
        break_even_revenue = startup_costs  # Simplified for this example

        financial_projection = FinancialProjection(
            revenue=p_and_l_statement["revenue"],
            cost_of_goods_sold=p_and_l_statement["cost_of_goods_sold"],
            gross_profit=p_and_l_statement["gross_profit"],
            operational_expenses=p_and_l_statement["operational_expenses"],
            net_profit=p_and_l_statement["net_profit"],
            break_even_revenue=break_even_revenue,
        )

        # Step 8: Risk Assessment
        risks = identify_potential_risks(corrected_business_idea, corrected_location)
        risk_assessment = assess_risks(risks)
        mitigation_strategies = suggest_mitigation_strategies(risk_assessment)

        # Save evaluation to the database
        evaluation = Evaluation(
            business_idea=corrected_business_idea,
            location=corrected_location,
            rating=rating,
            explanation=explanation,
        )
        db.add(evaluation)
        db.commit()
        db.refresh(evaluation)

        response = EvaluationResponse(
            rating=rating,
            explanation=explanation,
            competitors=competitors,
            corrected_location=corrected_location,
            corrected_business_idea=corrected_business_idea,
            new_location_added=new_location_added,
            new_business_idea_added=new_business_idea_added,
            trend_score=trend_score,
            economic_indicator=economic_indicator,
            financial_projection=financial_projection,
            risks=risk_assessment,
            mitigation_strategies=mitigation_strategies,
        )

        logging.info("Successfully processed the request.")
        return response

    except HTTPException:
        raise
    except requests.RequestException as e:
        logging.error(f"Request error during API call: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=502, detail="Bad Gateway: External API request failed."
        )
    except KeyError as e:
        logging.error(f"Key error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Internal Server Error: Unexpected response format."
        )
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error.")

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
def create_business_idea(
    business_idea: BusinessIdeaCreate, db: Session = Depends(get_db)
):
    if db.query(BusinessIdea).filter_by(name=business_idea.name).first():
        raise HTTPException(status_code=400, detail="Business idea already exists.")
    new_idea = BusinessIdea(name=business_idea.name)
    db.add(new_idea)
    db.commit()
    db.refresh(new_idea)
    return new_idea

@app.get("/routes")
def list_routes():
    routes = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            routes.append({
                "path": route.path,
                "methods": list(route.methods)
            })
    return routes

@app.get("/locations")
def get_locations(db: Session = Depends(get_db)):
    locations = db.query(Location).all()
    return [loc.name for loc in locations]

@app.get("/business-ideas")
def get_business_ideas(db: Session = Depends(get_db)):
    ideas = db.query(BusinessIdea).all()
    return [idea.name for idea in ideas]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
