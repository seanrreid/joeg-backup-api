# backend/seed.py
from db import SessionLocal, engine, Base
from models import Location, BusinessIdea
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("seed_debug.log")
    ]
)

def seed_database():
    db = SessionLocal()
    try:
        # Seed locations
        known_locations = [
            "New York, NY", "Los Angeles, CA", "Chicago, IL",
            "Houston, TX", "Phoenix, AZ", "Philadelphia, PA"
        ]
        for loc in known_locations:
            if not db.query(Location).filter_by(name=loc).first():
                db.add(Location(name=loc))

        # Seed business ideas
        known_business_ideas = [
            "Coffee Shop", "Book Store", "Gym",
            "Bakery", "Fitness Center", "Art Gallery"
        ]
        for idea in known_business_ideas:
            if not db.query(BusinessIdea).filter_by(name=idea).first():
                db.add(BusinessIdea(name=idea))

        db.commit()
        logging.info("Database seeding completed successfully.")
    except Exception as e:
        logging.error(f"Error seeding database: {str(e)}", exc_info=True)
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_database()
