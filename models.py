# models.py
from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from db import Base

class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    business_idea = Column(String, index=True)
    location = Column(String, index=True)
    rating = Column(String)
    explanation = Column(Text)

# Existing models
class Location(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    geo_code = Column(String, nullable=True)  # If added as per previous suggestions

class BusinessIdea(Base):
    __tablename__ = "business_ideas"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
