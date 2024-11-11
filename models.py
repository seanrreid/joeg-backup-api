# backend/models.py
from db import Base
from sqlalchemy import Column, Integer, String

class BusinessIdea(Base):
    __tablename__ = 'business_ideas'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)

class Location(Base):
    __tablename__ = 'locations'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    geo_code = Column(String, nullable=True)

class Evaluation(Base):
    __tablename__ = 'evaluations'
    id = Column(Integer, primary_key=True, index=True)
    business_idea = Column(String, nullable=False)
    location = Column(String, nullable=False)
    rating = Column(String, nullable=False)
    explanation = Column(String, nullable=False)
    # Add other fields as necessary
