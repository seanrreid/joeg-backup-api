# backend/models.py
from sqlalchemy import Column, Integer, String
from db import Base

class Location(Base):
    __tablename__ = 'locations'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)

class BusinessIdea(Base):
    __tablename__ = 'business_ideas'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
