# schemas.py
from pydantic import BaseModel
from typing import List, Optional

class FinancialProjection(BaseModel):
    revenue: float
    cost_of_goods_sold: float
    gross_profit: float
    operational_expenses: float
    net_profit: float
    break_even_revenue: float

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
    economic_indicator: float
    financial_projection: Optional[FinancialProjection] = None
    risks: Optional[List[RiskAssessment]] = []
    mitigation_strategies: Optional[List[str]] = []
