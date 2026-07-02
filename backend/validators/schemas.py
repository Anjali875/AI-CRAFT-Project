from pydantic import BaseModel, Field
from typing import Optional

class PCOSInput(BaseModel):
    age: float = Field(..., ge=12, le=60)
    weight: float = Field(..., ge=25, le=200)
    height: float = Field(..., ge=100, le=220)
    bmi: float = Field(..., ge=10, le=60)
    cycle_length: float = Field(..., ge=2, le=12)
    weight_gain: int = Field(..., ge=0, le=1)
    hair_growth: int = Field(..., ge=0, le=1)
    skin_darkening: int = Field(..., ge=0, le=1)
    hair_loss: int = Field(..., ge=0, le=1)
    pimples: int = Field(..., ge=0, le=1)
    fast_food: int = Field(..., ge=0, le=1)
    regular_exercise: int = Field(..., ge=0, le=1)

class EndoInput(BaseModel):
    age: float = Field(..., ge=15, le=55)
    weight: float = Field(..., ge=25, le=200)
    height: float = Field(..., ge=100, le=220)
    bmi: float = Field(..., ge=10, le=60)
    cycle_length: float = Field(..., ge=15, le=45)
    age_of_menarche: float = Field(..., ge=8, le=18)
    dysmenorrhea_score: float = Field(..., ge=0, le=10)
    urinary_symptoms_score: float = Field(..., ge=0, le=9)
    family_history: int = Field(..., ge=0, le=1)
    infertility_status: int = Field(..., ge=0, le=1)
    mental_health_score: float = Field(..., ge=0, le=10)

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|model)$")
    content: str = Field(..., min_length=1, max_length=2000)

class ChatInput(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    condition: Optional[str] = Field(default=None, pattern="^(pcos|endo)$")
    risk_level: Optional[str] = Field(default=None, pattern="^(Low|Moderate|High)$")
    risk_score: Optional[float] = Field(default=None, ge=0, le=1)
    contributing_factors: list[str] = Field(default=[])
    symptoms: dict = Field(default={})
    history: list[ChatMessage] = Field(default=[])

class ReportInput(BaseModel):
    condition: str = Field(..., pattern="^(pcos|endo)$")
    risk_level: str = Field(..., pattern="^(Low|Moderate|High)$")
    risk_percentage: float = Field(..., ge=0, le=100)
    contributing_factors: list[str] = Field(default=[])
    symptoms: dict = Field(default={})