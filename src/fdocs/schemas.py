"""Structured extraction schemas for financial documents."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


def _confidence_field() -> Field:
    return Field(default=None, ge=0.0, le=1.0)


class GuidanceItem(BaseModel):
    metric: Optional[str] = None
    metric_confidence: Optional[float] = _confidence_field()
    previous_value: Optional[str] = None
    previous_value_confidence: Optional[float] = _confidence_field()
    new_value: Optional[str] = None
    new_value_confidence: Optional[float] = _confidence_field()
    change_direction: Optional[str] = None
    change_direction_confidence: Optional[float] = _confidence_field()
    confidence: Optional[float] = _confidence_field()


class RiskFactor(BaseModel):
    category: Optional[str] = None
    category_confidence: Optional[float] = _confidence_field()
    description: Optional[str] = None
    description_confidence: Optional[float] = _confidence_field()
    severity: Optional[str] = None
    severity_confidence: Optional[float] = _confidence_field()
    is_new: Optional[bool] = None
    is_new_confidence: Optional[float] = _confidence_field()
    mentioned_by: Optional[str] = None
    mentioned_by_confidence: Optional[float] = _confidence_field()
    confidence: Optional[float] = _confidence_field()


class FinancialMetric(BaseModel):
    name: Optional[str] = None
    name_confidence: Optional[float] = _confidence_field()
    value: Optional[str] = None
    value_confidence: Optional[float] = _confidence_field()
    period: Optional[str] = None
    period_confidence: Optional[float] = _confidence_field()
    unit: Optional[str] = None
    unit_confidence: Optional[float] = _confidence_field()
    comparison_to_consensus: Optional[str] = None
    comparison_to_consensus_confidence: Optional[float] = _confidence_field()
    confidence: Optional[float] = _confidence_field()


class ManagementTone(BaseModel):
    overall_sentiment: Optional[str] = None
    overall_sentiment_confidence: Optional[float] = _confidence_field()
    confidence_indicators: Optional[List[str]] = None
    confidence_indicators_confidence: Optional[float] = _confidence_field()
    hedging_language: Optional[List[str]] = None
    hedging_language_confidence: Optional[float] = _confidence_field()
    confidence: Optional[float] = _confidence_field()


class AnalystConcern(BaseModel):
    analyst_name: Optional[str] = None
    analyst_name_confidence: Optional[float] = _confidence_field()
    firm: Optional[str] = None
    firm_confidence: Optional[float] = _confidence_field()
    topic: Optional[str] = None
    topic_confidence: Optional[float] = _confidence_field()
    sentiment: Optional[str] = None
    sentiment_confidence: Optional[float] = _confidence_field()
    confidence: Optional[float] = _confidence_field()


class ExtractedReport(BaseModel):
    company: Optional[str] = None
    document_path: Optional[str] = None
    document_hash: Optional[str] = None
    document_id: Optional[str] = None
    extraction_version: Optional[str] = None
    generated_at: Optional[datetime] = None
    guidance: List[GuidanceItem] = Field(default_factory=list)
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    financial_metrics: List[FinancialMetric] = Field(default_factory=list)
    management_tone: List[ManagementTone] = Field(default_factory=list)
    analyst_concerns: List[AnalystConcern] = Field(default_factory=list)
    metadata: Optional[dict] = None
    raw_sections: Optional[dict] = None

    @field_validator("metadata", mode="before")
    @classmethod
    def _default_metadata(cls, value):
        return value or {}
