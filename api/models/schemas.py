"""
UCMS API — models/schemas.py
Pydantic response/request schemas for all endpoints.
Phase II: Added ForecastRecord, PatrolSlot, AnomalyRecord, FeatureImportanceRecord schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from enum import Enum


# ─────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TrendDirection(str, Enum):
    RISING = "RISING"
    STABLE = "STABLE"
    FALLING = "FALLING"


# ─────────────────────────────────────────────────────────────
# ZONES
# ─────────────────────────────────────────────────────────────

class ZoneSummary(BaseModel):
    zone_id: int
    zone_name: Optional[str] = None
    district: Optional[str] = None

    mean_ccrp: float = Field(
        ...,
        ge=0,
        le=1,
        description="Mean CCRP score"
    )

    peak_ccrp: float = Field(
        ...,
        ge=0,
        le=1,
        description="Peak CCRP score"
    )

    risk_level: RiskLevel
    trend: TrendDirection

    peak_slot: Optional[str] = None

    alert_count: int = 0

    mean_uncertainty: float = 0.0

    lat: Optional[float] = None
    lon: Optional[float] = None


class ZoneListResponse(BaseModel):
    total: int
    zones: List[ZoneSummary]


class YearTrendRecord(BaseModel):
    year: int
    score: float


class ZoneDetail(BaseModel):
    zone_id: int
    zone_name: Optional[str] = None
    district: Optional[str] = None

    risk_level: RiskLevel
    trend: TrendDirection

    mean_ccrp: float
    peak_ccrp: float

    alert_count: int
    mean_uncertainty: float

    lat: Optional[float] = None
    lon: Optional[float] = None

    slot_scores: Dict[str, float]

    year_trend: List[YearTrendRecord]


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):

    zone_id: int = Field(
        ...,
        ge=1,
        le=500,
        description="Indian Urban Zone ID (1-500)"
    )

    year: int = Field(
        2025,
        ge=2000,
        le=2035
    )

    time_slot: int = Field(
        ...,
        ge=0,
        le=3,
        description="0=Night, 1=Morning, 2=Afternoon, 3=Evening"
    )

    crowd_density: Optional[float] = Field(None, ge=0, le=1)

    lighting_risk: Optional[float] = Field(None, ge=0, le=1)

    social_media: Optional[float] = Field(None, ge=0, le=1)

    suspicious_activity: Optional[float] = Field(None, ge=0, le=1)

    cctv_score: Optional[float] = Field(None, ge=0, le=1)


class PredictResponse(BaseModel):

    zone_id: int

    year: int

    time_slot: int

    slot_name: str

    ccrp_score: float = Field(..., ge=0, le=1)

    ccrp_bayesian: float = Field(..., ge=0, le=1)

    uncertainty_lower: float

    uncertainty_upper: float

    confidence_band_pct: float = Field(
        ...,
        description="Width of 90% credible interval as percentage"
    )

    risk_level: RiskLevel

    stream_weights: Dict[str, float]

    message: str


# ─────────────────────────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────────────────────────

class AlertRecord(BaseModel):

    zone_id: int

    year: int

    time_slot: int

    slot_name: str

    ccrp_score: float

    risk_level: RiskLevel

    lat: Optional[float] = None

    lon: Optional[float] = None


class AlertResponse(BaseModel):

    total_alerts: int

    threshold_used: float

    alerts: List[AlertRecord]


# ─────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────

class OverviewStats(BaseModel):

    total_zones: int

    critical_zones: int

    high_zones: int

    medium_zones: int

    low_zones: int

    total_alert_records: int

    mean_ccrp_citywide: float

    most_dangerous_zone: int

    safest_zone: int

    rising_trend_zones: int


# ─────────────────────────────────────────────────────────────
# PHASE II — FORECAST
# ─────────────────────────────────────────────────────────────

class ForecastSlot(BaseModel):
    time_slot: int
    slot_name: str
    predicted_cps: float = Field(..., ge=0, le=1)


class ForecastRecord(BaseModel):
    zone_id: int
    forecast_year: int
    mean_predicted_cps: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    slot_forecasts: List[ForecastSlot]
    lat: Optional[float] = None
    lon: Optional[float] = None


class ForecastResponse(BaseModel):
    forecast_year: int
    zones: List[ForecastRecord]


class CitywideForecast(BaseModel):
    forecast_year: int
    mean_ccrp: float
    risk_level: RiskLevel
    slot_breakdown: List[Dict]
    top_rising_zones: List[Dict]


# ─────────────────────────────────────────────────────────────
# PHASE II — PATROL
# ─────────────────────────────────────────────────────────────

class PatrolSlot(BaseModel):
    rank: int
    zone_id: int
    time_slot: int
    slot_name: str
    mean_ccrp: float
    risk_level: RiskLevel
    priority_score: float
    lat: Optional[float] = None
    lon: Optional[float] = None


class PatrolRecommendationResponse(BaseModel):
    shift: str
    total_zones: int
    recommendations: List[PatrolSlot]


class PatrolRouteStop(BaseModel):
    order: int
    zone_id: int
    lat: Optional[float] = None
    lon: Optional[float] = None
    mean_ccrp: float
    risk_level: RiskLevel


class PatrolRouteResponse(BaseModel):
    route_length: int
    total_risk_covered: float
    waypoints: List[PatrolRouteStop]


# ─────────────────────────────────────────────────────────────
# PHASE II — ANOMALY
# ─────────────────────────────────────────────────────────────

class AnomalyRecord(BaseModel):
    zone_id: int
    year: int
    time_slot: int
    slot_name: str
    ccrp_score: float
    anomaly_score: float
    is_anomaly: bool
    lat: Optional[float] = None
    lon: Optional[float] = None


class AnomalyResponse(BaseModel):
    total_anomalies: int
    threshold_used: float
    anomalies: List[AnomalyRecord]


# ─────────────────────────────────────────────────────────────
# PHASE II — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────

class FeatureImportanceRecord(BaseModel):
    feature: str
    importance: float
    rank: int


class FeatureImportanceResponse(BaseModel):
    zone_id: Optional[int] = None
    scope: str = Field(..., description="'zone' or 'citywide'")
    features: List[FeatureImportanceRecord]