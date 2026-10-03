"""
UCMS API — routers/forecast.py
Phase II Addition.

Endpoints:
GET /api/forecast/citywide        — City-wide aggregate forecast
GET /api/forecast/top-rising      — Top N zones predicted to increase most
GET /api/forecast/{zone_id}       — 1-year forecast for a specific zone
"""

from fastapi import APIRouter, Request, HTTPException, Query
from api.models.schemas import (
    ForecastRecord,
    ForecastResponse,
    ForecastSlot,
    CitywideForecast,
    RiskLevel,
)
from api.core.dependencies import get_master, get_summary

import pandas as pd
import numpy as np

router = APIRouter()

# ─────────────────────────────────────────────────────────────
# SLOT LABELS
# ─────────────────────────────────────────────────────────────

SLOT_NAMES = {
    0: "Night (0-6)",
    1: "Morning (6-12)",
    2: "Afternoon (12-18)",
    3: "Evening (18-24)",
}


# ─────────────────────────────────────────────────────────────
# RISK CLASSIFICATION
# ─────────────────────────────────────────────────────────────

def assign_risk(score: float) -> RiskLevel:
    if score > 0.75:
        return RiskLevel.CRITICAL
    elif score > 0.55:
        return RiskLevel.HIGH
    elif score > 0.35:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


# ─────────────────────────────────────────────────────────────
# GET FORECAST DATA FROM APP STATE
# ─────────────────────────────────────────────────────────────

def get_forecasts(request: Request):
    """Get forecast dataframe from app state (loaded at startup)."""
    return getattr(request.app.state, "forecasts", None)


def build_zone_forecast(zone_id: int, fc_df: pd.DataFrame,
                         summary_df: pd.DataFrame) -> ForecastRecord:
    """Build ForecastRecord for a single zone."""
    zfc = fc_df[fc_df["Community Area"] == zone_id]

    if zfc.empty:
        raise HTTPException(status_code=404,
                            detail=f"No forecast data for zone {zone_id}")

    forecast_year = int(zfc["year"].iloc[0]) if "year" in zfc.columns else 2026
    mean_cps = float(zfc["predicted_cps"].mean())

    slot_forecasts = []
    for slot in range(4):
        slot_df = zfc[zfc["time_slot"] == slot]
        cps = float(slot_df["predicted_cps"].mean()) if not slot_df.empty else mean_cps
        slot_forecasts.append(ForecastSlot(
            time_slot=slot,
            slot_name=SLOT_NAMES[slot],
            predicted_cps=round(cps, 4),
        ))

    # Lat/lon from summary
    lat, lon = None, None
    if summary_df is not None:
        srow = summary_df[summary_df["zone_id"] == zone_id]
        if not srow.empty:
            lat = float(srow.iloc[0].get("lat", None) or 0) or None
            lon = float(srow.iloc[0].get("lon", None) or 0) or None

    return ForecastRecord(
        zone_id=zone_id,
        forecast_year=forecast_year,
        mean_predicted_cps=round(mean_cps, 4),
        risk_level=assign_risk(mean_cps),
        slot_forecasts=slot_forecasts,
        lat=lat,
        lon=lon,
    )


# ─────────────────────────────────────────────────────────────
# GET /api/forecast/citywide
# ─────────────────────────────────────────────────────────────

@router.get(
    "/citywide",
    response_model=CitywideForecast,
    summary="City-wide aggregate forecast for next year"
)
def get_citywide_forecast(request: Request):

    fc_df      = get_forecasts(request)
    master_df  = get_master(request)
    summary_df = get_summary(request)

    if fc_df is None:
        raise HTTPException(
            status_code=503,
            detail="Forecast data not loaded. Run 13_temporal_forecasting.py first."
        )

    forecast_year = int(fc_df["year"].iloc[0]) if "year" in fc_df.columns else 2026
    mean_ccrp = float(fc_df["predicted_cps"].mean())

    # Slot breakdown
    slot_breakdown = []
    for slot in range(4):
        sdf = fc_df[fc_df["time_slot"] == slot]
        m   = float(sdf["predicted_cps"].mean()) if not sdf.empty else 0.0
        slot_breakdown.append({
            "time_slot": slot,
            "slot_name": SLOT_NAMES[slot],
            "mean_ccrp": round(m, 4),
            "risk_level": assign_risk(m).value,
        })

    # Top rising zones (forecast vs latest historical)
    top_rising = []
    if master_df is not None:
        hist_latest = (master_df.groupby("Community Area")["ccrp_bayesian"]
                       .mean().reset_index()
                       .rename(columns={"ccrp_bayesian": "hist_cps"}))
        fc_mean = (fc_df.groupby("Community Area")["predicted_cps"]
                   .mean().reset_index()
                   .rename(columns={"predicted_cps": "fc_cps"}))
        merged = hist_latest.merge(fc_mean, on="Community Area", how="inner")
        merged["delta"] = merged["fc_cps"] - merged["hist_cps"]
        rising = merged.sort_values("delta", ascending=False).head(10)

        for _, row in rising.iterrows():
            top_rising.append({
                "zone_id":  int(row["Community Area"]),
                "hist_cps": round(float(row["hist_cps"]), 4),
                "fc_cps":   round(float(row["fc_cps"]), 4),
                "delta":    round(float(row["delta"]), 4),
            })

    return CitywideForecast(
        forecast_year=forecast_year,
        mean_ccrp=round(mean_ccrp, 4),
        risk_level=assign_risk(mean_ccrp),
        slot_breakdown=slot_breakdown,
        top_rising_zones=top_rising,
    )


# ─────────────────────────────────────────────────────────────
# GET /api/forecast/top-rising
# ─────────────────────────────────────────────────────────────

@router.get(
    "/top-rising",
    summary="Top N zones predicted to increase risk most"
)
def get_top_rising(
    request: Request,
    n: int = Query(10, ge=1, le=500, description="Number of zones to return"),
):
    fc_df     = get_forecasts(request)
    master_df = get_master(request)

    if fc_df is None:
        raise HTTPException(status_code=503,
                            detail="Forecast data not loaded.")

    if master_df is None:
        raise HTTPException(status_code=503,
                            detail="Master dataset not loaded.")

    hist = (master_df.groupby("Community Area")["ccrp_bayesian"]
            .mean().reset_index()
            .rename(columns={"ccrp_bayesian": "hist_cps"}))
    fc   = (fc_df.groupby("Community Area")["predicted_cps"]
            .mean().reset_index()
            .rename(columns={"predicted_cps": "fc_cps"}))

    merged = hist.merge(fc, on="Community Area", how="inner")
    merged["delta"] = merged["fc_cps"] - merged["hist_cps"]
    rising = merged.sort_values("delta", ascending=False).head(n)

    return {
        "top_n": n,
        "zones": [
            {
                "zone_id":  int(r["Community Area"]),
                "hist_cps": round(float(r["hist_cps"]), 4),
                "fc_cps":   round(float(r["fc_cps"]),   4),
                "delta":    round(float(r["delta"]),     4),
                "risk_level": assign_risk(float(r["fc_cps"])).value,
            }
            for _, r in rising.iterrows()
        ]
    }


# ─────────────────────────────────────────────────────────────
# GET /api/forecast/{zone_id}
# ─────────────────────────────────────────────────────────────

@router.get(
    "/{zone_id}",
    response_model=ForecastRecord,
    summary="1-year risk forecast for a specific zone"
)
def get_zone_forecast(zone_id: int, request: Request):

    fc_df = get_forecasts(request)
    summary_df = get_summary(request)

    if fc_df is None:
        raise HTTPException(
            status_code=503,
            detail="Forecast data not loaded. Run 13_temporal_forecasting.py first."
        )

    return build_zone_forecast(zone_id, fc_df, summary_df)
