"""
UCMS API — routers/alerts.py
Phase II: Added /api/alerts/anomalies and /api/alerts/export (CSV download).

Endpoints:
GET /api/alerts
GET /api/alerts/critical
GET /api/alerts/summary
GET /api/alerts/hotspots
GET /api/alerts/anomalies    [Phase II]
GET /api/alerts/export       [Phase II]
"""

import io
from fastapi import APIRouter, Query, Request, HTTPException
from fastapi.responses import StreamingResponse
from api.models.schemas import (
    AlertRecord,
    AlertResponse,
    AnomalyRecord,
    AnomalyResponse,
    RiskLevel
)
from api.core.dependencies import (
    get_master,
    get_summary
)

import pandas as pd

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# SLOT LABELS
# ─────────────────────────────────────────────────────────────

SLOT_NAMES = {
    0: "Night(0-6)",
    1: "Morning(6-12)",
    2: "Afternoon(12-18)",
    3: "Evening(18-24)"
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
# GET ALL ALERTS
# ─────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=AlertResponse,
    summary="Get all alert records above threshold"
)
def get_alerts(
    request: Request,

    threshold: float = Query(
        0.70,
        ge=0,
        le=1,
        description="Minimum CCRP threshold"
    ),

    year: int = Query(
        None,
        description="Filter by year"
    ),

    time_slot: int = Query(
        None,
        ge=0,
        le=3,
        description="Filter by time slot"
    ),

    limit: int = Query(
        100,
        ge=1,
        le=500
    )
):

    master_df = get_master(request)

    if master_df is None:
        raise HTTPException(
            status_code=503,
            detail="Master dataset not loaded"
        )

    df = master_df[
        master_df["ccrp_bayesian"] > threshold
    ].copy()

    if year is not None:
        df = df[df["year"] == year]

    if time_slot is not None:
        df = df[df["time_slot"] == time_slot]

    df = df.sort_values(
        "ccrp_bayesian",
        ascending=False
    ).head(limit)

    alerts = []

    for _, row in df.iterrows():

        alerts.append(
            AlertRecord(
                zone_id=int(row["Community Area"]),

                year=int(row["year"]),

                time_slot=int(row["time_slot"]),

                slot_name=SLOT_NAMES.get(
                    int(row["time_slot"]),
                    "Unknown"
                ),

                ccrp_score=round(
                    float(row["ccrp_bayesian"]),
                    4
                ),

                risk_level=assign_risk(
                    float(row["ccrp_bayesian"])
                ),

                lat=float(row["lat"])
                if pd.notna(row.get("lat"))
                else None,

                lon=float(row["lon"])
                if pd.notna(row.get("lon"))
                else None,
            )
        )

    return AlertResponse(
        total_alerts=len(alerts),
        threshold_used=threshold,
        alerts=alerts
    )


# ─────────────────────────────────────────────────────────────
# CRITICAL ALERTS
# ─────────────────────────────────────────────────────────────

@router.get(
    "/critical",
    summary="Get CRITICAL risk zones only"
)
def get_critical_alerts(
    request: Request,

    year: int = Query(
        None,
        description="Optional year filter"
    )
):

    master_df = get_master(request)

    if master_df is None:
        raise HTTPException(
            status_code=503,
            detail="Master dataset not loaded"
        )

    df = master_df[
        master_df["ccrp_bayesian"] > 0.75
    ]

    if year is not None:
        df = df[df["year"] == year]

    zones = sorted(
        df["Community Area"].unique().tolist()
    )

    return {
        "critical_zone_count": len(zones),

        "critical_zones": zones,

        "records": int(len(df)),

        "mean_ccrp": round(
            float(df["ccrp_bayesian"].mean()),
            4
        ) if not df.empty else 0.0
    }


# ─────────────────────────────────────────────────────────────
# ALERT SUMMARY
# ─────────────────────────────────────────────────────────────

@router.get(
    "/summary",
    summary="Get alert count per zone"
)
def get_alert_summary(
    request: Request,

    threshold: float = Query(
        0.70,
        ge=0,
        le=1
    )
):

    master_df = get_master(request)

    summary_df = get_summary(request)

    if master_df is None:
        raise HTTPException(
            status_code=503,
            detail="Master dataset not loaded"
        )

    alert_counts = (
        master_df[
            master_df["ccrp_bayesian"] > threshold
        ]
        .groupby("Community Area")
        .agg(
            alert_count=("ccrp_bayesian", "count"),
            max_ccrp=("ccrp_bayesian", "max"),
            mean_ccrp=("ccrp_bayesian", "mean")
        )
        .reset_index()
        .sort_values(
            "alert_count",
            ascending=False
        )
    )

    result = []

    for _, row in alert_counts.iterrows():

        result.append({
            "zone_id": int(row["Community Area"]),

            "alert_count": int(row["alert_count"]),

            "max_ccrp": round(
                float(row["max_ccrp"]),
                4
            ),

            "mean_ccrp": round(
                float(row["mean_ccrp"]),
                4
            ),

            "risk_level": assign_risk(
                float(row["max_ccrp"])
            ).value
        })

    return {
        "threshold": threshold,

        "zones_with_alerts": len(result),

        "total_alerts": int(
            alert_counts["alert_count"].sum()
        ) if not alert_counts.empty else 0,

        "zone_breakdown": result
    }


# ─────────────────────────────────────────────────────────────
# HOTSPOTS
# ─────────────────────────────────────────────────────────────

@router.get(
    "/hotspots",
    summary="Get top hotspot zones"
)
def get_hotspots(
    request: Request,

    n: int = Query(
        10,
        ge=1,
        le=500,
        description="Number of hotspots"
    ),

    year: int = Query(
        None,
        description="Optional year filter"
    ),

    slot: int = Query(
        None,
        ge=0,
        le=3,
        description="Optional slot filter"
    )
):

    master_df = get_master(request)

    if master_df is None:
        raise HTTPException(
            status_code=503,
            detail="Master dataset not loaded"
        )

    df = master_df.copy()

    if year is not None:
        df = df[df["year"] == year]

    if slot is not None:
        df = df[df["time_slot"] == slot]

    hotspots = (
        df.groupby("Community Area")
        .agg(
            mean_ccrp=("ccrp_bayesian", "mean"),
            max_ccrp=("ccrp_bayesian", "max"),
            lat=("lat", "mean"),
            lon=("lon", "mean")
        )
        .reset_index()
        .sort_values(
            "mean_ccrp",
            ascending=False
        )
        .head(n)
    )

    return {
        "top_n": n,

        "filters": {
            "year": year,
            "slot": slot
        },

        "hotspots": [

            {
                "rank": i + 1,

                "zone_id": int(r["Community Area"]),

                "mean_ccrp": round(
                    float(r["mean_ccrp"]),
                    4
                ),

                "max_ccrp": round(
                    float(r["max_ccrp"]),
                    4
                ),

                "risk_level": assign_risk(
                    float(r["mean_ccrp"])
                ).value,

                "lat": float(r["lat"])
                if pd.notna(r.get("lat"))
                else None,

                "lon": float(r["lon"])
                if pd.notna(r.get("lon"))
                else None,
            }

            for i, (_, r)
            in enumerate(hotspots.iterrows())
        ]
    }


# ─────────────────────────────────────────────────────────────
# PHASE II — ANOMALY ALERTS
# ─────────────────────────────────────────────────────────────

@router.get(
    "/anomalies",
    response_model=AnomalyResponse,
    summary="[Phase II] Get anomaly-flagged records"
)
def get_anomaly_alerts(
    request: Request,

    threshold: float = Query(
        0.60,
        ge=0,
        le=1,
        description="Minimum anomaly_score threshold"
    ),

    year: int = Query(None, description="Optional year filter"),

    limit: int = Query(100, ge=1, le=500),
):
    anomaly_df = getattr(request.app.state, "anomalies", None)

    if anomaly_df is None:
        raise HTTPException(
            status_code=503,
            detail="Anomaly dataset not loaded. Run 14_anomaly_detection.py first."
        )

    df = anomaly_df[anomaly_df["anomaly_score"] >= threshold].copy()

    if year is not None:
        df = df[df["year"] == year]

    df = df.sort_values("anomaly_score", ascending=False).head(limit)

    anomalies = []
    for _, row in df.iterrows():
        lat_val = float(row["lat"]) if pd.notna(row.get("lat")) else None
        lon_val = float(row["lon"]) if pd.notna(row.get("lon")) else None

        anomalies.append(
            AnomalyRecord(
                zone_id=int(row["Community Area"]),
                year=int(row["year"]),
                time_slot=int(row["time_slot"]),
                slot_name=SLOT_NAMES.get(int(row["time_slot"]), "Unknown"),
                ccrp_score=round(float(row.get("ccrp_bayesian", 0)), 4),
                anomaly_score=round(float(row["anomaly_score"]), 4),
                is_anomaly=bool(row["is_anomaly"]),
                lat=lat_val,
                lon=lon_val,
            )
        )

    return AnomalyResponse(
        total_anomalies=len(anomalies),
        threshold_used=threshold,
        anomalies=anomalies,
    )


# ─────────────────────────────────────────────────────────────
# PHASE II — EXPORT ALERTS AS CSV
# ─────────────────────────────────────────────────────────────

@router.get(
    "/export",
    summary="[Phase II] Download current alerts as CSV file"
)
def export_alerts_csv(
    request: Request,
    threshold: float = Query(0.70, ge=0, le=1),
    year: int = Query(None),
    time_slot: int = Query(None, ge=0, le=3),
):
    master_df = get_master(request)

    if master_df is None:
        raise HTTPException(status_code=503, detail="Master dataset not loaded")

    df = master_df[master_df["ccrp_bayesian"] > threshold].copy()

    if year is not None:
        df = df[df["year"] == year]
    if time_slot is not None:
        df = df[df["time_slot"] == time_slot]

    export_cols = [
        c for c in [
            "Community Area", "year", "time_slot",
            "ccrp_bayesian", "lat", "lon",
            "crowd_density", "lighting_risk", "social_media",
            "suspicious_activity", "cctv_feed",
        ] if c in df.columns
    ]

    df_export = df[export_cols].sort_values("ccrp_bayesian", ascending=False)

    stream = io.StringIO()
    df_export.to_csv(stream, index=False)
    stream.seek(0)

    slot_str = str(time_slot) if time_slot is not None else "all"
    yr_str   = str(year)      if year      is not None else "all"
    filename = f"ucms_alerts_{yr_str}_{slot_str}.csv"

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )