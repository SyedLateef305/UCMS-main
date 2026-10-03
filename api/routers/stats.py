"""
UCMS API — routers/stats.py

Endpoints:
GET /api/stats/overview
GET /api/stats/trends
GET /api/stats/streams
GET /api/stats/risk-distribution
GET /api/stats/zone-comparison
GET /api/stats/slot-profile
"""

from fastapi import APIRouter, Query, Request, HTTPException

from api.models.schemas import OverviewStats

from api.core.dependencies import (
    get_master,
    get_summary,
    get_weights
)

import pandas as pd
import numpy as np

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# SLOT LABELS
# ─────────────────────────────────────────────────────────────

SLOT_NAMES = {
    0: "Night (0–6)",
    1: "Morning (6–12)",
    2: "Afternoon (12–18)",
    3: "Evening (18–24)",
}


# ─────────────────────────────────────────────────────────────
# RISK CLASSIFIER
# ─────────────────────────────────────────────────────────────

def assign_risk(score: float) -> str:

    if score > 0.75:
        return "CRITICAL"

    elif score > 0.55:
        return "HIGH"

    elif score > 0.35:
        return "MEDIUM"

    return "LOW"


# ─────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────

@router.get(
    "/overview",
    response_model=OverviewStats,
    summary="City-wide overview statistics"
)
def get_overview(request: Request):

    summary_df = get_summary(request)

    master_df = get_master(request)

    if summary_df is None:
        raise HTTPException(
            status_code=503,
            detail="Summary dataset not loaded"
        )

    rl = summary_df["risk_level"].value_counts()

    alerts = master_df[
        master_df["ccrp_bayesian"] > 0.70
    ]

    most_dangerous = (
        int(summary_df.iloc[0]["zone_id"])
        if not summary_df.empty else -1
    )

    safest = (
        int(summary_df.iloc[-1]["zone_id"])
        if not summary_df.empty else -1
    )

    rising = int(
        (summary_df["trend"] == "RISING").sum()
    )

    return OverviewStats(

        total_zones=int(
            summary_df["zone_id"].nunique()
        ),

        critical_zones=int(
            rl.get("CRITICAL", 0)
        ),

        high_zones=int(
            rl.get("HIGH", 0)
        ),

        medium_zones=int(
            rl.get("MEDIUM", 0)
        ),

        low_zones=int(
            rl.get("LOW", 0)
        ),

        total_alert_records=int(
            len(alerts)
        ),

        mean_ccrp_citywide=round(
            float(summary_df["mean_ccrp"].mean()),
            4
        ),

        most_dangerous_zone=most_dangerous,

        safest_zone=safest,

        rising_trend_zones=rising
    )


# ─────────────────────────────────────────────────────────────
# YEARLY TRENDS
# ─────────────────────────────────────────────────────────────

@router.get(
    "/trends",
    summary="Year-over-year CCRP trends"
)
def get_trends(

    request: Request,

    zone_id: int = Query(
        None,
        description="Optional zone filter"
    ),

    time_slot: int = Query(
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

    if zone_id is not None:

        df = df[
            df["Community Area"] == zone_id
        ]

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Zone {zone_id} not found"
            )

    if time_slot is not None:

        df = df[
            df["time_slot"] == time_slot
        ]

    trend = (
        df.groupby("year")
        .agg(
            mean_ccrp=(
                "ccrp_bayesian",
                "mean"
            ),

            max_ccrp=(
                "ccrp_bayesian",
                "max"
            ),

            alert_count=(
                "alert_flag",
                "sum"
            )
        )
        .reset_index()
        .sort_values("year")
    )

    data = []

    for _, row in trend.iterrows():

        data.append({

            "year": int(row["year"]),

            "mean_ccrp": round(
                float(row["mean_ccrp"]),
                4
            ),

            "max_ccrp": round(
                float(row["max_ccrp"]),
                4
            ),

            "alert_count": int(
                row["alert_count"]
            ),

            "risk_level": assign_risk(
                float(row["mean_ccrp"])
            )
        })

    if len(data) >= 2:

        delta = (
            data[-1]["mean_ccrp"] -
            data[0]["mean_ccrp"]
        )

        if delta > 0.03:
            direction = "RISING"

        elif delta < -0.03:
            direction = "FALLING"

        else:
            direction = "STABLE"

    else:

        direction = "STABLE"

    return {
        "zone_id": zone_id,

        "time_slot": time_slot,

        "overall_direction": direction,

        "years": len(data),

        "data": data
    }


# ─────────────────────────────────────────────────────────────
# STREAM WEIGHTS
# ─────────────────────────────────────────────────────────────

@router.get(
    "/streams",
    summary="Adaptive stream weights"
)
def get_stream_weights(request: Request):

    weights_df = get_weights(request)

    if weights_df is None:
        raise HTTPException(
            status_code=503,
            detail="Weights dataset not loaded"
        )

    STREAM_COLS = [
        "crowd_density",
        "lighting_risk",
        "social_media",
        "suspicious_activity",
        "cctv_feed",
        "historical_cps"
    ]

    result = []

    for _, row in weights_df.iterrows():

        slot_data = {
            "slot": str(
                row.get("slot", "Unknown")
            )
        }

        for col in STREAM_COLS:

            if col in row:

                slot_data[col] = round(
                    float(row[col]),
                    4
                )

        result.append(slot_data)

    avg_weights = {}

    for col in STREAM_COLS:

        if col in weights_df.columns:

            avg_weights[col] = round(
                float(weights_df[col].mean()),
                4
            )

    ranked = sorted(
        avg_weights.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {

        "slots": result,

        "average_weights": avg_weights,

        "importance_rank": [

            {
                "stream": s,
                "weight": w
            }

            for s, w in ranked
        ]
    }


# ─────────────────────────────────────────────────────────────
# RISK DISTRIBUTION
# ─────────────────────────────────────────────────────────────

@router.get(
    "/risk-distribution",
    summary="Risk score histogram"
)
def get_risk_distribution(

    request: Request,

    bins: int = Query(
        20,
        ge=5,
        le=50
    ),

    year: int = Query(
        None
    ),

    time_slot: int = Query(
        None,
        ge=0,
        le=3
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

    if time_slot is not None:

        df = df[
            df["time_slot"] == time_slot
        ]

    scores = df[
        "ccrp_bayesian"
    ].dropna().values

    counts, edges = np.histogram(
        scores,
        bins=bins,
        range=(0, 1)
    )

    hist = []

    for i in range(len(counts)):

        midpoint = (
            edges[i] + edges[i + 1]
        ) / 2

        hist.append({

            "bin_start": round(
                float(edges[i]),
                3
            ),

            "bin_end": round(
                float(edges[i + 1]),
                3
            ),

            "midpoint": round(
                float(midpoint),
                3
            ),

            "count": int(counts[i]),

            "risk_level": assign_risk(
                float(midpoint)
            )
        })

    rl_counts = {

        "CRITICAL": int(
            (scores > 0.75).sum()
        ),

        "HIGH": int(
            (
                (scores > 0.55) &
                (scores <= 0.75)
            ).sum()
        ),

        "MEDIUM": int(
            (
                (scores > 0.35) &
                (scores <= 0.55)
            ).sum()
        ),

        "LOW": int(
            (scores <= 0.35).sum()
        )
    }

    return {

        "total_records": len(scores),

        "mean_score": round(
            float(scores.mean()),
            4
        ),

        "std_score": round(
            float(scores.std()),
            4
        ),

        "filters": {
            "year": year,
            "time_slot": time_slot
        },

        "risk_counts": rl_counts,

        "histogram": hist
    }


# ─────────────────────────────────────────────────────────────
# ZONE COMPARISON
# ─────────────────────────────────────────────────────────────

@router.get(
    "/zone-comparison",
    summary="Compare multiple zones"
)
def get_zone_comparison(

    request: Request,

    zone_ids: str = Query(
        ...,
        description="Comma-separated zone IDs"
    ),

    year: int = Query(None)
):

    try:

        ids = [
            int(z.strip())
            for z in zone_ids.split(",")
        ]

    except ValueError:

        raise HTTPException(
            status_code=422,
            detail="zone_ids must be integers"
        )

    master_df = get_master(request)

    summary_df = get_summary(request)

    if master_df is None:
        raise HTTPException(
            status_code=503,
            detail="Master dataset not loaded"
        )

    df = master_df[
        master_df["Community Area"].isin(ids)
    ].copy()

    if year is not None:

        df = df[df["year"] == year]

    STREAM_COLS = [
        "crowd_density",
        "lighting_risk",
        "social_media",
        "suspicious_activity",
        "cctv_feed",
        "historical_cps"
    ]

    result = []

    for zone_id in ids:

        zdf = df[
            df["Community Area"] == zone_id
        ]

        if zdf.empty:

            result.append({
                "zone_id": zone_id,
                "error": "not found"
            })

            continue

        row = summary_df[
            summary_df["zone_id"] == zone_id
        ]

        streams = {}

        for col in STREAM_COLS:

            if col in zdf.columns:

                streams[col] = round(
                    float(zdf[col].mean()),
                    4
                )

        result.append({

            "zone_id": zone_id,

            "mean_ccrp": round(
                float(
                    zdf["ccrp_bayesian"].mean()
                ),
                4
            ),

            "risk_level": (
                row.iloc[0]["risk_level"]
                if not row.empty
                else "UNKNOWN"
            ),

            "trend": (
                row.iloc[0]["trend"]
                if not row.empty
                else "STABLE"
            ),

            "streams": streams,

            "uncertainty": round(
                float(
                    zdf["uncertainty"].mean()
                ),
                4
            )
        })

    return {

        "zones_requested": ids,

        "year": year,

        "comparison": result
    }


# ─────────────────────────────────────────────────────────────
# SLOT PROFILE
# ─────────────────────────────────────────────────────────────

@router.get(
    "/slot-profile",
    summary="24-hour slot risk profile"
)
def get_slot_profile(

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

    df = master_df.copy()

    if year is not None:

        df = df[df["year"] == year]

    profile = []

    for slot, name in SLOT_NAMES.items():

        sdf = df[
            df["time_slot"] == slot
        ]

        mean = (
            float(
                sdf["ccrp_bayesian"].mean()
            )
            if not sdf.empty else 0.0
        )

        top3 = (
            sdf.groupby("Community Area")[
                "ccrp_bayesian"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
            .head(3)
        )

        profile.append({

            "time_slot": slot,

            "slot_name": name,

            "mean_ccrp": round(
                mean,
                4
            ),

            "risk_level": assign_risk(
                mean
            ),

            "top3_zones": [
                int(z)
                for z in top3.index.tolist()
            ]
        })

    peak = max(
        profile,
        key=lambda x: x["mean_ccrp"]
    )

    safe = min(
        profile,
        key=lambda x: x["mean_ccrp"]
    )

    return {

        "year": year,

        "peak_slot": peak["slot_name"],

        "safest_slot": safe["slot_name"],

        "profile": profile
    }


# ─────────────────────────────────────────────────────────────
# PHASE II — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "mean_cps", "log_crime_count_norm", "avg_severity",
    "avg_arrest_rate", "avg_domestic_rate", "yoy_change_pct_norm",
    "lat_norm", "lon_norm",
    "cps_slot0_night", "cps_slot1_morning",
    "cps_slot2_afternoon", "cps_slot3_evening",
    "max_cps", "mean_cps_dup",
]

STREAM_FEATURE_COLS = [
    "crowd_density", "lighting_risk",
    "social_media", "suspicious_activity", "cctv_feed", "historical_cps"
]


@router.get(
    "/feature-importance",
    summary="[Phase II] Feature importance driving CCRP risk"
)
def get_feature_importance(
    request: Request,
    zone_id: int = Query(None, ge=1, le=500, description="Optional zone filter"),
    year: int = Query(None, description="Optional year filter"),
):
    master_df  = get_master(request)
    weights_df = get_weights(request)

    importance = {}

    # Option 1: Use learned stream weights
    if weights_df is not None:
        for col in STREAM_FEATURE_COLS:
            if col in weights_df.columns:
                importance[col] = float(weights_df[col].mean())

    # Option 2: Use correlation if master_df has columns
    if not importance and master_df is not None:
        numeric_cols = [
            c for c in master_df.select_dtypes(include=[np.number]).columns
            if c not in ["Community Area", "year", "time_slot", "ccrp_bayesian"]
        ]
        target = master_df["ccrp_bayesian"].fillna(0)
        for col in numeric_cols:
            vals = master_df[col].fillna(0)
            if vals.std() > 0:
                corr = abs(float(np.corrcoef(vals, target)[0, 1]))
                importance[col] = round(corr, 4)

    # Fallback default weights if neither is available
    if not importance:
        importance = {
            "historical_cps": 0.35,
            "suspicious_activity": 0.20,
            "cctv_feed": 0.18,
            "crowd_density": 0.12,
            "lighting_risk": 0.08,
            "social_media": 0.07,
        }

    # Normalise to sum to 1.0
    total = sum(importance.values()) or 1.0
    norm_importance = {k: round(v / total, 4) for k, v in importance.items()}
    ranked = sorted(norm_importance.items(), key=lambda x: x[1], reverse=True)

    return {
        "zone_id": zone_id,
        "year": year,
        "scope": "zone" if zone_id else "citywide",
        "features": [
            {"rank": i + 1, "feature": feat, "importance": imp}
            for i, (feat, imp) in enumerate(ranked)
        ]
    }


# ─────────────────────────────────────────────────────────────
# PHASE II — ZONE COMPARISON TREND
# ─────────────────────────────────────────────────────────────

@router.get(
    "/zone-comparison-trend",
    summary="[Phase II] Multi-zone year-over-year CCRP trend for comparison"
)
def get_zone_comparison_trend(
    request: Request,
    zone_ids: str = Query(..., description="Comma-separated zone IDs (max 4)"),
    time_slot: int = Query(None, ge=0, le=3),
):
    try:
        ids = [int(z.strip()) for z in zone_ids.split(",")][:4]
    except ValueError:
        raise HTTPException(status_code=422, detail="zone_ids must be comma-separated integers")

    master_df = get_master(request)
    if master_df is None:
        raise HTTPException(status_code=503, detail="Master dataset not loaded")

    df = master_df.copy()
    if time_slot is not None:
        df = df[df["time_slot"] == time_slot]

    result = {}
    for zone_id in ids:
        zdf = df[df["Community Area"] == zone_id]
        if zdf.empty:
            result[str(zone_id)] = []
            continue
        trend = (zdf.groupby("year")["ccrp_bayesian"]
                    .mean().reset_index().sort_values("year"))
        result[str(zone_id)] = [
            {"year": int(r["year"]), "ccrp": round(float(r["ccrp_bayesian"]), 4)}
            for _, r in trend.iterrows()
        ]

    return {
        "zones": ids,
        "time_slot": time_slot,
        "trends": result,
    }


# ─────────────────────────────────────────────────────────────
# PHASE II — HEATMAP MATRIX
# ─────────────────────────────────────────────────────────────

@router.get(
    "/heatmap",
    summary="[Phase II] Slot × Zone CCRP matrix for heatmap visualisation"
)
def get_heatmap_matrix(
    request: Request,
    year: int = Query(None, description="Filter year (defaults to latest)"),
    top_n: int = Query(20, ge=5, le=500, description="Number of zones to include"),
):
    master_df = get_master(request)
    if master_df is None:
        raise HTTPException(status_code=503, detail="Master dataset not loaded")

    df = master_df.copy()
    if year is not None:
        df = df[df["year"] == year]
    else:
        df = df[df["year"] == int(df["year"].max())]

    # Top-N highest risk zones
    top_zones = (df.groupby("Community Area")["ccrp_bayesian"]
                   .mean().sort_values(ascending=False)
                   .head(top_n).index.tolist())

    df = df[df["Community Area"].isin(top_zones)]

    matrix = []
    for slot, slot_name in SLOT_NAMES.items():
        slot_df = df[df["time_slot"] == slot]
        row = {"slot": slot_name}
        for zone in top_zones:
            val = slot_df[slot_df["Community Area"] == zone]["ccrp_bayesian"].mean()
            row[f"z{zone}"] = round(float(val), 4) if not pd.isna(val) else 0.0
        matrix.append(row)

    return {
        "year": year or int(master_df["year"].max()),
        "zones": top_zones,
        "matrix": matrix,
    }