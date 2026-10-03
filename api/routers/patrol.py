"""
UCMS API — routers/patrol.py
Phase II Addition.

Endpoints:
GET /api/patrol/recommend    — AI-prioritised patrol schedule for a shift
GET /api/patrol/hotspot-route — Ordered nearest-neighbour route through top-K zones
"""

from fastapi import APIRouter, Request, HTTPException, Query
from api.models.schemas import (
    PatrolSlot,
    PatrolRecommendationResponse,
    PatrolRouteStop,
    PatrolRouteResponse,
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

SLOT_LABELS = {
    "night":     0,
    "morning":   1,
    "afternoon": 2,
    "evening":   3,
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
# NEAREST-NEIGHBOUR ROUTE
# ─────────────────────────────────────────────────────────────

def nearest_neighbour_route(points: list[dict]) -> list[dict]:
    """
    Greedy nearest-neighbour TSP approximation.
    points: list of dicts with keys lat, lon (and any other fields).
    Returns ordered list.
    """
    if not points:
        return []

    # Filter points that have valid coordinates
    valid = [p for p in points if p.get("lat") is not None and p.get("lon") is not None]
    no_coords = [p for p in points if p.get("lat") is None or p.get("lon") is None]

    if not valid:
        return points   # can't route without coords

    coords = np.array([[p["lat"], p["lon"]] for p in valid])
    visited = [False] * len(valid)
    route_indices = [0]
    visited[0] = True

    for _ in range(len(valid) - 1):
        current = route_indices[-1]
        current_coord = coords[current]
        best_dist = float("inf")
        best_next = -1

        for j, coord in enumerate(coords):
            if visited[j]:
                continue
            dist = np.sqrt(((current_coord - coord) ** 2).sum())
            if dist < best_dist:
                best_dist = dist
                best_next = j

        if best_next >= 0:
            route_indices.append(best_next)
            visited[best_next] = True

    return [valid[i] for i in route_indices] + no_coords


# ─────────────────────────────────────────────────────────────
# GET /api/patrol/recommend
# ─────────────────────────────────────────────────────────────

@router.get(
    "/recommend",
    response_model=PatrolRecommendationResponse,
    summary="AI-recommended patrol priority list for a shift"
)
def get_patrol_recommendation(
    request: Request,
    shift: str = Query(
        "night",
        description="Shift name: night | morning | afternoon | evening | all"
    ),
    top_n: int = Query(
        15,
        ge=1,
        le=500,
        description="Number of zones to include in patrol list"
    ),
    year: int = Query(
        None,
        description="Filter by year (defaults to latest available)"
    ),
):
    master_df  = get_master(request)
    summary_df = get_summary(request)

    if master_df is None:
        raise HTTPException(status_code=503, detail="Master dataset not loaded")

    df = master_df.copy()

    # Year filter
    if year is not None:
        df = df[df["year"] == year]
    else:
        latest_year = int(df["year"].max())
        df = df[df["year"] == latest_year]

    # Slot filter
    shift_lower = shift.lower()
    if shift_lower in SLOT_LABELS:
        slot_id = SLOT_LABELS[shift_lower]
        df = df[df["time_slot"] == slot_id]
        shift_label = SLOT_NAMES[slot_id]
    else:
        shift_label = "All Shifts"

    if df.empty:
        raise HTTPException(status_code=404,
                            detail=f"No data for shift '{shift}'")

    # Aggregate per zone
    zone_stats = (
        df.groupby("Community Area")
        .agg(
            mean_ccrp    = ("ccrp_bayesian", "mean"),
            max_ccrp     = ("ccrp_bayesian", "max"),
            alert_count  = ("alert_flag",    "sum") if "alert_flag" in df.columns
                           else ("ccrp_bayesian", "count"),
            lat          = ("lat",           "mean") if "lat" in df.columns
                           else ("Community Area", "count"),
            lon          = ("lon",           "mean") if "lon" in df.columns
                           else ("Community Area", "count"),
        )
        .reset_index()
    )

    # Include lat/lon from summary if not in master
    if summary_df is not None and "lat" not in master_df.columns:
        coords = summary_df[["zone_id", "lat", "lon"]].rename(
            columns={"zone_id": "Community Area"}
        )
        zone_stats = zone_stats.merge(coords, on="Community Area", how="left")

    # Priority score: 60% mean CCRP + 40% max CCRP
    zone_stats["priority_score"] = (
        0.60 * zone_stats["mean_ccrp"] +
        0.40 * zone_stats["max_ccrp"]
    ).clip(0, 1)

    top = zone_stats.sort_values("priority_score", ascending=False).head(top_n)

    recommendations = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        slot_id = SLOT_LABELS.get(shift_lower, 0)
        recommendations.append(PatrolSlot(
            rank=rank,
            zone_id=int(row["Community Area"]),
            time_slot=slot_id,
            slot_name=shift_label,
            mean_ccrp=round(float(row["mean_ccrp"]), 4),
            risk_level=assign_risk(float(row["mean_ccrp"])),
            priority_score=round(float(row["priority_score"]), 4),
            lat=float(row["lat"]) if pd.notna(row.get("lat")) else None,
            lon=float(row["lon"]) if pd.notna(row.get("lon")) else None,
        ))

    return PatrolRecommendationResponse(
        shift=shift_label,
        total_zones=len(recommendations),
        recommendations=recommendations,
    )


# ─────────────────────────────────────────────────────────────
# GET /api/patrol/hotspot-route
# ─────────────────────────────────────────────────────────────

@router.get(
    "/hotspot-route",
    response_model=PatrolRouteResponse,
    summary="Ordered nearest-neighbour patrol route through top-K hotspots"
)
def get_hotspot_route(
    request: Request,
    k: int = Query(
        10,
        ge=2,
        le=30,
        description="Number of hotspot zones to include in route"
    ),
    year: int = Query(None, description="Year filter"),
    slot: int = Query(None, ge=0, le=3, description="Time slot filter"),
):
    master_df  = get_master(request)
    summary_df = get_summary(request)

    if master_df is None:
        raise HTTPException(status_code=503, detail="Master dataset not loaded")

    df = master_df.copy()

    if year is not None:
        df = df[df["year"] == year]
    else:
        df = df[df["year"] == int(df["year"].max())]

    if slot is not None:
        df = df[df["time_slot"] == slot]

    # Top-K zones by mean CCRP
    hotspots = (
        df.groupby("Community Area")
        .agg(
            mean_ccrp = ("ccrp_bayesian", "mean"),
            lat       = ("lat", "mean") if "lat" in df.columns
                        else ("Community Area", "first"),
            lon       = ("lon", "mean") if "lon" in df.columns
                        else ("Community Area", "first"),
        )
        .reset_index()
        .sort_values("mean_ccrp", ascending=False)
        .head(k)
    )

    # Include lat/lon from summary if missing
    if summary_df is not None and "lat" not in master_df.columns:
        coords = summary_df[["zone_id", "lat", "lon"]].rename(
            columns={"zone_id": "Community Area"}
        )
        hotspots = hotspots.merge(coords, on="Community Area", how="left")

    points = [
        {
            "zone_id":   int(row["Community Area"]),
            "lat":       float(row["lat"]) if pd.notna(row.get("lat")) else None,
            "lon":       float(row["lon"]) if pd.notna(row.get("lon")) else None,
            "mean_ccrp": round(float(row["mean_ccrp"]), 4),
        }
        for _, row in hotspots.iterrows()
    ]

    # Apply nearest-neighbour routing
    route = nearest_neighbour_route(points)

    waypoints = [
        PatrolRouteStop(
            order=i + 1,
            zone_id=p["zone_id"],
            lat=p.get("lat"),
            lon=p.get("lon"),
            mean_ccrp=p["mean_ccrp"],
            risk_level=assign_risk(p["mean_ccrp"]),
        )
        for i, p in enumerate(route)
    ]

    total_risk = float(np.mean([w.mean_ccrp for w in waypoints]))

    return PatrolRouteResponse(
        route_length=len(waypoints),
        total_risk_covered=round(total_risk, 4),
        waypoints=waypoints,
    )
