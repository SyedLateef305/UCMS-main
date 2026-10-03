"""
UCMS API — routers/zones.py

Endpoints:
GET /api/zones
GET /api/zones/{zone_id}
"""

from typing import Optional

import pandas as pd

from fastapi import (
    APIRouter,
    Query,
    HTTPException,
    Request
)

from api.models.schemas import (
    ZoneSummary,
    ZoneDetail,
    ZoneListResponse
)

from api.core.dependencies import (
    get_summary,
    get_master
)

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
# GET ALL ZONES
# ─────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=ZoneListResponse,
    summary="Get all zone summaries"
)
def get_all_zones(

    request: Request,

    risk_level: Optional[str] = Query(
        None,
        description="LOW | MEDIUM | HIGH | CRITICAL"
    ),

    trend: Optional[str] = Query(
        None,
        description="RISING | STABLE | FALLING"
    ),

    min_score: float = Query(
        0.0,
        ge=0,
        le=1
    ),

    max_score: float = Query(
        1.0,
        ge=0,
        le=1
    ),

    sort_by: str = Query(
        "mean_ccrp",
        description="mean_ccrp | peak_ccrp | alert_count | zone_id"
    ),

    descending: bool = Query(True),

    limit: int = Query(
        200,
        ge=1,
        le=500
    )
):

    df = get_summary(request)

    if df is None:
        raise HTTPException(
            status_code=503,
            detail="Summary dataset not loaded"
        )

    # ─────────────────────────────────────────
    # FILTERS
    # ─────────────────────────────────────────

    if risk_level:

        df = df[
            df["risk_level"] == risk_level.upper()
        ]

    if trend:

        df = df[
            df["trend"] == trend.upper()
        ]

    df = df[
        (df["mean_ccrp"] >= min_score) &
        (df["mean_ccrp"] <= max_score)
    ]

    # ─────────────────────────────────────────
    # SORTING
    # ─────────────────────────────────────────

    allowed_sorts = [
        "mean_ccrp",
        "peak_ccrp",
        "alert_count",
        "zone_id"
    ]

    if sort_by in allowed_sorts and sort_by in df.columns:

        df = df.sort_values(
            sort_by,
            ascending=not descending
        )

    # ─────────────────────────────────────────
    # LIMIT
    # ─────────────────────────────────────────

    df = df.head(limit)

    # ─────────────────────────────────────────
    # RESPONSE BUILD
    # ─────────────────────────────────────────

    zones = []

    for _, row in df.iterrows():

        zones.append(

            ZoneSummary(

                zone_id=int(
                    row["zone_id"]
                ),

                zone_name=(
                    str(row["zone_name"])
                    if pd.notna(row.get("zone_name"))
                    else None
                ),

                district=(
                    str(row["district"])
                    if pd.notna(row.get("district"))
                    else None
                ),

                mean_ccrp=round(
                    float(row["mean_ccrp"]),
                    4
                ),

                peak_ccrp=round(
                    float(row["peak_ccrp"]),
                    4
                ),

                risk_level=row["risk_level"],

                trend=row["trend"],

                peak_slot=str(
                    row.get("peak_slot", "N/A")
                ),

                alert_count=int(
                    row.get("alert_count", 0)
                ),

                mean_uncertainty=round(
                    float(
                        row.get(
                            "mean_uncertainty",
                            0
                        )
                    ),
                    4
                ),

                lat=(
                    float(row["lat"])
                    if pd.notna(row.get("lat"))
                    else None
                ),

                lon=(
                    float(row["lon"])
                    if pd.notna(row.get("lon"))
                    else None
                )
            )
        )

    return ZoneListResponse(
        total=len(zones),
        zones=zones
    )


# ─────────────────────────────────────────────────────────────
# GET SINGLE ZONE DETAIL
# ─────────────────────────────────────────────────────────────

@router.get(
    "/{zone_id}",
    response_model=ZoneDetail,
    summary="Get detailed information for a zone"
)
def get_zone_detail(

    zone_id: int,

    request: Request
):

    summary_df = get_summary(request)

    master_df = get_master(request)

    if summary_df is None or master_df is None:

        raise HTTPException(
            status_code=503,
            detail="Dataset not loaded"
        )

    # ─────────────────────────────────────────
    # FIND ZONE
    # ─────────────────────────────────────────

    row = summary_df[
        summary_df["zone_id"] == zone_id
    ]

    if row.empty:

        raise HTTPException(
            status_code=404,
            detail=f"Zone {zone_id} not found"
        )

    row = row.iloc[0]

    # ─────────────────────────────────────────
    # MASTER DATA FOR THIS ZONE
    # ─────────────────────────────────────────

    zdf = master_df[
        master_df["Community Area"] == zone_id
    ]

    # ─────────────────────────────────────────
    # SLOT SCORES
    # ─────────────────────────────────────────

    slot_scores = {}

    for slot, name in SLOT_NAMES.items():

        val = zdf[
            zdf["time_slot"] == slot
        ]["ccrp_bayesian"].mean()

        slot_scores[name] = (
            round(float(val), 4)
            if not pd.isna(val)
            else 0.0
        )

    # ─────────────────────────────────────────
    # YEAR TREND
    # ─────────────────────────────────────────

    year_trend = (

        zdf.groupby("year")[
            "ccrp_bayesian"
        ]

        .mean()

        .reset_index()

        .rename(
            columns={
                "ccrp_bayesian": "score"
            }
        )

        .sort_values("year")
    )

    year_trend_list = [

        {
            "year": int(r["year"]),

            "score": round(
                float(r["score"]),
                4
            )
        }

        for _, r in year_trend.iterrows()
    ]

    # ─────────────────────────────────────────
    # RESPONSE
    # ─────────────────────────────────────────

    return ZoneDetail(

        zone_id=int(zone_id),

        zone_name=(
            str(row["zone_name"])
            if pd.notna(row.get("zone_name"))
            else None
        ),

        district=(
            str(row["district"])
            if pd.notna(row.get("district"))
            else None
        ),

        risk_level=row["risk_level"],

        trend=row["trend"],

        mean_ccrp=round(
            float(row["mean_ccrp"]),
            4
        ),

        peak_ccrp=round(
            float(row["peak_ccrp"]),
            4
        ),

        alert_count=int(
            row.get("alert_count", 0)
        ),

        mean_uncertainty=round(
            float(
                row.get(
                    "mean_uncertainty",
                    0
                )
            ),
            4
        ),

        lat=(
            float(row["lat"])
            if pd.notna(row.get("lat"))
            else None
        ),

        lon=(
            float(row["lon"])
            if pd.notna(row.get("lon"))
            else None
        ),

        slot_scores=slot_scores,

        year_trend=year_trend_list
    )


# ─────────────────────────────────────────────────────────────
# GET ZONE INCIDENTS — /api/zones/{zone_id}/incidents
# ─────────────────────────────────────────────────────────────

@router.get("/{zone_id}/incidents")
def get_zone_incidents(
    zone_id: int,
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    year:  Optional[int] = Query(None, description="Filter by year (e.g. 2024)"),
    crime_type: Optional[str] = Query(None, description="Filter by crime type"),
):
    """
    Return a list of actual crime incidents for a specific zone.
    Data is sourced from the cleaned Indian crimes parquet.
    """
    inc_df: pd.DataFrame = getattr(request.app.state, "incidents", None)
    if inc_df is None:
        raise HTTPException(status_code=503, detail="Incidents dataset not available")

    summary_df = get_summary(request)
    zone_row = summary_df[summary_df["zone_id"] == zone_id]

    zone_name = str(zone_row["zone_name"].iloc[0]) if len(zone_row) and "zone_name" in zone_row.columns else f"Zone {zone_id}"
    district  = str(zone_row["district"].iloc[0])  if len(zone_row) and "district"  in zone_row.columns else ""

    # Filter by zone
    mask = inc_df["Community Area"] == zone_id
    if year:
        mask &= inc_df["year"] == year
    if crime_type:
        mask &= inc_df["Primary Type"].str.upper() == crime_type.upper()

    zdf = inc_df[mask].copy()
    total = len(zdf)

    # Sort by date descending (most recent first)
    if "Date" in zdf.columns:
        zdf = zdf.sort_values("Date", ascending=False)

    zdf = zdf.head(limit)

    incidents = []
    for _, row in zdf.iterrows():
        date_val = row.get("Date")
        if hasattr(date_val, "strftime"):
            date_str = date_val.strftime("%Y-%m-%d")
            time_str = date_val.strftime("%H:%M")
        else:
            date_str = str(date_val)[:10] if pd.notna(date_val) else "Unknown"
            time_str = ""

        incidents.append({
            "date":        date_str,
            "time":        time_str,
            "year":        int(row["year"]) if pd.notna(row.get("year")) else None,
            "crime_type":  str(row.get("Primary Type", "")).title(),
            "description": str(row.get("Description", "")),
            "location":    str(row.get("Location Description", "")),
            "severity":    round(float(row["severity_weight"]), 3) if pd.notna(row.get("severity_weight")) else None,
            "lat":         round(float(row["Latitude"]),  6) if pd.notna(row.get("Latitude"))  else None,
            "lon":         round(float(row["Longitude"]), 6) if pd.notna(row.get("Longitude")) else None,
        })

    # Crime type breakdown
    crime_counts = inc_df[inc_df["Community Area"] == zone_id]["Primary Type"].value_counts().head(8).to_dict()
    crime_breakdown = [{"type": k.title(), "count": int(v)} for k, v in crime_counts.items()]

    return {
        "zone_id":        zone_id,
        "zone_name":      zone_name,
        "district":       district,
        "total_records":  total,
        "filtered_by":    {"year": year, "crime_type": crime_type},
        "crime_breakdown": crime_breakdown,
        "incidents":      incidents,
    }