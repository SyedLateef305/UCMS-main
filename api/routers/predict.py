"""
UCMS API — routers/predict.py

Endpoints:
POST /api/predict
GET  /api/predict/zone/{zone_id}/latest
GET  /api/predict/zone/{zone_id}/history
"""

from fastapi import APIRouter, Request, HTTPException
from api.models.schemas import (
    PredictRequest,
    PredictResponse,
    RiskLevel
)
from api.core.dependencies import (
    get_master,
    get_weights
)

import pandas as pd
import numpy as np
from scipy.stats import beta as beta_dist

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
# STREAM COLUMNS
# ─────────────────────────────────────────────────────────────

STREAM_COLS = [
    "crowd_density",
    "lighting_risk",
    "social_media",
    "suspicious_activity",
    "cctv_feed",
    "historical_cps"
]


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
# POST /api/predict
# ─────────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=PredictResponse,
    summary="Predict CCRP score"
)
def predict(
    req: PredictRequest,
    request: Request
):

    master_df = get_master(request)

    weights_df = get_weights(request)

    if master_df is None:
        raise HTTPException(
            status_code=503,
            detail="Master dataset not loaded"
        )

    # ---------------------------------------------------------
    # Find matching record
    # ---------------------------------------------------------

    mask = (
        (master_df["Community Area"] == req.zone_id) &
        (master_df["year"] == req.year) &
        (master_df["time_slot"] == req.time_slot)
    )

    record = master_df[mask]

    # fallback to latest year
    if record.empty:

        zone_df = master_df[
            (master_df["Community Area"] == req.zone_id) &
            (master_df["time_slot"] == req.time_slot)
        ]

        if zone_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Zone {req.zone_id} not found"
            )

        record = zone_df.sort_values(
            "year"
        ).iloc[[-1]]

    row = record.iloc[0]

    # ---------------------------------------------------------
    # Live override values
    # ---------------------------------------------------------

    override_map = {
        "crowd_density": req.crowd_density,
        "lighting_risk": req.lighting_risk,
        "social_media": req.social_media,
        "suspicious_activity": req.suspicious_activity,
        "cctv_feed": req.cctv_score,
    }

    stream_vals = {}

    for col in STREAM_COLS:

        override = override_map.get(col)

        if override is not None:

            stream_vals[col] = float(override)

        else:

            if col in master_df.columns:
                stream_vals[col] = float(
                    row.get(col, 0.5)
                )
            else:
                stream_vals[col] = 0.5

    # ---------------------------------------------------------
    # Load weights
    # ---------------------------------------------------------

    slot_name = SLOT_NAMES[req.time_slot]

    if (
        weights_df is not None and
        "slot" in weights_df.columns
    ):

        slot_key = slot_name.split("(")[0].strip()

        wrow = weights_df[
            weights_df["slot"] == slot_key
        ]

        if not wrow.empty:

            weights = {
                col: float(
                    wrow.iloc[0].get(col, 1 / 6)
                )
                for col in STREAM_COLS
            }

        else:

            weights = {
                col: 1 / 6
                for col in STREAM_COLS
            }

    else:

        weights = {
            col: 1 / 6
            for col in STREAM_COLS
        }

    # ---------------------------------------------------------
    # Compute raw CCRP
    # ---------------------------------------------------------

    ccrp_raw = sum(
        weights[col] * stream_vals[col]
        for col in STREAM_COLS
    )

    ccrp_raw = float(
        np.clip(ccrp_raw, 0, 1)
    )

    # ---------------------------------------------------------
    # Bayesian recalibration
    # ---------------------------------------------------------

    has_override = any(
        v is not None
        for v in override_map.values()
    )

    if has_override:

        prior_alpha = float(
            row.get("prior_alpha", 2.0)
        )

        prior_beta = float(
            row.get("prior_beta", 2.0)
        )

        n_obs = 5

        post_alpha = (
            prior_alpha +
            ccrp_raw * n_obs
        )

        post_beta = (
            prior_beta +
            (1 - ccrp_raw) * n_obs
        )

        ccrp_bay = float(
            post_alpha /
            (post_alpha + post_beta)
        )

        lower_ci = float(
            beta_dist.ppf(
                0.05,
                post_alpha,
                post_beta
            )
        )

        upper_ci = float(
            beta_dist.ppf(
                0.95,
                post_alpha,
                post_beta
            )
        )

    else:

        ccrp_bay = float(
            row.get(
                "ccrp_bayesian",
                ccrp_raw
            )
        )

        lower_ci = float(
            row.get(
                "credible_lower_90",
                max(0, ccrp_bay - 0.1)
            )
        )

        upper_ci = float(
            row.get(
                "credible_upper_90",
                min(1, ccrp_bay + 0.1)
            )
        )

    # ---------------------------------------------------------
    # Final response
    # ---------------------------------------------------------

    risk = assign_risk(ccrp_bay)

    message = (
        f"Zone {req.zone_id} is "
        f"{risk.value} risk during "
        f"{slot_name}. "
        f"CCRP={ccrp_bay:.3f}"
    )

    confidence_band_pct = round((upper_ci - lower_ci) * 100, 2)

    return PredictResponse(
        zone_id=req.zone_id,

        year=req.year,

        time_slot=req.time_slot,

        slot_name=slot_name,

        ccrp_score=round(ccrp_raw, 4),

        ccrp_bayesian=round(ccrp_bay, 4),

        uncertainty_lower=round(lower_ci, 4),

        uncertainty_upper=round(upper_ci, 4),

        confidence_band_pct=confidence_band_pct,

        risk_level=risk,

        stream_weights={
            k: round(v, 4)
            for k, v in weights.items()
        },

        message=message
    )


# ─────────────────────────────────────────────────────────────
# GET LATEST ZONE SCORES
# ─────────────────────────────────────────────────────────────

@router.get(
    "/zone/{zone_id}/latest",
    summary="Latest CCRP score for a zone"
)
def get_latest(
    zone_id: int,
    request: Request
):

    master_df = get_master(request)

    if master_df is None:
        raise HTTPException(
            status_code=503,
            detail="Master dataset not loaded"
        )

    zdf = master_df[
        master_df["Community Area"] == zone_id
    ]

    if zdf.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Zone {zone_id} not found"
        )

    latest_year = int(
        zdf["year"].max()
    )

    latest = zdf[
        zdf["year"] == latest_year
    ]

    slots = []

    for _, row in latest.sort_values(
        "time_slot"
    ).iterrows():

        score = float(
            row["ccrp_bayesian"]
        )

        slots.append({

            "time_slot": int(
                row["time_slot"]
            ),

            "slot_name": SLOT_NAMES[
                int(row["time_slot"])
            ],

            "ccrp_bayesian": round(
                score,
                4
            ),

            "risk_level": assign_risk(
                score
            ).value,

            "uncertainty": round(
                float(
                    row.get("uncertainty", 0)
                ),
                4
            )
        })

    mean_score = float(
        latest["ccrp_bayesian"].mean()
    )

    return {
        "zone_id": zone_id,

        "year": latest_year,

        "mean_ccrp": round(
            mean_score,
            4
        ),

        "risk_level": assign_risk(
            mean_score
        ).value,

        "slots": slots
    }


# ─────────────────────────────────────────────────────────────
# GET ZONE HISTORY
# ─────────────────────────────────────────────────────────────

@router.get(
    "/zone/{zone_id}/history",
    summary="Year-by-year CCRP history"
)
def get_history(
    zone_id: int,
    request: Request,
    time_slot: int = None
):

    master_df = get_master(request)

    if master_df is None:
        raise HTTPException(
            status_code=503,
            detail="Master dataset not loaded"
        )

    zdf = master_df[
        master_df["Community Area"] == zone_id
    ]

    if zdf.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Zone {zone_id} not found"
        )

    if time_slot is not None:

        zdf = zdf[
            zdf["time_slot"] == time_slot
        ]

    history = (
        zdf.groupby("year")[
            "ccrp_bayesian"
        ]
        .mean()
        .reset_index()
        .sort_values("year")
    )

    return {
        "zone_id": zone_id,

        "time_slot": time_slot,

        "history": [

            {
                "year": int(r["year"]),

                "ccrp": round(
                    float(r["ccrp_bayesian"]),
                    4
                )
            }

            for _, r in history.iterrows()
        ]
    }


# ─────────────────────────────────────────────────────────────
# PHASE II — BATCH PREDICT
# ─────────────────────────────────────────────────────────────

from typing import List as TypingList
from pydantic import BaseModel as PydanticBaseModel


class BatchItem(PydanticBaseModel):
    zone_id: int
    year: int = 2025
    time_slot: int


class BatchPredictResponse(PydanticBaseModel):
    total: int
    results: list


@router.post(
    "/batch",
    response_model=BatchPredictResponse,
    summary="[Phase II] Bulk CCRP predictions for multiple zone+slot combinations"
)
def batch_predict(
    items: TypingList[BatchItem],
    request: Request
):
    if len(items) > 50:
        raise HTTPException(
            status_code=422,
            detail="Batch size limited to 50 items"
        )

    master_df  = get_master(request)
    weights_df = get_weights(request)

    if master_df is None:
        raise HTTPException(status_code=503, detail="Master dataset not loaded")

    results = []

    for item in items:
        mask = (
            (master_df["Community Area"] == item.zone_id) &
            (master_df["year"]           == item.year)    &
            (master_df["time_slot"]      == item.time_slot)
        )
        record = master_df[mask]

        if record.empty:
            # Fallback: latest available year for this zone/slot
            zone_df = master_df[
                (master_df["Community Area"] == item.zone_id) &
                (master_df["time_slot"]      == item.time_slot)
            ]
            if zone_df.empty:
                results.append({
                    "zone_id":   item.zone_id,
                    "year":      item.year,
                    "time_slot": item.time_slot,
                    "error":     "not found"
                })
                continue
            record = zone_df.sort_values("year").iloc[[-1]]

        row = record.iloc[0]
        ccrp_bay = round(float(row.get("ccrp_bayesian", 0.5)), 4)
        slot_name = SLOT_NAMES.get(item.time_slot, "Unknown")

        risk = assign_risk(ccrp_bay)

        results.append({
            "zone_id":       item.zone_id,
            "year":          item.year,
            "time_slot":     item.time_slot,
            "slot_name":     slot_name,
            "ccrp_bayesian": ccrp_bay,
            "risk_level":    risk.value,
            "uncertainty":   round(float(row.get("uncertainty", 0.1)), 4),
        })

    return BatchPredictResponse(total=len(results), results=results)