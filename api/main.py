"""
UCMS - FastAPI Backend (main.py)
================================
Phase II: v2.0.0
  - Loads forecasts_2026.parquet and anomaly_flags.parquet
  - WebSocket /ws/live for real-time score streaming
  - In-memory response caching (60s TTL)
  - New routers: /api/forecast, /api/patrol

Run server:
    uvicorn api.main:app --reload --port 8000

Swagger UI:
    http://localhost:8000/docs

ReDoc:
    http://localhost:8000/redoc
"""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ─────────────────────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────────────────────

from api.routers import (
    zones,
    predict,
    alerts,
    stats,
)
from api.routers import forecast as forecast_router
from api.routers import patrol  as patrol_router

# ─────────────────────────────────────────────────────────────
# BASE DIRECTORY
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────────────────────

MASTER_PATH = BASE_DIR / "ccrp_fusion" / "outputs" / "ccrp_final_master.parquet"
SUMMARY_PATH = BASE_DIR / "ccrp_fusion" / "outputs" / "ccrp_zone_summary.csv"
WEIGHTS_PATH = BASE_DIR / "ccrp_fusion" / "outputs" / "learned_stream_weights.csv"
INCIDENTS_PATH = BASE_DIR / "outputs" / "chicago_crimes_clean.parquet"

# Phase II: new datasets
FORECAST_PATH = BASE_DIR / "outputs" / "forecasts_2026.parquet"
ANOMALY_PATH  = BASE_DIR / "outputs" / "anomaly_flags.parquet"

# ─────────────────────────────────────────────────────────────
# GLOBAL CACHE  (simple TTL dict — thread-safe for single worker)
# ─────────────────────────────────────────────────────────────

_cache: Dict[str, Any] = {}
CACHE_TTL = 60  # seconds


def cache_get(key: str):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL:
        return entry["data"]
    return None


def cache_set(key: str, data: Any):
    _cache[key] = {"data": data, "ts": time.time()}


# ─────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────

app_state = {}

# WebSocket connection manager
active_ws_connections: list[WebSocket] = []


# ─────────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("\nLoading UCMS datasets...\n")

    try:
        # ── Phase I datasets ──────────────────────────────
        app_state["master"]  = pd.read_parquet(MASTER_PATH)
        app_state["summary"] = pd.read_csv(SUMMARY_PATH)
        app_state["weights"] = pd.read_csv(WEIGHTS_PATH)

        app.state.master  = app_state["master"]
        app.state.summary = app_state["summary"]
        app.state.weights = app_state["weights"]

        print(f"[OK] Master records loaded: {len(app_state['master']):,}")
        print(f"[OK] Summary zones loaded : {len(app_state['summary'])}")
        print(f"[OK] Weight rows loaded   : {len(app_state['weights'])}")

    except Exception as e:
        print("\n[ERROR] Failed to load Phase I datasets\n")
        print(str(e))
        app_state["master"]  = None
        app_state["summary"] = None
        app_state["weights"] = None
        app.state.master  = None
        app.state.summary = None
        app.state.weights = None

    try:
        inc_df = pd.read_parquet(INCIDENTS_PATH)
        # Keep only columns needed for incident display
        inc_cols = [c for c in ["Community Area","year","month","hour","Primary Type",
                                 "Description","Location Description","severity_weight",
                                 "Latitude","Longitude","Date"] if c in inc_df.columns]
        app_state["incidents"] = inc_df[inc_cols]
        app.state.incidents    = app_state["incidents"]
        print(f"[OK] Incident records loaded : {len(inc_df):,}")
    except Exception as e:
        print(f"[WARN] Incident data not found — {e}")
        app_state["incidents"] = None
        app.state.incidents    = None

    # -- Phase II datasets (optional -- graceful degradation) --
    try:
        fc_df = pd.read_parquet(FORECAST_PATH)
        app_state["forecasts"] = fc_df
        app.state.forecasts    = fc_df
        print(f"[OK] Forecast records loaded : {len(fc_df):,}")
    except Exception as e:
        print(f"[WARN] Forecast data not found ({FORECAST_PATH.name}) -- run 13_temporal_forecasting.py")
        app_state["forecasts"] = None
        app.state.forecasts    = None

    try:
        an_df = pd.read_parquet(ANOMALY_PATH)
        app_state["anomalies"] = an_df
        app.state.anomalies    = an_df
        print(f"[OK] Anomaly records loaded  : {len(an_df):,}")
    except Exception as e:
        print(f"[WARN] Anomaly data not found ({ANOMALY_PATH.name}) -- run 14_anomaly_detection.py")
        app_state["anomalies"] = None
        app.state.anomalies    = None

    print("\nUCMS API v2.0.0 is ready.\n")

    yield

    # ── Cleanup ───────────────────────────────────────────
    app_state.clear()
    _cache.clear()
    print("\nUCMS API shutdown complete.\n")


# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────

app = FastAPI(

    title="UCMS — Urban Crime Monitoring System API",

    description=(
        "Contextual Crime Risk Propagation (CCRP) API v2.0.0.\n\n"

        "Provides historical, real-time, and **predictive** "
        "crime-risk analytics using:\n"

        "- Graph Attention Networks (GAT)\n"
        "- Multi-stream sensor fusion\n"
        "- Bayesian recalibration\n"
        "- Zone-based risk analysis\n"
        "- **[Phase II]** LSTM temporal forecasting\n"
        "- **[Phase II]** Isolation Forest anomaly detection\n"
        "- **[Phase II]** AI patrol routing\n"
        "- **[Phase II]** WebSocket live updates"
    ),

    version="2.0.0",

    lifespan=lifespan
)

# ─────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─────────────────────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────────────────────

app.include_router(zones.router,           prefix="/api/zones",    tags=["Zones"])
app.include_router(predict.router,         prefix="/api/predict",  tags=["Predict"])
app.include_router(alerts.router,          prefix="/api/alerts",   tags=["Alerts"])
app.include_router(stats.router,           prefix="/api/stats",    tags=["Statistics"])
app.include_router(forecast_router.router, prefix="/api/forecast", tags=["Forecast (Phase II)"])
app.include_router(patrol_router.router,   prefix="/api/patrol",   tags=["Patrol (Phase II)"])

# ─────────────────────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():

    return {

        "system":
            "Urban Crime Monitoring System (UCMS)",

        "algorithm":
            "CCRP — Contextual Crime Risk Propagation",

        "version": "2.0.0",

        "phase": "Phase II",

        "status":
            "operational",

        "zones_loaded":
            len(app_state.get("summary", []))
            if app_state.get("summary") is not None
            else 0,

        "records_loaded":
            len(app_state.get("master", []))
            if app_state.get("master") is not None
            else 0,

        "forecast_loaded":
            app_state.get("forecasts") is not None,

        "anomaly_loaded":
            app_state.get("anomalies") is not None,

        "available_endpoints": {
            "zones":              "/api/zones",
            "zone_detail":        "/api/zones/{zone_id}",
            "predict":            "/api/predict",
            "predict_batch":      "/api/predict/batch",
            "alerts":             "/api/alerts",
            "anomaly_alerts":     "/api/alerts/anomalies",
            "export_alerts":      "/api/alerts/export",
            "stats":              "/api/stats/overview",
            "forecast_zone":      "/api/forecast/{zone_id}",
            "forecast_citywide":  "/api/forecast/citywide",
            "forecast_rising":    "/api/forecast/top-rising",
            "patrol_recommend":   "/api/patrol/recommend",
            "patrol_route":       "/api/patrol/hotspot-route",
            "websocket_live":     "/ws/live",
            "swagger_docs":       "/docs",
            "redoc":              "/redoc",
        }
    }


# ─────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health():

    master    = app_state.get("master")
    summary   = app_state.get("summary")
    weights   = app_state.get("weights")
    forecasts = app_state.get("forecasts")
    anomalies = app_state.get("anomalies")

    return {

        "status": "ok",

        "version": "2.0.0",

        "datasets_loaded": {
            "master":    master    is not None,
            "summary":   summary   is not None,
            "weights":   weights   is not None,
            "forecasts": forecasts is not None,
            "anomalies": anomalies is not None,
        },

        "records_loaded":
            len(master)   if master   is not None else 0,

        "forecast_records":
            len(forecasts) if forecasts is not None else 0,

        "anomaly_records":
            len(anomalies) if anomalies is not None else 0,

        "zones_loaded":
            len(summary)  if summary  is not None else 0,

        "cache_entries": len(_cache),
    }


# ─────────────────────────────────────────────────────────────
# WEBSOCKET — LIVE UPDATES  (Phase II)
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    WebSocket endpoint that streams live CCRP snapshots every 10 seconds.
    Sends JSON:  { "type": "snapshot", "data": [ {zone_id, mean_ccrp, risk_level}, ... ] }
    """
    await websocket.accept()
    active_ws_connections.append(websocket)

    try:
        while True:
            # Build snapshot from latest-year summary
            summary = app_state.get("summary")
            if summary is not None:
                snapshot = []
                for _, row in summary.head(200).iterrows():
                    snapshot.append({
                        "zone_id":   int(row.get("zone_id", 0)),
                        "mean_ccrp": round(float(row.get("mean_ccrp", 0)), 4),
                        "risk_level": str(row.get("risk_level", "LOW")),
                    })
                payload = json.dumps({"type": "snapshot", "data": snapshot})
                await websocket.send_text(payload)

            await asyncio.sleep(10)   # push every 10 seconds

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in active_ws_connections:
            active_ws_connections.remove(websocket)