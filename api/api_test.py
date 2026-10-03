"""
UCMS - API Test Suite (Phase I & II)
=====================================
Supports both live server and in-process TestClient testing.

Run:
    python api/api_test.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient
from api.main import app

TESTS = [
    # ── Phase I: Root & Health ──
    ("GET", "/", None, 200, "Root endpoint"),
    ("GET", "/health", None, 200, "Health check"),

    # ── Phase I: Zones ──
    ("GET", "/api/zones", None, 200, "All zones"),
    ("GET", "/api/zones?risk_level=HIGH", None, 200, "HIGH risk zones"),
    ("GET", "/api/zones/25", None, 200, "Zone 25 details"),
    ("GET", "/api/zones/999", None, 404, "Invalid zone"),

    # ── Phase I: Predict ──
    ("POST", "/api/predict", {"zone_id": 25, "year": 2024, "time_slot": 3}, 200, "Predict zone 25"),
    ("GET", "/api/predict/zone/25/latest", None, 200, "Latest prediction"),
    ("GET", "/api/predict/zone/25/history", None, 200, "Zone history"),

    # ── Phase II: Predict Batch ──
    ("POST", "/api/predict/batch", [{"zone_id": 25, "year": 2024, "time_slot": 3}, {"zone_id": 68, "year": 2024, "time_slot": 1}], 200, "Batch predict"),

    # ── Phase I & II: Alerts ──
    ("GET", "/api/alerts", None, 200, "All alerts"),
    ("GET", "/api/alerts/critical", None, 200, "Critical alerts"),
    ("GET", "/api/alerts/summary", None, 200, "Alert summary"),
    ("GET", "/api/alerts/hotspots?n=10", None, 200, "Top 10 hotspots"),
    ("GET", "/api/alerts/anomalies", None, 200, "[Phase II] Anomaly alerts"),
    ("GET", "/api/alerts/export?threshold=0.7", None, 200, "[Phase II] Export alerts CSV"),

    # ── Phase I & II: Stats ──
    ("GET", "/api/stats/overview", None, 200, "Overview stats"),
    ("GET", "/api/stats/trends", None, 200, "City trends"),
    ("GET", "/api/stats/streams", None, 200, "Stream weights"),
    ("GET", "/api/stats/risk-distribution", None, 200, "Risk distribution"),
    ("GET", "/api/stats/slot-profile", None, 200, "Slot profile"),
    ("GET", "/api/stats/feature-importance?zone_id=25", None, 200, "[Phase II] Feature importance"),
    ("GET", "/api/stats/zone-comparison-trend?zone_ids=25,68,71", None, 200, "[Phase II] Zone comparison trend"),
    ("GET", "/api/stats/heatmap", None, 200, "[Phase II] Heatmap matrix"),

    # ── Phase II: Forecast ──
    ("GET", "/api/forecast/25", None, 200, "[Phase II] Zone 25 forecast"),
    ("GET", "/api/forecast/citywide", None, 200, "[Phase II] Citywide forecast"),
    ("GET", "/api/forecast/top-rising?n=5", None, 200, "[Phase II] Top rising zones"),

    # ── Phase II: Patrol ──
    ("GET", "/api/patrol/recommend?shift=night&top_n=10", None, 200, "[Phase II] Patrol recommendations"),
    ("GET", "/api/patrol/hotspot-route?k=6", None, 200, "[Phase II] Patrol hotspot route"),
]


def run():
    print("=" * 70)
    print("UCMS API v2.0.0 - Test Suite (Phase I & II)")
    print("=" * 70)

    passed = 0
    failed = 0
    errors = []

    with TestClient(app) as client:
        for method, path, body, expected, desc in TESTS:
            try:
                if method == "GET":
                    response = client.get(path)
                else:
                    response = client.post(path, json=body)

                success = response.status_code == expected
                icon = "[PASS]" if success else "[FAIL]"
                print(f"{icon} [{response.status_code}] {method:<5} {path:<55} {desc}")

                if success:
                    passed += 1
                else:
                    failed += 1
                    errors.append(f"{method} {path} -> got {response.status_code}, expected {expected}")
                    try:
                        print("    Details:", response.json())
                    except Exception:
                        pass
            except Exception as e:
                failed += 1
                print(f"[ERR]  {method:<5} {path:<55} {desc} -> {e}")
                errors.append(f"{method} {path} -> {e}")

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed | {failed} failed | {len(TESTS)} total")

    if errors:
        print("\nFailed tests:")
        for err in errors:
            print(" -", err)
    else:
        print("\n[OK] All Phase I and Phase II tests passed successfully!")

    print("=" * 70 + "\n")
    return failed == 0


if __name__ == "__main__":
    success = run()
    sys.exit(0 if success else 1)