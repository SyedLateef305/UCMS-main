// src/api/ucmsApi.js
// Phase II: Added forecast, patrol, anomaly, feature-importance, heatmap, WebSocket helpers

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const get = (path) =>
  fetch(`${BASE}${path}`).then((r) => {
    if (!r.ok) throw new Error(`API ${r.status}: ${path}`);
    return r.json();
  });

const post = (path, body) =>
  fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then((r) => {
    if (!r.ok) throw new Error(`API ${r.status}: ${path}`);
    return r.json();
  });

// ── ZONES ──────────────────────────────────────────────────────
export const getAllZones       = (params = {}) => {
  const qs = new URLSearchParams(params).toString();
  return get(`/api/zones${qs ? "?" + qs : ""}`);
};

export const getZoneDetail     = (zoneId)       => get(`/api/zones/${zoneId}`);
export const getZoneIncidents  = (zoneId, params = {}) => {
  const qs = new URLSearchParams(params).toString();
  return get(`/api/zones/${zoneId}/incidents${qs ? "?" + qs : ""}`);
};

// ── PREDICT ────────────────────────────────────────────────────
export const predictZone       = (body)          => post("/api/predict", body);
export const getLatestZone     = (zoneId)        => get(`/api/predict/zone/${zoneId}/latest`);
export const getZoneHistory    = (zoneId, slot)  =>
  get(`/api/predict/zone/${zoneId}/history${slot != null ? "?time_slot=" + slot : ""}`);

/** Phase II: batch predict up to 50 zone+slot pairs */
export const batchPredict      = (items)         => post("/api/predict/batch", items);

// ── ALERTS ─────────────────────────────────────────────────────
export const getAlerts         = (params = {})   => {
  const qs = new URLSearchParams(params).toString();
  return get(`/api/alerts${qs ? "?" + qs : ""}`);
};

export const getCriticalAlerts = (year)          =>
  get(`/api/alerts/critical${year ? "?year=" + year : ""}`);

export const getAlertSummary   = (threshold = 0.70) =>
  get(`/api/alerts/summary?threshold=${threshold}`);

export const getHotspots       = (n = 10, year, slot) => {
  const params = new URLSearchParams({ n });
  if (year != null) params.append("year", year);
  if (slot != null) params.append("slot", slot);
  return get(`/api/alerts/hotspots?${params}`);
};

/** Phase II: anomaly-flagged records */
export const getAnomalyAlerts  = (threshold = 0.60, year) => {
  const params = new URLSearchParams({ threshold });
  if (year != null) params.append("year", year);
  return get(`/api/alerts/anomalies?${params}`);
};

/** Phase II: download alerts CSV — returns blob URL */
export const exportAlerts      = (threshold = 0.70, year, slot) => {
  const params = new URLSearchParams({ threshold });
  if (year  != null) params.append("year",      year);
  if (slot  != null) params.append("time_slot", slot);
  return `${BASE}/api/alerts/export?${params}`;
};

// ── STATS ──────────────────────────────────────────────────────
export const getOverview       = ()              => get("/api/stats/overview");
export const getTrends         = (zoneId, slot)  => {
  const params = new URLSearchParams();
  if (zoneId != null) params.append("zone_id",   zoneId);
  if (slot   != null) params.append("time_slot", slot);
  return get(`/api/stats/trends${params.toString() ? "?" + params : ""}`);
};

export const getStreamWeights  = ()              => get("/api/stats/streams");

export const getSlotProfile    = (year)          =>
  get(`/api/stats/slot-profile${year ? "?year=" + year : ""}`);

export const getRiskDistribution = (bins = 20)  =>
  get(`/api/stats/risk-distribution?bins=${bins}`);

export const getZoneComparison  = (zoneIds, year) => {
  const params = new URLSearchParams({ zone_ids: zoneIds.join(",") });
  if (year != null) params.append("year", year);
  return get(`/api/stats/zone-comparison?${params}`);
};

/** Phase II: feature importance */
export const getFeatureImportance = (zoneId, year) => {
  const params = new URLSearchParams();
  if (zoneId != null) params.append("zone_id", zoneId);
  if (year   != null) params.append("year",    year);
  return get(`/api/stats/feature-importance?${params}`);
};

/** Phase II: multi-zone comparison trend */
export const getZoneComparisonTrend = (zoneIds, slot) => {
  const params = new URLSearchParams({ zone_ids: zoneIds.join(",") });
  if (slot != null) params.append("time_slot", slot);
  return get(`/api/stats/zone-comparison-trend?${params}`);
};

/** Phase II: slot × zone heatmap matrix */
export const getHeatmapMatrix = (year, topN = 20) => {
  const params = new URLSearchParams({ top_n: topN });
  if (year != null) params.append("year", year);
  return get(`/api/stats/heatmap?${params}`);
};

// ── FORECAST (Phase II) ────────────────────────────────────────
export const getZoneForecast    = (zoneId)       => get(`/api/forecast/${zoneId}`);
export const getCitywideForecast = ()            => get("/api/forecast/citywide");
export const getTopRisingZones  = (n = 10)       => get(`/api/forecast/top-rising?n=${n}`);

// ── PATROL (Phase II) ──────────────────────────────────────────
export const getPatrolRecommendation = (shift = "night", topN = 15, year) => {
  const params = new URLSearchParams({ shift, top_n: topN });
  if (year != null) params.append("year", year);
  return get(`/api/patrol/recommend?${params}`);
};

export const getPatrolRoute     = (k = 10, year, slot) => {
  const params = new URLSearchParams({ k });
  if (year != null) params.append("year", year);
  if (slot != null) params.append("slot", slot);
  return get(`/api/patrol/hotspot-route?${params}`);
};

// ── WEBSOCKET (Phase II) ───────────────────────────────────────
const WS_BASE = (BASE.replace("http://", "ws://").replace("https://", "wss://"));

/**
 * Connect to live WebSocket stream.
 * @param {function} onMessage  called with each parsed JSON message
 * @returns {WebSocket}         the WS instance (call .close() to disconnect)
 */
export const connectLiveWebSocket = (onMessage) => {
  const ws = new WebSocket(`${WS_BASE}/ws/live`);
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (_) {}
  };
  ws.onerror = (e) => console.warn("UCMS WebSocket error:", e);
  return ws;
};