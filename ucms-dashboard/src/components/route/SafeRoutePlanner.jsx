// src/components/route/SafeRoutePlanner.jsx
// Interactive Safety Route Navigator with arbitrary Indian address search (Nominatim Geocoding),
// map-click coordinate selection, quick city chips, multi-style map layers, and corridor safety analysis.
import React, { useState, useMemo, useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Polyline,
  Popup,
  Marker,
  Tooltip,
  useMap,
  useMapEvents,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useApi } from "../../hooks/useApi";
import { getAllZones } from "../../api/ucmsApi";

// Custom pin icons
const createPinIcon = (color, title, subtitle) =>
  L.divIcon({
    className: "custom-route-pin",
    html: `<div style="
      background: ${color};
      color: white;
      font-size: 11px;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 14px;
      border: 2px solid #ffffff;
      box-shadow: 0 4px 12px rgba(0,0,0,0.6);
      white-space: nowrap;
      display: flex;
      flex-direction: column;
      align-items: center;
      line-height: 1.1;
    ">
      <span>${title}</span>
      ${subtitle ? `<span style="font-size: 9px; opacity: 0.85; font-weight: normal;">${subtitle}</span>` : ""}
    </div>`,
    iconSize: [110, 32],
    iconAnchor: [55, 16],
  });

const RISK_COLOR = {
  CRITICAL: "#ff1e40",
  HIGH:     "#fa8c16",
  MEDIUM:   "#fadb14",
  LOW:      "#52c41a",
};

// Great-circle distance calculation in km
function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// Distance from point P to line segment AB in km
function distToSegmentKm(pLat, pLon, aLat, aLon, bLat, bLon) {
  const dAB = haversineKm(aLat, aLon, bLat, bLon);
  if (dAB === 0) return haversineKm(pLat, pLon, aLat, aLon);
  const dAP = haversineKm(aLat, aLon, pLat, pLon);
  const dBP = haversineKm(bLat, bLon, pLat, pLon);
  if (dAP * dAP >= dBP * dBP + dAB * dAB) return dBP;
  if (dBP * dBP >= dAP * dAP + dAB * dAB) return dAP;
  const s = (dAB + dAP + dBP) / 2;
  const area = Math.sqrt(Math.max(0, s * (s - dAB) * (s - dAP) * (s - dBP)));
  return (2 * area) / dAB;
}

function MapRecenter({ start, end }) {
  const map = useMap();
  useEffect(() => {
    if (start && end) {
      const bounds = L.latLngBounds([start.lat, start.lon], [end.lat, end.lon]);
      map.fitBounds(bounds, { padding: [60, 60], maxZoom: 13 });
    }
  }, [start?.lat, start?.lon, end?.lat, end?.lon, map]);
  return null;
}

function MapClickHandler({ clickMode, onLocationPicked }) {
  useMapEvents({
    click(e) {
      if (!clickMode) return;
      onLocationPicked(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

export default function SafeRoutePlanner() {
  const { data } = useApi(() => getAllZones({ limit: 200 }), []);
  const zones = data?.zones || [];

  const [startPoint, setStartPoint] = useState({
    name: "Bangalore",
    address: "Bengaluru, Karnataka, India",
    lat: 12.9716,
    lon: 77.5946,
  });

  const [endPoint, setEndPoint] = useState({
    name: "Mysuru",
    address: "Mysuru, Karnataka, India",
    lat: 12.2958,
    lon: 76.6394,
  });

  const [travelSlot, setTravelSlot] = useState(1);
  const [mapStyle, setMapStyle] = useState("streets");
  const [clickMode, setClickMode] = useState(null);

  const [fromQuery, setFromQuery] = useState("Bangalore");
  const [toQuery, setToQuery]     = useState("Mysuru");
  const [fromSuggestions, setFromSuggestions] = useState([]);
  const [toSuggestions, setToSuggestions]     = useState([]);
  const [searchingFrom, setSearchingFrom]     = useState(false);
  const [searchingTo, setSearchingTo]         = useState(false);

  const searchAddress = async (query, setSuggestions, setLoading) => {
    if (!query || query.trim().length < 2) {
      setSuggestions([]);
      return;
    }
    setLoading(true);
    try {
      const url = "https://nominatim.openstreetmap.org/search?format=json&q=" +
        encodeURIComponent(query) + "&countrycodes=in&limit=5";
      const res = await fetch(url, { headers: { "User-Agent": "UCMS-RoutePlanner/2.0" } });
      const d = await res.json();
      setSuggestions(d || []);
    } catch (err) {
      console.warn("Geocoding error:", err);
      setSuggestions([]);
    } finally {
      setLoading(false);
    }
  };

  const selectCityPreset = (city, isFrom) => {
    const matched = zones.find((z) => (z.district || "").toLowerCase() === city.toLowerCase()) ||
      zones.find((z) => (z.zone_name || "").toLowerCase().includes(city.toLowerCase()));

    const coords = matched
      ? { lat: matched.lat, lon: matched.lon }
      : { lat: 12.9716, lon: 77.5946 };

    const point = {
      name: city,
      address: city + ", India",
      lat: coords.lat,
      lon: coords.lon,
    };

    if (isFrom) {
      setStartPoint(point);
      setFromQuery(city);
      setFromSuggestions([]);
    } else {
      setEndPoint(point);
      setToQuery(city);
      setToSuggestions([]);
    }
  };

  const handleMapLocationPicked = (lat, lon) => {
    const formatted = lat.toFixed(4) + ", " + lon.toFixed(4);
    if (clickMode === "from") {
      setStartPoint({ name: "Location (" + formatted + ")", address: "Coordinates " + formatted, lat, lon });
      setFromQuery(formatted);
      setClickMode(null);
    } else if (clickMode === "to") {
      setEndPoint({ name: "Location (" + formatted + ")", address: "Coordinates " + formatted, lat, lon });
      setToQuery(formatted);
      setClickMode(null);
    }
  };

  const routeAnalysis = useMemo(() => {
    if (!startPoint || !endPoint) return null;

    const directDist = haversineKm(startPoint.lat, startPoint.lon, endPoint.lat, endPoint.lon);
    const bufferKm = Math.min(100, Math.max(40, directDist * 0.15));

    const corridorZones = zones.filter((z) => {
      if (!z.lat || !z.lon) return false;
      const d = distToSegmentKm(
        z.lat, z.lon,
        startPoint.lat, startPoint.lon,
        endPoint.lat, endPoint.lon
      );
      return d <= bufferKm;
    });

    const criticalZones = corridorZones.filter((z) => z.risk_level === "CRITICAL");
    const highZones     = corridorZones.filter((z) => z.risk_level === "HIGH");
    const mediumZones   = corridorZones.filter((z) => z.risk_level === "MEDIUM");
    const lowZones      = corridorZones.filter((z) => z.risk_level === "LOW");

    const avgScore = corridorZones.length
      ? corridorZones.reduce((acc, z) => acc + (z.mean_ccrp || 0.3), 0) / corridorZones.length
      : 0.3;

    let safetyScore = Math.round(
      Math.max(20, Math.min(98, 100 - avgScore * 105 - criticalZones.length * 4.5))
    );
    if (criticalZones.length === 0 && highZones.length <= 1) {
      safetyScore = Math.max(82, safetyScore);
    }

    const slotRecommendations = [
      { slot: 1, name: "Morning (06:00–12:00)", score: "Low Risk", tip: "Safest transit window with highest highway patrol visibility and open public stops." },
      { slot: 2, name: "Afternoon (12:00–18:00)", score: "Moderate Risk", tip: "Good daylight visibility. Moderate crime risk near city borders." },
      { slot: 3, name: "Evening (18:00–24:00)", score: "Elevated Risk", tip: "Increased snatching & vehicle theft risk near toll gates and outskirts." },
      { slot: 0, name: "Night (00:00–06:00)", score: "Highest Risk", tip: "Peak nighttime violent crime window. Stay strictly on national expressways." },
    ];

    return {
      directDist: Math.round(directDist),
      corridorZones,
      criticalZones,
      highZones,
      mediumZones,
      lowZones,
      avgScore,
      safetyScore,
      slotRecommendations,
    };
  }, [startPoint, endPoint, zones]);

  const routeColor =
    !routeAnalysis ? "#388bfd" :
    routeAnalysis.safetyScore >= 80 ? "#238636" :
    routeAnalysis.safetyScore >= 60 ? "#d29922" : "#da3633";

  const POPULAR_CITIES = [
    "Bangalore", "Mysuru", "Mangalore", "Hubli", "Belagavi",
    "Mumbai", "Pune", "Delhi", "Hyderabad", "Chennai", "Kolkata"
  ];

  return (
    <div style={{ display: "flex", height: "100%", width: "100%", background: "#0d1117" }}>
      <div
        style={{
          width: 410,
          flexShrink: 0,
          borderRight: "1px solid #21262d",
          display: "flex",
          flexDirection: "column",
          background: "#161b22",
          overflowY: "auto",
          color: "#e6edf3",
        }}
      >
        <div style={{ padding: "16px 18px", borderBottom: "1px solid #21262d" }}>
          <div style={{ fontSize: 16, fontWeight: 700, display: "flex", alignItems: "center", gap: 8 }}>
            <span>🛡️</span> Safe Route Planner
          </div>
          <div style={{ fontSize: 11, color: "#8b949e", marginTop: 3 }}>
            Enter any Indian street, locality, landmark, or city to calculate transit safety
          </div>
        </div>

        <div style={{ padding: "14px 18px", borderBottom: "1px solid #21262d", display: "flex", flexDirection: "column", gap: 14 }}>
          {/* FROM INPUT */}
          <div style={{ position: "relative" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
              <label style={{ fontSize: 11, fontWeight: 700, color: "#7ee787" }}>
                🟢 Origin (From Address / Landmark)
              </label>
              <button
                onClick={() => setClickMode(clickMode === "from" ? null : "from")}
                style={{
                  fontSize: 10,
                  background: clickMode === "from" ? "#238636" : "#21262d",
                  border: "1px solid #30363d",
                  borderRadius: 4,
                  padding: "1px 6px",
                  color: clickMode === "from" ? "#fff" : "#8b949e",
                  cursor: "pointer",
                }}
              >
                {clickMode === "from" ? "Click map now..." : "📍 Pick on map"}
              </button>
            </div>

            <div style={{ display: "flex", gap: 6 }}>
              <input
                type="text"
                value={fromQuery}
                placeholder="e.g. Koramangala Bangalore, Marine Drive..."
                onChange={(e) => {
                  setFromQuery(e.target.value);
                  searchAddress(e.target.value, setFromSuggestions, setSearchingFrom);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") searchAddress(fromQuery, setFromSuggestions, setSearchingFrom);
                }}
                style={{
                  flex: 1,
                  padding: "7px 10px",
                  background: "#0d1117",
                  border: "1px solid #30363d",
                  borderRadius: 6,
                  color: "#e6edf3",
                  fontSize: 12,
                }}
              />
              <button
                onClick={() => searchAddress(fromQuery, setFromSuggestions, setSearchingFrom)}
                style={{
                  padding: "7px 12px",
                  background: "#21262d",
                  border: "1px solid #30363d",
                  borderRadius: 6,
                  color: "#58a6ff",
                  fontSize: 11,
                  cursor: "pointer",
                  fontWeight: 600,
                }}
              >
                {searchingFrom ? "..." : "Search"}
              </button>
            </div>

            {fromSuggestions.length > 0 && (
              <div
                style={{
                  position: "absolute",
                  top: "100%",
                  left: 0,
                  right: 0,
                  zIndex: 2000,
                  background: "#161b22",
                  border: "1px solid #388bfd",
                  borderRadius: 6,
                  marginTop: 4,
                  boxShadow: "0 8px 24px rgba(0,0,0,0.7)",
                  maxHeight: 180,
                  overflowY: "auto",
                }}
              >
                {fromSuggestions.map((item, idx) => (
                  <div
                    key={idx}
                    onClick={() => {
                      setStartPoint({
                        name: item.display_name.split(",")[0],
                        address: item.display_name,
                        lat: parseFloat(item.lat),
                        lon: parseFloat(item.lon),
                      });
                      setFromQuery(item.display_name.split(",")[0]);
                      setFromSuggestions([]);
                    }}
                    style={{
                      padding: "8px 10px",
                      fontSize: 11,
                      borderBottom: "1px solid #21262d",
                      cursor: "pointer",
                      color: "#c9d1d9",
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = "#1f6feb22")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    <strong style={{ color: "#58a6ff" }}>{item.display_name.split(",")[0]}</strong>
                    <div style={{ fontSize: 10, color: "#8b949e", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {item.display_name}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* TO INPUT */}
          <div style={{ position: "relative" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
              <label style={{ fontSize: 11, fontWeight: 700, color: "#f85149" }}>
                🏁 Destination (To Address / Landmark)
              </label>
              <button
                onClick={() => setClickMode(clickMode === "to" ? null : "to")}
                style={{
                  fontSize: 10,
                  background: clickMode === "to" ? "#da3633" : "#21262d",
                  border: "1px solid #30363d",
                  borderRadius: 4,
                  padding: "1px 6px",
                  color: clickMode === "to" ? "#fff" : "#8b949e",
                  cursor: "pointer",
                }}
              >
                {clickMode === "to" ? "Click map now..." : "📍 Pick on map"}
              </button>
            </div>

            <div style={{ display: "flex", gap: 6 }}>
              <input
                type="text"
                value={toQuery}
                placeholder="e.g. Mysuru Palace, Connaught Place..."
                onChange={(e) => {
                  setToQuery(e.target.value);
                  searchAddress(e.target.value, setToSuggestions, setSearchingTo);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter") searchAddress(toQuery, setToSuggestions, setSearchingTo);
                }}
                style={{
                  flex: 1,
                  padding: "7px 10px",
                  background: "#0d1117",
                  border: "1px solid #30363d",
                  borderRadius: 6,
                  color: "#e6edf3",
                  fontSize: 12,
                }}
              />
              <button
                onClick={() => searchAddress(toQuery, setToSuggestions, setSearchingTo)}
                style={{
                  padding: "7px 12px",
                  background: "#21262d",
                  border: "1px solid #30363d",
                  borderRadius: 6,
                  color: "#58a6ff",
                  fontSize: 11,
                  cursor: "pointer",
                  fontWeight: 600,
                }}
              >
                {searchingTo ? "..." : "Search"}
              </button>
            </div>

            {toSuggestions.length > 0 && (
              <div
                style={{
                  position: "absolute",
                  top: "100%",
                  left: 0,
                  right: 0,
                  zIndex: 2000,
                  background: "#161b22",
                  border: "1px solid #388bfd",
                  borderRadius: 6,
                  marginTop: 4,
                  boxShadow: "0 8px 24px rgba(0,0,0,0.7)",
                  maxHeight: 180,
                  overflowY: "auto",
                }}
              >
                {toSuggestions.map((item, idx) => (
                  <div
                    key={idx}
                    onClick={() => {
                      setEndPoint({
                        name: item.display_name.split(",")[0],
                        address: item.display_name,
                        lat: parseFloat(item.lat),
                        lon: parseFloat(item.lon),
                      });
                      setToQuery(item.display_name.split(",")[0]);
                      setToSuggestions([]);
                    }}
                    style={{
                      padding: "8px 10px",
                      fontSize: 11,
                      borderBottom: "1px solid #21262d",
                      cursor: "pointer",
                      color: "#c9d1d9",
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = "#1f6feb22")}
                    onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                  >
                    <strong style={{ color: "#58a6ff" }}>{item.display_name.split(",")[0]}</strong>
                    <div style={{ fontSize: 10, color: "#8b949e", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {item.display_name}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div>
            <div style={{ fontSize: 10, color: "#8b949e", marginBottom: 5 }}>
              QUICK PICKS (Click to set Origin, Double-click for Destination):
            </div>
            <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
              {POPULAR_CITIES.map((c) => (
                <button
                  key={c}
                  onClick={() => selectCityPreset(c, true)}
                  onDoubleClick={() => selectCityPreset(c, false)}
                  style={{
                    fontSize: 10,
                    padding: "2px 8px",
                    borderRadius: 12,
                    background: "#0d1117",
                    border: "1px solid #30363d",
                    color: startPoint.name === c ? "#7ee787" : endPoint.name === c ? "#f85149" : "#8b949e",
                    cursor: "pointer",
                  }}
                  title="Click = Origin, Double-click = Destination"
                >
                  {c}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label style={{ display: "block", fontSize: 11, fontWeight: 600, color: "#8b949e", marginBottom: 4 }}>
              🕒 Planned Travel Time Window
            </label>
            <select
              value={travelSlot}
              onChange={(e) => setTravelSlot(Number(e.target.value))}
              style={{
                width: "100%",
                padding: "6px 10px",
                background: "#0d1117",
                border: "1px solid #30363d",
                borderRadius: 6,
                color: "#e6edf3",
                fontSize: 11,
              }}
            >
              <option value={1}>🌅 Morning (06:00 – 12:00) — Safest</option>
              <option value={2}>☀️ Afternoon (12:00 – 18:00) — Moderate</option>
              <option value={3}>🌆 Evening (18:00 – 24:00) — Elevated Caution</option>
              <option value={0}>🌙 Night (00:00 – 06:00) — High Risk</option>
            </select>
          </div>
        </div>

        {routeAnalysis && (
          <div style={{ padding: "16px 18px", borderBottom: "1px solid #21262d" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: "#8b949e", textTransform: "uppercase" }}>
                Transit Safety Index
              </span>
              <span style={{ fontSize: 11, color: "#8b949e" }}>
                Direct: {routeAnalysis.directDist} km
              </span>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <div
                style={{
                  width: 58,
                  height: 58,
                  borderRadius: "50%",
                  border: "4px solid " + routeColor,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 18,
                  fontWeight: 800,
                  color: routeColor,
                }}
              >
                {routeAnalysis.safetyScore}
              </div>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: routeColor }}>
                  {routeAnalysis.safetyScore >= 80
                    ? "Safe Transit Corridor"
                    : routeAnalysis.safetyScore >= 60
                    ? "Moderate Risk Corridor"
                    : "High Danger Alert Corridor"}
                </div>
                <div style={{ fontSize: 11, color: "#8b949e", marginTop: 2 }}>
                  {startPoint.name} ➔ {endPoint.name}
                </div>
              </div>
            </div>

            <div style={{ marginTop: 14, background: "#0d1117", padding: "10px 12px", borderRadius: 8, border: "1px solid #21262d" }}>
              <div style={{ fontSize: 10, color: "#8b949e", marginBottom: 6, fontWeight: 600 }}>
                NEARBY CRIME CLUSTERS ({routeAnalysis.corridorZones.length} zones within transit buffer)
              </div>
              <div style={{ display: "flex", height: 6, borderRadius: 3, overflow: "hidden", marginBottom: 6 }}>
                <div style={{ width: ((routeAnalysis.criticalZones.length / Math.max(1, routeAnalysis.corridorZones.length)) * 100) + "%", background: RISK_COLOR.CRITICAL }} />
                <div style={{ width: ((routeAnalysis.highZones.length / Math.max(1, routeAnalysis.corridorZones.length)) * 100) + "%", background: RISK_COLOR.HIGH }} />
                <div style={{ width: ((routeAnalysis.mediumZones.length / Math.max(1, routeAnalysis.corridorZones.length)) * 100) + "%", background: RISK_COLOR.MEDIUM }} />
                <div style={{ width: ((routeAnalysis.lowZones.length / Math.max(1, routeAnalysis.corridorZones.length)) * 100) + "%", background: RISK_COLOR.LOW }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#8b949e" }}>
                <span style={{ color: RISK_COLOR.CRITICAL }}>🔴 {routeAnalysis.criticalZones.length} Crit</span>
                <span style={{ color: RISK_COLOR.HIGH }}>🟠 {routeAnalysis.highZones.length} High</span>
                <span style={{ color: RISK_COLOR.MEDIUM }}>🟡 {routeAnalysis.mediumZones.length} Med</span>
                <span style={{ color: RISK_COLOR.LOW }}>🟢 {routeAnalysis.lowZones.length} Low</span>
              </div>
            </div>
          </div>
        )}

        {routeAnalysis && (
          <div style={{ padding: "14px 18px", flex: 1 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: "#e6edf3", marginBottom: 8 }}>
              🚨 Route Safety Advisories
            </div>

            {routeAnalysis.criticalZones.length > 0 ? (
              <div
                style={{
                  background: "#ff1e4018",
                  border: "1px solid #ff1e4044",
                  borderRadius: 6,
                  padding: "10px 12px",
                  fontSize: 11,
                  color: "#ff7b92",
                  marginBottom: 10,
                }}
              >
                ⚠️ <strong>High Crime Clusters along path:</strong>{" "}
                {routeAnalysis.criticalZones.map((z) => z.zone_name).join(", ")}. Exercise elevated vigilance at night and avoid unlit stops.
              </div>
            ) : (
              <div
                style={{
                  background: "#23863618",
                  border: "1px solid #23863644",
                  borderRadius: 6,
                  padding: "10px 12px",
                  fontSize: 11,
                  color: "#7ee787",
                  marginBottom: 10,
                }}
              >
                ✓ No critical risk clusters directly on path. Standard highway precautions advised.
              </div>
            )}

            <div
              style={{
                background: "#1f6feb18",
                border: "1px solid #1f6feb44",
                borderRadius: 6,
                padding: "10px 12px",
                fontSize: 11,
                color: "#79c0ff",
              }}
            >
              💡 <strong>Travel Timing Recommendation:</strong>{" "}
              {routeAnalysis.slotRecommendations.find((s) => s.slot === travelSlot)?.tip}
            </div>
          </div>
        )}
      </div>

      <div style={{ flex: 1, position: "relative" }}>
        <div
          style={{
            position: "absolute",
            top: 12,
            right: 12,
            zIndex: 1000,
            display: "flex",
            gap: 4,
            background: "rgba(22,27,34,.94)",
            padding: "4px",
            borderRadius: 8,
            border: "1px solid #30363d",
            backdropFilter: "blur(8px)",
          }}
        >
          {[
            { id: "streets",   label: "🗺️ Streets & Places" },
            { id: "dark",      label: "🌙 Contrast Dark" },
            { id: "satellite", label: "🛰️ Satellite" },
          ].map((style) => (
            <button
              key={style.id}
              onClick={() => setMapStyle(style.id)}
              style={{
                padding: "4px 10px",
                borderRadius: 6,
                fontSize: 11,
                fontWeight: 600,
                border: "none",
                cursor: "pointer",
                background: mapStyle === style.id ? "#1f6feb" : "transparent",
                color: mapStyle === style.id ? "#ffffff" : "#8b949e",
                transition: "all .15s",
              }}
            >
              {style.label}
            </button>
          ))}
        </div>

        {clickMode && (
          <div
            style={{
              position: "absolute",
              top: 12,
              left: "50%",
              transform: "translateX(-50%)",
              zIndex: 1000,
              background: "#1f6feb",
              color: "white",
              padding: "6px 16px",
              borderRadius: 20,
              fontSize: 12,
              fontWeight: 600,
              boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
            }}
          >
            📍 Click anywhere on the map to set {clickMode === "from" ? "Origin" : "Destination"}
          </div>
        )}

        <MapContainer
          center={[startPoint.lat, startPoint.lon]}
          zoom={7}
          minZoom={4}
          maxZoom={19}
          style={{ height: "100%", width: "100%", background: "#0d1117" }}
        >
          {mapStyle === "streets" && (
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              maxZoom={19}
            />
          )}

          {mapStyle === "dark" && (
            <>
              <TileLayer
                url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
                attribution="&copy; Esri &mdash; Esri, DeLorme, NAVTEQ"
                maxZoom={16}
              />
              <TileLayer
                url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}"
                attribution="&copy; Esri"
                maxZoom={16}
              />
            </>
          )}

          {mapStyle === "satellite" && (
            <>
              <TileLayer
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                attribution="&copy; Esri World Imagery"
                maxZoom={19}
              />
              <TileLayer
                url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
                attribution="&copy; Esri Boundaries"
                maxZoom={19}
              />
            </>
          )}

          <MapRecenter start={startPoint} end={endPoint} />
          <MapClickHandler clickMode={clickMode} onLocationPicked={handleMapLocationPicked} />

          {startPoint && endPoint && (
            <>
              <Polyline
                positions={[
                  [startPoint.lat, startPoint.lon],
                  [endPoint.lat, endPoint.lon],
                ]}
                pathOptions={{
                  color: routeColor,
                  weight: 8,
                  opacity: 0.38,
                }}
              />
              <Polyline
                positions={[
                  [startPoint.lat, startPoint.lon],
                  [endPoint.lat, endPoint.lon],
                ]}
                pathOptions={{
                  color: routeColor,
                  weight: 3.5,
                  dashArray: "6, 8",
                  opacity: 0.95,
                }}
              />

              <Marker
                position={[startPoint.lat, startPoint.lon]}
                icon={createPinIcon("#238636", "🟢 " + startPoint.name, "Origin")}
              >
                <Popup>
                  <div style={{ color: "#000" }}>
                    <strong>Origin: {startPoint.name}</strong>
                    <br />
                    <span style={{ fontSize: 11 }}>{startPoint.address}</span>
                  </div>
                </Popup>
              </Marker>

              <Marker
                position={[endPoint.lat, endPoint.lon]}
                icon={createPinIcon("#da3633", "🏁 " + endPoint.name, "Destination")}
              >
                <Popup>
                  <div style={{ color: "#000" }}>
                    <strong>Destination: {endPoint.name}</strong>
                    <br />
                    <span style={{ fontSize: 11 }}>{endPoint.address}</span>
                  </div>
                </Popup>
              </Marker>
            </>
          )}

          {zones.map((z) => {
            if (!z.lat || !z.lon) return null;
            const isCorridor = routeAnalysis?.corridorZones.some((cz) => cz.zone_id === z.zone_id);
            const color = RISK_COLOR[z.risk_level] || "#8b949e";
            const radius = z.risk_level === "CRITICAL" ? 9 : 5 + (z.mean_ccrp || 0.3) * 10;

            return (
              <CircleMarker
                key={z.zone_id}
                center={[z.lat, z.lon]}
                radius={isCorridor ? radius + 3 : radius}
                pathOptions={{
                  color: isCorridor ? "#ffffff" : color,
                  weight: isCorridor ? 2 : 0.8,
                  fillColor: color,
                  fillOpacity: isCorridor ? 0.92 : 0.65,
                }}
              >
                <Tooltip direction="top" offset={[0, -6]} opacity={0.92}>
                  <div style={{ fontSize: 11, fontWeight: 600 }}>
                    {z.zone_name} <span style={{ color, fontWeight: 700 }}>({z.risk_level})</span>
                  </div>
                </Tooltip>
                <Popup>
                  <div style={{ minWidth: 170, color: "#e6edf3" }}>
                    <strong style={{ fontSize: 13 }}>{z.zone_name}</strong>
                    <div style={{ fontSize: 10, color: "#8b949e" }}>{z.district}</div>
                    <div style={{ marginTop: 6, fontSize: 11 }}>
                      Risk Level: <strong style={{ color }}>{z.risk_level}</strong>
                    </div>
                    <div style={{ fontSize: 11 }}>
                      CCRP Score: <strong>{z.mean_ccrp?.toFixed(4)}</strong>
                    </div>
                  </div>
                </Popup>
              </CircleMarker>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}
