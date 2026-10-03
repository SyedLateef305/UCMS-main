import React, { useEffect, useRef, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, Tooltip, Polyline, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { useApi } from "../../hooks/useApi";
import { getAllZones } from "../../api/ucmsApi";
import ZoneCrimeDrawer from "../zones/ZoneCrimeDrawer";

const RISK_COLOR = {
  CRITICAL: "#ff1e40",
  HIGH:     "#fa8c16",
  MEDIUM:   "#fadb14",
  LOW:      "#52c41a",
};

const SLOT_OPTIONS = [
  { value: null,  label: "All Slots"       },
  { value: 0,     label: "Night (0–6)"     },
  { value: 1,     label: "Morning (6–12)"  },
  { value: 2,     label: "Afternoon (12–18)"},
  { value: 3,     label: "Evening (18–24)" },
];

function Legend() {
  return (
    <div style={{
      position: "absolute", bottom: 24, right: 10, zIndex: 1000,
      background: "rgba(22,27,34,.92)", border: "1px solid #30363d",
      borderRadius: 8, padding: "10px 14px", backdropFilter: "blur(8px)",
    }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: "#8b949e",
                    textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 8 }}>
        Risk Level
      </div>
      {Object.entries(RISK_COLOR).map(([level, color]) => (
        <div key={level} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 5 }}>
          <div style={{ width: 10, height: 10, borderRadius: "50%",
                        background: color, boxShadow: `0 0 6px ${color}60` }} />
          <span style={{ fontSize: 11, color: "#e6edf3" }}>{level}</span>
        </div>
      ))}
      <div style={{ borderTop: "1px solid #30363d", marginTop: 8, paddingTop: 8 }}>
        <div style={{ fontSize: 10, color: "#8b949e" }}>Circle size = CCRP score</div>
      </div>
    </div>
  );
}

export default function CrimeHeatMap({
  selectedZone,
  onZoneClick,
  selectedSlot,
  selectedYear,
  liveZones,
}) {
  const { data, loading } = useApi(() => getAllZones({ limit: 200 }), []);
  const [filter,       setFilter]       = useState("ALL");
  const [slotFilter,   setSlotFilter]   = useState(null);
  const [drawerZone,   setDrawerZone]   = useState(null);
  const [mapStyle,     setMapStyle]     = useState("streets"); // "streets" | "dark" | "satellite"

  // Use liveZones if available, otherwise fall back to API data
  const baseZones = liveZones?.length ? liveZones : (data?.zones || []);

  const filtered = baseZones.filter((z) => {
    if (filter !== "ALL" && z.risk_level !== filter) return false;
    return true;
  });

  return (
    <div style={{ position: "relative", height: "100%", width: "100%" }}>
      {/* ── Top controls bar ── */}
      <div style={{
        position: "absolute", top: 10, left: 10, right: 10, zIndex: 1000,
        display: "flex", justifyContent: "space-between", alignItems: "center",
        flexWrap: "wrap", gap: 8, pointerEvents: "none"
      }}>
        {/* Left: Risk Filters & Time Slot */}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", pointerEvents: "auto" }}>
          {["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"].map((lvl) => (
            <button key={lvl} onClick={() => setFilter(lvl)} style={{
              padding: "4px 10px", borderRadius: 16, fontSize: 10,
              fontWeight: 600, cursor: "pointer", border: "1px solid",
              borderColor: filter === lvl ? (RISK_COLOR[lvl] || "#388bfd") : "#30363d",
              background:  filter === lvl ? (lvl === "ALL" ? "#1f6feb22" : `${RISK_COLOR[lvl]}22`) : "rgba(22,27,34,.9)",
              color:       filter === lvl ? (RISK_COLOR[lvl] || "#388bfd") : "#8b949e",
              backdropFilter: "blur(8px)",
            }}>
              {lvl}
            </button>
          ))}

          <select
            value={slotFilter ?? ""}
            onChange={(e) => setSlotFilter(e.target.value === "" ? null : Number(e.target.value))}
            style={{
              padding: "3px 8px", borderRadius: 12, fontSize: 10,
              background: "rgba(22,27,34,.92)", border: "1px solid #30363d",
              color: "#8b949e", cursor: "pointer", backdropFilter: "blur(8px)",
            }}
          >
            {SLOT_OPTIONS.map((opt) => (
              <option key={opt.value ?? "all"} value={opt.value ?? ""}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Right: Map Style Switcher (Streets / Dark / Satellite) */}
        <div style={{
          display: "flex", gap: 4, background: "rgba(22,27,34,.94)",
          padding: "3px", borderRadius: 8, border: "1px solid #30363d",
          backdropFilter: "blur(8px)", pointerEvents: "auto",
        }}>
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
      </div>

      {loading && (
        <div style={{
          position: "absolute", top: "50%", left: "50%",
          transform: "translate(-50%,-50%)", zIndex: 999,
          color: "#8b949e", fontSize: 13,
        }}>
          Loading zones…
        </div>
      )}

      <MapContainer
        center={[22.5, 80.0]}
        zoom={5}
        minZoom={4}
        maxZoom={18}
        maxBounds={[[6.0, 68.0], [37.6, 97.5]]}
        maxBoundsViscosity={1.0}
        style={{ height: "100%", width: "100%", background: "#0d1117" }}
        zoomControl={true}
      >
        {/* Layer 1: Streets & Places (OpenStreetMap — full street and town names) */}
        {mapStyle === "streets" && (
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            maxZoom={19}
          />
        )}

        {/* Layer 2: Contrast Dark (Esri Dark Base + High-Contrast Reference Labels) */}
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

        {/* Layer 3: Satellite Hybrid (Esri Imagery + Boundaries & Names) */}
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

        {filtered.map((zone) => {
          if (!zone.lat || !zone.lon) return null;
          const color      = RISK_COLOR[zone.risk_level] || "#8b949e";
          const radius     = zone.risk_level === "CRITICAL" ? 9 + zone.mean_ccrp * 10 : 5 + zone.mean_ccrp * 12;
          const isSelected = selectedZone?.zone_id === zone.zone_id;
          const isCritical = zone.risk_level === "CRITICAL";

          return (
            <React.Fragment key={zone.zone_id}>
              {/* Outer danger halo for CRITICAL zones */}
              {isCritical && (
                <CircleMarker
                  center={[zone.lat, zone.lon]}
                  radius={radius + 6}
                  pathOptions={{
                    color:       "#ff1e40",
                    weight:      1.5,
                    dashArray:   "3, 3",
                    fillColor:   "#ff1e40",
                    fillOpacity: 0.16,
                  }}
                />
              )}

              <CircleMarker
                center={[zone.lat, zone.lon]}
                radius={isSelected ? radius + 4 : radius}
                pathOptions={{
                  color:       isSelected ? "#ffffff" : isCritical ? "#ffffff" : "#0d1117",
                  weight:      isSelected ? 2.5 : isCritical ? 1.6 : 0.8,
                  fillColor:   color,
                  fillOpacity: isSelected ? 0.95 : 0.82,
                }}
                eventHandlers={{
                  click: () => {
                    onZoneClick(zone);
                    setDrawerZone(zone);
                  },
                }}
              >
                {/* Instant Tooltip on Hover */}
                <Tooltip direction="top" offset={[0, -6]} opacity={0.92}>
                  <div style={{ fontSize: 11, fontWeight: 600 }}>
                    {zone.zone_name || `Zone ${zone.zone_id}`}
                    <span style={{ marginLeft: 5, color, fontWeight: 700 }}>({zone.risk_level})</span>
                  </div>
                </Tooltip>
                <Popup>
                  <div style={{ minWidth: 210 }}>
                    <div style={{ fontWeight: 700, fontSize: 14, color: "#e6edf3", marginBottom: 2 }}>
                      {zone.zone_name || `Zone ${zone.zone_id}`}
                    </div>
                    {zone.district && (
                      <div style={{ fontSize: 10, color: "#8b949e", marginBottom: 6 }}>
                        {zone.district} District · Zone {zone.zone_id}
                      </div>
                    )}
                    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      <Row label="CCRP Score"  value={zone.mean_ccrp?.toFixed(4)} color={color} />
                      <Row label="Risk Level"  value={zone.risk_level} color={color} />
                      <Row label="Trend"       value={zone.trend} />
                      <Row label="Peak Slot"   value={zone.peak_slot} />
                      <Row label="Alerts"      value={zone.alert_count} />
                      <Row label="Uncertainty" value={zone.mean_uncertainty?.toFixed(4)} />
                    </div>

                    <div style={{ display: "flex", flexDirection: "column", gap: 5, marginTop: 10 }}>
                      <button
                        onClick={() => setDrawerZone(zone)}
                        style={{
                          width: "100%", padding: "6px 0",
                          background: "#da3633", border: "none", borderRadius: 6,
                          color: "#fff", fontSize: 11, fontWeight: 600, cursor: "pointer",
                        }}
                      >
                        🚨 View Crime Records
                      </button>
                      <button
                        onClick={() => onZoneClick(zone)}
                        style={{
                          width: "100%", padding: "5px 0",
                          background: "#21262d", border: "1px solid #30363d", borderRadius: 6,
                          color: "#c9d1d9", fontSize: 11, cursor: "pointer",
                        }}
                      >
                        Open Risk Details
                      </button>
                    </div>
                  </div>
                </Popup>
              </CircleMarker>
            </React.Fragment>
          );
        })}
      </MapContainer>

      <Legend />

      {/* Crime Records Drawer */}
      {drawerZone && (
        <ZoneCrimeDrawer zone={drawerZone} onClose={() => setDrawerZone(null)} />
      )}

      {/* Zone count badge */}
      <div style={{
        position: "absolute", bottom: 24, left: 10, zIndex: 1000,
        background: "rgba(22,27,34,.85)", border: "1px solid #30363d",
        borderRadius: 8, padding: "6px 12px", fontSize: 11, color: "#8b949e",
      }}>
        Showing <strong style={{ color: "#e6edf3" }}>{filtered.length}</strong> zones
        {liveZones?.length > 0 && (
          <span style={{ marginLeft: 6, color: "#52c41a", fontSize: 9 }}>● LIVE</span>
        )}
      </div>
    </div>
  );
}

function Row({ label, value, color }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, color: "#e6edf3" }}>
      <span style={{ color: "#8b949e" }}>{label}</span>
      <span style={{ fontWeight: 600, color: color || "#e6edf3" }}>{value}</span>
    </div>
  );
}