// src/components/forecast/ForecastPanel.jsx
// Phase II: 1-year CCRP forecast panel with sparklines, top rising zones, city-wide summary.
import React, { useState } from "react";
import { useApi } from "../../hooks/useApi";
import { getCitywideForecast, getTopRisingZones, getZoneForecast } from "../../api/ucmsApi";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, BarChart, Bar, Cell, ReferenceLine,
} from "recharts";

const RISK_COLOR = {
  CRITICAL: "#ff4d4f",
  HIGH:     "#fa8c16",
  MEDIUM:   "#fadb14",
  LOW:      "#52c41a",
};

const SLOT_NAMES = ["Night", "Morning", "Afternoon", "Evening"];

const tooltip = {
  contentStyle: { background: "#161b22", border: "1px solid #30363d", borderRadius: 6, fontSize: 11 },
  labelStyle:   { color: "#8b949e" },
};

function SectionTitle({ title }) {
  return (
    <div style={{ fontSize: 11, fontWeight: 600, color: "#8b949e",
                  textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 8 }}>
      {title}
    </div>
  );
}

function RiskBadge({ level }) {
  const color = RISK_COLOR[level] || "#8b949e";
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 10,
      background: color + "22", color, border: `1px solid ${color}55`,
    }}>
      {level}
    </span>
  );
}

// ── City-Wide Forecast Card ───────────────────────────────────────────────────
function CitywideForecastCard() {
  const { data, loading, error } = useApi(getCitywideForecast);

  if (loading) return <div style={{ color: "#8b949e", fontSize: 12, padding: 10 }}>Loading city forecast…</div>;
  if (error)   return (
    <div className="card" style={{ marginBottom: 10 }}>
      <SectionTitle title="City-Wide Forecast 2026" />
      <div style={{ color: "#ff4d4f", fontSize: 11 }}>
        ⚠️ Forecast not available. Run <code>13_temporal_forecasting.py</code> first.
      </div>
    </div>
  );

  const slotData = (data?.slot_breakdown || []).map((s) => ({
    slot:      s.slot_name?.split(" ")[0] || `Slot ${s.time_slot}`,
    mean_ccrp: s.mean_ccrp,
    risk:      s.risk_level,
  }));

  return (
    <div className="card" style={{ marginBottom: 10 }}>
      <SectionTitle title="City-Wide Forecast 2026" />
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
        <div style={{ fontSize: 28, fontWeight: 700, color: RISK_COLOR[data?.risk_level] || "#388bfd" }}>
          {data?.mean_ccrp?.toFixed(4)}
        </div>
        <div>
          <div style={{ fontSize: 10, color: "#8b949e" }}>Mean CCRP (predicted)</div>
          <RiskBadge level={data?.risk_level} />
        </div>
      </div>
      <ResponsiveContainer width="100%" height={110}>
        <BarChart data={slotData} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
          <XAxis dataKey="slot" tick={{ fontSize: 9, fill: "#484f58" }} tickLine={false} axisLine={false} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: "#484f58" }} tickLine={false} axisLine={false} />
          <Tooltip {...tooltip} formatter={(v) => [v.toFixed(4), "Predicted CPS"]} />
          <Bar dataKey="mean_ccrp" radius={[4, 4, 0, 0]}>
            {slotData.map((e, i) => (
              <Cell key={i} fill={RISK_COLOR[e.risk] || "#1f6feb"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Top Rising Zones ──────────────────────────────────────────────────────────
function TopRisingZones() {
  const { data, loading, error } = useApi(() => getTopRisingZones(8));
  const zones = data?.zones || [];

  return (
    <div className="card" style={{ marginBottom: 10 }}>
      <SectionTitle title="Top Rising Zones (2026 Forecast)" />
      {loading ? <div style={{ color: "#8b949e", fontSize: 11 }}>Loading…</div>
      : error   ? <div style={{ color: "#ff4d4f", fontSize: 11 }}>Forecast data unavailable</div>
      : zones.length === 0 ? <div style={{ color: "#8b949e", fontSize: 11 }}>No data</div>
      : (
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          {zones.map((z, i) => {
            const delta = z.delta || 0;
            const color = delta >= 0 ? "#ff4d4f" : "#52c41a";
            const pct   = Math.abs(delta * 100).toFixed(1);
            return (
              <div key={z.zone_id} style={{
                display: "flex", justifyContent: "space-between",
                alignItems: "center", padding: "5px 8px",
                background: "#161b22", borderRadius: 6,
                border: "1px solid #21262d",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 10, color: "#484f58", width: 16 }}>#{i + 1}</span>
                  <span style={{ fontSize: 12, color: "#e6edf3", fontWeight: 600 }}>
                    Zone {z.zone_id}
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 10, color: "#8b949e" }}>{z.hist_cps?.toFixed(3)} → {z.fc_cps?.toFixed(3)}</span>
                  <span style={{ fontSize: 11, fontWeight: 700, color }}>
                    {delta >= 0 ? "↑" : "↓"}{pct}%
                  </span>
                  <RiskBadge level={z.risk_level} />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ── Zone Forecast Sparkline ───────────────────────────────────────────────────
function ZoneForecastCard({ zoneId }) {
  const { data, loading, error } = useApi(() => getZoneForecast(zoneId), [zoneId]);

  if (loading) return <div style={{ color: "#8b949e", fontSize: 11 }}>Loading zone forecast…</div>;
  if (error)   return null;

  const slotData = (data?.slot_forecasts || []).map((s) => ({
    slot:  SLOT_NAMES[s.time_slot] || `Slot ${s.time_slot}`,
    score: s.predicted_cps,
    risk:  data?.risk_level,
  }));

  return (
    <div className="card" style={{ marginBottom: 10 }}>
      <SectionTitle title={`Zone ${zoneId} — 2026 Forecast`} />
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
        <span style={{ fontSize: 20, fontWeight: 700, color: RISK_COLOR[data?.risk_level] || "#388bfd" }}>
          {data?.mean_predicted_cps?.toFixed(4)}
        </span>
        <RiskBadge level={data?.risk_level} />
      </div>
      <ResponsiveContainer width="100%" height={90}>
        <BarChart data={slotData} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
          <XAxis dataKey="slot" tick={{ fontSize: 8, fill: "#484f58" }} tickLine={false} axisLine={false} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 8, fill: "#484f58" }} tickLine={false} axisLine={false} />
          <Tooltip {...tooltip} formatter={(v) => [v.toFixed(4), "Forecast CPS"]} />
          <Bar dataKey="score" radius={[3, 3, 0, 0]}>
            {slotData.map((e, i) => (
              <Cell key={i} fill={RISK_COLOR[e.risk] || "#1f6feb"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Main ForecastPanel ────────────────────────────────────────────────────────
export default function ForecastPanel({ selectedZone }) {
  const [searchZone, setSearchZone] = useState("");
  const [queriedZone, setQueriedZone] = useState(null);

  const handleSearch = () => {
    const n = parseInt(searchZone, 10);
    if (!isNaN(n) && n >= 1 && n <= 77) setQueriedZone(n);
  };

  return (
    <div style={{ height: "100%", overflow: "auto", padding: 10 }}>
      <CitywideForecastCard />
      <TopRisingZones />

      {/* Zone Lookup */}
      <div className="card">
        <SectionTitle title="Zone-Specific Forecast" />
        <div style={{ display: "flex", gap: 6, marginBottom: 10 }}>
          <input
            type="number" min={1} max={77}
            placeholder="Zone ID (1–77)"
            value={searchZone}
            onChange={(e) => setSearchZone(e.target.value)}
            style={{
              flex: 1, padding: "5px 8px", borderRadius: 6,
              background: "#0d1117", border: "1px solid #30363d",
              color: "#e6edf3", fontSize: 12,
            }}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          />
          <button onClick={handleSearch} style={{
            padding: "5px 12px", borderRadius: 6,
            background: "#1f6feb", border: "none",
            color: "#fff", fontSize: 11, cursor: "pointer",
          }}>
            Forecast
          </button>
        </div>
        {queriedZone && <ZoneForecastCard zoneId={queriedZone} />}
        {selectedZone && selectedZone.zone_id !== queriedZone && (
          <ZoneForecastCard zoneId={selectedZone.zone_id} />
        )}
      </div>
    </div>
  );
}
