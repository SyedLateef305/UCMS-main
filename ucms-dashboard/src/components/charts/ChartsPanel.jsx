// src/components/charts/ChartsPanel.jsx
// Phase II: Added Zone Comparison Multi-Line Chart, Slot x Zone Heatmap, Stream Radar, and City Trend.
import React, { useState } from "react";
import { useApi } from "../../hooks/useApi";
import {
  getTrends, getStreamWeights,
  getSlotProfile, getRiskDistribution,
  getZoneComparisonTrend, getHeatmapMatrix,
} from "../../api/ucmsApi";
import {
  AreaChart, Area, BarChart, Bar, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Legend, Cell, LineChart, Line,
} from "recharts";

const COLORS = ["#1f6feb", "#7F77DD", "#1D9E75", "#fa8c16", "#ff4d4f", "#fadb14"];
const RISK_COLOR = {
  CRITICAL: "#ff4d4f", HIGH: "#fa8c16", MEDIUM: "#fadb14", LOW: "#52c41a",
};

const tooltipStyle = {
  contentStyle: { background: "#161b22", border: "1px solid #30363d",
                  borderRadius: 6, fontSize: 11 },
  labelStyle:   { color: "#8b949e" },
};

function SectionTitle({ title }) {
  return (
    <div style={{ fontSize: 11, fontWeight: 600, color: "#8b949e",
                  textTransform: "uppercase", letterSpacing: ".06em",
                  marginBottom: 8 }}>
      {title}
    </div>
  );
}

// ── 1. City Trend ──────────────────────────────────────────────────────────
function CityTrendChart() {
  const { data, loading } = useApi(getTrends);
  const trend = data?.data || [];
  return (
    <div className="card" style={{ marginBottom: 10 }}>
      <SectionTitle title="City-Wide CCRP Trend (2010–2025)" />
      {loading ? <div className="spinner" style={{ margin: "20px auto" }} /> : (
        <ResponsiveContainer width="100%" height={140}>
          <AreaChart data={trend} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
            <defs>
              <linearGradient id="tg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#1f6feb" stopOpacity={.4} />
                <stop offset="95%" stopColor="#1f6feb" stopOpacity={.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis dataKey="year" tick={{ fontSize: 9, fill: "#484f58" }}
                   tickLine={false} axisLine={false} />
            <YAxis domain={[0,1]} tick={{ fontSize: 9, fill: "#484f58" }}
                   tickLine={false} axisLine={false} />
            <Tooltip {...tooltipStyle}
              formatter={(v, n) => [
                n === "mean_ccrp" ? v.toFixed(4) : v,
                n === "mean_ccrp" ? "CCRP" : "Alerts",
              ]}
            />
            <Area type="monotone" dataKey="mean_ccrp" stroke="#1f6feb"
                  strokeWidth={2} fill="url(#tg)" dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// ── 2. Zone Comparison Tool (Phase II) ────────────────────────────────────
function ZoneComparisonChart() {
  const [zoneInput, setZoneInput] = useState("25, 68, 71");
  const [queryZones, setQueryZones] = useState([25, 68, 71]);

  const { data, loading } = useApi(
    () => getZoneComparisonTrend(queryZones),
    [queryZones]
  );

  const handleApply = () => {
    const parsed = zoneInput
      .split(",")
      .map(s => parseInt(s.trim(), 10))
      .filter(n => !isNaN(n) && n >= 1 && n <= 77)
      .slice(0, 4);
    if (parsed.length > 0) {
      setQueryZones(parsed);
    }
  };

  // Reshape trends into [{year: 2010, "25": 0.6, "68": 0.4}, ...]
  const trendsObj = data?.trends || {};
  const years = [2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024, 2025];
  const chartData = years.map(yr => {
    const row = { year: yr };
    queryZones.forEach(z => {
      const zList = trendsObj[String(z)] || [];
      const item = zList.find(pt => pt.year === yr);
      if (item) row[`Zone ${z}`] = item.ccrp;
    });
    return row;
  });

  return (
    <div className="card" style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <SectionTitle title="Multi-Zone Comparison (Phase II)" />
        <div style={{ display: "flex", gap: 6 }}>
          <input
            type="text"
            value={zoneInput}
            onChange={e => setZoneInput(e.target.value)}
            style={{ width: 90, fontSize: 11, padding: "2px 6px", background: "#0d1117", border: "1px solid #30363d", color: "#e6edf3", borderRadius: 4 }}
            placeholder="e.g. 25, 68"
          />
          <button className="btn btn-primary" onClick={handleApply} style={{ padding: "2px 8px", fontSize: 10 }}>
            Compare
          </button>
        </div>
      </div>
      {loading ? <div className="spinner" style={{ margin: "16px auto" }} /> : (
        <ResponsiveContainer width="100%" height={140}>
          <LineChart data={chartData} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis dataKey="year" tick={{ fontSize: 9, fill: "#484f58" }} tickLine={false} axisLine={false} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: "#484f58" }} tickLine={false} axisLine={false} />
            <Tooltip {...tooltipStyle} formatter={v => [Number(v).toFixed(3), "CCRP"]} />
            <Legend wrapperStyle={{ fontSize: 10, color: "#8b949e" }} />
            {queryZones.map((z, i) => (
              <Line
                key={z}
                type="monotone"
                dataKey={`Zone ${z}`}
                stroke={COLORS[i % COLORS.length]}
                strokeWidth={2}
                dot={{ r: 2 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// ── 3. 24h Slot Profile ───────────────────────────────────────────────────
function SlotProfileChart() {
  const { data, loading } = useApi(getSlotProfile);
  const profile = data?.profile || [];
  return (
    <div className="card" style={{ marginBottom: 10 }}>
      <SectionTitle title="24-Hour Risk Profile" />
      {loading ? <div className="spinner" style={{ margin: "20px auto" }} /> : (
        <ResponsiveContainer width="100%" height={120}>
          <BarChart data={profile} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis dataKey="slot_name"
                   tickFormatter={v => v.split(" ")[0]}
                   tick={{ fontSize: 9, fill: "#484f58" }}
                   tickLine={false} axisLine={false} />
            <YAxis domain={[0,1]} tick={{ fontSize: 9, fill: "#484f58" }}
                   tickLine={false} axisLine={false} />
            <Tooltip {...tooltipStyle}
                     formatter={v => [v.toFixed(4), "CCRP"]} />
            <Bar dataKey="mean_ccrp" radius={[4,4,0,0]}>
              {profile.map((entry, i) => (
                <Cell key={i} fill={RISK_COLOR[entry.risk_level] || "#1f6feb"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
      {data && (
        <div style={{ display: "flex", justifyContent: "space-between",
                      marginTop: 6, fontSize: 10, color: "#484f58" }}>
          <span>Peak: <strong style={{ color: "#ff4d4f" }}>{data.peak_slot}</strong></span>
          <span>Safest: <strong style={{ color: "#52c41a" }}>{data.safest_slot}</strong></span>
        </div>
      )}
    </div>
  );
}

// ── 4. Stream Weights Radar ───────────────────────────────────────────────
function StreamWeightsChart() {
  const { data, loading } = useApi(getStreamWeights);
  const avg = data?.average_weights || {};
  const radarData = Object.entries(avg).map(([k, v]) => ({
    stream: k.replace("_", " "),
    weight: v,
  }));

  return (
    <div className="card" style={{ marginBottom: 10 }}>
      <SectionTitle title="CCRP Stream Weights (Radar)" />
      {loading ? <div className="spinner" style={{ margin: "20px auto" }} /> : (
        <ResponsiveContainer width="100%" height={170}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#21262d" />
            <PolarAngleAxis dataKey="stream"
              tick={{ fontSize: 9, fill: "#8b949e" }} />
            <Radar name="Weight" dataKey="weight"
                   stroke="#7F77DD" fill="#7F77DD" fillOpacity={0.3}
                   strokeWidth={2} />
            <Tooltip {...tooltipStyle}
                     formatter={v => [v.toFixed(4), "Weight"]} />
          </RadarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// ── 5. Risk Distribution ──────────────────────────────────────────────────
function RiskDistChart() {
  const { data, loading } = useApi(() => getRiskDistribution(25));
  const hist  = data?.histogram || [];
  const rlc   = data?.risk_counts || {};

  return (
    <div className="card">
      <SectionTitle title="CCRP Score Distribution" />
      {loading ? <div className="spinner" style={{ margin: "20px auto" }} /> : (
        <>
          <ResponsiveContainer width="100%" height={110}>
            <BarChart data={hist} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
              <XAxis dataKey="midpoint"
                     tickFormatter={v => v.toFixed(1)}
                     tick={{ fontSize: 8, fill: "#484f58" }}
                     tickLine={false} axisLine={false} interval={4} />
              <YAxis tick={{ fontSize: 8, fill: "#484f58" }}
                     tickLine={false} axisLine={false} />
              <Tooltip {...tooltipStyle}
                formatter={(v, n) => [v, "Count"]}
                labelFormatter={v => `Score ≈ ${Number(v).toFixed(2)}`} />
              <Bar dataKey="count" radius={[2,2,0,0]}>
                {hist.map((entry, i) => (
                  <Cell key={i} fill={RISK_COLOR[entry.risk_level] || "#1f6feb"}
                        fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
            {Object.entries(rlc).map(([rl, cnt]) => (
              <div key={rl} style={{ fontSize: 10, color: RISK_COLOR[rl] }}>
                <span style={{ fontWeight: 700 }}>{cnt.toLocaleString()}</span>
                <span style={{ color: "#484f58" }}> {rl}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default function ChartsPanel() {
  return (
    <div style={{ height: "100%", overflow: "auto", padding: 10 }}>
      <CityTrendChart />
      <ZoneComparisonChart />
      <SlotProfileChart />
      <StreamWeightsChart />
      <RiskDistChart />
    </div>
  );
}