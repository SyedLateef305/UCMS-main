// src/components/layout/Header.jsx
// Phase II: Added Live Mode toggle, animated CRITICAL pulse badge, Export Alerts button.
import React from "react";
import { useApi } from "../../hooks/useApi";
import { getOverview, exportAlerts } from "../../api/ucmsApi";

const RISK_COLORS = {
  CRITICAL: "#ff4d4f",
  HIGH:     "#fa8c16",
  MEDIUM:   "#fadb14",
  LOW:      "#52c41a",
};

function StatCard({ label, value, color, sub }) {
  return (
    <div style={{
      background: "#161b22", border: "1px solid #30363d",
      borderRadius: 10, padding: "10px 16px", minWidth: 110, flex: 1,
    }}>
      <div style={{ fontSize: 11, color: "#8b949e", marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || "#e6edf3", lineHeight: 1.2 }}>
        {value ?? "—"}
      </div>
      {sub && <div style={{ fontSize: 10, color: "#484f58", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

// Phase II: animated pulse for CRITICAL risk
function CityRiskBadge({ riskLevel }) {
  if (!riskLevel) return null;
  const color = RISK_COLORS[riskLevel] || "#8b949e";
  const isCritical = riskLevel === "CRITICAL";

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 5,
      padding: "3px 10px", borderRadius: 12,
      background: color + "22", border: `1px solid ${color}55`,
    }}>
      <div style={{
        width: 7, height: 7, borderRadius: "50%", background: color,
        boxShadow: isCritical ? `0 0 8px ${color}` : "none",
        animation: isCritical ? "pulse 1.2s infinite" : "none",
      }} />
      <span style={{ fontSize: 10, fontWeight: 700, color }}>{riskLevel}</span>
    </div>
  );
}

export default function Header({
  selectedYear,
  onYearChange,
  liveMode,
  onLiveModeToggle,
  liveConnected,
}) {
  const { data: ov, loading } = useApi(getOverview);

  // Determine city-wide risk level from overview
  const cityRisk = ov
    ? (ov.critical_zones > 5 ? "CRITICAL"
      : ov.high_zones > 15   ? "HIGH"
      : ov.medium_zones > 30 ? "MEDIUM" : "LOW")
    : null;

  const handleExport = () => {
    const url = exportAlerts(0.70, selectedYear, null);
    window.open(url, "_blank");
  };

  return (
    <header style={{
      display: "flex", alignItems: "center", gap: 10,
      padding: "10px 16px", background: "#0d1117",
      borderBottom: "1px solid #21262d", flexShrink: 0,
    }}>
      {/* Logo + version */}
      <div style={{ marginRight: 8 }}>
        <div style={{ fontSize: 15, fontWeight: 700, color: "#e6edf3", letterSpacing: "-.01em" }}>
          UCMS
        </div>
        <div style={{ fontSize: 9, color: "#8b949e", whiteSpace: "nowrap" }}>
          India Urban Crime Monitoring &nbsp;<span style={{ color: "#388bfd" }}>v2.0</span>
        </div>
      </div>

      <div style={{ width: 1, height: 36, background: "#21262d", flexShrink: 0 }} />

      {/* Stat cards */}
      {loading ? (
        <div style={{ color: "#8b949e", fontSize: 12 }}>Loading stats…</div>
      ) : ov ? (
        <div style={{ display: "flex", gap: 8, flex: 1, overflow: "hidden" }}>
          <StatCard label="Total Zones"  value={ov.total_zones} />
          <StatCard label="Critical"     value={ov.critical_zones} color={RISK_COLORS.CRITICAL} />
          <StatCard label="High"         value={ov.high_zones}     color={RISK_COLORS.HIGH} />
          <StatCard label="Medium"       value={ov.medium_zones}   color={RISK_COLORS.MEDIUM} />
          <StatCard label="Low"          value={ov.low_zones}      color={RISK_COLORS.LOW} />
          <StatCard label="Alerts"       value={ov.total_alert_records} color="#ff4d4f" sub="CPS > 0.70" />
          <StatCard label="City CCRP"    value={ov.mean_ccrp_citywide?.toFixed(3)} sub="avg all zones" />
          <StatCard label="Rising"       value={ov.rising_trend_zones} color="#fa8c16" sub="trend zones" />
        </div>
      ) : null}

      {/* Right controls */}
      <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center", flexShrink: 0 }}>

        {/* City risk badge */}
        {cityRisk && <CityRiskBadge riskLevel={cityRisk} />}

        {/* Year selector */}
        <select
          value={selectedYear}
          onChange={(e) => onYearChange(Number(e.target.value))}
          style={{ width: 90 }}
        >
          {[2025, 2024, 2023, 2022, 2021, 2020].map((y) => (
            <option key={y} value={y}>{y}</option>
          ))}
        </select>

        {/* [Phase II] Export alerts button */}
        <button
          onClick={handleExport}
          title="Export alerts as CSV"
          style={{
            padding: "4px 10px", borderRadius: 6,
            background: "#21262d", border: "1px solid #30363d",
            color: "#8b949e", fontSize: 10, cursor: "pointer",
            transition: "color .15s",
          }}
          onMouseEnter={(e) => (e.target.style.color = "#e6edf3")}
          onMouseLeave={(e) => (e.target.style.color = "#8b949e")}
        >
          ⬇ Export CSV
        </button>

        {/* [Phase II] Live Mode toggle */}
        <button
          onClick={onLiveModeToggle}
          title={liveMode ? "Disable live updates" : "Enable live updates"}
          style={{
            padding: "4px 10px", borderRadius: 6, fontSize: 10,
            fontWeight: 600, cursor: "pointer", border: "1px solid",
            borderColor: liveMode ? "#52c41a" : "#30363d",
            background:  liveMode ? "#52c41a22" : "transparent",
            color:       liveMode ? "#52c41a"   : "#8b949e",
            display: "flex", alignItems: "center", gap: 5,
          }}
        >
          <span style={{
            width: 7, height: 7, borderRadius: "50%",
            background: liveMode && liveConnected ? "#52c41a" : "#484f58",
            boxShadow:  liveMode && liveConnected ? "0 0 6px #52c41a" : "none",
          }} />
          Live
        </button>

        {/* Connection indicator */}
        <div style={{
          width: 8, height: 8, borderRadius: "50%",
          background: "#52c41a", alignSelf: "center",
          boxShadow: "0 0 6px #52c41a",
        }} title="API connected" />
      </div>
    </header>
  );
}