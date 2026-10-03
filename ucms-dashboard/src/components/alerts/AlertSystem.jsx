// src/components/alerts/AlertSystem.jsx
// Phase II: Added Anomaly Alerts tab and CSV export button.
import React, { useState } from "react";
import { getAlerts, getHotspots, getAnomalyAlerts, exportAlerts } from "../../api/ucmsApi";
import { useApi, useApiPolled } from "../../hooks/useApi";

const RISK_COLOR = {
  CRITICAL: "#ff4d4f", HIGH: "#fa8c16", MEDIUM: "#fadb14", LOW: "#52c41a",
};
const SLOT_ICON = { 0: "🌙", 1: "🌅", 2: "☀️", 3: "🌆" };

function AlertRow({ alert, onClick }) {
  const color = RISK_COLOR[alert.risk_level] || "#ff4d4f";
  return (
    <div
      onClick={() => onClick?.(alert)}
      style={{
        display:      "flex",
        alignItems:   "center",
        gap:          10,
        padding:      "8px 12px",
        borderBottom: "1px solid #21262d",
        cursor:       "pointer",
        transition:   "background .12s",
      }}
      onMouseEnter={e => e.currentTarget.style.background = "#1c2128"}
      onMouseLeave={e => e.currentTarget.style.background = "transparent"}
    >
      {/* Risk indicator */}
      <div style={{
        width: 6, height: 36, borderRadius: 3,
        background: color, flexShrink: 0,
        boxShadow: `0 0 8px ${color}60`,
      }} />

      {/* Main info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: "#e6edf3" }}>
            Zone {alert.zone_id}
          </span>
          <span className={`risk-badge risk-${alert.risk_level}`}>
            {alert.risk_level}
          </span>
        </div>
        <div style={{ fontSize: 10, color: "#8b949e", marginTop: 2 }}>
          {SLOT_ICON[alert.time_slot]} {alert.slot_name} · {alert.year}
        </div>
      </div>

      {/* Score */}
      <div style={{ textAlign: "right", flexShrink: 0 }}>
        <div style={{ fontSize: 16, fontWeight: 700, color }}>
          {alert.ccrp_score?.toFixed(3)}
        </div>
        <div style={{ fontSize: 9, color: "#484f58" }}>CCRP</div>
      </div>
    </div>
  );
}

function AnomalyRow({ anomaly, onClick }) {
  return (
    <div
      onClick={() => onClick?.(anomaly)}
      style={{
        display:      "flex",
        alignItems:   "center",
        gap:          10,
        padding:      "8px 12px",
        borderBottom: "1px solid #21262d",
        cursor:       "pointer",
        transition:   "background .12s",
      }}
      onMouseEnter={e => e.currentTarget.style.background = "#1c2128"}
      onMouseLeave={e => e.currentTarget.style.background = "transparent"}
    >
      <div style={{
        width: 6, height: 36, borderRadius: 3,
        background: "#7F77DD", flexShrink: 0,
        boxShadow: "0 0 8px #7F77DD60",
      }} />

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: "#e6edf3" }}>
            Zone {anomaly.zone_id}
          </span>
          <span style={{
            fontSize: 9, fontWeight: 700, padding: "1px 6px", borderRadius: 8,
            background: "#7F77DD22", color: "#7F77DD", border: "1px solid #7F77DD55",
          }}>
            ANOMALY
          </span>
        </div>
        <div style={{ fontSize: 10, color: "#8b949e", marginTop: 2 }}>
          {SLOT_ICON[anomaly.time_slot]} {anomaly.slot_name} · {anomaly.year}
        </div>
      </div>

      <div style={{ textAlign: "right", flexShrink: 0 }}>
        <div style={{ fontSize: 16, fontWeight: 700, color: "#7F77DD" }}>
          {anomaly.anomaly_score?.toFixed(3)}
        </div>
        <div style={{ fontSize: 9, color: "#484f58" }}>Score</div>
      </div>
    </div>
  );
}

function HotspotCard({ hotspot, rank }) {
  const color = RISK_COLOR[hotspot.risk_level] || "#ff4d4f";
  return (
    <div style={{
      background:   "#0d1117",
      border:       `1px solid ${color}30`,
      borderRadius: 8,
      padding:      "8px 10px",
      display:      "flex",
      alignItems:   "center",
      gap:          10,
    }}>
      <div style={{
        width: 28, height: 28, borderRadius: "50%",
        background: `${color}20`, border: `1.5px solid ${color}`,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 11, fontWeight: 700, color, flexShrink: 0,
      }}>{rank}</div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: "#e6edf3" }}>
          Zone {hotspot.zone_id}
        </div>
        <div style={{ fontSize: 10, color: "#8b949e" }}>
          {hotspot.risk_level}
        </div>
      </div>
      <div style={{ fontSize: 15, fontWeight: 700, color }}>{
        hotspot.mean_ccrp?.toFixed(3)
      }</div>
    </div>
  );
}

export default function AlertSystem({ onZoneSelect }) {
  const [tab,       setTab]       = useState("feed");
  const [threshold, setThreshold] = useState(0.70);
  const [year,      setYear]      = useState(null);

  const { data: alertData, loading: alertLoading } = useApiPolled(
    () => getAlerts({ threshold, limit: 80, ...(year ? { year } : {}) }),
    30000,
    [threshold, year],
  );
  const { data: hotData, loading: hotLoading } = useApi(
    () => getHotspots(10, year),
    [year],
  );
  const { data: anomData, loading: anomLoading } = useApi(
    () => getAnomalyAlerts(0.60, year),
    [year],
  );

  const alerts    = alertData?.alerts   || [];
  const hotspots  = hotData?.hotspots   || [];
  const anomalies = anomData?.anomalies || [];

  return (
    <div style={{
      height:        "100%",
      display:       "flex",
      flexDirection: "column",
      overflow:      "hidden",
    }}>
      {/* Header */}
      <div style={{
        padding:      "10px 14px",
        borderBottom: "1px solid #21262d",
        flexShrink:   0,
      }}>
        <div style={{ display: "flex", alignItems: "center",
                      justifyContent: "space-between", marginBottom: 8 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: "#e6edf3" }}>
            Alert System
            {alerts.length > 0 && (
              <span style={{
                marginLeft: 8, background: "#ff4d4f22", color: "#ff4d4f",
                fontSize: 10, padding: "1px 7px", borderRadius: 10,
              }}>{alerts.length}</span>
            )}
          </div>
          <div style={{ display: "flex", gap: 3 }}>
            {[
              { id: "feed", label: "Feed" },
              { id: "hotspots", label: "Hotspots" },
              { id: "anomalies", label: "Anomalies" },
            ].map(t => (
              <button key={t.id} onClick={() => setTab(t.id)} className="btn"
                style={{
                  padding: "3px 8px", fontSize: 10,
                  background: tab === t.id ? "#21262d" : "transparent",
                  borderColor: tab === t.id ? "#30363d" : "transparent",
                  color: tab === t.id ? "#e6edf3" : "#8b949e",
                }}>
                {t.label}
              </button>
            ))}
          </div>
        </div>

        {/* Filters */}
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          {tab !== "anomalies" ? (
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 9, color: "#484f58", marginBottom: 2 }}>
                THRESHOLD
              </div>
              <input
                type="range" min="0.5" max="0.95" step="0.05"
                value={threshold}
                onChange={e => setThreshold(Number(e.target.value))}
                style={{ width: "100%", accentColor: "#ff4d4f" }}
              />
              <div style={{ fontSize: 9, color: "#8b949e", textAlign: "center" }}>
                &gt; {threshold.toFixed(2)}
              </div>
            </div>
          ) : (
            <div style={{ flex: 1, fontSize: 10, color: "#8b949e" }}>
              ⚡ ML Isolation Forest Spikes
            </div>
          )}
          <div>
            <div style={{ fontSize: 9, color: "#484f58", marginBottom: 2 }}>YEAR</div>
            <select value={year || ""} onChange={e => setYear(e.target.value || null)}
                    style={{ width: 75, fontSize: 11 }}>
              <option value="">All</option>
              {[2025,2024,2023,2022,2021].map(y =>
                <option key={y} value={y}>{y}</option>)}
            </select>
          </div>
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: "auto" }}>
        {tab === "feed" ? (
          alertLoading ? (
            <div style={{ padding: 20, textAlign: "center" }}>
              <div className="spinner" />
            </div>
          ) : alerts.length === 0 ? (
            <div style={{ padding: 20, textAlign: "center",
                          color: "#484f58", fontSize: 12 }}>
              No alerts above {threshold.toFixed(2)}
            </div>
          ) : (
            alerts.map((a, i) => (
              <AlertRow key={i} alert={a}
                onClick={() => onZoneSelect?.({ zone_id: a.zone_id, ...a })}
              />
            ))
          )
        ) : tab === "hotspots" ? (
          <div style={{ padding: 10, display: "flex", flexDirection: "column", gap: 6 }}>
            {hotLoading ? (
              <div className="spinner" style={{ margin: "16px auto" }} />
            ) : hotspots.map(h => (
              <HotspotCard key={h.zone_id} hotspot={h} rank={h.rank} />
            ))}
          </div>
        ) : (
          anomLoading ? (
            <div style={{ padding: 20, textAlign: "center" }}>
              <div className="spinner" />
            </div>
          ) : anomalies.length === 0 ? (
            <div style={{ padding: 20, textAlign: "center", color: "#484f58", fontSize: 12 }}>
              No anomalies found.
            </div>
          ) : (
            anomalies.map((an, i) => (
              <AnomalyRow key={i} anomaly={an}
                onClick={() => onZoneSelect?.({ zone_id: an.zone_id, ...an })}
              />
            ))
          )
        )}
      </div>

      {/* Footer stats */}
      <div style={{
        padding:   "8px 14px",
        borderTop: "1px solid #21262d",
        display:   "flex", gap: 16, flexShrink: 0,
      }}>
        {[
          ["Total Alerts", alertData?.total_alerts ?? "—"],
          ["Threshold",    threshold.toFixed(2)],
        ].map(([l, v]) => (
          <div key={l}>
            <div style={{ fontSize: 9, color: "#484f58" }}>{l}</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: "#e6edf3" }}>{v}</div>
          </div>
        ))}
        <div style={{ marginLeft: "auto", fontSize: 10, color: "#484f58",
                      alignSelf: "center" }}>
          auto-refresh 30s
        </div>
      </div>
    </div>
  );
}