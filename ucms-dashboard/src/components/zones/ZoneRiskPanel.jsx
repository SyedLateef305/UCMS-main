// src/components/zones/ZoneRiskPanel.jsx
// Phase II: Added 2026 Forecast insight, Feature Importance drivers, and fixed predictZone call.
import React, { useState } from "react";
import { useApi } from "../../hooks/useApi";
import { getZoneDetail, getZoneHistory, predictZone, getZoneForecast, getFeatureImportance } from "../../api/ucmsApi";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";

const RISK_COLOR = {
  CRITICAL: "#ff4d4f", HIGH: "#fa8c16", MEDIUM: "#fadb14", LOW: "#52c41a",
};
const SLOT_LABELS = ["Night (0-6)", "Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)"];

function SlotGrid({ slotScores }) {
  if (!slotScores) return null;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
      {Object.entries(slotScores).map(([name, score]) => {
        const risk  = score > .75 ? "CRITICAL" : score > .55 ? "HIGH"
                    : score > .35 ? "MEDIUM" : "LOW";
        const color = RISK_COLOR[risk];
        return (
          <div key={name} style={{
            background: "#0d1117", border: "1px solid #21262d",
            borderRadius: 8, padding: "8px 10px",
          }}>
            <div style={{ fontSize: 10, color: "#484f58", marginBottom: 4 }}>
              {name.split("(")[0].trim()}
            </div>
            <div style={{ fontSize: 18, fontWeight: 700, color }}>
              {score.toFixed(3)}
            </div>
            <div style={{
              fontSize: 9, fontWeight: 600, color,
              textTransform: "uppercase", letterSpacing: ".06em",
            }}>{risk}</div>
          </div>
        );
      })}
    </div>
  );
}

function ZoneForecastMini({ zoneId }) {
  const { data: fc } = useApi(() => getZoneForecast(zoneId), [zoneId]);
  if (!fc) return null;

  const color = RISK_COLOR[fc.risk_level] || "#388bfd";
  return (
    <div style={{
      marginTop: 12, padding: "10px 12px", background: "#0d1117",
      border: "1px solid #21262d", borderRadius: 8,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: "#8b949e", textTransform: "uppercase" }}>
          🔮 2026 AI Forecast
        </span>
        <span style={{
          fontSize: 10, fontWeight: 700, padding: "1px 6px", borderRadius: 8,
          background: color + "22", color, border: `1px solid ${color}55`,
        }}>
          {fc.risk_level}
        </span>
      </div>
      <div style={{ fontSize: 20, fontWeight: 700, color }}>
        {fc.mean_predicted_cps?.toFixed(4)}
      </div>
      <div style={{ fontSize: 10, color: "#484f58", marginTop: 2 }}>
        Predicted annual risk profile based on LSTM projection.
      </div>
    </div>
  );
}

function FeatureDriversMini({ zoneId }) {
  const { data: fi } = useApi(() => getFeatureImportance(zoneId), [zoneId]);
  const features = fi?.features || [];
  if (features.length === 0) return null;

  return (
    <div style={{ marginTop: 12 }}>
      <div className="section-label">Top Risk Drivers</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
        {features.slice(0, 3).map((f) => (
          <div key={f.feature} style={{ fontSize: 11 }}>
            <div style={{ display: "flex", justifyContent: "space-between", color: "#8b949e", marginBottom: 2 }}>
              <span>{f.feature.replace("_", " ")}</span>
              <span style={{ color: "#e6edf3", fontWeight: 600 }}>{(f.importance * 100).toFixed(1)}%</span>
            </div>
            <div style={{ height: 4, background: "#21262d", borderRadius: 2, overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${f.importance * 100}%`, background: "#7F77DD", borderRadius: 2 }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PredictPanel({ zoneId }) {
  const [slot, setSlot]     = useState(3);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const r = await predictZone({
        zone_id:   zoneId,
        year:      2025,
        time_slot: slot,
      });
      setResult(r);
    } catch (e) {
      setResult({ error: e.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: 12 }}>
      <div className="section-label">Live Predict</div>
      <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
        <select value={slot} onChange={e => setSlot(Number(e.target.value))}
                style={{ flex: 1 }}>
          {SLOT_LABELS.map((l, i) => <option key={i} value={i}>{l}</option>)}
        </select>
        <button className="btn btn-primary" onClick={handlePredict}
                style={{ whiteSpace: "nowrap" }}>
          {loading ? "…" : "Predict"}
        </button>
      </div>
      {result && !result.error && (
        <div style={{
          background: "#0d1117", border: "1px solid #21262d",
          borderRadius: 8, padding: "10px 12px",
        }}>
          <div style={{ display: "flex", justifyContent: "space-between",
                        alignItems: "center", marginBottom: 6 }}>
            <span style={{ fontSize: 24, fontWeight: 700,
                           color: RISK_COLOR[result.risk_level] }}>
              {result.ccrp_bayesian?.toFixed(4)}
            </span>
            <span className={`risk-badge risk-${result.risk_level}`}>
              {result.risk_level}
            </span>
          </div>
          <div style={{ fontSize: 10, color: "#8b949e" }}>
            90% CI: [{result.uncertainty_lower?.toFixed(3)}, {result.uncertainty_upper?.toFixed(3)}]
            {result.confidence_band_pct != null && ` (±${result.confidence_band_pct}%)`}
          </div>
          <div style={{ fontSize: 10, color: "#484f58", marginTop: 4 }}>
            {result.message}
          </div>
        </div>
      )}
      {result?.error && (
        <div style={{ color: "#ff4d4f", fontSize: 11 }}>{result.error}</div>
      )}
    </div>
  );
}

export default function ZoneRiskPanel({ zone, onClose }) {
  const { data: detail, loading } = useApi(
    () => getZoneDetail(zone.zone_id), [zone.zone_id]
  );
  const { data: hist } = useApi(
    () => getZoneHistory(zone.zone_id), [zone.zone_id]
  );

  const color    = RISK_COLOR[zone.risk_level] || "#8b949e";
  const histData = hist?.history || [];

  return (
    <div style={{
      height:      "100%",
      display:     "flex",
      flexDirection:"column",
      overflow:    "hidden",
    }}>
      {/* Panel header */}
      <div style={{
        display:      "flex",
        alignItems:   "center",
        justifyContent:"space-between",
        padding:      "12px 14px",
        borderBottom: "1px solid #21262d",
        flexShrink:   0,
      }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 700, color: "#e6edf3" }}>
            {zone.zone_name || `Zone ${zone.zone_id}`}
          </div>
          <div style={{ fontSize: 11, color: "#8b949e", marginTop: 1 }}>
            {zone.district ? `${zone.district} District · ` : ""}
            Zone {zone.zone_id} &nbsp;·&nbsp; {zone.lat?.toFixed(4)}, {zone.lon?.toFixed(4)}
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span className={`risk-badge risk-${zone.risk_level}`}>
            {zone.risk_level}
          </span>
          <span className={`trend-${zone.trend}`} style={{ fontSize: 16 }}>
            {zone.trend === "RISING" ? "↑" : zone.trend === "FALLING" ? "↓" : "→"}
          </span>
          <button onClick={onClose} className="btn"
                  style={{ padding: "3px 8px", fontSize: 14 }}>✕</button>
        </div>
      </div>

      {/* Scrollable content */}
      <div style={{ flex: 1, overflow: "auto", padding: "12px 14px" }}>
        {/* Score summary */}
        <div style={{
          display: "grid", gridTemplateColumns: "1fr 1fr",
          gap: 8, marginBottom: 14,
        }}>
          {[
            ["Mean CCRP", zone.mean_ccrp?.toFixed(4), color],
            ["Peak CCRP", zone.peak_ccrp?.toFixed(4), "#ff4d4f"],
            ["Alerts",    zone.alert_count, "#fa8c16"],
            ["Uncertainty", zone.mean_uncertainty?.toFixed(4), "#8b949e"],
          ].map(([lbl, val, c]) => (
            <div key={lbl} style={{
              background: "#0d1117", border: "1px solid #21262d",
              borderRadius: 8, padding: "8px 10px",
            }}>
              <div style={{ fontSize: 10, color: "#484f58", marginBottom: 2 }}>{lbl}</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: c }}>{val}</div>
            </div>
          ))}
        </div>

        {/* Phase II: 2026 Forecast mini card */}
        <ZoneForecastMini zoneId={zone.zone_id} />

        {/* Phase II: Risk drivers */}
        <FeatureDriversMini zoneId={zone.zone_id} />

        {/* Year trend chart */}
        {histData.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <div className="section-label">CCRP Year Trend</div>
            <div style={{ height: 110, marginBottom: 14 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={histData}
                           margin={{ top: 5, right: 5, left: -28, bottom: 0 }}>
                  <defs>
                    <linearGradient id="cg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={color} stopOpacity={0.4} />
                      <stop offset="95%" stopColor={color} stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="year" tick={{ fontSize: 9, fill: "#484f58" }}
                         tickLine={false} axisLine={false} />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: "#484f58" }}
                         tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{ background: "#161b22", border: "1px solid #30363d",
                                    borderRadius: 6, fontSize: 11 }}
                    labelStyle={{ color: "#8b949e" }}
                    itemStyle={{ color }}
                    formatter={v => [v.toFixed(4), "CCRP"]}
                  />
                  <ReferenceLine y={0.55} stroke="#fa8c16" strokeDasharray="3 3"
                                  strokeWidth={1} />
                  <Area type="monotone" dataKey="ccrp" stroke={color}
                        strokeWidth={2} fill="url(#cg)" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Slot scores */}
        {loading ? (
          <div className="spinner" style={{ margin: "16px auto" }} />
        ) : detail?.slot_scores ? (
          <div style={{ marginTop: 12 }}>
            <div className="section-label">Score by Time Slot</div>
            <SlotGrid slotScores={detail.slot_scores} />
          </div>
        ) : null}

        {/* Live predict */}
        <PredictPanel zoneId={zone.zone_id} />
      </div>
    </div>
  );
}