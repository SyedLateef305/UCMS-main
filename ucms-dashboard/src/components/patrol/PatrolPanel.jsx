// src/components/patrol/PatrolPanel.jsx
// Phase II: AI patrol recommendation panel with shift selector and route map overlay.
import React, { useState } from "react";
import { useApi } from "../../hooks/useApi";
import { getPatrolRecommendation, getPatrolRoute, getAllZones } from "../../api/ucmsApi";

const RISK_COLOR = {
  CRITICAL: "#ff4d4f",
  HIGH:     "#fa8c16",
  MEDIUM:   "#fadb14",
  LOW:      "#52c41a",
};

const SHIFTS = [
  { id: "night",     label: "🌙 Night (0–6)"      },
  { id: "morning",   label: "🌅 Morning (6–12)"    },
  { id: "afternoon", label: "☀️ Afternoon (12–18)" },
  { id: "evening",   label: "🌆 Evening (18–24)"   },
];

function RiskBadge({ level }) {
  const color = RISK_COLOR[level] || "#8b949e";
  return (
    <span style={{
      fontSize: 9, fontWeight: 700, padding: "1px 6px", borderRadius: 8,
      background: color + "22", color, border: `1px solid ${color}55`,
    }}>
      {level}
    </span>
  );
}

function PriorityBar({ score }) {
  const color = score > 0.75 ? "#ff4d4f" : score > 0.55 ? "#fa8c16" : score > 0.35 ? "#fadb14" : "#52c41a";
  return (
    <div style={{ flex: 1, height: 4, background: "#21262d", borderRadius: 2 }}>
      <div style={{ width: `${score * 100}%`, height: "100%", background: color, borderRadius: 2 }} />
    </div>
  );
}

// ── Patrol Recommendation List ────────────────────────────────────────────────
function PatrolRecommendList({ shift, topN }) {
  const { data, loading, error } = useApi(
    () => getPatrolRecommendation(shift, topN),
    [shift, topN]
  );
  const { data: zonesData } = useApi(() => getAllZones({ limit: 77 }), []);
  const zoneNameMap = React.useMemo(() => {
    const map = {};
    (zonesData?.zones || []).forEach((z) => {
      map[z.zone_id] = z.zone_name || `Zone ${z.zone_id}`;
    });
    return map;
  }, [zonesData]);

  const recs = data?.recommendations || [];

  if (loading) return (
    <div style={{ color: "#8b949e", fontSize: 11, padding: 8 }}>
      Loading patrol recommendations…
    </div>
  );

  if (error) return (
    <div style={{ color: "#ff4d4f", fontSize: 11, padding: 8 }}>
      ⚠️ Patrol data unavailable: {error}
    </div>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ fontSize: 10, color: "#8b949e", marginBottom: 4 }}>
        {data?.shift} — {recs.length} zones recommended
      </div>
      {recs.map((rec) => (
        <div key={`${rec.zone_id}-${rec.time_slot}`} style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "6px 10px", background: "#161b22",
          borderRadius: 6, border: "1px solid #21262d",
        }}>
          <span style={{
            width: 22, height: 22, borderRadius: "50%", display: "flex",
            alignItems: "center", justifyContent: "center",
            background: "#21262d", fontSize: 9, fontWeight: 700,
            color: "#8b949e", flexShrink: 0,
          }}>
            {rec.rank}
          </span>

          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <div>
                <span style={{ fontSize: 12, fontWeight: 600, color: "#e6edf3" }}>
                  {zoneNameMap[rec.zone_id] || `Zone ${rec.zone_id}`}
                </span>
                <span style={{ fontSize: 9, color: "#484f58", marginLeft: 5 }}>
                  #{rec.zone_id}
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <span style={{ fontSize: 10, color: "#8b949e" }}>
                  {rec.mean_ccrp?.toFixed(3)}
                </span>
                <RiskBadge level={rec.risk_level} />
              </div>
            </div>
            <PriorityBar score={rec.priority_score} />
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Patrol Route ──────────────────────────────────────────────────────────────
function PatrolRouteCard({ k }) {
  const { data, loading } = useApi(() => getPatrolRoute(k), [k]);
  const waypoints = data?.waypoints || [];

  return (
    <div className="card" style={{ marginTop: 10 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: "#8b949e",
                    textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 8 }}>
        Optimal Patrol Route (Top {k} Hotspots)
      </div>
      {loading ? (
        <div style={{ color: "#8b949e", fontSize: 11 }}>Computing route…</div>
      ) : (
        <>
          <div style={{ fontSize: 10, color: "#484f58", marginBottom: 8 }}>
            Avg risk covered: <strong style={{ color: "#e6edf3" }}>
              {data?.total_risk_covered?.toFixed(4)}
            </strong>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {waypoints.map((wp, i) => {
              const color = RISK_COLOR[wp.risk_level] || "#8b949e";
              return (
                <React.Fragment key={wp.zone_id}>
                  <div style={{
                    padding: "3px 8px", borderRadius: 12,
                    background: color + "22", color,
                    border: `1px solid ${color}55`,
                    fontSize: 11, fontWeight: 600,
                  }}>
                    {wp.order}. Z{wp.zone_id}
                  </div>
                  {i < waypoints.length - 1 && (
                    <span style={{ color: "#30363d", alignSelf: "center" }}>→</span>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

// ── Main PatrolPanel ──────────────────────────────────────────────────────────
export default function PatrolPanel() {
  const [shift, setShift]   = useState("night");
  const [topN,  setTopN]    = useState(12);
  const [routeK, setRouteK] = useState(8);

  return (
    <div style={{ height: "100%", overflow: "auto", padding: 10 }}>
      {/* Shift Selector */}
      <div className="card" style={{ marginBottom: 10 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: "#8b949e",
                      textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 8 }}>
          Select Patrol Shift
        </div>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {SHIFTS.map((s) => (
            <button key={s.id} onClick={() => setShift(s.id)} style={{
              padding: "5px 10px", borderRadius: 16, fontSize: 10,
              fontWeight: 600, cursor: "pointer", border: "1px solid",
              borderColor: shift === s.id ? "#1f6feb" : "#30363d",
              background:  shift === s.id ? "#1f6feb22" : "rgba(22,27,34,.85)",
              color:        shift === s.id ? "#388bfd"  : "#8b949e",
              transition: "all .15s",
            }}>
              {s.label}
            </button>
          ))}
        </div>

        {/* Top-N slider */}
        <div style={{ marginTop: 10, display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 10, color: "#8b949e", whiteSpace: "nowrap" }}>
            Top zones: <strong style={{ color: "#e6edf3" }}>{topN}</strong>
          </span>
          <input type="range" min={5} max={30} value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
            style={{ flex: 1 }}
          />
        </div>
      </div>

      {/* Recommendations */}
      <div className="card" style={{ marginBottom: 10 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: "#8b949e",
                      textTransform: "uppercase", letterSpacing: ".06em", marginBottom: 8 }}>
          🚔 Priority Patrol List
        </div>
        <PatrolRecommendList shift={shift} topN={topN} />
      </div>

      {/* Route */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
        <span style={{ fontSize: 10, color: "#8b949e" }}>Route stops: <strong style={{ color: "#e6edf3" }}>{routeK}</strong></span>
        <input type="range" min={3} max={20} value={routeK}
          onChange={(e) => setRouteK(Number(e.target.value))}
          style={{ flex: 1 }}
        />
      </div>
      <PatrolRouteCard k={routeK} />
    </div>
  );
}
