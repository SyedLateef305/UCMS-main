// src/components/zones/ZoneCrimeDrawer.jsx
import React, { useState, useEffect } from "react";
import { getZoneIncidents } from "../../api/ucmsApi";

const RISK_BADGE = {
  CRITICAL: { bg: "#ff1e4022", border: "#ff1e40", text: "#ff4d6d" },
  HIGH:     { bg: "#fa8c1622", border: "#fa8c16", text: "#ffa940" },
  MEDIUM:   { bg: "#fadb1422", border: "#fadb14", text: "#ffec3d" },
  LOW:      { bg: "#52c41a22", border: "#52c41a", text: "#73d13d" },
};

const SEVERITY_COLOR = (s) => {
  if (s >= 0.8) return "#ff4d6d";
  if (s >= 0.6) return "#ffa940";
  if (s >= 0.4) return "#ffec3d";
  return "#73d13d";
};

export default function ZoneCrimeDrawer({ zone, onClose }) {
  const [incidents, setIncidents] = useState([]);
  const [breakdown, setBreakdown] = useState([]);
  const [totalRecords, setTotalRecords] = useState(0);
  const [loading, setLoading] = useState(true);
  const [yearFilter, setYearFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const [page, setPage] = useState(1);
  const pageSize = 15;

  useEffect(() => {
    if (!zone?.zone_id) return;
    setLoading(true);
    const params = { limit: 100 };
    if (yearFilter !== "all") params.year = yearFilter;
    if (typeFilter !== "all") params.crime_type = typeFilter;

    getZoneIncidents(zone.zone_id, params)
      .then((data) => {
        setIncidents(data.incidents || []);
        setBreakdown(data.crime_breakdown || []);
        setTotalRecords(data.total_records || 0);
        setPage(1);
      })
      .catch((err) => console.error("Error fetching incidents:", err))
      .finally(() => setLoading(false));
  }, [zone?.zone_id, yearFilter, typeFilter]);

  if (!zone) return null;

  const badge = RISK_BADGE[zone.risk_level] || RISK_BADGE.MEDIUM;
  const totalPages = Math.ceil(incidents.length / pageSize);
  const currentIncidents = incidents.slice((page - 1) * pageSize, page * pageSize);

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        right: 0,
        bottom: 0,
        width: 440,
        maxWidth: "92vw",
        background: "rgba(13, 17, 23, 0.96)",
        borderLeft: "1px solid #30363d",
        zIndex: 1100,
        display: "flex",
        flexDirection: "column",
        boxShadow: "-8px 0 24px rgba(0,0,0,0.6)",
        backdropFilter: "blur(12px)",
        color: "#e6edf3",
      }}
    >
      {/* ── Top Header ── */}
      <div
        style={{
          padding: "16px 18px 12px",
          borderBottom: "1px solid #21262d",
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          background: "rgba(22, 27, 34, 0.7)",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 16, fontWeight: 700, letterSpacing: "-0.01em" }}>
              {zone.zone_name || `Zone ${zone.zone_id}`}
            </span>
            <span
              style={{
                fontSize: 10,
                fontWeight: 700,
                padding: "2px 8px",
                borderRadius: 12,
                background: badge.bg,
                border: `1px solid ${badge.border}`,
                color: badge.text,
              }}
            >
              {zone.risk_level}
            </span>
          </div>
          <div style={{ fontSize: 11, color: "#8b949e", marginTop: 2 }}>
            {zone.district ? `${zone.district} · ` : ""}Zone #{zone.zone_id} · CCRP Risk Score:{" "}
            <strong style={{ color: badge.text }}>{zone.mean_ccrp?.toFixed(4)}</strong>
          </div>
        </div>

        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "none",
            color: "#8b949e",
            fontSize: 20,
            cursor: "pointer",
            padding: "2px 6px",
            lineHeight: 1,
          }}
          title="Close"
        >
          ✕
        </button>
      </div>

      {/* ── Filter Bar ── */}
      <div
        style={{
          padding: "10px 18px",
          borderBottom: "1px solid #21262d",
          background: "rgba(22, 27, 34, 0.4)",
          display: "flex",
          gap: 10,
          alignItems: "center",
        }}
      >
        <div style={{ flex: 1 }}>
          <label style={{ display: "block", fontSize: 10, color: "#8b949e", marginBottom: 3 }}>
            Crime Category
          </label>
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value)}
            style={{
              width: "100%",
              padding: "4px 8px",
              borderRadius: 6,
              background: "#0d1117",
              border: "1px solid #30363d",
              color: "#c9d1d9",
              fontSize: 11,
            }}
          >
            <option value="all">All Categories ({totalRecords})</option>
            {breakdown.map((b) => (
              <option key={b.type} value={b.type}>
                {b.type} ({b.count})
              </option>
            ))}
          </select>
        </div>

        <div style={{ width: 105 }}>
          <label style={{ display: "block", fontSize: 10, color: "#8b949e", marginBottom: 3 }}>
            Year
          </label>
          <select
            value={yearFilter}
            onChange={(e) => setYearFilter(e.target.value)}
            style={{
              width: "100%",
              padding: "4px 8px",
              borderRadius: 6,
              background: "#0d1117",
              border: "1px solid #30363d",
              color: "#c9d1d9",
              fontSize: 11,
            }}
          >
            <option value="all">All Years</option>
            {[2025, 2024, 2023, 2022, 2021, 2020].map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* ── Category Quick Badges ── */}
      {breakdown.length > 0 && (
        <div
          style={{
            padding: "8px 18px",
            display: "flex",
            gap: 6,
            overflowX: "auto",
            borderBottom: "1px solid #21262d",
            whiteSpace: "nowrap",
          }}
        >
          {breakdown.slice(0, 5).map((b) => (
            <button
              key={b.type}
              onClick={() => setTypeFilter(typeFilter === b.type ? "all" : b.type)}
              style={{
                fontSize: 10,
                padding: "3px 8px",
                borderRadius: 12,
                cursor: "pointer",
                background: typeFilter === b.type ? "#1f6feb33" : "#161b22",
                border: typeFilter === b.type ? "1px solid #388bfd" : "1px solid #30363d",
                color: typeFilter === b.type ? "#58a6ff" : "#8b949e",
              }}
            >
              {b.type}: <strong>{b.count}</strong>
            </button>
          ))}
        </div>
      )}

      {/* ── Incidents Feed ── */}
      <div style={{ flex: 1, overflowY: "auto", padding: "10px 18px" }}>
        {loading ? (
          <div style={{ padding: 40, textAlign: "center", color: "#8b949e", fontSize: 12 }}>
            Loading crime incidents...
          </div>
        ) : incidents.length === 0 ? (
          <div style={{ padding: 40, textAlign: "center", color: "#8b949e", fontSize: 12 }}>
            No recorded crimes found matching filters.
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {currentIncidents.map((inc, i) => (
              <div
                key={i}
                style={{
                  background: "#161b22",
                  border: "1px solid #21262d",
                  borderRadius: 8,
                  padding: "10px 12px",
                  transition: "border-color .15s",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: "#e6edf3" }}>
                    {inc.crime_type}
                  </span>
                  <span
                    style={{
                      fontSize: 10,
                      fontWeight: 600,
                      color: SEVERITY_COLOR(inc.severity),
                      background: `${SEVERITY_COLOR(inc.severity)}18`,
                      padding: "1px 6px",
                      borderRadius: 10,
                    }}
                  >
                    Sev: {inc.severity ? inc.severity.toFixed(2) : "N/A"}
                  </span>
                </div>

                <div style={{ fontSize: 11, color: "#c9d1d9", marginBottom: 6 }}>
                  {inc.description}
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#8b949e" }}>
                  <span>📍 {inc.location || "Urban Area"}</span>
                  <span>🗓 {inc.date} {inc.time ? `· ${inc.time}` : ""}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── Footer / Pagination ── */}
      <div
        style={{
          padding: "10px 18px",
          borderTop: "1px solid #21262d",
          background: "rgba(22, 27, 34, 0.8)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontSize: 11,
          color: "#8b949e",
        }}
      >
        <span>
          Showing {incidents.length > 0 ? (page - 1) * pageSize + 1 : 0}–
          {Math.min(page * pageSize, incidents.length)} of {totalRecords} incidents
        </span>
        <div style={{ display: "flex", gap: 6 }}>
          <button
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
            style={{
              padding: "3px 8px",
              background: "#21262d",
              border: "1px solid #30363d",
              borderRadius: 4,
              color: page <= 1 ? "#484f58" : "#c9d1d9",
              cursor: page <= 1 ? "default" : "pointer",
              fontSize: 10,
            }}
          >
            Prev
          </button>
          <span style={{ padding: "3px 6px", color: "#e6edf3" }}>
            {page}/{totalPages || 1}
          </span>
          <button
            disabled={page >= totalPages}
            onClick={() => setPage((p) => p + 1)}
            style={{
              padding: "3px 8px",
              background: "#21262d",
              border: "1px solid #30363d",
              borderRadius: 4,
              color: page >= totalPages ? "#484f58" : "#c9d1d9",
              cursor: page >= totalPages ? "default" : "pointer",
              fontSize: 10,
            }}
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
