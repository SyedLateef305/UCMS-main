// src/App.jsx
// Phase II: Added Forecast (🔮) and Patrol (🚔) tabs + Live Mode toggle + useLiveZones
import React, { useState } from "react";
import "./index.css";
import Header       from "./components/layout/Header";
import CrimeHeatMap from "./components/map/CrimeHeatMap";
import ZoneRiskPanel from "./components/zones/ZoneRiskPanel";
import AlertSystem   from "./components/alerts/AlertSystem";
import ChartsPanel   from "./components/charts/ChartsPanel";
import ForecastPanel from "./components/forecast/ForecastPanel";
import PatrolPanel   from "./components/patrol/PatrolPanel";
import SafeRoutePlanner from "./components/route/SafeRoutePlanner";
import { useApi }    from "./hooks/useApi";
import { getAllZones } from "./api/ucmsApi";
import { useLiveZones } from "./hooks/useApi";

export default function App() {
  const [selectedZone, setSelectedZone] = useState(null);
  const [selectedYear, setSelectedYear] = useState(2025);
  const [activeTab,    setActiveTab]    = useState("map");
  const [selectedSlot, setSelectedSlot] = useState(null);
  const [liveMode,     setLiveMode]     = useState(false);

  // Fetch base zones once; merge live overrides when liveMode is on
  const { data: zonesData } = useApi(() => getAllZones({ limit: 200 }), []);
  const { zones: liveZones, liveConnected } = useLiveZones(zonesData?.zones, liveMode);

  const handleZoneClick = (zone) => setSelectedZone(zone);

  const TABS = [
    { id: "map",      label: "🗺  Crime Map"   },
    { id: "route",    label: "🛡️  Safe Route"  },
    { id: "charts",   label: "📊  Analytics"   },
    { id: "forecast", label: "🔮  Forecast"     },
    { id: "patrol",   label: "🚔  Patrol"       },
  ];

  return (
    <div style={{
      height: "100vh", display: "flex", flexDirection: "column",
      overflow: "hidden", background: "#0d1117",
    }}>
      {/* ── Top header bar ─────────────────────────────────────────────── */}
      <Header
        selectedYear={selectedYear}
        onYearChange={setSelectedYear}
        liveMode={liveMode}
        onLiveModeToggle={() => setLiveMode((v) => !v)}
        liveConnected={liveConnected}
      />

      {/* ── Tab switcher ───────────────────────────────────────────────── */}
      <div style={{
        display: "flex", gap: 2, padding: "6px 16px 0",
        borderBottom: "1px solid #21262d", flexShrink: 0, background: "#0d1117",
      }}>
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: "6px 16px", fontSize: 12, fontWeight: 500,
              border: "none",
              borderBottom: activeTab === tab.id
                ? "2px solid #1f6feb" : "2px solid transparent",
              background: "transparent",
              color: activeTab === tab.id ? "#388bfd" : "#8b949e",
              cursor: "pointer", borderRadius: "4px 4px 0 0",
              transition: "color .15s",
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* ── Main content ───────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden", minHeight: 0 }}>

        {/* Left column — main content area */}
        <div style={{ flex: 1, minWidth: 0, position: "relative", overflow: "hidden" }}>
          {activeTab === "map" && (
            <CrimeHeatMap
              selectedZone={selectedZone}
              onZoneClick={handleZoneClick}
              selectedSlot={selectedSlot}
              selectedYear={selectedYear}
              liveZones={liveZones}
            />
          )}
          {activeTab === "route"    && <SafeRoutePlanner />}
          {activeTab === "charts"   && <ChartsPanel selectedYear={selectedYear} />}
          {activeTab === "forecast" && <ForecastPanel selectedZone={selectedZone} />}
          {activeTab === "patrol"   && <PatrolPanel />}
        </div>

        {/* Right sidebar — visible on non-route tabs */}
        {activeTab !== "route" && (
          <div style={{
            width: 340, flexShrink: 0, display: "flex",
            flexDirection: "column", borderLeft: "1px solid #21262d", overflow: "hidden",
          }}>

          {/* Zone detail panel */}
          {selectedZone ? (
            <div style={{ flex: 1, minHeight: 0, borderBottom: "1px solid #21262d", overflow: "hidden" }}>
              <ZoneRiskPanel
                zone={selectedZone}
                onClose={() => setSelectedZone(null)}
              />
            </div>
          ) : (
            <div style={{
              padding: 20, color: "#484f58", fontSize: 12,
              textAlign: "center", borderBottom: "1px solid #21262d", flexShrink: 0,
            }}>
              <div style={{ fontSize: 28, marginBottom: 8 }}>🗺</div>
              Click any zone on the map<br/>to view detailed risk info
            </div>
          )}

          {/* Alert system — always visible below */}
          <div style={{ flex: 1, minHeight: 0, overflow: "hidden" }}>
            <AlertSystem
              onZoneSelect={(zone) => {
                setSelectedZone(zone);
                setActiveTab("map");
              }}
            />
          </div>
        </div>
        )}
      </div>
    </div>
  );
}