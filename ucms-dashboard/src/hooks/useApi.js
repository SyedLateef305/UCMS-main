// src/hooks/useApi.js
// Phase II: Added useWebSocket and useLiveZones hooks.
import { useState, useEffect, useCallback, useRef } from "react";
import { connectLiveWebSocket } from "../api/ucmsApi";

/**
 * Generic data-fetching hook.
 * Usage:
 *   const { data, loading, error, refetch } = useApi(getOverview);
 *   const { data } = useApi(() => getZoneDetail(zoneId), [zoneId]);
 */
export function useApi(apiFn, deps = []) {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await apiFn();
      setData(result);
    } catch (err) {
      setError(err?.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => { fetch(); }, [fetch]);

  return { data, loading, error, refetch: fetch };
}

/**
 * Auto-refresh hook — polls every `intervalMs` milliseconds.
 * Usage:
 *   const { data } = useApiPolled(getAlerts, 30000); // refresh every 30s
 */
export function useApiPolled(apiFn, intervalMs = 30000, deps = []) {
  const result = useApi(apiFn, deps);

  useEffect(() => {
    const id = setInterval(result.refetch, intervalMs);
    return () => clearInterval(id);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, ...deps]);

  return result;
}

/**
 * [Phase II] WebSocket hook — connects to /ws/live and calls onMessage for each update.
 *
 * Usage:
 *   const { connected } = useWebSocket(true, (msg) => console.log(msg));
 *
 * @param {boolean}  active    Connect only when true (controlled by Live Mode toggle)
 * @param {function} onMessage Callback receiving parsed JSON payload
 */
export function useWebSocket(active, onMessage) {
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!active) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
        setConnected(false);
      }
      return;
    }

    const ws = connectLiveWebSocket((data) => {
      setConnected(true);
      onMessage(data);
    });

    ws.onopen  = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    wsRef.current = ws;
    return () => { ws.close(); wsRef.current = null; };
  }, [active]); // eslint-disable-line react-hooks/exhaustive-deps

  return { connected };
}

/**
 * [Phase II] useLiveZones — merges WebSocket snapshots into zone state.
 *
 * @param {Array}   baseZones   Initial zones list from REST API
 * @param {boolean} liveMode    Whether WebSocket is active
 * @returns {{ zones, liveConnected }}
 */
export function useLiveZones(baseZones, liveMode) {
  const [liveOverrides, setLiveOverrides] = useState({});

  const { connected: liveConnected } = useWebSocket(liveMode, (msg) => {
    if (msg.type === "snapshot" && Array.isArray(msg.data)) {
      const overrides = {};
      msg.data.forEach((z) => {
        overrides[z.zone_id] = {
          mean_ccrp:  z.mean_ccrp,
          risk_level: z.risk_level,
        };
      });
      setLiveOverrides(overrides);
    }
  });

  // Merge live overrides into base zone list
  const zones = (baseZones || []).map((zone) => {
    const ov = liveOverrides[zone.zone_id];
    return ov ? { ...zone, ...ov } : zone;
  });

  return { zones, liveConnected };
}