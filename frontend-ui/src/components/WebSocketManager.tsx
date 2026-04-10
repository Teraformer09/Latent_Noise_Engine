import { useEffect, useRef } from "react";
import { decode } from "@msgpack/msgpack";
import { useSimulationStore } from "../store/useSimulationStore";

const _API_BASE = (import.meta as any).env?.VITE_API_BASE ?? "http://localhost:8000";
const WS_URL = _API_BASE.replace(/^http/, "ws") + "/ws/stream";
const BACKOFF_BASE_MS = 500;
const BACKOFF_MAX_MS = 10_000;

export default function WebSocketManager(): null {
  const wsRef = useRef<WebSocket | null>(null);
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const unmountedRef = useRef(false);

  const { setConnected, updateFromTelemetry, startStatusPolling } = useSimulationStore.getState();

  useEffect(() => {
    unmountedRef.current = false;
    const stopPolling = startStatusPolling(_API_BASE);

    function connect() {
      if (unmountedRef.current) return;

      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        retryCountRef.current = 0;
        setConnected(true);
      };

      ws.onmessage = (event: MessageEvent<ArrayBuffer>) => {
        try {
          const decoded = decode(new Uint8Array(event.data)) as Record<string, unknown>;
          updateFromTelemetry(decoded);
        } catch (err) {
          console.warn("[WebSocketManager] Failed to decode msgpack frame:", err);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;
        scheduleReconnect();
      };

      ws.onerror = () => {
        // onclose will fire after onerror — no double-schedule needed
        setConnected(false);
      };
    }

    function scheduleReconnect() {
      if (unmountedRef.current) return;
      const delay = Math.min(
        BACKOFF_BASE_MS * 2 ** retryCountRef.current,
        BACKOFF_MAX_MS
      );
      retryCountRef.current += 1;
      retryTimerRef.current = setTimeout(connect, delay);
    }

    connect();

    return () => {
      unmountedRef.current = true;
      stopPolling();
      if (retryTimerRef.current !== null) {
        clearTimeout(retryTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return null;
}
