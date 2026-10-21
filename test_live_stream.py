"""
Interactive Live WebSocket Audit Stream Client
Connects to ws://localhost:8000/ws/live-audit, displays live telemetry (FPS, object counts),
and renders throttled LangChain Gemini 1.5-flash audit alerts in real time.
"""

import asyncio
import json
import sys
import websockets

WS_URL = "ws://localhost:8000/ws/live-audit"


async def monitor_live_stream(source: str = "0", preview: bool = True):
    full_url = f"{WS_URL}?source={source}&preview={str(preview).lower()}"
    print(f"[*] Connecting to live stream WebSocket: {full_url}")

    try:
        async with websockets.connect(full_url) as ws:
            print("[+] Successfully connected to VisionAudit Live Stream.")
            print("[*] Listening for real-time telemetry and throttled LLM audits...\n")

            while True:
                msg_raw = await ws.recv()
                data = json.loads(msg_raw)
                event = data.get("event")
                ts = data.get("timestamp")

                if event == "status":
                    print(f"\033[94m[STATUS] {ts} - {data.get('message')}\033[0m")

                elif event == "telemetry":
                    fps = data.get("fps", 0.0)
                    telem = data.get("telemetry", {})
                    counts = telem.get("class_counts", {})
                    total = telem.get("total_count", 0)
                    has_preview = "YES" if (data.get("preview") or telem.get("annotated_image_base64")) else "NO"
                    # Pretty inline telemetry string
                    counts_str = ", ".join([f"{k}: {v}" for k, v in counts.items()]) if counts else "None"
                    print(f"\r[LIVE] FPS: {fps:.1f} | Detections: {total:2d} | Counts: [{counts_str}] | Preview: {has_preview}", end="", flush=True)

                elif event in ("critical_alert", "periodic_audit"):
                    reason = data.get("trigger_reason", "Audit Event")
                    report = data.get("report", {})
                    findings = report.get("findings", [])

                    header_color = "\033[91m" if event == "critical_alert" else "\033[93m"
                    reset_color = "\033[0m"

                    print(f"\n\n{header_color}{'='*70}")
                    print(f"[{event.upper()}] Trigger: {reason}")
                    print(f"Timestamp: {ts} | Summary: {report.get('summary')}")
                    print(f"{'-'*70}{reset_color}")

                    for idx, finding in enumerate(findings, 1):
                        sev = finding.get("severity", "LOW")
                        sev_color = "\033[91m" if sev == "HIGH" else ("\033[93m" if sev == "MEDIUM" else "\033[92m")
                        print(f"  {idx}. [{finding.get('category')}] {sev_color}{sev}{reset_color}: {finding.get('observation')}")
                        print(f"     Action: {finding.get('recommended_action')}")

                    print(f"{header_color}{'='*70}{reset_color}\n")

                elif event == "error":
                    print(f"\n\033[91m[ERROR] {data.get('message')}\033[0m\n")

    except websockets.exceptions.ConnectionClosed:
        print("\n[*] WebSocket connection closed.")
    except ConnectionRefusedError:
        print(f"\n[-] Could not connect to {WS_URL}. Ensure FastAPI is running on port 8000.")
    except KeyboardInterrupt:
        print("\n[*] Exiting live monitor.")


if __name__ == "__main__":
    src_arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    preview_arg = sys.argv[2].lower() != "false" if len(sys.argv) > 2 else True
    asyncio.run(monitor_live_stream(source=src_arg, preview=preview_arg))
