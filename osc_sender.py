from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 8000)

def send_wave_data(freqs, direction):
    client.send_message("/wave/freq_low", float(freqs["low"]))
    client.send_message("/wave/freq_mid", float(freqs["mid"]))
    client.send_message("/wave/freq_high", float(freqs["high"]))
    client.send_message("/wave/direction", float(direction))


def send_fused_wave_data(analysis):
    smooth = analysis["smoothed"]
    raw = analysis["raw"]

    client.send_message("/wave/frequency_hz", float(smooth["wave_frequency_hz"]))
    client.send_message("/wave/bump_size_common", float(smooth["bump_size_common"]))
    client.send_message("/wave/bump_size_spread", float(smooth["bump_size_spread"]))
    client.send_message("/wave/movement_direction_deg", float(smooth["movement_direction_deg"]))
    client.send_message("/wave/movement_speed_norm", float(smooth["movement_speed_norm"]))
    client.send_message("/wave/activity", float(smooth["activity"]))
    client.send_message("/wave/confidence", float(smooth["confidence"]))

    # Wavelength channels (scaled to match on-screen WL slider gain).
    wl_data = analysis.get("wavelength_data", {}) or {}
    wl_px = wl_data.get("wavelength_px")
    if wl_px is not None:
        wl_scaled = float(max(0.0, wl_px) * 1.3)
        client.send_message("/wave/wavelength_px", wl_scaled)
        client.send_message("/wave/wavelength_norm", float(min(wl_scaled / 300.0, 1.0)))

    wl_quads = wl_data.get("quadrants", {}) if isinstance(wl_data, dict) else {}
    for q in ("UL", "UR", "LL", "LR"):
        q_wl = (wl_quads.get(q) or {}).get("wavelength_px")
        if q_wl is not None:
            q_scaled = float(max(0.0, q_wl) * 1.3)
            client.send_message(f"/wave/wavelength/{q.lower()}_px", q_scaled)
            client.send_message(f"/wave/wavelength/{q.lower()}_norm", float(min(q_scaled / 300.0, 1.0)))

    # Pyramid S/T channels (scaled for more expressive control range).
    pyr = analysis.get("pyramid_data", {}) or {}
    g_s = pyr.get("global_bands", [0.0, 0.0, 0.0])
    g_t = pyr.get("temporal_band_activity", [0.0, 0.0, 0.0])
    for i, v in enumerate(g_s):
        client.send_message(f"/wave/pyramid/global/s{i}", float(min(max(float(v) * 1.3, 0.0), 1.0)))
    for i, v in enumerate(g_t):
        client.send_message(f"/wave/pyramid/global/t{i}", float(min(max(float(v) * 1.3, 0.0), 1.0)))

    q_s = pyr.get("quadrant_bands", {}) if isinstance(pyr, dict) else {}
    q_t = pyr.get("quadrant_temporal_bands", {}) if isinstance(pyr, dict) else {}
    for q in ("UL", "UR", "LL", "LR"):
        q_key = q.lower()
        for i, v in enumerate(q_s.get(q, [0.0, 0.0, 0.0])):
            client.send_message(f"/wave/pyramid/{q_key}/s{i}", float(min(max(float(v) * 1.3, 0.0), 1.0)))
        for i, v in enumerate(q_t.get(q, [0.0, 0.0, 0.0])):
            client.send_message(f"/wave/pyramid/{q_key}/t{i}", float(min(max(float(v) * 1.3, 0.0), 1.0)))

    # Optional debug channels for inspectability in downstream tools.
    client.send_message("/wave/raw/frequency_hz", float(raw["wave_frequency_hz"]))
    client.send_message("/wave/raw/activity", float(raw["activity"]))

    # Per-scale and adaptive flow direction channels.
    flow = analysis.get("flow_data", {})
    client.send_message("/wave/flow/fast_direction_deg", float(flow.get("fast_direction_deg", 0.0)))
    client.send_message("/wave/flow/fast_activity", float(flow.get("fast_activity", 0.0)))
    client.send_message("/wave/flow/fast_coherence", float(flow.get("fast_coherence", 0.0)))
    client.send_message("/wave/flow/slow_direction_deg", float(flow.get("slow_direction_deg", 0.0)))
    client.send_message("/wave/flow/slow_activity", float(flow.get("slow_activity", 0.0)))
    client.send_message("/wave/flow/slow_coherence", float(flow.get("slow_coherence", 0.0)))
    client.send_message("/wave/flow/adaptive_direction_deg", float(flow.get("adaptive_direction_deg", 0.0)))
    client.send_message("/wave/flow/adaptive_activity", float(flow.get("adaptive_activity", 0.0)))
    client.send_message("/wave/flow/adaptive_coherence", float(flow.get("adaptive_coherence", 0.0)))
    client.send_message("/wave/flow/direction_quality", float(flow.get("direction_quality", 0.0)))