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