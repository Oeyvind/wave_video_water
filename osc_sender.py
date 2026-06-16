import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 8100)

SLIT_SAMPLE_COUNT = 512
SLIT_CHUNK_SIZE = 16
_slit_frame_id = 0


def _extract_center_slit_samples(analysis, sample_count=SLIT_SAMPLE_COUNT):
    slit_fft = analysis.get("slit_fft_data") if isinstance(analysis, dict) else None
    waveform = None
    if isinstance(slit_fft, dict):
        waveform = slit_fft.get("waveform")
    if waveform is not None:
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if waveform.size > 0:
            if waveform.size == sample_count:
                return np.clip(waveform, 0.0, 1.0)
            x_src = np.linspace(0.0, 1.0, waveform.size, dtype=np.float32)
            x_dst = np.linspace(0.0, 1.0, int(sample_count), dtype=np.float32)
            return np.clip(np.interp(x_dst, x_src, waveform), 0.0, 1.0)

    src = analysis.get("roi_gray") if isinstance(analysis, dict) else None
    if src is None or getattr(src, "size", 0) == 0:
        return None

    if len(src.shape) != 2:
        return None

    h, _w = src.shape[:2]
    if h <= 0:
        return None

    center_y = h // 2
    slit = src[center_y:center_y + 1, :]
    if slit.size == 0:
        return None

    # Area resampling gives stable anti-aliased downsampling for wide slits.
    slit_ds = cv2.resize(slit, (int(sample_count), 1), interpolation=cv2.INTER_AREA)
    return np.clip(slit_ds.astype(np.float32).reshape(-1) / 255.0, 0.0, 1.0)


def _send_center_slit_chunks(analysis):
    global _slit_frame_id

    samples = _extract_center_slit_samples(analysis, sample_count=SLIT_SAMPLE_COUNT)
    if samples is None:
        return

    chunk_size = int(max(1, SLIT_CHUNK_SIZE))
    total = int(samples.size)
    if total <= 0 or (total % chunk_size) != 0:
        return

    chunk_count = total // chunk_size
    frame_id = int(_slit_frame_id)
    _slit_frame_id += 1

    for chunk_idx in range(chunk_count):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk_vals = [float(v) for v in samples[start:end]]
        payload = [frame_id, int(chunk_idx), int(chunk_count)] + chunk_vals
        client.send_message("/slit/chunk", payload)

def send_wave_data(freqs, direction):
    client.send_message("/wave/freq_low", float(freqs["low"]))
    client.send_message("/wave/freq_mid", float(freqs["mid"]))
    client.send_message("/wave/freq_high", float(freqs["high"]))
    client.send_message("/wave/direction", float(direction))


def send_fused_wave_data(analysis):
    smooth = analysis["smoothed"]
    raw = analysis["raw"]
    slit_fft = analysis.get("slit_fft_data") or {}

    def _scale_a(v):
        return float(np.clip(float(v) * 3.0, 0.0, 1.0))

    def _clip01(v):
        return float(np.clip(float(v), 0.0, 1.0))

    client.send_message("/wave/frequency_hz", float(smooth["wave_frequency_hz"]))
    client.send_message("/wave/bump_size_common", float(smooth["bump_size_common"]))
    client.send_message("/wave/bump_size_spread", float(smooth["bump_size_spread"]))
    client.send_message("/wave/movement_direction_deg", float(smooth["movement_direction_deg"]))
    client.send_message("/wave/movement_speed_norm", float(smooth["movement_speed_norm"]))
    client.send_message("/wave/activity", _scale_a(smooth["activity"]))
    client.send_message("/wave/confidence", float(smooth["confidence"]))

    slit_fft_centroid = _clip01(slit_fft.get("fft_centroid_norm", 0.0))
    client.send_message("/wave/slit_fft/centroid", slit_fft_centroid)

    def _clip(v, lo, hi):
        return float(min(max(float(v), float(lo)), float(hi)))

    # Wavelength channels (scaled to match on-screen WL slider gain).
    wl_data = analysis.get("wavelength_data", {}) or {}
    wl_px = wl_data.get("wavelength_px")
    wl0_norm = 0.0
    if wl_px is not None:
        wl_scaled = float(max(0.0, wl_px) * 1.3)
        wl0_norm = float(min(wl_scaled / 300.0, 1.0))

    wl_quads = wl_data.get("quadrants", {}) if isinstance(wl_data, dict) else {}
    wl_pack = [wl0_norm]
    for q in ("UL", "UR", "LL", "LR"):
        q_wl = (wl_quads.get(q) or {}).get("wavelength_px")
        q_norm = 0.0
        if q_wl is not None:
            q_scaled = float(max(0.0, q_wl) * 1.3)
            q_norm = float(min(q_scaled / 300.0, 1.0))
        wl_pack.append(q_norm)

    # Packed wavelength message: global + UL/UR/LL/LR normalized values.
    client.send_message("/wave/wavelength/pack", wl_pack)

    # Pyramid S/T channels (scaled for more expressive control range).
    pyr = analysis.get("pyramid_data", {}) or {}
    g_s = pyr.get("global_bands", [0.0, 0.0, 0.0])
    g_t = pyr.get("temporal_band_activity", [0.0, 0.0, 0.0])
    g_s_ctr = float(pyr.get("scale_centroid_renorm", pyr.get("scale_centroid", 0.0)))
    g_t_ctr = float(pyr.get("temporal_scale_centroid", 0.0))
    g_pack = [
        _clip(g_s[0] * 1.3, 0.0, 1.0),
        _clip(g_s[1] * 1.3, 0.0, 1.0),
        _clip(g_s[2] * 1.3, 0.0, 1.0),
        _clip(g_t[0] * 1.3, 0.0, 1.0),
        _clip(g_t[1] * 1.3, 0.0, 1.0),
        _clip(g_t[2] * 1.3, 0.0, 1.0),
        _clip(g_s_ctr, 0.0, 1.0),
        _clip(g_t_ctr, 0.0, 1.0),
    ]
    client.send_message("/wave/pyramid/global/pack", g_pack)

    q_s = pyr.get("quadrant_bands", {}) if isinstance(pyr, dict) else {}
    q_t = pyr.get("quadrant_temporal_bands", {}) if isinstance(pyr, dict) else {}
    q_s_ctr = pyr.get("quadrant_scale_centroids_renorm", pyr.get("quadrant_scale_centroids", {})) if isinstance(pyr, dict) else {}
    q_t_ctr = pyr.get("quadrant_temporal_scale_centroids", {}) if isinstance(pyr, dict) else {}
    for q in ("UL", "UR", "LL", "LR"):
        q_key = q.lower()
        qsv = q_s.get(q, [0.0, 0.0, 0.0])
        qtv = q_t.get(q, [0.0, 0.0, 0.0])
        q_pack = [
            _clip(qsv[0] * 1.3, 0.0, 1.0),
            _clip(qsv[1] * 1.3, 0.0, 1.0),
            _clip(qsv[2] * 1.3, 0.0, 1.0),
            _clip(qtv[0] * 1.3, 0.0, 1.0),
            _clip(qtv[1] * 1.3, 0.0, 1.0),
            _clip(qtv[2] * 1.3, 0.0, 1.0),
            _clip(float(q_s_ctr.get(q, 0.0)), 0.0, 1.0),
            _clip(float(q_t_ctr.get(q, 0.0)), 0.0, 1.0),
        ]
        client.send_message(f"/wave/pyramid/{q_key}/pack", q_pack)

    # LBP channels (global + quadrants).
    lbp = analysis.get("lbp_data", {}) or {}
    lbp_q = lbp.get("quadrants", {}) if isinstance(lbp, dict) else {}

    g_smooth = _clip(lbp.get("lbp_smooth", 0.0), 0.0, 1.0)
    g_order = float(lbp.get("lbp_order", 0.0))
    g_chaos = float(lbp.get("lbp_chaos", 0.0))
    g_order_chaos = _clip(g_order - g_chaos, -1.0, 1.0)

    q_smooth_pack = [g_smooth]
    q_orderchaos_pack = [g_order_chaos]
    for q in ("UL", "UR", "LL", "LR"):
        qd = lbp_q.get(q, {}) if isinstance(lbp_q, dict) else {}
        q_smooth = _clip(qd.get("lbp_smooth", 0.0), 0.0, 1.0)
        q_order = float(qd.get("lbp_order", 0.0))
        q_chaos = float(qd.get("lbp_chaos", 0.0))
        q_order_chaos = _clip(q_order - q_chaos, -1.0, 1.0)
        q_smooth_pack.append(q_smooth)
        q_orderchaos_pack.append(q_order_chaos)

    # Packed LBP messages (global + 4 quadrants).
    client.send_message("/wave/lbp/smooth_pack", q_smooth_pack)
    client.send_message("/wave/lbp/orderchaos_pack", q_orderchaos_pack)

    # Optional debug channels for inspectability in downstream tools.
    client.send_message("/wave/raw/frequency_hz", float(raw["wave_frequency_hz"]))
    client.send_message("/wave/raw/activity", _scale_a(raw["activity"]))

    # Per-scale and adaptive flow direction channels.
    flow = analysis.get("flow_data", {})
    client.send_message("/wave/flow/fast_pack", [
        float(flow.get("fast_direction_deg", 0.0)),
        float(flow.get("fast_activity", 0.0)),
        float(flow.get("fast_coherence", 0.0)),
    ])
    client.send_message("/wave/flow/slow_pack", [
        float(flow.get("slow_direction_deg", 0.0)),
        float(flow.get("slow_activity", 0.0)),
        float(flow.get("slow_coherence", 0.0)),
    ])
    client.send_message("/wave/flow/adaptive_direction_deg", float(flow.get("adaptive_direction_deg", 0.0)))
    client.send_message("/wave/flow/adaptive_activity", float(flow.get("adaptive_activity", 0.0)))
    client.send_message("/wave/flow/adaptive_coherence", float(flow.get("adaptive_coherence", 0.0)))
    client.send_message("/wave/flow/direction_quality", float(flow.get("direction_quality", 0.0)))

    # Per-quadrant fast flow: [UL_dir, UL_act, UR_dir, UR_act, LL_dir, LL_act, LR_dir, LR_act]
    qfm = flow.get("quadrant_fast_metrics") or {}
    fast_quad_pack = []
    for q in ("UL", "UR", "LL", "LR"):
        m = qfm.get(q) or {}
        fast_quad_pack += [float(m.get("direction_deg", 0.0)), float(m.get("activity", 0.0))]
    client.send_message("/wave/flow/fast_quad_pack", fast_quad_pack)

    # Per-quadrant slow flow: [UL_dir, UL_act, UR_dir, UR_act, LL_dir, LL_act, LR_dir, LR_act]
    qsm = flow.get("quadrant_slow_metrics") or {}
    slow_quad_pack = []
    for q in ("UL", "UR", "LL", "LR"):
        m = qsm.get(q) or {}
        slow_quad_pack += [float(m.get("direction_deg", 0.0)), float(m.get("activity", 0.0))]
    client.send_message("/wave/flow/slow_quad_pack", slow_quad_pack)

    # Fused activity: global + UL/UR/LL/LR
    act = analysis.get("activity_data") or {}
    q_act = act.get("quadrant_activity") or {}
    act_pack = [
        _scale_a(act.get("global_activity", 0.0)),
        _scale_a(q_act.get("UL", 0.0)),
        _scale_a(q_act.get("UR", 0.0)),
        _scale_a(q_act.get("LL", 0.0)),
        _scale_a(q_act.get("LR", 0.0)),
    ]
    client.send_message("/wave/activity/pack", act_pack)

    client.send_message("/wave/slit_fft/centroid_pack", [slit_fft_centroid] * 5)

    # Central horizontal slit waveform transport: 512 samples in 16-float chunks.
    _send_center_slit_chunks(analysis)