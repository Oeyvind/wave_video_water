import argparse
import statistics
import threading
import time
from typing import Dict, Tuple

from pythonosc import dispatcher
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient


def make_frame(frame_id: int):
    return [((idx + (frame_id * 7)) % 512) / 511.0 for idx in range(512)]


class AckStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.data: Dict[int, Tuple[float, int, float]] = {}

    def put(self, frame_id: int, recv_count: int, maxerr: float):
        with self.lock:
            self.data[frame_id] = (time.perf_counter(), int(recv_count), float(maxerr))

    def get(self, frame_id: int):
        with self.lock:
            return self.data.get(frame_id)


def build_chunk_message(frame_id: int, chunk_idx: int, chunk_count: int, chunk_values):
    msg = OscMessageBuilder(address="/slit/chunk")
    msg.add_arg(int(frame_id), arg_type="i")
    msg.add_arg(int(chunk_idx), arg_type="i")
    msg.add_arg(int(chunk_count), arg_type="i")
    for v in chunk_values:
        msg.add_arg(float(v), arg_type="f")
    return msg.build()


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (len(sorted_vals) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]))


def main():
    parser = argparse.ArgumentParser(description="512-sample slit chunk sender with ACK-based reliability/latency stats")
    parser.add_argument("--send-host", default="127.0.0.1")
    parser.add_argument("--send-port", type=int, default=8031)
    parser.add_argument("--ack-host", default="127.0.0.1")
    parser.add_argument("--ack-port", type=int, default=8032)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--frame-interval", type=float, default=0.03)
    parser.add_argument("--ack-timeout", type=float, default=0.5)
    args = parser.parse_args()

    if args.chunk_size <= 0 or 512 % args.chunk_size != 0:
        raise ValueError("chunk-size must be a positive divisor of 512")

    chunk_count = 512 // args.chunk_size

    acks = AckStore()
    disp = dispatcher.Dispatcher()

    def on_ack(_address, frame_id, recv_count, maxerr):
        acks.put(int(frame_id), int(recv_count), float(maxerr))

    disp.map("/slit/ack", on_ack)

    server = ThreadingOSCUDPServer((args.ack_host, args.ack_port), disp)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    client = SimpleUDPClient(args.send_host, args.send_port)

    print(
        f"Sending {args.frames} frames, 512 samples/frame, {chunk_count} chunks/frame, "
        f"chunk_size={args.chunk_size} to {args.send_host}:{args.send_port}, ack={args.ack_host}:{args.ack_port}"
    )

    sent_frames = 0
    acked_frames = 0
    maxerr_failures = 0
    chunk_mismatch = 0
    latencies_ms = []

    for frame_id in range(args.frames):
        frame = make_frame(frame_id)
        t0 = time.perf_counter()
        sent_frames += 1

        for chunk_idx in range(chunk_count):
            start = chunk_idx * args.chunk_size
            end = start + args.chunk_size
            msg = build_chunk_message(frame_id, chunk_idx, chunk_count, frame[start:end])
            client.send(msg)

        deadline = t0 + args.ack_timeout
        ack = None
        while time.perf_counter() < deadline:
            ack = acks.get(frame_id)
            if ack is not None:
                break
            time.sleep(0.001)

        if ack is not None:
            tack, recv_count, maxerr = ack
            rtt_ms = (tack - t0) * 1000.0
            latencies_ms.append(rtt_ms)
            acked_frames += 1
            if recv_count != chunk_count:
                chunk_mismatch += 1
            if maxerr > 1e-5:
                maxerr_failures += 1
            print(
                f"frame={frame_id:03d} acked rtt_ms={rtt_ms:7.3f} recv_chunks={recv_count:2d}/{chunk_count} maxerr={maxerr:.8f}"
            )
        else:
            print(f"frame={frame_id:03d} TIMEOUT after {args.ack_timeout:.3f}s")

        if args.frame_interval > 0.0:
            elapsed = time.perf_counter() - t0
            remain = args.frame_interval - elapsed
            if remain > 0.0:
                time.sleep(remain)

    server.shutdown()
    server.server_close()

    reliability = (acked_frames / sent_frames) if sent_frames > 0 else 0.0
    lat_sorted = sorted(latencies_ms)

    mean_ms = statistics.fmean(latencies_ms) if latencies_ms else 0.0
    median_ms = statistics.median(latencies_ms) if latencies_ms else 0.0
    p95_ms = percentile(lat_sorted, 0.95)
    max_ms = max(latencies_ms) if latencies_ms else 0.0

    print("\n=== RESULT ===")
    print(f"frames_sent={sent_frames}")
    print(f"frames_acked={acked_frames}")
    print(f"reliability={reliability * 100.0:.2f}%")
    print(f"maxerr_failures={maxerr_failures}")
    print(f"chunk_mismatch_frames={chunk_mismatch}")
    print(f"latency_rtt_mean_ms={mean_ms:.3f}")
    print(f"latency_rtt_median_ms={median_ms:.3f}")
    print(f"latency_rtt_p95_ms={p95_ms:.3f}")
    print(f"latency_rtt_max_ms={max_ms:.3f}")
    print(f"latency_one_way_est_ms={mean_ms * 0.5:.3f} (RTT/2 estimate)")


if __name__ == "__main__":
    main()
