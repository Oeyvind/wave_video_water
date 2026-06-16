import argparse
import time
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.udp_client import SimpleUDPClient


def checksum16(payload: bytes) -> int:
    return int(sum(payload) & 0xFFFF)


def make_payload(size: int, include_zero: bool) -> bytes:
    if include_zero:
        # Full byte range repeats (includes zero).
        return bytes([i % 256 for i in range(size)])
    # 1..255 repeats (avoids NUL byte) to test string-truncation behavior.
    return bytes([((i % 255) + 1) for i in range(size)])


def send_blob(client: SimpleUDPClient, address: str, payload: bytes, seq: int, label: str) -> None:
    b = OscMessageBuilder(address=address)
    b.add_arg(payload, arg_type="b")
    b.add_arg(seq, arg_type="i")
    b.add_arg(len(payload), arg_type="i")
    b.add_arg(checksum16(payload), arg_type="i")
    b.add_arg(label)
    msg = b.build()
    client.send(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send OSC blob test packets to Csound OSCraw")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()

    client = SimpleUDPClient(args.host, args.port)

    print(f"Sending to {args.host}:{args.port} size={args.size} count={args.count}")
    for seq in range(args.count):
        include_zero = (seq % 2) == 1
        payload = make_payload(args.size, include_zero=include_zero)
        label = "with_zero" if include_zero else "no_zero"
        send_blob(client, "/blob/test", payload, seq, label)
        print(
            f"sent seq={seq:02d} label={label:9s} len={len(payload)} checksum={checksum16(payload)}"
        )
        time.sleep(max(0.0, args.interval))

    print("Done.")


if __name__ == "__main__":
    main()
