"""Record Rokoko Studio's custom livestream (UDP, JSON v3, plain or LZ4) to a file, exactly as sent.

    .venv/bin/python tools/record_studio.py [--port 14043] [--out recordings/NAME.rkkudp]

File: repeated records of  <double seconds since start> <uint32 length> <datagram bytes>  (little-endian).
Ctrl-C stops. Prints the frame rate and both hand positions (forward, left, up from the chest, metres) each second.
Only one program can receive the stream at a time: stop the bridge first.
"""
import argparse, json, socket, struct, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from remote_hands.rokoko import RokokoStudioSource

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=14043)
ap.add_argument("--out", default=None)
a = ap.parse_args()
out = Path(a.out or f"recordings/suit_{time.strftime('%H%M%S')}.rkkudp")
out.parent.mkdir(parents=True, exist_ok=True)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", a.port))
sock.settimeout(1.0)
print(f"recording to {out}  (Ctrl-C to stop)", flush=True)
t0, n, last, n_last = time.monotonic(), 0, time.monotonic(), 0
with open(out, "wb") as f:
    try:
        while True:
            try:
                data, _ = sock.recvfrom(1 << 20)
            except socket.timeout:
                print("  no data from Studio (is the livestream on?)", flush=True)
                continue
            now = time.monotonic()
            f.write(struct.pack("<dI", now - t0, len(data)) + data)
            n += 1
            if now - last >= 1.0:
                f.flush()
                try:
                    if data[:4] == b"\x04\x22\x4d\x18":
                        import lz4.frame
                        data = lz4.frame.decompress(data)
                    w = RokokoStudioSource.wrists_from_frame(json.loads(data))
                    hands = "  ".join(f"{s} ({w[s][0]:+.2f} {w[s][1]:+.2f} {w[s][2]:+.2f})" for s in w)
                except Exception as e:
                    hands = f"frame not readable: {e!r}"
                print(f"  {now - t0:6.1f} s  {n - n_last:3d} frames/s  {hands}", flush=True)
                last, n_last = now, n
    except KeyboardInterrupt:
        pass
print(f"saved {n} frames, {time.monotonic() - t0:.0f} s, {out.stat().st_size / 1e6:.1f} MB -> {out}")
