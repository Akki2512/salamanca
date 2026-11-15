#!/bin/bash
# Tunnel between this laptop and the pod, restarted whenever it drops. Needs POD_HOST and POD_PORT (not in git):
#   laptop :8765  -> the robot node on the pod          (worker page, bridge, scripts)
#   pod    :18021 -> the session server on this laptop  (the robot node's link back)
while true; do
    echo "[$(date +%H:%M:%S)] tunnel up"
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -o ConnectTimeout=10 \
        -p "${POD_PORT:?set POD_PORT}" -L 8765:127.0.0.1:8765 -R 127.0.0.1:18021:127.0.0.1:8000 "root@${POD_HOST:?set POD_HOST}"
    echo "[$(date +%H:%M:%S)] tunnel dropped, reconnecting in 2 s"
    sleep 2
done
