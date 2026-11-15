#!/bin/bash
# Keep the Muse headband streaming to LSL (OpenMuse), reconnecting after a Bluetooth drop. Run it in Terminal.app
# (macOS only grants Bluetooth to apps you allow). Set MUSE_ADDRESS to your headband's address (`OpenMuse find`).
cd "$(dirname "$0")" && source .venv/bin/activate
ADDR="${MUSE_ADDRESS:?Set MUSE_ADDRESS to your headband address (run: OpenMuse find)}"
mkdir -p results
while true; do
  echo "=== $(date '+%F %T') starting OpenMuse stream ===" | tee -a results/stream-log.txt
  PYTHONUNBUFFERED=1 OpenMuse stream --address "$ADDR" 2>&1 | tee -a results/stream-log.txt
  echo "=== $(date '+%F %T') stream exited; retrying in 5 s (Ctrl+C twice to quit) ===" | tee -a results/stream-log.txt
  sleep 5
done
