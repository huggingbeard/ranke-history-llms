#!/bin/bash
# Local backup loop (plan v2.6): copy pod /workspace/run/out and logs to the run folder every 10 minutes.
# Usage: bash sync_loop.sh <local_run_dir>
export MSYS_NO_PATHCONV=1
DEST="$1"; mkdir -p "$DEST"
cd /g/detritus/ranke1929_talkie_sft_20260928
while true; do
  python scripts/v2/pod.py get /workspace/run/out "$DEST/" >/dev/null 2>&1 && \
  python scripts/v2/pod.py get /workspace/run/logs "$DEST/" >/dev/null 2>&1 && \
  echo "synced $(date '+%F %T')" >> "$DEST/sync.log" || echo "sync failed $(date '+%F %T')" >> "$DEST/sync.log"
  [ -f "$DEST/STOP_SYNC" ] && exit 0
  sleep 600
done
