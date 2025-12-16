#!/bin/bash
# Path to CSV produced earlier or predictions file written by your model.
# The model is expected to periodically (e.g. every 5s) overwrite this file with new predictions.
export PREDICTIONS_FILE="./predictions.csv"
# keep PREFETCH_CSV for backward compatibility with older scripts/tools
export PREFETCH_CSV="./predictions.csv"

# Target GPU device for prefetch (default 0)
export PREFETCH_DEVICE=0

# If you want prefetch to block until finishes before app starts, set PREFETCH_SYNC=1
# For the streaming/prediction mode we recommend asynchronous prefetch so the app can start
# while predictions/updates happen; default to 0 (async).
export PREFETCH_SYNC=0

# optional delay (seconds) before the first prefetch runs. Use if target program does early allocations.
export PREFETCH_DELAY_SEC=0

# Poll interval (ms) for the preload watcher to check for file updates (model overwrites file)
export PREFETCH_POLL_MS=1000

# Make LD_PRELOAD absolute and put preload_replayer first so its constructor runs early
# Make LD_PRELOAD absolute and put preload_replayer first so its constructor runs early
export LD_PRELOAD="./preload_replayer.so${LD_PRELOAD:+:}$LD_PRELOAD"

# Optionally also include your data_collector if you want additional runtime collection.
# If you wish to include it, uncomment the following line. Keep preload_replayer first.
# export LD_PRELOAD="/root/AI4Memv2/preload_replayer.so:/root/AI4Memv2/data_collector.so${LD_PRELOAD:+:}$LD_PRELOAD"

# Now exec the target program with its arguments
# Run the auto collection loop and tee stdout/stderr into prefetch_file.log
# Use bash -c so exec replaces the shell but keeps the pipe/tee behavior.
exec bash -c './auto_collect_loop.py --interval 5 --test-cmd "./test" --model ./lstm_model.pth \
  --total-log --total-sep --prune-rotated 2>&1 | tee -a prefetch_file.log'