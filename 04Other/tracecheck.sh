#!/bin/bash

OUTPUT_FILE="matmul_gvsoc_rv32imf.txt"
MAX_SIZE=10485760 
SLEEP_INTERVAL=1

./install/bin/gvsoc \
  --target=siracusa \
  --binary /app/Deeploy/DeeployTest/TEST_SIRACUSA/build/bin/testFloatMatmul \
  run --trace-level=6 --trace=/chip/cluster/pe0/ > "$OUTPUT_FILE" &
PID=$!



echo "🚀 Start gvsoc，PID=$PID"

while kill -0 $PID 2>/dev/null; do
    if [ -f "$OUTPUT_FILE" ]; then
        SIZE=$(stat -c%s "$OUTPUT_FILE")
        if [ "$SIZE" -gt "$MAX_SIZE" ]; then
            echo "🛑 Kill gvsoc"
            pkill -f gvsoc
            break
        fi
    fi
    sleep $SLEEP_INTERVAL
done

echo "✅ Finished"
