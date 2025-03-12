#!/bin/bash

BASE_PATH="/app/Deeploy/DeeployTest/Tests"

if [ -z "$1" ]; then
    echo "❌ Please provide a test name as an argument. Example: ./visualizeCCTonnx.sh testFloatGemm"
    exit 1
fi

TEST_NAME="$1"


if [[ "$TEST_NAME" == /* ]]; then
    ONNX_PATH="$TEST_NAME"
else
    ONNX_PATH="${BASE_PATH}/${TEST_NAME}/network.onnx"
fi

if [ ! -f "$ONNX_PATH" ]; then
    echo "❌ The ONNX file does not exist: $ONNX_PATH"
    exit 1
fi


PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

if ! python3 -c "import netron" &> /dev/null; then
    echo "❌ Netron is not installed. Installing now..."
    pip install netron || { echo "❌ Failed to install Netron."; exit 1; }
fi


pkill -f "netron"

echo "🚀 Starting Netron in the background on http://localhost:$PORT ..."
nohup netron -p $PORT "$ONNX_PATH" --browser false > /dev/null 2>&1 &

echo "✅ Netron is running in the background."
echo "🌐 Access it at: http://localhost:$PORT"
