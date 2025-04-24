
BASE_PATH="/app/Deeploy/DeeployTest/Tests"
if [ -z "$1" ]; then
    echo "❌ Please provide a test name as an argument. Example: ./visualizeCCTonnx.sh testFloatGemm"
    exit 1
fi
TEST_NAME="$1"

ONNX_PATH=${TEST_NAME}


if [ ! -f "$ONNX_PATH" ]; then
    echo "❌ The ONNX file does not exist: $ONNX_PATH"
    exit 1
fi


PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")


echo "🚀 Starting Netron in the background on http://localhost:$PORT ..."
nohup /usr/local/bin/netron -p $PORT "$ONNX_PATH" > /dev/null 2>&1 &


echo "✅ Netron is running in the background."
echo "🌐 Access it at: http://localhost:$PORT"
