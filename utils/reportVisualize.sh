#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DEFAULT_PORT=8000
HTML_FILE="/app/Onnx4Deeploy/Report/csv/benchmark_visualize.html"
CSV_DIR="/app/Onnx4Deeploy/Report/csv"

show_help() {
    echo -e "${BLUE}Benchmark CSV Visualizer${NC}"
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help            Show this help message"
    echo "  -p, --port PORT       Use specified port number (default: 8000)"
    echo "  -d, --directory DIR   CSV directory path (default: $CSV_DIR)"
    echo ""
    echo "Examples:"
    echo "  $0                            Use default settings"
    echo "  $0 -p 8080                    Use port 8080"
    echo "  $0 -d /path/to/csv/dir        Use specified CSV directory"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--port)
            DEFAULT_PORT="$2"
            shift 2
            ;;
        -d|--directory)
            CSV_DIR="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Check if CSV directory exists
if [ ! -d "$CSV_DIR" ]; then
    echo -e "${RED}CSV directory '$CSV_DIR' does not exist${NC}"
    exit 1
fi

# Check if HTML file exists
if [ ! -f "$HTML_FILE" ]; then
    echo -e "${RED}HTML file '$HTML_FILE' does not exist${NC}"
    exit 1
fi

# Create server directory
SERVER_DIR="$CSV_DIR/viz_server"
mkdir -p "$SERVER_DIR"
echo -e "${BLUE}Created server directory: $SERVER_DIR${NC}"

# Copy HTML file to server directory
cp "$HTML_FILE" "$SERVER_DIR/index.html"

# Check if port is already in use
while netstat -tuln | grep -q ":$DEFAULT_PORT "; do
    echo -e "${YELLOW}Port $DEFAULT_PORT is already in use, trying another port...${NC}"
    DEFAULT_PORT=$((DEFAULT_PORT + 1))
done

# Get container IP if possible
if command -v hostname &> /dev/null; then
    HOST_IP=$(hostname -i 2>/dev/null || echo "localhost")
else
    HOST_IP="localhost"
fi

# Modify HTML to automatically switch to directory mode and list CSV files
sed -i 's/document.addEventListener("DOMContentLoaded", function() {/document.addEventListener("DOMContentLoaded", function() {\n    // Auto-switch to directory mode\n    dirMode.click();/' "$SERVER_DIR/index.html"

echo -e "${BLUE}Starting server from Docker container...${NC}"
echo -e "${GREEN}Access at:${NC}"
echo -e "  - http://localhost:$DEFAULT_PORT (if port is exposed)"
echo -e "  - http://$HOST_IP:$DEFAULT_PORT (from within Docker network)"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"

# Start HTTP server from the CSV directory, not the viz_server directory
# This way all CSV files are accessible at the root level
cd "$CSV_DIR" || exit 1
python3 -m http.server $DEFAULT_PORT