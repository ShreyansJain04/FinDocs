#!/bin/bash
set -euo pipefail
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
python3 -m src.fdocs.services.llm_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model llama3 \
  --ollama-base-url "${OLLAMA_BASE_URL:-http://127.0.0.1:11434}" \
  --temperature 0.1 \
  --max-new-tokens 512 \
  "$@" \
  >"$LOG_DIR/llm_server_${TIMESTAMP}.log" 2>&1 &
PID=$!
echo $PID > "$LOG_DIR/llm_server.pid"
echo "LLM server started (PID=$PID). Logs: $LOG_DIR/llm_server_${TIMESTAMP}.log"
