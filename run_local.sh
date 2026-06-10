#!/usr/bin/env bash

# 选择后端端口（占用则顺延）
BACKEND_PORT="${BACKEND_PORT:-8000}"
while lsof -i :"$BACKEND_PORT" >/dev/null 2>&1; do
  echo "Port $BACKEND_PORT in use, trying next..."
  BACKEND_PORT=$((BACKEND_PORT + 1))
done

# 启动后端（带 access log）
cd chatbox-backend
if [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi
"$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port "$BACKEND_PORT" --reload --log-level info --access-log &
BACK_PID=$!
echo "Backend started on port $BACKEND_PORT with PID $BACK_PID"

# 启动前端（Vite），指向选中的后端端口
cd ../chatbox-frontend
VITE_API_URL="http://127.0.0.1:${BACKEND_PORT}" npm run dev

# 前端退出后，停止后端
echo "Frontend exited. Stopping backend..."
if kill -0 "$BACK_PID" >/dev/null 2>&1; then
  kill "$BACK_PID"
  echo "Backend (PID $BACK_PID) stopped."
else
  echo "Backend process already stopped."
fi
