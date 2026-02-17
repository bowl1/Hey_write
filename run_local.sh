#!/usr/bin/env bash
set -euo pipefail

# 后端
cd chatbox-backend

if [ ! -f ".venv/bin/activate" ]; then
  echo "Creating backend virtualenv..."
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload &
BACK_PID=$!
echo "Backend started with PID $BACK_PID"

# 前端
cd ../chatbox-frontend
npm install
npm start

# 前端关闭后，再杀后端
echo "Frontend exited. Stopping backend..."
kill $BACK_PID
