#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
export $(cat .env | grep -v '^#' | xargs)

echo ""
echo "Starting server at http://127.0.0.1:8000"
echo "Press Ctrl+C to stop. Leave this window open while testing in another window."
echo ""

uvicorn app.api.main:app --reload --reload-dir app
