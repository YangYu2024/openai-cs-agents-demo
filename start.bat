@echo off
REM Start Python backend
start cmd /k "cd python-backend && .venv\Scripts\activate && python api.py"
REM Start frontend
start cmd /k "cd ui && npm run dev"
