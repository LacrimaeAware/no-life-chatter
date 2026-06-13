@echo off
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo NoLifeChatter - train intent probes
echo.
echo Uses the private oracle labels in data\unsynced\oracle and writes:
echo   data\unsynced\intent_probes.pkl
echo   _private\INTENT_PROBES_REPORT.md
echo.
echo LM Studio embeddings are used when available; otherwise the script falls back to TF-IDF.
echo.
"%PY%" scripts\train_intent_probes.py
echo.
pause
