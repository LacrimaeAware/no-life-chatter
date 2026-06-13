@echo off
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo NoLifeChatter - build intent-axis review queue v2
echo.
echo Requires data\unsynced\intent_probes.pkl from 12-train-intent-probes.bat.
echo Writes a private review queue to the ai-prompt-engineering review_queues folder.
echo Default sample is 700 archive candidates, which can still take a minute or two.
echo.
"%PY%" scripts\build_intent_axis_queue.py
echo.
pause
