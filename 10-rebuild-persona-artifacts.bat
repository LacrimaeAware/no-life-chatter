@echo off
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo NoLifeChatter - rebuild persona artifacts
echo.
echo This can take a while and the embeddings step needs LM Studio embeddings running.
echo It rebuilds classifier, style profiles, semantic vectors, and trait-axis smoke output.
echo.
set /p OK=Type YES to start, or anything else to cancel:
if /I not "%OK%"=="YES" (
  echo Cancelled.
  pause
  exit /b 0
)
echo.
"%PY%" scripts\rebuild_persona_artifacts.py
echo.
pause
