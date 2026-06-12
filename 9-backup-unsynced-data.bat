@echo off
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo NoLifeChatter - backup data\unsynced
echo.
"%PY%" scripts\backup_unsynced.py
echo.
pause
