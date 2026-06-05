@echo off
REM ============================================================
REM  NoLifeChatter - one-time setup
REM  Double-click this once. It installs everything and builds
REM  the database. Re-running it is safe.
REM ============================================================
cd /d "%~dp0"

echo.
echo [1/3] Creating a private Python environment (.venv)...
python -m venv .venv
if errorlevel 1 (
  echo.
  echo ERROR: Python was not found. Install Python 3.12+ from https://www.python.org/downloads/
  echo Make sure to tick "Add Python to PATH" during install, then run this again.
  pause
  exit /b 1
)

echo.
echo [2/3] Installing dependencies (this can take a few minutes)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo ERROR: Something went wrong installing dependencies. Scroll up to see why.
  pause
  exit /b 1
)

echo.
echo [3/3] Setting up the database...
".venv\Scripts\python.exe" scripts\init_db.py

echo.
echo ============================================================
echo  Setup complete!
echo  Next: double-click  2-login.bat  to log the bot in.
echo ============================================================
pause
