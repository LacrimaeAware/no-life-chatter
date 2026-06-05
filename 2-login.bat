@echo off
REM ============================================================
REM  NoLifeChatter - log the bot in to Twitch (one-time)
REM  Opens your browser, you approve, it saves the token.
REM ============================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Run 1-setup.bat first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" scripts\get_initial_token.py
echo.
echo If that worked, you're logged in. Next: double-click  3-run.bat
pause
