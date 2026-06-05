@echo off
REM ============================================================
REM  NoLifeChatter - run the bot
REM  Double-click to start. Close this window to stop the bot.
REM ============================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Run 1-setup.bat first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" chatbot.py
echo.
echo The bot stopped. Scroll up to see why if it wasn't on purpose.
pause
