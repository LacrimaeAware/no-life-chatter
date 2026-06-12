@echo off
REM Internal runner — don't double-click this directly (use run-background.vbs).
REM Runs the bot, restarts it if it crashes, and logs everything to data\bot.log.
cd /d "%~dp0"
if not exist data mkdir data
REM UTF-8 stdout so emoji in chat/personas never break logging on Windows.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

REM Ensure only one instance: stop any bot that's already running.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*chatbot.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
del /q "data\STOP" >nul 2>&1

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

:loop
if exist "data\STOP" goto end
"%PY%" scripts\rotate_logs.py --quiet >nul 2>&1
echo. >> "data\bot.log"
echo [%date% %time%] starting bot >> "data\bot.log"
"%PY%" -u chatbot.py >> "data\bot.log" 2>&1
if exist "data\STOP" goto end
echo [%date% %time%] bot exited - restarting in 10s >> "data\bot.log"
timeout /t 10 /nobreak >nul
goto loop

:end
echo [%date% %time%] bot stopped by user >> "data\bot.log"
del /q "data\STOP" >nul 2>&1
