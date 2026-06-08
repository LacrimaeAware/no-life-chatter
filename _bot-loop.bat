@echo off
REM Internal runner — don't double-click this directly (use run-background.vbs).
REM Runs the bot, restarts it if it crashes, and logs everything to data\bot.log.
cd /d "%~dp0"
if not exist data mkdir data

REM Ensure only one instance: stop any bot that's already running.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*chatbot.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
del /q "data\STOP" >nul 2>&1

:loop
if exist "data\STOP" goto end
echo. >> "data\bot.log"
echo [%date% %time%] starting bot >> "data\bot.log"
".venv\Scripts\python.exe" -u chatbot.py >> "data\bot.log" 2>&1
if exist "data\STOP" goto end
echo [%date% %time%] bot exited - restarting in 10s >> "data\bot.log"
timeout /t 10 /nobreak >nul
goto loop

:end
echo [%date% %time%] bot stopped by user >> "data\bot.log"
del /q "data\STOP" >nul 2>&1
