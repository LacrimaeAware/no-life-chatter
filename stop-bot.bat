@echo off
REM Stops the background bot (and prevents the auto-restart loop from relaunching).
cd /d "%~dp0"
echo Stopping NoLifeChatter...
if not exist data mkdir data
type nul > "data\STOP"
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*chatbot.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
echo Stopped. (It will start again next time you log in, unless you remove auto-start.)
timeout /t 3 /nobreak >nul
