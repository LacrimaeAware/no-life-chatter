@echo off
REM Permanently stops the bot: kills it, stops the keep-alive loop, AND
REM removes the login autostart. It will NOT come back until you run
REM run-background.vbs again (and re-add autostart yourself if wanted).
cd /d "%~dp0"
if not exist data mkdir data
type nul > "data\STOP"
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*chatbot.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >/dev/null 2>&1
del /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\NoLifeChatter.lnk" >/dev/null 2>&1
echo Bot stopped, loop stopped, login autostart removed. Dead until you revive it.
pause
