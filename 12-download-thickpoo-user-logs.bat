@echo off
setlocal
cd /d "%~dp0"

echo.
echo NoLifeChatter - download external logs for #thickpoo
echo.
echo Enter one or more Twitch logins, separated by spaces or commas.
echo Example: ebbel forsenstares 99froxy
echo.
set /p USERS=Users: 
if "%USERS%"=="" (
    echo No users entered.
    pause
    exit /b 1
)

echo.
set /p IMPORT=Also import into the local chat archive after downloading? [Y/n]: 
set "IMPORT_FLAG=--import-archive"
if /I "%IMPORT%"=="n" set "IMPORT_FLAG="
if /I "%IMPORT%"=="no" set "IMPORT_FLAG="

echo.
set /p LIMIT=Newest N months only for a test run? Leave blank for ALL logs: 
set "LIMIT_FLAG="
if not "%LIMIT%"=="" set "LIMIT_FLAG=--limit-months %LIMIT%"

echo.
echo Downloading from logs.zonian.dev for #thickpoo...
echo Raw private files: data\unsynced\external_logs\zonian\raw\thickpoo
echo.

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" scripts\download_zonian_user_logs.py --channel thickpoo --users "%USERS%" %IMPORT_FLAG% %LIMIT_FLAG%
set "ERR=%ERRORLEVEL%"

echo.
if not "%ERR%"=="0" (
    echo Download failed with exit code %ERR%.
) else (
    echo Done. Summary:
    echo data\unsynced\external_logs\zonian\thickpoo_download_summary.json
)
pause
exit /b %ERR%
