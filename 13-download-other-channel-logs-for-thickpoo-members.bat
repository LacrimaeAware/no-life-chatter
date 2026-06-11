@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo NoLifeChatter - cross-channel logs for ThickPoo members
echo.
echo This downloads logs from other channels, but only for users already found
echo in your local #thickpoo archive.
echo.
echo Target channels:
echo duardo1 forsen nymn fernardo earnestsinceresugmamale ebbel fabzeef
echo heyimbee huni4president erobb221 sodapoppin skippypoppin moonmoon vei
echo jerma985 grubby pajlada mizkif avoidingthepuddle nl_kripp asmongold
echo zackrawrr jokerdtv
echo.

set "TARGETS=duardo1 forsen nymn fernardo earnestsinceresugmamale ebbel fabzeef heyimbee huni4president erobb221 sodapoppin skippypoppin moonmoon vei jerma985 grubby pajlada mizkif avoidingthepuddle nl_kripp asmongold zackrawrr jokerdtv"

set /p MIN=Minimum local #thickpoo messages per user [25]:
if "%MIN%"=="" set "MIN=25"

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
set /p INCLUDE_EXCLUDED=Include configured bot/noise accounts too? [y/N]:
set "EXCLUDED_FLAG="
if /I "%INCLUDE_EXCLUDED%"=="y" set "EXCLUDED_FLAG=--include-excluded"
if /I "%INCLUDE_EXCLUDED%"=="yes" set "EXCLUDED_FLAG=--include-excluded"

echo.
echo Raw private files: data\unsynced\external_logs\zonian\raw\CHANNEL\USER
echo Summaries: data\unsynced\external_logs\zonian\CHANNEL_download_summary.json
echo.

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "ERRORS=0"
for %%C in (%TARGETS%) do (
    echo.
    echo ============================================================
    echo Channel: %%C  Users: local #thickpoo members
    echo ============================================================
    "%PY%" scripts\download_zonian_user_logs.py --channel %%C --from-archive --users-from-channel thickpoo --min-archive-messages %MIN% %EXCLUDED_FLAG% %IMPORT_FLAG% %LIMIT_FLAG%
    if errorlevel 1 (
        echo Channel %%C failed.
        set /a ERRORS+=1
    )
)

echo.
if "!ERRORS!"=="0" (
    echo Done. All channel downloads finished.
) else (
    echo Done, but !ERRORS! channel run(s) failed. Check the output above.
)
pause
exit /b !ERRORS!
