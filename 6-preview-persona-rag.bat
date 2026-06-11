@echo off
REM ============================================================
REM  NoLifeChatter - preview persona RAG retrieval
REM
REM  Double-click this to see which archived lines would be fed
REM  into a persona prompt. This does not call LM Studio and does
REM  not post anything to Twitch.
REM ============================================================
cd /d "%~dp0"
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv not found. Run 1-setup.bat first.
  pause
  exit /b 1
)

echo.
set /p PERSONA_USER=Persona/user to inspect:
if "%PERSONA_USER%"=="" (
  echo ERROR: no user entered.
  pause
  exit /b 1
)

set /p PERSONA_QUERY=Topic/message to retrieve against:
set /p PERSONA_CHANNEL=Optional recent-context channel, or press Enter:

echo.
echo ============================================================
echo  Persona RAG preview
echo ============================================================
echo.

if "%PERSONA_CHANNEL%"=="" (
  ".venv\Scripts\python.exe" scripts\persona_rag_preview.py "%PERSONA_USER%" "%PERSONA_QUERY%"
) else (
  ".venv\Scripts\python.exe" scripts\persona_rag_preview.py "%PERSONA_USER%" "%PERSONA_QUERY%" --channel "%PERSONA_CHANNEL%"
)

echo.
pause
