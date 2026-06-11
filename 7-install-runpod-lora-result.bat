@echo off
setlocal
cd /d "%~dp0"

set "DEST_DIR=data\unsynced\fine_tune"
set "DEST_ZIP=%DEST_DIR%\persona_lora_result.zip"
set "DEST_EXTRACT=%DEST_DIR%\persona_lora_result"
set "DOWNLOAD_ZIP=%USERPROFILE%\Downloads\persona_lora_result.zip"

echo NoLifeChatter RunPod LoRA result installer
echo.

if not exist "%DEST_DIR%" mkdir "%DEST_DIR%"

if exist "%DOWNLOAD_ZIP%" (
  echo Found "%DOWNLOAD_ZIP%"
  copy /Y "%DOWNLOAD_ZIP%" "%DEST_ZIP%" >nul
) else if exist "%DEST_ZIP%" (
  echo Using existing "%DEST_ZIP%"
) else (
  echo ERROR: Could not find persona_lora_result.zip.
  echo.
  echo Download it from RunPod Jupyter:
  echo   /workspace/nlc_persona/persona_lora_result.zip
  echo.
  echo Put it here, then run this again:
  echo   %DEST_ZIP%
  echo.
  pause
  exit /b 1
)

echo Extracting to "%DEST_EXTRACT%"...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$dest='%DEST_EXTRACT%';" ^
  "if (Test-Path $dest) { Remove-Item -LiteralPath $dest -Recurse -Force };" ^
  "New-Item -ItemType Directory -Force -Path $dest | Out-Null;" ^
  "Expand-Archive -Force -LiteralPath '%DEST_ZIP%' -DestinationPath $dest"

if errorlevel 1 (
  echo ERROR: Extraction failed.
  pause
  exit /b 1
)

echo.
echo Installed locally:
echo   %DEST_ZIP%
echo   %DEST_EXTRACT%
echo.
echo Next: ask Codex to inspect/test the LoRA adapter, then convert or serve it.
echo You can now stop/terminate the RunPod pod if you have not already.
echo.
pause
