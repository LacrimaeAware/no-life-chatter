@echo off
setlocal
cd /d "%~dp0"

if not exist "scripts\runpod_smoke_test_command.txt" (
  echo ERROR: scripts\runpod_smoke_test_command.txt is missing.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$cmd = Get-Content -Raw -LiteralPath 'scripts\runpod_smoke_test_command.txt';" ^
  "Set-Clipboard -Value $cmd;" ^
  "Write-Host 'Copied RunPod smoke-test command to clipboard.';" ^
  "Write-Host 'Paste it into the RunPod terminal while the pod/volume is available.'"

echo.
echo Copied. Paste into the RunPod terminal.
echo.
pause
