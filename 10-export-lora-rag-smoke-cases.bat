@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv\Scripts\python.exe was not found.
  pause
  exit /b 1
)

echo Exporting RAG-backed smoke-test cases for the trained LoRA.
echo.
echo Output:
echo   data\unsynced\fine_tune\persona_lora_rag_smoke_cases.json
echo.

".venv\Scripts\python.exe" scripts\export_lora_rag_smoke_cases.py %*

echo.
echo Upload this file to RunPod:
echo   data\unsynced\fine_tune\persona_lora_rag_smoke_cases.json
echo.
echo Destination on RunPod:
echo   /workspace/nlc_persona/persona_lora_rag_smoke_cases.json
echo.
pause
