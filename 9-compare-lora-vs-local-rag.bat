@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv\Scripts\python.exe was not found.
  pause
  exit /b 1
)

echo Comparing RunPod LoRA smoke output against local LM Studio + RAG.
echo.
echo Optional input:
echo   data\unsynced\fine_tune\persona_lora_smoke_test.txt
echo.
echo Output:
echo   data\unsynced\fine_tune\persona_lora_vs_local_rag.md
echo.
echo Make sure LM Studio's local server is running before continuing.
echo.
pause

".venv\Scripts\python.exe" scripts\compare_lora_smoke_with_local_rag.py %*

echo.
echo Done.
pause
