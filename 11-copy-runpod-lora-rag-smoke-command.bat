@echo off
setlocal
cd /d "%~dp0"

if not exist "scripts\runpod_lora_rag_smoke_command.txt" (
  echo ERROR: scripts\runpod_lora_rag_smoke_command.txt was not found.
  pause
  exit /b 1
)

type "scripts\runpod_lora_rag_smoke_command.txt" | clip

echo Copied the LoRA+RAG RunPod smoke-test command to your clipboard.
echo.
echo Before pasting it into RunPod, upload this local file:
echo   data\unsynced\fine_tune\persona_lora_rag_smoke_cases.json
echo.
echo To this RunPod folder:
echo   /workspace/nlc_persona/
echo.
echo The RunPod command will write:
echo   /workspace/nlc_persona/persona_lora_rag_smoke_test.txt
echo.
pause
