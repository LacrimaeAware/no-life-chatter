@echo off
REM ============================================================
REM  NoLifeChatter - export persona fine-tuning pilot dataset
REM
REM  Double-click this on your Windows machine.
REM  It exports a curated ThickPoo persona pilot, merges known alt accounts,
REM  and builds a zip for RunPod.
REM
REM  Private output:
REM    data\unsynced\fine_tune\persona_sft_runpod.zip
REM ============================================================
cd /d "%~dp0"
setlocal

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv not found. Run 1-setup.bat first.
  pause
  exit /b 1
)

if not exist "data\unsynced\fine_tune" mkdir "data\unsynced\fine_tune"

echo.
echo [1/3] Exporting curated ThickPoo training examples...
echo       Criteria: ThickPoo only, max 5,000 examples per author.
echo.
".venv\Scripts\python.exe" scripts\export_persona_sft.py ^
  --channels thickpoo ^
  --authors earnestsinceresugmamale,gero_30,apu_wrc,forsenstares,99froxy,ebbel,satanisteen,normanbiz,bluepigman5000,bander423,crais1n,danklipse,huni4president,poggerooskii,theobr0mine ^
  --user-aliases fernardo=earnestsinceresugmamale,q0z3=earnestsinceresugmamale,danklipseofthemind=danklipse ^
  --exclude-users supibot,nightbot,potatbotat,bluepagmanbot,weirdfarts1ave ^
  --max-examples-per-author 5000
if errorlevel 1 (
  echo.
  echo ERROR: Export failed. Scroll up to see why.
  pause
  exit /b 1
)

echo.
echo [2/3] Copying RunPod helper scripts into the private bundle folder...
copy /Y "scripts\train_persona_lora_unsloth.py" "data\unsynced\fine_tune\train_persona_lora_unsloth.py" >nul
copy /Y "scripts\runpod_train_persona_lora.sh" "data\unsynced\fine_tune\runpod_train_persona_lora.sh" >nul
copy /Y "scripts\RUN_ME_ON_RUNPOD.sh" "data\unsynced\fine_tune\RUN_ME_ON_RUNPOD.sh" >nul
copy /Y "docs\RUNPOD_FINE_TUNE_README.txt" "data\unsynced\fine_tune\RUNPOD_FINE_TUNE_README.txt" >nul
if errorlevel 1 (
  echo.
  echo ERROR: Could not copy RunPod helper scripts.
  pause
  exit /b 1
)

echo.
echo [3/3] Creating RunPod zip...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Compress-Archive -Force -Path 'data\unsynced\fine_tune\persona_train.jsonl','data\unsynced\fine_tune\persona_val.jsonl','data\unsynced\fine_tune\train_persona_lora_unsloth.py','data\unsynced\fine_tune\runpod_train_persona_lora.sh','data\unsynced\fine_tune\RUN_ME_ON_RUNPOD.sh','data\unsynced\fine_tune\RUNPOD_FINE_TUNE_README.txt' -DestinationPath 'data\unsynced\fine_tune\persona_sft_runpod.zip'"
if errorlevel 1 (
  echo.
  echo ERROR: Zip creation failed.
  pause
  exit /b 1
)

echo.
echo ============================================================
echo  Done.
echo.
echo  Upload this file to RunPod Jupyter:
echo  data\unsynced\fine_tune\persona_sft_runpod.zip
echo.
echo  In RunPod terminal after upload:
echo  cd /workspace ^&^& rm -rf nlc_persona ^&^& python -m zipfile -e persona_sft_runpod.zip nlc_persona ^&^& bash nlc_persona/RUN_ME_ON_RUNPOD.sh
echo ============================================================
pause
