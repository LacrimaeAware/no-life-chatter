@echo off
REM Opens the fine-tuning guide and the private folder where the RunPod zip is created.
cd /d "%~dp0"

if not exist "data\unsynced\fine_tune" mkdir "data\unsynced\fine_tune"

start "" explorer "%cd%\data\unsynced\fine_tune"
start "" notepad "%cd%\docs\FINE_TUNING.md"
