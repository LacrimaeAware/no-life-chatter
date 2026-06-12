@echo off
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo NoLifeChatter - unit tests
echo.
"%PY%" -m unittest discover -s tests -p "test_*.py"
echo.
pause
