@echo off
REM Live view of the background bot's output. Close this window any time —
REM it does NOT stop the bot, it's just watching the log.
cd /d "%~dp0"
title NoLifeChatter - live log (closing this does not stop the bot)
if not exist "data\bot.log" (
  echo No log yet. Has the bot been started? ^(run-background.vbs^)
  echo.
  pause
  exit /b
)
powershell -NoProfile -Command "Get-Content -LiteralPath 'data\bot.log' -Tail 50 -Wait"
