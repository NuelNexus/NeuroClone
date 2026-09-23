@echo off
rem Start NeuroClone on Windows. Double-click to chat in this window (the dashboard runs too).
rem   start.bat run      go live: Twitch/YouTube chat, voice, avatar, games (see config\local.yaml)
rem   start.bat doctor   check that everything is ready
rem   start.bat bench    measure reply speed on this PC
cd /d "%~dp0"
if not exist ".venv\Scripts\neuroclone.exe" (
    echo NeuroClone is not installed yet. Double-click install.bat first.
    pause
    exit /b 1
)
set MODE=%1
if "%MODE%"=="" set MODE=chat
if "%MODE%"=="run" (
    ".venv\Scripts\neuroclone.exe" run --console %2 %3 %4 %5 %6 %7 %8 %9
) else (
    ".venv\Scripts\neuroclone.exe" %MODE% %2 %3 %4 %5 %6 %7 %8 %9
)
pause
