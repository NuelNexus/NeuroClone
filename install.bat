@echo off
rem NeuroClone one-click installer for Windows: double-click this file.
rem Options: install.bat -Prefer quality -Mic -Creator "Your Name"
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\install.ps1" %*
echo.
pause
