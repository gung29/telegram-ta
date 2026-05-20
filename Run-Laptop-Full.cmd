@echo off
setlocal

title HateSpeechModerator - Core API and Tunnel
cd /d "%~dp0"

echo Starting Core API and laptop tunnel...
echo Script directory: %CD%
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Run-Laptop.ps1"

echo.
echo Script stopped. Press any key to close this window.
pause >nul
