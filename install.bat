@echo off
rem Double-click entry point - runs install.ps1 without ever hitting
rem PowerShell's execution-policy prompt/error. All real logic lives in
rem install.ps1; this just launches it safely.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
