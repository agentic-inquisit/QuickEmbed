@echo off
rem Double-click entry point - runs init.ps1 without ever hitting
rem PowerShell's execution-policy prompt/error. All real logic lives in
rem init.ps1; this just launches it safely.
rem Usage: init.bat            (dashboard)
rem        init.bat -Mode mcp  (MCP server)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0init.ps1" %*
