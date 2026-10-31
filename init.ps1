# QuickEmbed - Windows app init/launch script
# Activates the venv and starts one of the two entry points. Run
# .\install.ps1 first (once per machine) - this script only launches, it
# does not install anything.
#
# Usage:
#   .\init.ps1                 # terminal dashboard (default)
#   .\init.ps1 -Mode dashboard # same, explicit
#   .\init.ps1 -Mode mcp       # MCP server (stdio)

param(
    [ValidateSet("dashboard", "mcp")]
    [string]$Mode = "dashboard"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    throw "No .venv found - run .\install.ps1 first."
}

& .\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"

if ($Mode -eq "mcp") {
    Write-Host "==> Starting MCP server (stdio)"
    python -m embedding_service.mcp_server
} else {
    Write-Host "==> Starting terminal dashboard"
    python -m embedding_service.dashboard
}
