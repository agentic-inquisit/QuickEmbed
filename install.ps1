# QuickEmbed - Windows install script
# Creates a venv, installs dependencies, downloads the model, then verifies
# the install. Run once per machine. Run .\init.ps1 afterward to launch the
# app - this script only installs, it does not start anything.

param(
    [string]$ModelDir = "models/embeddinggemma-2"
)

$ErrorActionPreference = "Stop"
# PowerShell 7.3+ promotes ANY stderr output from a native command into a
# terminating NativeCommandError when $ErrorActionPreference is "Stop" -
# even just pip's harmless warnings. Disable that so our own $LASTEXITCODE
# checks (below) are what decide success/failure, not stderr content.
$PSNativeCommandUseErrorActionPreference = $false

Write-Host "==> Creating virtual environment (.venv)"
python -m venv .venv
if ($LASTEXITCODE -ne 0) {
    throw "python -m venv failed - is Python installed and on PATH?"
}

Write-Host "==> Activating virtual environment"
& .\.venv\Scripts\Activate.ps1

Write-Host "==> Checking dependencies (skipping anything already installed)"
# One pip list call instead of one pip show call per package - avoids
# repeated native-command invocations inside the loop entirely (see the
# NativeCommandError note above) and is faster.
$installed = @{}
pip list --disable-pip-version-check --format=freeze 2>$null | ForEach-Object {
    $name = ($_ -split '==', 2)[0]
    if ($name) { $installed[$name.ToLower().Replace('_', '-')] = $true }
}

$toInstall = @()
Get-Content requirements.txt | ForEach-Object {
    $line = ($_ -split '#')[0].Trim()
    if ($line -eq "") { return }
    $pkgName = ($line -split '[<>=!~\[]')[0].Trim()
    if ($pkgName -eq "") { return }
    $normalized = $pkgName.ToLower().Replace('_', '-')
    if ($installed.ContainsKey($normalized)) {
        Write-Host "  [already installed] $pkgName"
    } else {
        Write-Host "  [missing]           $pkgName"
        $toInstall += $line
    }
}

if ($toInstall.Count -gt 0) {
    Write-Host "==> Installing $($toInstall.Count) missing package(s)"
    pip install @toInstall
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed - see output above."
    }
} else {
    Write-Host "==> Nothing to install - all dependencies already satisfied"
}

Write-Host "==> Verifying torch/transformers actually import"
# transformers degrades gracefully when torch fails to import (common
# Windows cause: missing Microsoft Visual C++ Redistributable, needed by
# torch's compiled backend even for CPU-only use) - that failure otherwise
# stays silent until some unrelated line deep inside transformers throws a
# confusing NameError. Surface the real error here instead, before the
# 1.5GB model download.
python -c "import torch, transformers; assert transformers.utils.is_torch_available(), 'transformers cannot see torch'; print(f'torch {torch.__version__} OK')"
if ($LASTEXITCODE -ne 0) {
    throw "torch/transformers import failed - see the real error above. On Windows this is almost always a missing Microsoft Visual C++ Redistributable (torch needs it even for CPU-only use): install it from https://aka.ms/vs/17/release/vc_redist.x64.exe, then re-run this script."
}

if (Test-Path "$ModelDir\model.safetensors") {
    Write-Host "==> Model already present at $ModelDir - skipping download"
} else {
    Write-Host "==> Downloading google/embeddinggemma-2 (~1.5GB, first run only)"
    python -c "from huggingface_hub import snapshot_download; snapshot_download('google/embeddinggemma-2', local_dir='$ModelDir')"
    if ($LASTEXITCODE -ne 0) {
        throw "Model download failed - check internet access and try again."
    }
}

Write-Host "==> Verifying config"
if (-not (Test-Path "$ModelDir\model.safetensors")) {
    throw "Model weights not found at $ModelDir\model.safetensors - download step above did not complete."
}
$env:PYTHONPATH = "src"
python -m pytest tests/test_corpus.py -v
if ($LASTEXITCODE -ne 0) {
    throw "tests/test_corpus.py failed - fix before running the app."
}

Write-Host ""
Write-Host "Install verified. Run .\init.ps1 to start QuickEmbed."
