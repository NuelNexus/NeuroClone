# NeuroClone installer for Windows 10/11.
# Installs Python packages into .venv, installs Ollama if needed, tunes it, then runs `neuroclone setup`,
# which detects your GPU, picks the models, writes config\local.yaml and downloads everything once.
# After that NeuroClone runs fully offline and for free.
#
# Usually started by double-clicking install.bat. Options (install.bat passes them through):
#   -Prefer speed|balanced|quality   pick snappier or smarter models (default: balanced)
#   -Mic                             talk to her with your microphone (downloads Whisper)
#   -Creator "Your Name"             what the characters call you
#   -NoOllamaTuning                  don't set Ollama's memory-saving environment variables
param(
    [ValidateSet("speed", "balanced", "quality")] [string]$Prefer = "balanced",
    [switch]$Mic,
    [string]$Creator = "",
    [switch]$NoOllamaTuning
)
$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

function Install-WithWinget($id, $manual) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "winget is not available on this PC. Install it by hand from $manual, then run install.bat again."
        exit 1
    }
    try { winget install -e --id $id --accept-source-agreements --accept-package-agreements }
    catch { Write-Host "winget could not install $id ($_). Install it from $manual, then run install.bat again."; exit 1 }
}

function Find-Python {
    # The py launcher knows every installed Python; prefer versions with wheels for every dependency.
    foreach ($version in @("3.12", "3.11", "3.13", "3.10")) {
        try {
            & py "-$version" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { return @("py", "-$version") }
        } catch { }
    }
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python -and $python.Source -notlike "*WindowsApps*") { return @("python") }
    return $null
}

Step "Python"
$py = Find-Python
if (-not $py) {
    Write-Host "Python 3.10-3.13 was not found. Installing Python 3.12 with winget..."
    Install-WithWinget "Python.Python.3.12" "https://www.python.org/downloads/ (tick 'Add python.exe to PATH')"
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
    $py = Find-Python
    if (-not $py) { Write-Host "Python installed. Close this window and run install.bat again."; exit 1 }
}
$pyExe = $py[0]
$pyArgs = @()
if ($py.Count -gt 1) { $pyArgs = $py[1..($py.Count - 1)] }
Write-Host "using: $($py -join ' ')"
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $pyExe @pyArgs -m venv .venv }
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"

Step "NeuroClone packages (local voice, microphone, screen vision)"
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -e ".[local]"
if ($LASTEXITCODE -ne 0) { Write-Host "pip failed; see the messages above."; exit 1 }

Step "Ollama (runs the AI brain on your GPU)"
$ollamaApp = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue) -and -not (Test-Path $ollamaApp)) {
    Write-Host "Installing Ollama with winget..."
    Install-WithWinget "Ollama.Ollama" "https://ollama.com/download"
}
if (-not $NoOllamaTuning) {
    # Half-size context cache (q8_0) and room for the brain + the memory model at the same time.
    [Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q8_0", "User")
    [Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")
    [Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "2", "User")
    Write-Host "set OLLAMA_KV_CACHE_TYPE=q8_0, OLLAMA_FLASH_ATTENTION=1, OLLAMA_MAX_LOADED_MODELS=2 (restarting Ollama to apply)"
    Get-Process -Name "ollama app", "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}
if (Test-Path $ollamaApp) { Start-Process $ollamaApp -WindowStyle Hidden }
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 2 | Out-Null; $ready = $true; break }
    catch { Start-Sleep -Seconds 1 }
}
if (-not $ready) { Write-Host "Ollama did not start. Open the Ollama app from the Start menu, then run install.bat again."; exit 1 }

Step "Detect this PC, pick models, download them (one time)"
$setupArgs = @("setup", "--yes", "--prefer", $Prefer)
if ($Mic) { $setupArgs += "--mic" }
if ($Creator) { $setupArgs += @("--creator", $Creator) }
& (Join-Path $Root ".venv\Scripts\neuroclone.exe") @setupArgs
$code = $LASTEXITCODE

Step "Done"
Write-Host "Start chatting:  double-click start.bat   (or: start.bat run   to go live with chat sources)"
Write-Host "Control room:    http://127.0.0.1:8080 while it runs"
exit $code
