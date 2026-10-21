# ==============================================================================
# VisionAudit Development Launcher & Environment Synchronizer
# Enforces Python 3.12 virtual environment isolation and dependency integrity
# ==============================================================================

[CmdletBinding()]
param (
    [switch]$SkipInstall = $false,
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$WorkspaceDir = $PSScriptRoot
Set-Location $WorkspaceDir

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  VisionAudit AI Microservice - Dev Environment Launcher  " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$VenvDir = Join-Path $WorkspaceDir "venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

# 1. Verify or create Python 3.12 virtual environment
if (-not (Test-Path $VenvPython)) {
    Write-Host "[*] Virtual environment not found. Initializing Python 3.12 venv..." -ForegroundColor Yellow
    
    # Try py -3.12 launcher first
    try {
        py -3.12 -m venv venv
    } catch {
        Write-Warning "Could not invoke 'py -3.12'. Attempting fallback to 'python -m venv venv'..."
        python -m venv venv
    }

    if (-not (Test-Path $VenvPython)) {
        Write-Error "Failed to initialize virtual environment at '$VenvDir'. Please ensure Python 3.12 is installed."
        exit 1
    }
    Write-Host "[+] Created isolated virtual environment at '$VenvDir'." -ForegroundColor Green
} else {
    $VenvVer = & $VenvPython --version
    Write-Host "[+] Found active virtual environment: $VenvVer" -ForegroundColor Green
}

# 2. Force dependency installation into the isolated virtual environment
if (-not $SkipInstall) {
    Write-Host "[*] Upgrading pip and installing required dependencies via venv pip..." -ForegroundColor Yellow
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r (Join-Path $WorkspaceDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "pip install returned non-zero exit code ($LASTEXITCODE). Check package versions."
        exit $LASTEXITCODE
    }
    Write-Host "[+] Dependencies synchronized successfully." -ForegroundColor Green
} else {
    Write-Host "[!] Skipping dependency installation (-SkipInstall specified)." -ForegroundColor Gray
}

# 3. Sanity verification: Import critical modules inside venv
Write-Host "[*] Verifying critical modules inside venv..." -ForegroundColor Yellow
$VerifyScript = @"
import sys
modules = [
    'fastapi',
    'uvicorn',
    'pydantic',
    'ultralytics',
    'cv2',
    'langchain_core',
    'langchain_google_genai',
    'websockets'
]
missing = []
for mod in modules:
    try:
        __import__(mod)
    except ImportError as e:
        missing.append((mod, str(e)))

if missing:
    print('MISSING:' + ', '.join([m[0] for m in missing]))
    for m, err in missing:
        print(f'  - {m}: {err}')
    sys.exit(1)
else:
    print('ALL_MODULES_OK')
"@

$VerifyOutput = & $VenvPython -c $VerifyScript
if ($LASTEXITCODE -ne 0) {
    Write-Host $VerifyOutput -ForegroundColor Red
    Write-Error "Module verification failed inside '$VenvPython'. Review missing packages above."
    exit 1
}
Write-Host "[+] All critical modules verified successfully inside isolated venv." -ForegroundColor Green

# 4. Launch FastAPI server directly using venv python to bypass global binary collisions
Write-Host "`n[*] Starting VisionAudit FastAPI server on http://$($HostAddress):$Port..." -ForegroundColor Cyan
Write-Host "[*] Docs available at: http://localhost:$Port/docs" -ForegroundColor DarkCyan
Write-Host "[*] Live WebSocket at: ws://localhost:$Port/ws/live-audit" -ForegroundColor DarkCyan
Write-Host "Press Ctrl+C to terminate the server.`n" -ForegroundColor Gray

& $VenvPython -m uvicorn main:app --host $HostAddress --port $Port --reload
