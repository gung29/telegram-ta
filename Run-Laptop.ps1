param(
  [string]$VpsHost = "root@8.215.192.229",
  [string]$KeyPath = "D:\Data cokagung\Skripsi\tele-new\key.pem",
  [int]$LocalPort = 8000,
  [int]$RemotePort = 8000,
  [switch]$SkipApi
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LogsDir = Join-Path $Root "logs"
$ApiLog = Join-Path $LogsDir "laptop-core-api.out"
$ApiErr = Join-Path $LogsDir "laptop-core-api.err"

New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null

function Write-Step {
  param([string]$Message)
  Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-HttpOk {
  param([string]$Url)
  try {
    $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
    return ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500)
  } catch {
    return $false
  }
}

function Stop-Api {
  if ($script:ApiProcess -and -not $script:ApiProcess.HasExited) {
    Write-Step "Stopping Core API"
    Stop-Process -Id $script:ApiProcess.Id -Force -ErrorAction SilentlyContinue
  }
}

if (!(Test-Path $KeyPath)) {
  throw "SSH key not found: $KeyPath"
}

if (!$SkipApi -and !(Test-Path $Python)) {
  throw "Python venv not found: $Python. Run: py -3 -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt"
}

if (!(Test-Path (Join-Path $Root ".env"))) {
  Write-Warning ".env not found in $Root"
}

try {
  [Console]::add_CancelKeyPress({
    param($sender, $eventArgs)
    $eventArgs.Cancel = $true
    if ($script:ApiProcess -and -not $script:ApiProcess.HasExited) {
      Stop-Process -Id $script:ApiProcess.Id -Force -ErrorAction SilentlyContinue
    }
    exit 0
  })
} catch {
  Write-Warning "Could not register Ctrl+C cleanup handler. The script will still try to clean up on normal exit."
}

try {
  Set-Location $Root

  if (!$SkipApi) {
    if (Test-HttpOk "http://127.0.0.1:$LocalPort/healthz") {
      Write-Step "Core API already reachable on 127.0.0.1:$LocalPort"
    } else {
      Write-Step "Starting Core API on 127.0.0.1:$LocalPort"
      $script:ApiProcess = Start-Process `
        -FilePath $Python `
        -ArgumentList @("-m", "uvicorn", "api.app:app", "--host", "127.0.0.1", "--port", "$LocalPort") `
        -WorkingDirectory $Root `
        -RedirectStandardOutput $ApiLog `
        -RedirectStandardError $ApiErr `
        -PassThru `
        -WindowStyle Hidden

      Write-Step "Waiting for Core API healthcheck"
      $ready = $false
      for ($i = 1; $i -le 90; $i++) {
        if ($script:ApiProcess.HasExited) {
          Write-Host "Core API exited. Last stderr:" -ForegroundColor Red
          if (Test-Path $ApiErr) { Get-Content $ApiErr -Tail 80 }
          throw "Core API failed to start."
        }
        if (Test-HttpOk "http://127.0.0.1:$LocalPort/healthz") {
          $ready = $true
          break
        }
        Start-Sleep -Seconds 2
      }
      if (!$ready) {
        throw "Core API did not become ready. Check logs: $ApiLog and $ApiErr"
      }
    }
  }

  Write-Step "Core API health"
  try {
    Invoke-RestMethod "http://127.0.0.1:$LocalPort/healthz" | ConvertTo-Json -Compress
  } catch {
    Write-Warning "Could not read Core API health locally."
  }

  Write-Step "Starting SSH reverse tunnel to $VpsHost"
  Write-Host "Remote VPS will access Core API at http://127.0.0.1:$RemotePort"
  Write-Host "Press Ctrl+C to stop."

  while ($true) {
    & ssh `
      -i $KeyPath `
      -N `
      -o ExitOnForwardFailure=yes `
      -o ServerAliveInterval=30 `
      -o ServerAliveCountMax=3 `
      -R "127.0.0.1:$RemotePort`:127.0.0.1:$LocalPort" `
      $VpsHost

    Write-Warning "SSH tunnel stopped. Reconnecting in 5 seconds..."
    Start-Sleep -Seconds 5
  }
} finally {
  Stop-Api
}
