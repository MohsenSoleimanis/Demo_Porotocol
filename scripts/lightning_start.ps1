# Start the remote heavy-parser server on Lightning AI and open a local tunnel.
#
# What this does:
#   1. SSHes into the Lightning AI Studio and starts the FastAPI server in a
#      detached tmux session named 'fmls-server'. If it's already running,
#      it skips that step (no double-start).
#   2. Opens a local SSH tunnel (background process) forwarding localhost:8000
#      on this machine to the server on Lightning. The tunnel PID is written
#      to .lightning_tunnel.pid so lightning_stop.ps1 can clean it up.
#   3. Pings http://localhost:8000/health and prints the result.
#
# Usage:
#   .\scripts\lightning_start.ps1
#
# Tear it down with:
#   .\scripts\lightning_stop.ps1

$ErrorActionPreference = "Stop"

$LightningHost = "s_01ks2jqeq9bgfgg0pz5vqnwf3c@ssh.lightning.ai"
$RemoteDir     = "/teamspace/studios/this_studio/fmls-remote"
$LocalPort     = 8000
$RemotePort    = 8000
$PidFile       = Join-Path $PSScriptRoot "..\.lightning_tunnel.pid"

# --- 1. start server in tmux on remote if not already running ---
Write-Host "==> ensuring remote server is up"
$startCmd = @"
if tmux has-session -t fmls-server 2>/dev/null; then
  echo 'tmux session fmls-server already running, leaving it alone'
else
  tmux new-session -d -s fmls-server "cd $RemoteDir && source .venv/bin/activate && uvicorn fmls_parser.remote.server:app --host 0.0.0.0 --port $RemotePort 2>&1 | tee /tmp/fmls-server.log"
  echo 'started tmux session fmls-server'
fi
"@
ssh $LightningHost $startCmd

Start-Sleep -Seconds 2

# --- 2. open SSH tunnel as a detached background process ---
# Kill any prior tunnel we started.
if (Test-Path $PidFile) {
    $oldPid = Get-Content $PidFile | Select-Object -First 1
    if ($oldPid -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
        Write-Host "==> killing previous tunnel pid=$oldPid"
        Stop-Process -Id $oldPid -Force
    }
    Remove-Item $PidFile -Force
}

Write-Host "==> opening tunnel localhost:$LocalPort -> $LightningHost`:$RemotePort"
$proc = Start-Process -FilePath ssh `
    -ArgumentList @("-N", "-L", "${LocalPort}:localhost:${RemotePort}", $LightningHost) `
    -WindowStyle Hidden -PassThru
$proc.Id | Out-File -FilePath $PidFile -Encoding ascii
Write-Host "    tunnel pid=$($proc.Id) (written to $PidFile)"

Start-Sleep -Seconds 3

# --- 3. health check ---
Write-Host "==> probing /health"
try {
    $r = Invoke-RestMethod -Uri "http://localhost:$LocalPort/health" -TimeoutSec 10
    Write-Host "    status: $($r.status)"
    Write-Host "    available_parsers: $($r.available_parsers -join ', ')"
    if ($r.notes) { Write-Host "    notes: $($r.notes -join ' | ')" }
} catch {
    Write-Host "    /health unreachable yet: $_"
    Write-Host "    note: first start can take ~30s while uvicorn boots and lazy-loads modules"
}

Write-Host ""
Write-Host "Done. Set this in the shell where you run Streamlit:"
Write-Host "  `$env:FMLS_REMOTE_URL = 'http://localhost:$LocalPort'"
Write-Host ""
Write-Host "Then:"
Write-Host "  .\.venv\Scripts\python.exe -m streamlit run app.py"
