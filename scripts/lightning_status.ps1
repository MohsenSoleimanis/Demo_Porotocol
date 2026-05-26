# Quick status check: is the tunnel up? is the remote server responding?
#
# Usage:
#   .\scripts\lightning_status.ps1

$ErrorActionPreference = "Continue"

$LightningHost = "s_01ks2jqeq9bgfgg0pz5vqnwf3c@ssh.lightning.ai"
$PidFile       = Join-Path $PSScriptRoot "..\.lightning_tunnel.pid"
$LocalPort     = 8000

Write-Host "==> local tunnel"
if (Test-Path $PidFile) {
    $tunnelPid = Get-Content $PidFile | Select-Object -First 1
    $p = Get-Process -Id $tunnelPid -ErrorAction SilentlyContinue
    if ($p) {
        Write-Host "    alive (pid=$tunnelPid)"
    } else {
        Write-Host "    pid file exists but process is dead (pid=$tunnelPid)"
    }
} else {
    Write-Host "    not started"
}

Write-Host ""
Write-Host "==> remote tmux session"
ssh $LightningHost "if tmux has-session -t fmls-server 2>/dev/null; then echo '    running'; tail -5 /tmp/fmls-server.log 2>/dev/null | sed 's/^/      | /'; else echo '    not running'; fi"

Write-Host ""
Write-Host "==> /health endpoint via tunnel"
try {
    $r = Invoke-RestMethod -Uri "http://localhost:$LocalPort/health" -TimeoutSec 5
    Write-Host "    status: $($r.status)"
    Write-Host "    available_parsers: $($r.available_parsers -join ', ')"
    if ($r.notes) { foreach ($n in $r.notes) { Write-Host "    note: $n" } }
} catch {
    Write-Host "    unreachable: $_"
}
