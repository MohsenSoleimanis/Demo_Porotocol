# Stop the remote server (kills tmux session on Lightning AI) and close the
# local SSH tunnel. Safe to run when nothing is running — it just no-ops.
#
# Usage:
#   .\scripts\lightning_stop.ps1

$ErrorActionPreference = "Continue"

$LightningHost = "s_01ks2jqeq9bgfgg0pz5vqnwf3c@ssh.lightning.ai"
$PidFile       = Join-Path $PSScriptRoot "..\.lightning_tunnel.pid"

# --- 1. close local tunnel ---
if (Test-Path $PidFile) {
    $tunnelPid = Get-Content $PidFile | Select-Object -First 1
    if ($tunnelPid -and (Get-Process -Id $tunnelPid -ErrorAction SilentlyContinue)) {
        Write-Host "==> closing tunnel pid=$tunnelPid"
        Stop-Process -Id $tunnelPid -Force
    } else {
        Write-Host "==> no live tunnel process for pid=$tunnelPid"
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "==> no .lightning_tunnel.pid found (tunnel was not started by this script)"
}

# --- 2. stop tmux session on remote ---
Write-Host "==> stopping tmux session fmls-server on Lightning AI"
ssh $LightningHost "tmux kill-session -t fmls-server 2>/dev/null && echo 'tmux session killed' || echo 'no tmux session running'"
