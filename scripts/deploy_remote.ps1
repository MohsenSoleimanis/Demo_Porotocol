# One-shot deploy of the remote heavy-parser to Lightning AI (PowerShell).
# Pushes the server code + remote requirements via scp, installs deps,
# then leaves you with the commands to start the server and open the tunnel.
#
# Edit $LightningHost if your endpoint changes.

$ErrorActionPreference = "Stop"

$LightningHost = "s_01ks2jqeq9bgfgg0pz5vqnwf3c@ssh.lightning.ai"
$RemoteDir     = "~/fmls"

Write-Host "==> creating remote dir $RemoteDir on $LightningHost"
ssh $LightningHost "mkdir -p $RemoteDir/fmls_parser/remote"

Write-Host "==> copying server code + requirements"
scp requirements-remote.txt "${LightningHost}:${RemoteDir}/requirements-remote.txt"
scp fmls_parser/__init__.py "${LightningHost}:${RemoteDir}/fmls_parser/__init__.py"
scp fmls_parser/models.py "${LightningHost}:${RemoteDir}/fmls_parser/models.py"
scp fmls_parser/remote/__init__.py "${LightningHost}:${RemoteDir}/fmls_parser/remote/__init__.py"
scp fmls_parser/remote/schemas.py "${LightningHost}:${RemoteDir}/fmls_parser/remote/schemas.py"
scp fmls_parser/remote/server.py "${LightningHost}:${RemoteDir}/fmls_parser/remote/server.py"

Write-Host "==> installing remote dependencies (this may take a few minutes the first time)"
ssh $LightningHost "cd $RemoteDir && python -m pip install --upgrade pip && pip install -r requirements-remote.txt"

Write-Host ""
Write-Host "================================================================"
Write-Host " Deploy complete."
Write-Host ""
Write-Host " 1. Start the server on Lightning AI (in one terminal):"
Write-Host "    ssh $LightningHost"
Write-Host "    cd $RemoteDir"
Write-Host "    uvicorn fmls_parser.remote.server:app --host 0.0.0.0 --port 8000"
Write-Host ""
Write-Host " 2. Open the SSH tunnel from this machine (in another terminal):"
Write-Host "    ssh -N -L 8000:localhost:8000 $LightningHost"
Write-Host ""
Write-Host " 3. Point the local pipeline at it:"
Write-Host "    `$env:FMLS_REMOTE_URL = 'http://localhost:8000'"
Write-Host "    .venv\Scripts\python.exe -m streamlit run app.py"
Write-Host ""
Write-Host " Health check (after tunnel up):"
Write-Host "   curl http://localhost:8000/health"
Write-Host "================================================================"
