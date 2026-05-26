# Start the FMLS viewer on http://localhost:8600
# Replaces the Streamlit UI. Stateless, server-rendered HTML; survives heavy data.

$ErrorActionPreference = "Stop"
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    & .\.venv\Scripts\python.exe -m uvicorn viewer.app:app --reload --port 8600
} finally {
    Pop-Location
}
