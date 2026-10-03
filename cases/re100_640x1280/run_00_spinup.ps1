$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python .\main.py `
    --config .\cases\re100_640x1280\config_00_stationary_spinup_t0_t100.txt `
    --experiment-config .\cases\re100_640x1280\experimental_config_fixed_body.txt `
    --post-config .\cases\re100_640x1280\post_config_snapshot_only.txt

