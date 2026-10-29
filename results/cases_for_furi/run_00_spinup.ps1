$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python .\main.py `
    --config .\results\cases_for_furi\config_00_stationary_spinup_t0_t100.txt `
    --experiment-config .\results\cases_for_furi\experimental_config_fixed_body.txt `
    --post-config .\results\cases_for_furi\post_config_snapshot_only.txt
