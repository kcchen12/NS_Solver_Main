$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

$Restart = ".\cases\cases_for_furi\output\00_stationary_spinup_t0_t100\snap_100.9001.npz"
if (-not (Test-Path $Restart)) {
    throw "Missing spin-up restart file: $Restart. Run run_00_spinup.ps1 first."
}

$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python .\main.py `
    --config .\cases\cases_for_furi\config_01_stationary_from_t100_t225.txt `
    --experiment-config .\cases\cases_for_furi\experimental_config_fixed_body.txt `
    --post-config .\cases\cases_for_furi\post_config_snapshot_only.txt
