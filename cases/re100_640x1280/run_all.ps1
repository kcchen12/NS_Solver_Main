$ErrorActionPreference = "Stop"

& "$PSScriptRoot\run_00_spinup.ps1"
& "$PSScriptRoot\run_01_stationary.ps1"
& "$PSScriptRoot\run_02_y_oscillation.ps1"
& "$PSScriptRoot\run_03_rotation.ps1"

