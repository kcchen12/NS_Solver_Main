param(
    [int]$Jobs = 4,
    [string]$Python = "",
    [switch]$Check
)
$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path "$PSScriptRoot\..\..").Path
if (-not $Python) {
    $Python = Join-Path $Repo ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) { $Python = "python" }
}
$RunnerArgs = @("$PSScriptRoot\run_all.py", "--jobs", "$Jobs")
if ($Check) { $RunnerArgs += "--check" }
& $Python @RunnerArgs
if ($LASTEXITCODE -ne 0) { throw "Spacing sweep failed. See the per-case logs under runs/." }
