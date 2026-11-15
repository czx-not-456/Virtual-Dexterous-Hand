param(
    [ValidateSet("pinch","wrap","neutral")][string]$Task = "pinch",
    [int]$Trial = 0
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$Python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
$Camera = if ($Task -eq "pinch") { "closeup" } else { "overview" }
& $Python -m src.main --sim mujoco --render --realtime --input dataset --dataset-trial $Trial --task $Task --steps 800 --camera $Camera
if ($LASTEXITCODE -ne 0) { throw "Simulation exited with code $LASTEXITCODE" }
