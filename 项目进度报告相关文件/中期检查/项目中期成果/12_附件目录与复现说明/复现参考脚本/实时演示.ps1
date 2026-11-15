param(
    [ValidateSet("pinch","wrap","neutral")][string]$Task = "pinch",
    [int]$Trial = 0
)
$ErrorActionPreference = "Stop"
$Root = Get-Item -LiteralPath $PSScriptRoot
while ($null -ne $Root -and -not ((Test-Path -LiteralPath (Join-Path $Root.FullName "pyproject.toml") -PathType Leaf) -and (Test-Path -LiteralPath (Join-Path $Root.FullName "src") -PathType Container) -and (Test-Path -LiteralPath (Join-Path $Root.FullName "datasets") -PathType Container))) {
    $Root = $Root.Parent
}
if ($null -eq $Root) { throw "Project root not found (pyproject.toml, src, datasets)." }
Set-Location -LiteralPath $Root.FullName
$Python = Join-Path $Root.FullName ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
$Camera = if ($Task -eq "pinch") { "closeup" } else { "overview" }
& $Python -m src.main --sim mujoco --render --realtime --input dataset --dataset-trial $Trial --task $Task --steps 800 --camera $Camera
if ($LASTEXITCODE -ne 0) { throw "Simulation exited with code $LASTEXITCODE" }
