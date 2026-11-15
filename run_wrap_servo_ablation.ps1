param(
    [string]$Trials = "0",
    [int]$Steps = 800,
    [switch]$NoAutoResetObject,
    [switch]$NoInactiveObjectReset
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "请先在项目根目录激活/创建 .venv：py -3.12 -m venv .venv；然后安装 requirements.txt"
}
$argsList = @("experiments\benchmark_wrap_servo_policy.py", "--trials", $Trials, "--steps", "$Steps")
if ($NoAutoResetObject) { $argsList += "--no-auto-reset-object" }
if ($NoInactiveObjectReset) { $argsList += "--no-inactive-object-reset" }
& $python @argsList
if ($LASTEXITCODE -ne 0) { throw "MuJoCo WRAP 消融实验运行失败，退出码 $LASTEXITCODE" }
