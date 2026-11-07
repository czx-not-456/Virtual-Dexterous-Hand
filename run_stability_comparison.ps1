# 兼容入口：请优先使用 实验10_StageC稳定性对比/run_stability_comparison.ps1
$script = Join-Path $PSScriptRoot 'experiments/实验10_StageC稳定性对比/run_stability_comparison.ps1'
& $script @args
if ($LASTEXITCODE) { exit $LASTEXITCODE }
