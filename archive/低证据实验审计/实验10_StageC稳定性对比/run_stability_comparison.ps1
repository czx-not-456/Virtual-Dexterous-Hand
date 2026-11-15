# Run from ANY PowerShell directory; always execute the unpacked project code.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Set-Location $projectRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (!(Test-Path $python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (!$cmd) { throw '未找到Python环境：请先建立 .venv 或将 Python 加入 PATH。' }
    $python = $cmd.Source
}
$glove = '数据手套数据\CHM6_pinch_hold_diagnostic_only.xlsx'
if (!(Test-Path $glove)) { throw "未找到诊断输入：$glove；请先从旧项目复制360帧Excel。" }
if (!(Test-Path 'configs\tqhg_online_v1.json')) { throw '缺少 configs\tqhg_online_v1.json' }
$outDir = Join-Path $PSScriptRoot 'results\generated\stability'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$methods = @('ridge6','pca4','fixed-blend','single-gate','tqhg')
foreach ($method in $methods) {
    Write-Host "===== Stability v2: $method =====" -ForegroundColor Cyan
    & $python -m src.main --input glove --glove-samples $glove `
        --sim mujoco --task pinch --steps 360 --output-dir $outDir `
        --run-name "stable_v2_${method}_hold_pinch" `
        --mapping-method $method --tqhg-calibration 'configs/tqhg_online_v1.json' `
        --coupled-servo-ablation hold --servo-trace
    if ($LASTEXITCODE -ne 0) { throw "失败：$method，退出码 $LASTEXITCODE" }
}
$results = foreach ($method in $methods) {
    $path = Join-Path $outDir "stable_v2_${method}_hold_pinch_summary.json"
    $s = Get-Content $path -Raw -Encoding UTF8 | ConvertFrom-Json
    [PSCustomObject]@{
        Method = $method
        ContactPct = [math]::Round(100 * [double]$s.contact_frame_ratio, 2)
        HoldStablePct = [math]::Round(100 * [double]$s.stable_frame_ratio, 2)
        HoldStableAtEnd = $s.stable_success_proxy
        HoldEverReached = $s.ever_hold_stable_proxy
        LegacyStablePct = [math]::Round(100 * [double]$s.legacy_stable_frame_ratio, 2)
        FreeContactFrames = $s.free_object_contact_frames
        P95Ms = [math]::Round([double]$s.p95_pipeline_latency_ms, 2)
    }
}
$results | Format-Table -AutoSize
$results | Export-Csv (Join-Path $outDir 'stable_v2_method_comparison.csv') -Encoding UTF8 -NoTypeInformation
Write-Host "Done. Summary saved to $outDir\stable_v2_method_comparison.csv"
