param(
    [int]$Repeats = 3,
    [switch]$RunMujoco,
    [switch]$Render
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    Write-Host '=== Scheme E E1-E3: strict diagnostics and fixed-dimension paired comparison ===' -ForegroundColor Cyan
    $e3 = @(
        'experiments/实验08_任务效应谱方案对比/benchmark_scheme_e_research.py',
        '--steps', '24', '--repeats', "$Repeats",
        '--modes', 'E_dynamic,E_fixed4,E_fixed5,E_fixed6',
        '--out', 'outputs/e_fair_dimension_controls_user'
    )
    & python @e3
    if ($LASTEXITCODE -ne 0) { throw "E1-E3 failed with exit code $LASTEXITCODE" }

    Write-Host '=== Scheme E E4: all captured sessions, resumable ===' -ForegroundColor Cyan
    $e4 = @(
        'experiments/benchmark_scheme_e_sessions.py',
        '--modes', 'E_dynamic,E_fixed6',
        '--out', 'outputs/scheme_e_cross_recording',
        '--resume'
    )
    & python @e4
    if ($LASTEXITCODE -ne 0) { throw "E4 failed with exit code $LASTEXITCODE" }

    $sample = '数据手套数据/records/chm6_calibrations/20260904_113022_G15RB010901_right/captured_samples.csv'
    if ($RunMujoco) {
        Write-Host '=== Scheme E E5: MuJoCo PINCH/WRAP validation ===' -ForegroundColor Cyan
        $e5 = @(
            'experiments/benchmark_scheme_e_mujoco.py',
            '--samples', $sample,
            '--tasks', 'pinch,wrap',
            '--modes', 'E_dynamic,E_fixed6',
            '--out', 'outputs/scheme_e_mujoco'
        )
        if ($Render) { $e5 += @('--render', '--realtime') }
    } else {
        Write-Host '=== Scheme E E5: MuJoCo command dry-run (use -RunMujoco to execute) ===' -ForegroundColor Yellow
        $e5 = @(
            'experiments/benchmark_scheme_e_mujoco.py',
            '--samples', $sample,
            '--tasks', 'pinch,wrap',
            '--modes', 'E_dynamic,E_fixed6',
            '--out', 'outputs/scheme_e_mujoco',
            '--dry-run'
        )
    }
    & python @e5
    if ($LASTEXITCODE -ne 0) { throw "E5 step failed with exit code $LASTEXITCODE" }

    Write-Host ''
    Write-Host 'Scheme E E1-E5 workflow completed.' -ForegroundColor Green
    Write-Host 'E1-E3: outputs/e_fair_dimension_controls_user/report.md'
    Write-Host 'E4:    outputs/scheme_e_cross_recording/report.md'
    if ($RunMujoco) { Write-Host 'E5:    outputs/scheme_e_mujoco/report.md' }
    else { Write-Host 'E5 was dry-run only. Run with -RunMujoco on a machine with MuJoCo installed.' }
}
finally {
    Pop-Location
}
