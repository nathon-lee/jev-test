param(
    [ValidateSet("toxigen", "hatecheck", "aegis2", "all")]
    [string]$Dataset = "all",
    [ValidateSet("v1", "v2")]
    [string]$PromptVersion = "v1",
    [int]$Limit,
    [switch]$Evaluate,
    [switch]$DryRun
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root

$datasets = if ($Dataset -eq "all") { @("toxigen", "hatecheck", "aegis2") } else { @($Dataset) }

foreach ($item in $datasets) {
    $cmd = @("python", "scripts/run_benchmark.py", "--dataset", $item, "--prompt-version", $PromptVersion)
    if ($Limit) { $cmd += @("--limit", [string]$Limit) }

    Write-Host "-> " + ($cmd -join " ")
    if (-not $DryRun) {
        & python @cmd
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        $src = Join-Path $root "outputs/$($item)_$PromptVersion`_predictions.csv"
        $dst = Join-Path $root "outputs/$($item)_predictions.csv"
        if (Test-Path $src) {
            Copy-Item $src $dst -Force
            Write-Host "Copied compatibility file: $src -> $dst"
        }

        if ($Evaluate) {
            $evalCmd = @("python", "scripts/evaluate.py", "--dataset", $item)
            Write-Host "-> " + ($evalCmd -join " ")
            & python @evalCmd
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
}

if ($DryRun) {
    Write-Host "Dry run complete; no commands executed."
} else {
    Write-Host "Benchmark step complete. Outputs are under outputs/."
}
