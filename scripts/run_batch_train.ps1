param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [string]$MatrixJson = "",
    [string]$Python = "python",
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($MatrixJson)) {
    $MatrixJson = Join-Path $ProjectDir "configs\experiment_matrix.json"
}
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Experiments = Get-Content -Raw -Encoding UTF8 $MatrixJson | ConvertFrom-Json

foreach ($Experiment in $Experiments) {
    $RunDir = Join-Path $OutputRoot $Experiment.name
    $Arguments = @(
        (Join-Path $ProjectDir "train_diffusion.py"),
        "--dataset_root", $DatasetRoot,
        "--output_dir", $RunDir,
        "--device", $Device
    )
    foreach ($Property in $Experiment.PSObject.Properties) {
        if ($Property.Name -eq "name") {
            continue
        }
        $Arguments += "--$($Property.Name)"
        if ($Property.Value -is [bool]) {
            $Arguments += $(if ($Property.Value) { "true" } else { "false" })
        } elseif ($Property.Value -is [System.Array]) {
            $Arguments += ($Property.Value -join ",")
        } else {
            $Arguments += [string]$Property.Value
        }
    }
    Write-Host "Running $($Experiment.name)"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment $($Experiment.name) failed with exit code $LASTEXITCODE"
    }
}
