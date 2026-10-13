param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,
    [string]$ConditionDir = "",
    [string]$MatrixJson = "",
    [string]$Python = "python",
    [string]$Device = "cuda:0",
    [string]$GpuIds = "0,1",
    [int]$NumSamples = 100,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($MatrixJson)) {
    $MatrixJson = Join-Path $ProjectDir "configs\experiment_matrix.json"
}
if ([string]::IsNullOrWhiteSpace($ConditionDir)) {
    $validationCondition = Join-Path $DatasetRoot "validation\contour"
    $rootCondition = Join-Path $DatasetRoot "contour"
    if (Test-Path $validationCondition) {
        $ConditionDir = $validationCondition
    } elseif (Test-Path $rootCondition) {
        $ConditionDir = $rootCondition
    } else {
        throw "Cannot find a contour directory under $DatasetRoot. Pass -ConditionDir explicitly."
    }
}
if (-not (Test-Path $DatasetRoot)) {
    throw "Dataset root does not exist: $DatasetRoot"
}
if (-not (Test-Path $ConditionDir)) {
    throw "Condition directory does not exist: $ConditionDir"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Experiments = Get-Content -Raw -Encoding UTF8 $MatrixJson | ConvertFrom-Json

foreach ($Experiment in $Experiments) {
    $lrText = ([double]$Experiment.lr).ToString("0.########", [Globalization.CultureInfo]::InvariantCulture)
    $augText = if ([double]$Experiment.augmentation_prob -eq 0) { "0" } else { "1" }
    $runName = "{0}__ep{1}_bs{2}_lr{3}_T{4}_S{5}_loss-{6}_aug{7}_dp-{8}" -f `
        $Experiment.name, $Experiment.epochs, $Experiment.batch_size, $lrText, `
        $Experiment.timesteps, $Experiment.ddim_steps, $Experiment.loss_type, $augText, $GpuIds.Replace(',', '-')
    $RunDir = Join-Path $OutputRoot $runName
    $PredictionDir = Join-Path $RunDir "prediction"

    $TrainArguments = @(
        (Join-Path $ProjectDir "train_diffusion.py"),
        "--dataset_root", $DatasetRoot,
        "--output_dir", $RunDir,
        "--device", $Device,
        "--use_data_parallel", "true",
        "--gpu_ids", $GpuIds,
        "--seed", [string]$Seed
    )
    foreach ($Property in $Experiment.PSObject.Properties) {
        if ($Property.Name -eq "name") {
            continue
        }
        $TrainArguments += "--$($Property.Name)"
        if ($Property.Value -is [bool]) {
            $TrainArguments += $(if ($Property.Value) { "true" } else { "false" })
        } elseif ($Property.Value -is [System.Array]) {
            $TrainArguments += ($Property.Value -join ",")
        } else {
            $TrainArguments += [string]$Property.Value
        }
    }

    Write-Host "Training $runName"
    & $Python @TrainArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $runName with exit code $LASTEXITCODE"
    }

    $Checkpoint = Join-Path $RunDir "best.pt"
    if (-not (Test-Path $Checkpoint)) {
        $Checkpoint = Join-Path $RunDir "last.pt"
    }
    if (-not (Test-Path $Checkpoint)) {
        throw "No checkpoint found for $runName"
    }

    $EvalArguments = @(
        (Join-Path $ProjectDir "evaluate_diffusion.py"),
        "--checkpoint", $Checkpoint,
        "--condition_dir", $ConditionDir,
        "--output_dir", $PredictionDir,
        "--num_samples", [string]$NumSamples,
        "--device", $Device,
        "--seed", [string]$Seed
    )
    Write-Host "Predicting $runName"
    & $Python @EvalArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Prediction failed for $runName with exit code $LASTEXITCODE"
    }
}
