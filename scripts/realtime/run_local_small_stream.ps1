param(
    [string]$ComPort = "COM1",
    [string]$Checkpoint = "pretrained_models\emg2pose\egoemg-incre-small-8ch-runtime.pt",
    [int]$StrideSamples = 200,
    [double]$OutputDelayS = 0.5,
    [double]$InputScale = 0.001,
    [string]$Device = "cuda",
    [string]$EnvName = "diffusers_env",
    [string]$SaveJsonl = "",
    [switch]$VisualizeMesh,
    [switch]$OnlineFullAdapt,
    [int]$TeacherCamera = 1,
    [string]$TeacherHand = "right",
    [string]$TeacherDevice = "cuda",
    [string]$TeacherMapperCheckpoint = "pretrained_models\mano_to_umetrack_mapper.pt",
    [double]$AdaptLr = 0.0001,
    [int]$AdaptBatchSize = 4,
    [int]$AdaptMinSamples = 8,
    [double]$AdaptMatchToleranceS = 0.12,
    [double]$AdaptUpdateIntervalS = 1.0,
    [int]$AdaptStepsPerUpdate = 1,
    [double]$AdaptGradClip = 1.0,
    [double]$AdaptKeepWeight = 0.05,
    [string]$AdaptSaveCheckpoint = "",
    [double]$AdaptLogIntervalS = 2.0,
    [string]$OnlineLogPath = "",
    [switch]$ShowTeacherCamera,
    [switch]$VisualizeTeacherMesh
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot
if ($OnlineFullAdapt -and $OnlineLogPath -eq "") {
    New-Item -ItemType Directory -Force logs | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OnlineLogPath = "logs\online_adapt_$stamp.log"
}

$argsList = @(
    "run", "--no-capture-output", "-n", $EnvName,
    "python", "-u", "scripts\realtime\local_small_stream.py",
    "--checkpoint", $Checkpoint,
    "--device", $Device,
    "--stride-samples", "$StrideSamples",
    "--output-delay-s", "$OutputDelayS",
    "--input-scale", "$InputScale",
    "--com-port", $ComPort
)

if ($SaveJsonl -ne "") {
    $argsList += @("--save-jsonl", $SaveJsonl)
}
if ($VisualizeMesh) {
    $argsList += @("--visualize-mesh")
}
if ($OnlineFullAdapt) {
    $argsList += @(
        "--online-full-adapt",
        "--teacher-camera", "$TeacherCamera",
        "--teacher-hand", $TeacherHand,
        "--teacher-device", $TeacherDevice,
        "--teacher-mapper-checkpoint", $TeacherMapperCheckpoint,
        "--adapt-lr", "$AdaptLr",
        "--adapt-batch-size", "$AdaptBatchSize",
        "--adapt-min-samples", "$AdaptMinSamples",
        "--adapt-match-tolerance-s", "$AdaptMatchToleranceS",
        "--adapt-update-interval-s", "$AdaptUpdateIntervalS",
        "--adapt-steps-per-update", "$AdaptStepsPerUpdate",
        "--adapt-grad-clip", "$AdaptGradClip",
        "--adapt-keep-weight", "$AdaptKeepWeight",
        "--adapt-log-interval-s", "$AdaptLogIntervalS"
    )
    if ($OnlineLogPath -ne "") {
        $argsList += @("--online-log-path", $OnlineLogPath)
    }
    if ($AdaptSaveCheckpoint -ne "") {
        $argsList += @("--adapt-save-checkpoint", $AdaptSaveCheckpoint)
    }
    if ($ShowTeacherCamera) {
        $argsList += @("--show-teacher-camera")
    }
    if ($VisualizeTeacherMesh) {
        $argsList += @("--visualize-teacher-mesh")
    }
}

Write-Host "Starting local small EMGFormer stream..."
Write-Host "  env:        $EnvName"
Write-Host "  checkpoint: $Checkpoint"
Write-Host "  com port:   $ComPort"
Write-Host "  stride:     $StrideSamples samples"
Write-Host "  delay:      $OutputDelayS s"
Write-Host "  inputScale: $InputScale"
Write-Host "  mesh viz:   $VisualizeMesh"
Write-Host "  adapt:      $OnlineFullAdapt"
if ($OnlineFullAdapt) {
    Write-Host "  teacherCam: $TeacherCamera"
    Write-Host "  adapt lr:   $AdaptLr"
    Write-Host "  camera dbg: $ShowTeacherCamera"
    Write-Host "  target mesh:$VisualizeTeacherMesh"
    Write-Host "  log path:   $OnlineLogPath"
}
Write-Host ""

& conda @argsList
