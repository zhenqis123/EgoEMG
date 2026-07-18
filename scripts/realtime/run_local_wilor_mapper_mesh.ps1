param(
    [int]$Camera = 0,
    [int]$Width = 1280,
    [int]$Height = 720,
    [double]$Fps = 30,
    [string]$Hand = "right",
    [ValidateSet("mapper", "lbfgs")]
    [string]$PoseSource = "mapper",
    [string]$MapperCheckpoint = "pretrained_models\mano_to_umetrack_mapper.pt",
    [string]$Device = "cuda",
    [string]$DType = "float16",
    [string]$EnvName = "diffusers_env",
    [int]$DetectInterval = 1,
    [int]$MaxBBoxAge = 3,
    [int]$LbfgsMaxIter = 3,
    [double]$LbfgsLr = 0.5,
    [int]$LbfgsHistorySize = 10,
    [int]$MaxFrames = 0,
    [double]$YoloConf = 0.1,
    [int]$YoloInputHeight = 512,
    [switch]$VisualizeManoMesh,
    [switch]$NoCameraPreview
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$argsList = @(
    "run", "-n", $EnvName,
    "python", "scripts\realtime\local_wilor_mapper_mesh.py",
    "--camera", "$Camera",
    "--width", "$Width",
    "--height", "$Height",
    "--fps", "$Fps",
    "--hand", "$Hand",
    "--pose-source", "$PoseSource",
    "--mapper-checkpoint", "$MapperCheckpoint",
    "--device", "$Device",
    "--dtype", "$DType",
    "--detect-interval", "$DetectInterval",
    "--max-bbox-age", "$MaxBBoxAge",
    "--lbfgs-max-iter", "$LbfgsMaxIter",
    "--lbfgs-lr", "$LbfgsLr",
    "--lbfgs-history-size", "$LbfgsHistorySize",
    "--max-frames", "$MaxFrames",
    "--yolo-conf", "$YoloConf",
    "--yolo-input-height", "$YoloInputHeight"
)

if ($VisualizeManoMesh) {
    $argsList += @("--visualize-mano-mesh")
}
if ($NoCameraPreview) {
    $argsList += @("--no-camera-preview")
}

Write-Host "Starting realtime WiLoR -> mapper -> UmeTrack mesh..."
Write-Host "  env:        $EnvName"
Write-Host "  camera:     $Camera"
Write-Host "  checkpoint: $MapperCheckpoint"
Write-Host "  hand:       $Hand"
Write-Host "  poseSource: $PoseSource"
Write-Host "  device:     $Device"
Write-Host "  mano mesh:  $VisualizeManoMesh"
Write-Host ""

& conda @argsList
