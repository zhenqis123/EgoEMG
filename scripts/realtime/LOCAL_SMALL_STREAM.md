# Local Small EMGFormer Runtime

This is the local-only deployment path for the Incre small EMGFormer model.
It runs in-process: serial EMG -> realtime preprocessing -> ring buffer ->
CUDA inference -> terminal/status output or optional JSONL.
With `-VisualizeMesh`, it also opens a local Open3D window that renders the
predicted UmeTrack mesh from the 20 FK angles. Open3D is created and polled from
the main process, while model callbacks only enqueue the latest angles, so GUI
refresh does not backlog inference.

## Smoke Test

```powershell
cd D:\develop\code\robot-data-collector
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\realtime\smoke_local_small_runtime.ps1
```

Expected output includes:

- `torch ... cuda True`
- `protocol d2d2d2 29 0xaa 0xbb 5`
- `output (1, 22, 230)`

## Run Streaming

```powershell
cd D:\develop\code\robot-data-collector
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\realtime\run_local_small_stream.ps1 -ComPort COM1
```

Use the actual EMG serial port in place of `COM1` when the device is attached.
The default output delay is `0.5s`, which was the best point measured on the
small-only45 validation split under the streaming causal preprocessing path.
The live serial input scale defaults to `0.001`, converting WeiLi int24 device
values to the millivolt-scale raw EMG used by the training memmap. Memmap replay
defaults to `1.0` because `emg_right_raw` is already stored at that scale.

The runtime uses `diffusers_env` and the exported runtime checkpoint:

```text
pretrained_models\emg2pose\egoemg-incre-small-8ch-runtime.pt
```

The full Lightning checkpoint is not needed for deployment.

## Run With UmeTrack Mesh Visualization

Run this from an interactive PowerShell session on the Windows desktop so the
Open3D window can attach to the logged-in display:

```powershell
cd D:\develop\code\robot-data-collector
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\realtime\run_local_small_stream.ps1 -ComPort COM1 -VisualizeMesh
```
