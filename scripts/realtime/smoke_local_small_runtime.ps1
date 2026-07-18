param(
    [string]$Checkpoint = "pretrained_models\emg2pose\egoemg-incre-small-only45-runtime.pt",
    [string]$EnvName = "diffusers_env"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $RepoRoot

$code = @"
import sys, time, torch
sys.path.insert(0, "data_collect")
from emg2pose.realtime_local.serial import SerialProtocol
from emg2pose.realtime_local.small_model import load_small_emgformer

print("python", sys.executable)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
p = SerialProtocol.from_collect_or_args(None, None, None, None)
print("protocol", p.header.hex(), p.packet_len, hex(p.emg_type), hex(p.imu_type), p.payload_offset)
m = load_small_emgformer("$Checkpoint", "cuda")
x = torch.zeros(1, 16, 12000, device="cuda")
torch.cuda.synchronize()
t0 = time.perf_counter()
y = m({"emg": x})
torch.cuda.synchronize()
print("output", tuple(y.shape), "forward_ms", round((time.perf_counter() - t0) * 1000, 3))
"@

$tmp = Join-Path $env:TEMP "emg2pose_small_runtime_smoke.py"
Set-Content -Path $tmp -Value $code -Encoding UTF8
try {
    conda run -n $EnvName python $tmp
}
finally {
    Remove-Item -Force $tmp -ErrorAction SilentlyContinue
}
