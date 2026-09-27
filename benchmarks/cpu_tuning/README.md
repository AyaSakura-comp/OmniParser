# Experimental CPU tuning snapshot

This is an opt-in research artifact, not the default detector or service backend.
The live GPU service is unchanged. The model is YOLOv9-E, not YOLOv5.

See REPORT.md for measured speedups, cosine checks, changed detections, mixed-size
failure to reach 2x, and warmup costs. JSON files preserve per-image timings and
raw-head cosine measurements. No screenshots, model weights or credentials are
included. The scripts retain the original local /home/chihmin paths and one
external /tmp screenshot fixture; edit those paths for a different host. The
validation script writes results to the original experiment directory. It is
not a portable benchmark package.

CPUForward requires an isolated process, batch one, fixed image size, and
explicit torch.jit.enable_onednn_fusion(True). It uses deprecated TorchScript
APIs and is pinned/tested with PyTorch 2.12.1+rocm7.2. Do not enable its global
thread/fusion settings in the shared GPU service.

Verification: three adapter unit tests and an actual detector CLI smoke run.
No independent reviewer was invoked (delegation was not authorized).
