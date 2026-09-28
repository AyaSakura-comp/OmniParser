# Opt-in YOLOv9-E OpenVINO INT8 CPU service

This fork adds a **detector-only** localhost API. It does not replace the complete
upstream OCR/caption pipeline, and does not change the default `YOLOv9Detector`.
The deployed backend is CPU, not AMD GPU. It frees the GPU from this detector's
workload, but was **slower** than GPU FP16 in the recorded 20-site comparison.

## Accuracy / performance decision

The operator explicitly selected this lossy version after inspecting examples.
It does **not** meet the earlier numeric cosine >=0.99 requirement:

| 20-site test, 1024 input | Median complete prediction | Raw-head cosine >=.99 | Probability cosine >=.99 |
|---|---:|---:|---:|
| PyTorch GPU FP16 | 71 ms | 20/20 | 20/20 |
| OpenVINO CPU BF16 | 151 ms | 20/20 | 19/20 |
| **OpenVINO CPU INT8** | **114 ms** | **0/20** | **1/20** |

INT8 minimum raw cosine was 0.956565 and minimum class-probability cosine was
0.571836. Against FP32's 1243 boxes, greedy one-to-one IoU>=.5 matching found
130 missing and 113 extra boxes. FP32 is not labelled ground truth: these are
agreement measurements, not mAP or task success. Small icons may be missed;
blank/background boxes and duplicates can occur in the reference too.

Measurements use identical 1280x633 page crops, 1024 square model inputs,
confidence .05, NMS IoU .7 and max_det 300, with 3 warmups/5 samples per image.
Prediction includes preprocessing/NMS and returning boxes to CPU, not HTTP,
PNG annotation, screenshot capture or model loading. Sequential backend runs
on a shared host are not strict interleaved performance certification.

## Installation

Use a separate Python 3.12 environment with compatible `torch`, `torchvision`,
Pillow and huggingface_hub already installed. The tested host uses
PyTorch 2.12.1+rocm7.2 / torchvision 0.27.1+rocm7.2 **on CPU**. No GPU access is
required by this backend. Check that the environment imports these modules
before installing additions; do not overwrite a shared inference environment.

```sh
python -m pip install -r requirements-openvino.txt
# Export only; not needed to serve an existing IR:
python -m pip install nncf==3.4.0
```

On the existing host we installed **only** OpenVINO into a repository-local
overlay because all other runtime dependencies were already present:

```sh
uv pip install --python "$PYTHON" --target "$PWD/.deps-openvino" \
  --no-deps openvino==2026.4.0
export PYTHONPATH="$PWD/.deps-openvino${PYTHONPATH:+:$PYTHONPATH}"
```

`.deps-openvino`, virtual environments and weights are git-ignored. The earlier
experimental NNCF/benchmark directory is not a runtime dependency.

## Model generation

Use the official inference-only YOLOv9-E `weights/icon_detect_v3/model.pt`
from the upstream README, plus the repository's existing `imgs` fixtures.
Only deserialize model files you trust. The export is CPU-only and uses 16
threads; schedule it away from latency-sensitive production traffic.

```sh
python benchmarks/cpu_tuning/export_int8.py \
  --model weights/icon_detect_v3/model.pt \
  --output weights/icon_detect_v3/openvino-int8/int8-1024.xml
```

The exporter uses FP32 TorchScript freezing then OpenVINO conversion and NNCF
MIXED post-training quantization. Calibration: `teams`, `windows_vm`, `onenote`,
`windows`, `excel`, `word`, `windows_home`, `ios` PNGs (eight images at 1024).
The 20 website screenshots were **not** calibration data. This small calibration
set is experimental. Export refuses to overwrite an existing XML/BIN pair.
No model weights or private screenshots are committed to this fork.

The exact deployed artifacts have SHA256:

- XML: `f82c45b74d2eadebfbc14ed20af9a1d3b83035c3e7963d8c05e4cce7e20a1b27`
- BIN: `37095beb70409d5cd8e196cec71e40b440b390110351e82f25d2c36c89c54e60`

A fresh export need not be byte-identical across toolchains; verify outputs.

## Run / interface

```sh
python detector_service.py --backend openvino-int8 \
  --model weights/icon_detect_v3/openvino-int8/int8-1024.xml \
  --device cpu --threads 16 --threshold 0.05 --image-size 1024 \
  --host 127.0.0.1 --port 8012
```

`GET /health` reports `backend: openvino-int8`, `device: cpu`, the default size,
and `precision: INT8 mixed; floating operations use BF16 hint`. INT8 does not
mean every operation is integer. No fallback to FP16/GPU or another model.
Missing/incompatible IR prevents startup rather than hiding the failure.

`POST /parse` keeps the existing JSON contract:
`image_base64`, optional `threshold`, optional `image_size`; response includes
original-image-pixel boxes, centres, confidence, annotated PNG and latency.
Request image size is bounded; OpenVINO accepts square model sizes 800..1280 in
multiples of 32. This covers the browser's adaptive size choices. Compilation
is cached for two shapes (LRU), with thread-safe infer-request access and copied
outputs. Startup warms the default size. A new/evicted size incurs compilation
(~1.3–1.8 s on the tested host). Repeated 1024 API calls measured ~148 ms service
latency / 160 ms HTTP wall time in one smoke test.

The numeric accuracy comparison applies to 1024 only. 800/1280 reshaping and
coordinate return were smoke-tested, not accuracy-certified. Dynamic input-size
support must not be presented as a >=.99 accuracy guarantee.

Keep localhost binding unless adding authentication, concurrency/request limits
and a reviewed reverse proxy. The app limits encoded payload to 32 MiB and
images to 16 megapixels; these are not a substitute for public-service hardening.

## Existing-host systemd deployment / rollback

Preserve the existing unit before changing it. Use the selected interpreter,
absolute model path, working directory, and if using the overlay:
`Environment=PYTHONPATH=/absolute/OmniParser/.deps-openvino`.
Only change OmniParser's unit; do not restart Qwen, the browser or other services.
On this host, route restart through the existing restart-service hub:

```sh
bash ~/.hermes/skills/restart-service/scripts/restart-omniparser.sh
curl -fsS http://127.0.0.1:8012/health
```

Rollback to the saved GPU configuration:

```sh
cp ~/.config/systemd/user/omniparser.service.bak-before-openvino-int8 \
   ~/.config/systemd/user/omniparser.service
bash ~/.hermes/skills/restart-service/scripts/restart-omniparser.sh
```

The shared service still supports `--backend pytorch --device cuda --model ...model.pt`.
Reverting the unit is sufficient; no source-code reset is needed.

## Verification

```sh
python -m unittest discover -s tests -v
```

Six tests cover API formatting/health/request bounds and real OpenVINO
shape compilation, cache eviction, independent copied outputs and rejection
of floating IR. Deployment adapter was compared to all 20 saved INT8 outputs:
20/20 matching box shapes and coordinates (atol .05, rtol 1e-5).
Live `/parse` passed at 800,1024,1280; the browser's `vision-mark omni` returned
13 regions on Wikipedia with reported detector latency 0.1705 s. This is a smoke
test, not a full Bad UI or action-success benchmark. No independent delegated
review was performed; only local tests and direct code/security review.
