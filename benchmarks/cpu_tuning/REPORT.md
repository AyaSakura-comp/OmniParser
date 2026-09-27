# OmniParser CPU tuning — YOLOv9-E, not YOLOv5

## Scope and result
On the tested four images, fixed 800 or 1024 input, warmed batch-one CPU inference exceeds 2x against the original FP32 detector. This is NOT a claim about mixed-resolution service throughput, cold start, or every future image. The live GPU service and source model are unchanged.

## Environment
- AMD Ryzen AI MAX+ 395, 16 physical cores / 32 threads.
- PyTorch 2.12.1+rocm7.2, CPU only; oneDNN 3.11.2.
- CPU already dispatched AVX512 before tuning. `onednn.log` confirms `jit_bf16:avx512_core_bf16`; simply enabling AVX512 was not the speedup.
- Both baseline and candidate in the final runs use 16 threads, interop=1 and taskset 0-15. No service stopped, frequency locked, or system-wide scheduler changed. Other workloads may cause variance.

## Candidate
FP32 TorchScript freeze/fold, convert frozen floating constants to BF16, enable oneDNN fusion and optimized JIT execution, convert six raw heads back to FP32 before original decode/NMS. Keep original preprocessing, image size, confidence .05, IoU .7, max_det 300. No resizing shortcut or candidate pruning.

## Measurement
Four images: PChome screenshot cropped to page, google_page.png, windows_multitab.png, mobile.png. SHA256 input hashes are in JSON (PChome crop separately specified in validate.py). Six warmups per model/case; eight measured predictions each in alternating AB/BA order. Timing includes preprocessing, forward, decode, NMS; excludes model loading and warmup. Raw forward times also retained. Compare every one of six raw head tensors independently in float64 cosine, not just a concatenated coordinate-dominated vector. Class sigmoid probabilities checked separately.

| Run | Image | Size | FP32 ms | Candidate ms | Speedup | Min head cosine |
|---|---|---:|---:|---:|---:|---:|
| validation-fusion-16-800.json | screenshot.jpg | 800 | 407.4 | 120.8 | 3.37x | 0.999729 |
| validation-fusion-16-800.json | google_page.png | 800 | 508.4 | 152.9 | 3.32x | 0.999719 |
| validation-fusion-16-800.json | windows_multitab.png | 800 | 450.5 | 159.1 | 2.83x | 0.999651 |
| validation-fusion-16-800.json | mobile.png | 800 | 440.4 | 140.7 | 3.13x | 0.999703 |
| validation-fusion-16-1024-run1.json | screenshot.jpg | 1024 | 843.7 | 299.5 | 2.82x | 0.999410 |
| validation-fusion-16-1024-run1.json | google_page.png | 1024 | 873.4 | 309.5 | 2.82x | 0.999176 |
| validation-fusion-16-1024-run1.json | windows_multitab.png | 1024 | 924.1 | 366.7 | 2.52x | 0.999503 |
| validation-fusion-16-1024-run1.json | mobile.png | 1024 | 1124.2 | 422.2 | 2.66x | 0.998979 |
| validation-fusion-16-1024.json | screenshot.jpg | 1024 | 903.1 | 271.7 | 3.32x | 0.999410 |
| validation-fusion-16-1024.json | google_page.png | 1024 | 966.6 | 300.0 | 3.22x | 0.999176 |
| validation-fusion-16-1024.json | windows_multitab.png | 1024 | 967.7 | 287.7 | 3.36x | 0.999503 |
| validation-fusion-16-1024.json | mobile.png | 1024 | 882.5 | 272.2 | 3.24x | 0.998979 |

## Accuracy and limitations
- Minimum raw-head cosine: 0.998979270; minimum class-probability cosine: 0.998739942. Both exceed 0.99.
- BF16 is not identical detections: threshold crossings / NMS changes alter a few boxes. Baseline-box coverage at IoU>=.5 was 96.77–100%; this is nearest-box coverage, NOT labelled mAP or a one-to-one matching metric. No downstream browser success guarantee.
- Mixed-resolution run `validation-fusion-16.json` had a worst-case 1.89x; it does NOT pass an all-cases 2x gate. Use a dedicated process per resolution; arbitrary dynamically resized production traffic is not validated.
- 1280 accepted by CLI but not benchmarked; no 2x claim for it.
- Cold CLI load + six warmups took 5.26 s on one smoke run; warmed CLI median 439 ms at 1024. Performance varies with host load.
- Uses deprecated TorchScript graph APIs; pin runtime and revalidate on upgrades.
- Global Torch thread/fusion settings belong only in the isolated process. No live service deployment or restart performed.

## Reproduce
```bash
cd /home/chihmin/omni-cpu-tuning
PY=/home/chihmin/jev-comparison/.venv-rocm72/bin/python
"$PY" -m unittest test_cpu_adapter -q
taskset -c 0-15 "$PY" validate.py 16 800
taskset -c 0-15 "$PY" validate.py 16 1024
taskset -c 0-15 "$PY" run_cpu.py /home/chihmin/src/OmniParser/imgs/google_page.png --size 1024
```

Adapter TDD: missing adapter / missing CPUForward failures observed before implementation; final 3 tests passed. CLI smoke verifies usable detector and FP32 NMS interface. Screening logs include failed candidates and experiment errors, not silently removed.
