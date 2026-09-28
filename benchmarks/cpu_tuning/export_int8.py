"""Offline calibration recipe for the experimental INT8 detector (not lossless)."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import openvino as ov
    import nncf
    from PIL import Image
    from util.yolov9 import YOLOv9Detector

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'weights/icon_detect_v3/model.pt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != '.xml':
        parser.error('--output must end with .xml')
    if args.output.exists() or args.output.with_suffix('.bin').exists():
        parser.error('Refusing to overwrite existing model artifacts')
    torch.set_num_threads(16)
    detector = YOLOv9Detector(args.model, device='cpu')
    names = ['teams.png', 'windows_vm.png', 'onenote.png', 'windows.png',
             'excel.png', 'word.png', 'windows_home.png', 'ios.png']
    inputs = [detector._preprocess(Image.open(ROOT/'imgs'/name).convert('RGB'), 1024)[0].numpy()
              for name in names]
    model = ov.convert_model(torch.jit.freeze(detector.model), input=[1,3,1024,1024])
    quantized = nncf.quantize(model, nncf.Dataset(inputs), subset_size=len(inputs),
                              preset=nncf.QuantizationPreset.MIXED,
                              target_device=nncf.TargetDevice.CPU)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, args.output, compress_to_fp16=False)
    print(f'Created lossy INT8 model: {args.output}; validate before deployment')


if __name__ == '__main__':
    main()
