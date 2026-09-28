#!/usr/bin/env python3
import argparse
import base64
import io
import time
from pathlib import Path

from PIL import Image, ImageDraw


class DetectorService:
    def __init__(self, detector, threshold=0.05, image_size=800):
        self.detector = detector
        self.threshold = threshold
        self.image_size = image_size

    def parse_image(self, image, threshold=None, image_size=None):
        image = image.convert('RGB')
        started = time.perf_counter()
        conf = self.threshold if threshold is None else threshold
        imgsz = self.image_size if image_size is None else image_size
        result = self.detector.predict(
            image, conf=conf, imgsz=imgsz
        )[0]
        elements = []
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        for index, (box_tensor, confidence_tensor) in enumerate(
            zip(result.boxes.xyxy, result.boxes.conf)
        ):
            box = [round(float(value), 2) for value in box_tensor.tolist()]
            confidence = round(float(confidence_tensor), 4)
            x1, y1, x2, y2 = box
            center = [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)]
            elements.append({
                'id': index,
                'confidence': confidence,
                'box': box,
                'center': center,
            })
            color = '#00e5ff'
            draw.rectangle((x1, y1, x2, y2), outline='#111111', width=5)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            label = str(index)
            label_box = draw.textbbox((0, 0), label)
            label_width = label_box[2] - label_box[0] + 8
            label_height = label_box[3] - label_box[1] + 6
            label_y = max(0, y1 - label_height)
            draw.rectangle((x1, label_y, x1 + label_width, label_y + label_height), fill='#111111')
            draw.text((x1 + 4, label_y + 2), label, fill='#ffffff')

        output = io.BytesIO()
        annotated.save(output, format='PNG')
        return {
            'width': image.width,
            'height': image.height,
            'elements': elements,
            'annotated_image_base64': base64.b64encode(output.getvalue()).decode('ascii'),
            'latency': round(time.perf_counter() - started, 4),
        }


def create_app(service):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class ParseRequest(BaseModel):
        image_base64: str = Field(max_length=32 * 1024 * 1024)
        threshold: float | None = Field(default=None, ge=0, le=1)
        image_size: int | None = Field(default=None, ge=32, le=1280, strict=True)

    app = FastAPI(title='OmniParser V3 Detector API')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'device': str(service.detector.device),
                'backend': getattr(service.detector, 'backend', 'pytorch'),
                'precision': getattr(service.detector, 'precision',
                                     'FP16 autocast' if str(service.detector.device).startswith('cuda') else 'FP32'),
                'default_image_size': service.image_size}

    @app.post('/parse')
    def parse(request: ParseRequest):
        try:
            image = Image.open(io.BytesIO(base64.b64decode(request.image_base64, validate=True)))
            if image.width * image.height > 16_000_000:
                raise ValueError('Image exceeds 16 megapixels')
            return service.parse_image(
                image,
                threshold=request.threshold,
                image_size=request.image_size,
            )
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='weights/icon_detect_v3/model.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--backend', choices=['pytorch', 'openvino-int8'], default='pytorch')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--image-size', type=int, default=800)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8012)
    args = parser.parse_args()

    from util.yolov9 import YOLOv9Detector
    import uvicorn

    if not 1 <= args.threads <= 32:
        parser.error('--threads must be 1..32')
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.backend == 'openvino-int8':
        if args.device != 'cpu':
            parser.error('openvino-int8 supports --device cpu only; no GPU fallback')
        from util.yolov9_openvino import OpenVINOINT8Detector
        detector = OpenVINOINT8Detector(Path(args.model), threads=args.threads)
        detector.predict(Image.new('RGB', (1280, 633)), conf=args.threshold, imgsz=args.image_size)
    else:
        detector = YOLOv9Detector(Path(args.model), device=args.device)
    service = DetectorService(detector, args.threshold, args.image_size)
    uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
