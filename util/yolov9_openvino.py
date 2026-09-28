"""Opt-in OpenVINO INT8 CPU runtime for the calibrated YOLOv9-E IR.

Uses the original image preprocessing, decoding and NMS. No backend fallback.
INT8 is lossy: see docs/CPU_INT8_INSTALL.md before enabling it.
"""
from collections import OrderedDict
from pathlib import Path
from threading import Lock

import torch
from util.yolov9 import YOLOv9Detector


class OpenVINOINT8Detector(YOLOv9Detector):
    backend = 'openvino-int8'
    precision = 'INT8 mixed; floating operations use BF16 hint'

    def __init__(self, model_path, threads=16, cache_size=2):
        import openvino as ov
        if not 1 <= threads <= 32 or not 1 <= cache_size <= 4:
            raise ValueError('threads must be 1..32 and cache_size 1..4')
        self.device = torch.device('cpu')
        self.model_path = Path(model_path)
        self._core = ov.Core()
        self._ir = self._core.read_model(self.model_path)
        if not any(op.get_type_name() == 'FakeQuantize' for op in self._ir.get_ops()):
            raise ValueError('Expected a calibrated FakeQuantize INT8 IR, not a floating model')
        if len(self._ir.inputs) != 1 or len(self._ir.outputs) != 6:
            raise ValueError('Expected one RGB input and six YOLOv9 heads')
        self._config = {'PERFORMANCE_HINT':'LATENCY', 'INFERENCE_NUM_THREADS':threads,
                        'NUM_STREAMS':1, 'INFERENCE_PRECISION_HINT':'bf16'}
        self._cache = OrderedDict()
        self._cache_size = cache_size
        self._lock = Lock()
        self.model = self._forward

    def predict(self, source, conf=0.25, imgsz=1024, iou=0.7, max_det=300):
        if isinstance(imgsz, bool) or not isinstance(imgsz, int) or not 800 <= imgsz <= 1280 or imgsz % 32:
            raise ValueError('OpenVINO image size must be a multiple of 32 within 800..1280')
        return super().predict(source, conf=conf, imgsz=imgsz, iou=iou, max_det=max_det)

    def _forward(self, image_tensor):
        shape = tuple(image_tensor.shape)
        if (image_tensor.device.type != 'cpu' or len(shape) != 4 or shape[:2] != (1,3)
                or shape[2] != shape[3] or not 800 <= shape[2] <= 1280 or shape[2] % 32):
            raise ValueError('Expected batch-one CPU RGB input, square size 800..1280, multiple of 32')
        # FastAPI runs requests in threads. Requests share buffers in OpenVINO,
        # so compilation, inference and copying results are serialized together.
        with self._lock:
            if shape not in self._cache:
                model = self._ir.clone()
                model.reshape(list(shape))
                compiled = self._core.compile_model(model, 'CPU', self._config)
                self._cache[shape] = compiled.create_infer_request()
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(shape)
            request = self._cache[shape]
            request.infer([image_tensor.detach().float().contiguous().numpy()])
            return [torch.from_numpy(request.get_output_tensor(i).data.copy()).float()
                    for i in range(6)]
