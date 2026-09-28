import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

class OpenVINOTests(unittest.TestCase):
    def test_backend_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('util.yolov9_openvino'))

    def test_real_runtime_shape_cache_and_independent_outputs(self):
        if importlib.util.find_spec('util.yolov9_openvino') is None:
            self.skipTest('backend absent')
        import openvino as ov
        from openvino import opset13 as ops
        from util.yolov9_openvino import OpenVINOINT8Detector
        x = ops.parameter([1,3,800,800], np.float32)
        fq = ops.fake_quantize(x, ops.constant(np.float32(0)), ops.constant(np.float32(1)), ops.constant(np.float32(0)), ops.constant(np.float32(1)), 256)
        model = ov.Model([fq for _ in range(6)], [x])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.xml'
            ov.save_model(model, path, compress_to_fp16=False)
            detector = OpenVINOINT8Detector(path, threads=2, cache_size=2)
            first = detector.model(torch.zeros(1,3,800,800))
            detector.model(torch.ones(1,3,800,800))
            self.assertTrue(torch.equal(first[0], torch.zeros_like(first[0])))
            for size in (1024,1280):
                outputs = detector.model(torch.zeros(1,3,size,size))
                self.assertEqual(tuple(outputs[0].shape), (1,3,size,size))
            self.assertEqual(len(detector._cache), 2)
            with self.assertRaises(ValueError):detector.model(torch.zeros(1,3,32,32))
            with self.assertRaises(ValueError):detector.predict(np.zeros((8,8,3),dtype=np.uint8),imgsz=8192)

    def test_rejects_float_model(self):
        if importlib.util.find_spec('util.yolov9_openvino') is None:self.skipTest('backend absent')
        import openvino as ov
        from openvino import opset13 as ops
        from util.yolov9_openvino import OpenVINOINT8Detector
        x=ops.parameter([1,3,800,800],np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'float.xml';ov.save_model(ov.Model([x],[x]),path)
            with self.assertRaises(ValueError):OpenVINOINT8Detector(path)
