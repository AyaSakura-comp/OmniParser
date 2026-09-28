import base64
import io
import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from detector_service import DetectorService, create_app


class FakeDetector:
    def predict(self, image, conf, imgsz):
        boxes = SimpleNamespace(
            xyxy=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            conf=torch.tensor([0.75]),
        )
        return [SimpleNamespace(boxes=boxes)]


class DetectorServiceTests(unittest.TestCase):
    def test_returns_screen_pixel_boxes_centers_and_annotated_image(self):
        image = Image.new('RGB', (100, 80), 'white')
        service = DetectorService(FakeDetector(), threshold=0.1, image_size=640)

        result = service.parse_image(image)

        self.assertEqual(result['width'], 100)
        self.assertEqual(result['height'], 80)
        self.assertEqual(result['elements'], [{
            'id': 0,
            'confidence': 0.75,
            'box': [10.0, 20.0, 30.0, 40.0],
            'center': [20.0, 30.0],
        }])
        annotated = Image.open(io.BytesIO(base64.b64decode(result['annotated_image_base64'])))
        self.assertEqual(annotated.size, (100, 80))
        self.assertNotEqual(annotated.getpixel((10, 20)), (255, 255, 255))


class APIBackendTests(unittest.TestCase):
    def test_health_reports_actual_backend(self):
        from fastapi.testclient import TestClient
        detector = FakeDetector()
        detector.device = 'cpu'
        detector.backend = 'openvino-int8'
        detector.precision = 'INT8 mixed'
        client = TestClient(create_app(DetectorService(detector)))
        self.assertEqual(client.get('/health').json().get('backend'), 'openvino-int8')

    def test_rejects_unbounded_image_size(self):
        from fastapi.testclient import TestClient
        client = TestClient(create_app(DetectorService(FakeDetector())))
        response = client.post('/parse', json={'image_base64':'abc', 'image_size':999999})
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
