import unittest
from starlette.testclient import TestClient
from backend.main import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "Operational")

    def test_fans_endpoint(self):
        resp = self.client.get("/api/fans")
        self.assertEqual(resp.status_code, 200)
        fans = resp.json()
        self.assertTrue(len(fans) >= 4)
        self.assertIn("id", fans[0])
        self.assertIn("status", fans[0])

    def test_acoustic_map_endpoint(self):
        resp = self.client.get("/api/acoustic-map")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("points", data)
        self.assertTrue(len(data["points"]) > 0)
        p0 = data["points"][0]
        self.assertIn("pc1", p0)
        self.assertIn("pc2", p0)
        self.assertIn("severity", p0)

    def test_model_info_endpoint(self):
        resp = self.client.get("/api/model-info")
        self.assertEqual(resp.status_code, 200)
        info = resp.json()
        self.assertEqual(info["primary_model"]["name"], "Isolation Forest")
        self.assertEqual(info["comparison_model"]["name"], "One-Class SVM")

    def test_dataset_info_endpoint(self):
        resp = self.client.get("/api/dataset-info")
        self.assertEqual(resp.status_code, 200)
        d_info = resp.json()
        self.assertIn("dataset_name", d_info)
        self.assertIn("status_label", d_info)


if __name__ == "__main__":
    unittest.main()
