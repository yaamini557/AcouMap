import unittest
import numpy as np
from backend.ml.pipeline import FanAnomalyPipeline


class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        self.pipe = FanAnomalyPipeline()

    def test_fit_and_inference(self):
        np.random.seed(42)
        # 30 normal windows of 20 dimensions
        X_normal = np.random.normal(loc=0.0, scale=1.0, size=(30, 20))
        names = [f"f_{i}" for i in range(20)]

        meta = self.pipe.fit_normal_baseline(X_normal, names, fan_id="test_fan")
        self.assertEqual(meta["training_samples"], 30)
        self.assertEqual(meta["feature_dimension"], 20)

        # Compute scores on normal vectors
        X_norm = (X_normal - self.pipe.baseline_mean) / self.pipe.baseline_std
        if_scores, svm_scores, severities = self.pipe.compute_anomaly_scores(X_norm)

        self.assertEqual(len(if_scores), 30)
        # Normal data should have low anomaly scores (< 0.50)
        self.assertTrue(np.mean(if_scores) < 0.50)

        # An anomalous point far away
        X_anomaly = (np.ones((1, 20)) * 10.0 - self.pipe.baseline_mean) / self.pipe.baseline_std
        anom_scores, _, anom_sev = self.pipe.compute_anomaly_scores(X_anomaly)
        self.assertTrue(anom_scores[0] > 0.50)


if __name__ == "__main__":
    unittest.main()
