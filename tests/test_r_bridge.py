import unittest
import numpy as np
from backend.services.r_bridge import RVectorAnalysisBridge


class TestRBridge(unittest.TestCase):
    def setUp(self):
        self.bridge = RVectorAnalysisBridge()

    def test_r_execution_and_math(self):
        np.random.seed(42)
        N = 10
        P = 20
        feature_matrix = np.random.rand(N, P)
        names = [f"feat_{i+1}" for i in range(P)]
        baseline_mean = np.mean(feature_matrix[:5], axis=0)
        baseline_std = np.std(feature_matrix[:5], axis=0) + 1e-4

        result = self.bridge.run_analysis(
            feature_matrix=feature_matrix,
            feature_names=names,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["vector_dimension"], P)
        self.assertEqual(result["sample_windows_count"], N)
        self.assertIn("mean_euclidean_distance", result)
        self.assertIn("mean_mahalanobis_distance", result)
        self.assertIn("mean_similarity_score", result)
        self.assertIn("windows", result)
        self.assertEqual(len(result["windows"]), N)

        # Check first window contents
        w0 = result["windows"][0]
        self.assertIn("euclidean_distance", w0)
        self.assertIn("mahalanobis_distance", w0)
        self.assertIn("pc1", w0)
        self.assertIn("pc2", w0)
        self.assertIn("top_deviations", w0)


if __name__ == "__main__":
    unittest.main()
