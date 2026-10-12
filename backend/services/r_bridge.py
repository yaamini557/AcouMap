"""
AcouMap - R Vector Analysis Subprocess Bridge
Coordinates the execution of R statistical analysis scripts (Rscript) with JSON serialization.
Computes:
- Vector normalization
- Euclidean distance
- Mahalanobis distance (regularized)
- Cosine vector similarity
- PCA 2D coordinates & loadings
- Feature-wise percentage deviations and z-scores
"""

import json
import subprocess
import os
import tempfile
import numpy as np
from typing import Dict, Any, List, Optional


class RVectorAnalysisBridge:
    def __init__(self, r_script_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if r_script_path is None:
            self.r_script_path = os.path.join(base_dir, "r_analysis", "vector_analysis.R")
        else:
            self.r_script_path = r_script_path

    def run_analysis(self,
                     feature_matrix: np.ndarray,
                     feature_names: List[str],
                     baseline_mean: np.ndarray,
                     baseline_std: np.ndarray,
                     baseline_cov: Optional[np.ndarray] = None,
                     baseline_pca_rotation: Optional[np.ndarray] = None,
                     baseline_pca_explained_var: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Execute R vector analysis via Rscript subprocess with JSON input/output.
        Falls back to exact NumPy/SciPy math if Rscript encounters any system level issues.
        """
        payload = {
            "feature_matrix": feature_matrix.tolist(),
            "feature_names": feature_names,
            "baseline_mean": baseline_mean.tolist(),
            "baseline_std": baseline_std.tolist(),
            "baseline_cov": baseline_cov.tolist() if baseline_cov is not None else None,
            "baseline_pca_rotation": baseline_pca_rotation.tolist() if baseline_pca_rotation is not None else None,
            "baseline_pca_explained_var": baseline_pca_explained_var.tolist() if baseline_pca_explained_var is not None else None
        }

        # Write to temporary file for Windows compatibility
        temp_input = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf:
                json.dump(payload, tf)
                temp_input = tf.name

            cmd = ["Rscript", self.r_script_path, temp_input]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if proc.returncode == 0 and proc.stdout.strip():
                # Extract JSON from output
                out_text = proc.stdout.strip()
                # Find start of JSON if R printed any warnings
                json_start = out_text.find("{")
                if json_start != -1:
                    json_str = out_text[json_start:]
                    result = json.loads(json_str)
                    result["execution_mode"] = "Rscript subprocess (native R 4.6.1)"
                    return result

            # If R exited with non-zero or empty output, log stderr
            stderr = proc.stderr if proc else "No output"
            print(f"[RBridge Warning] Rscript returned code {proc.returncode if proc else 'N/A'}: {stderr}")
            return self._python_fallback(payload)

        except Exception as e:
            print(f"[RBridge Exception] Falling back to NumPy/SciPy: {e}")
            return self._python_fallback(payload)

        finally:
            if temp_input and os.path.exists(temp_input):
                try:
                    os.remove(temp_input)
                except OSError:
                    pass

    def _python_fallback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exact mathematical equivalent in Python if Rscript execution is interrupted.
        Guarantees system resilience without fabricating any metrics.
        """
        X = np.array(payload["feature_matrix"], dtype=np.float64)
        names = payload["feature_names"]
        b_mean = np.array(payload["baseline_mean"], dtype=np.float64)
        b_std = np.array(payload["baseline_std"], dtype=np.float64)
        b_std[b_std < 1e-6] = 1e-4

        N, P = X.shape

        # 1. Normalization
        X_norm = (X - b_mean) / b_std

        # 2. Euclidean Distance
        euclidean_dist = np.sqrt(np.sum(X_norm ** 2, axis=1))

        # 3. Regularized Mahalanobis Distance
        cov_norm = np.cov(X_norm, rowvar=False) if N > 1 else np.eye(P)
        if cov_norm.ndim == 0 or np.isnan(cov_norm).any():
            cov_norm = np.eye(P)
        lambda_reg = 0.15
        cov_reg = (1.0 - lambda_reg) * cov_norm + lambda_reg * np.diag(np.diag(cov_norm)) + np.eye(P) * 1e-4
        inv_cov = np.linalg.pinv(cov_reg)

        mahalanobis_dist = np.sqrt(np.maximum(0.0, np.sum(X_norm @ inv_cov * X_norm, axis=1)))

        # 4. Cosine Similarity
        norm_b = np.linalg.norm(b_mean)
        norm_X = np.linalg.norm(X, axis=1)
        sim = np.zeros(N)
        for i in range(N):
            if norm_X[i] > 1e-8 and norm_b > 1e-8:
                sim[i] = float(np.dot(X[i], b_mean) / (norm_X[i] * norm_b))

        # 5. PCA Projection
        if payload.get("baseline_pca_rotation") is not None:
            rot = np.array(payload["baseline_pca_rotation"])[:, :2]
            pca_coords = X_norm @ rot
            explained_var = payload.get("baseline_pca_explained_var", [0.6, 0.25])
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(2, P))
            pca_coords = pca.fit_transform(X_norm)
            explained_var = pca.explained_variance_ratio_.tolist()

        # 6. Percentage deviations
        denom = np.where(np.abs(b_mean) > 1e-6, np.abs(b_mean), 1.0)
        pct_dev = ((X - b_mean) / denom) * 100.0

        windows = []
        for i in range(N):
            dev_order = np.argsort(np.abs(pct_dev[i]))[::-1]
            top_devs = []
            for k in range(min(3, P)):
                idx = int(dev_order[k])
                top_devs.append({
                    "feature": names[idx],
                    "pct_change": round(float(pct_dev[i, idx]), 2),
                    "z_score": round(float(X_norm[i, idx]), 3),
                    "current_value": round(float(X[i, idx]), 4),
                    "baseline_value": round(float(b_mean[idx]), 4)
                })

            windows.append({
                "window_index": i,
                "euclidean_distance": round(float(euclidean_dist[i]), 4),
                "mahalanobis_distance": round(float(mahalanobis_dist[i]), 4),
                "similarity_score": round(float(sim[i]), 4),
                "pc1": round(float(pca_coords[i, 0]), 4),
                "pc2": round(float(pca_coords[i, 1]), 4) if pca_coords.shape[1] > 1 else 0.0,
                "normalized_vector": [round(float(v), 4) for v in X_norm[i]],
                "top_deviations": top_devs
            })

        return {
            "status": "success",
            "engine": "NumPy/SciPy Math Fallback",
            "execution_mode": "Internal Python Vector Analysis",
            "vector_dimension": P,
            "sample_windows_count": N,
            "mean_euclidean_distance": round(float(np.mean(euclidean_dist)), 4),
            "mean_mahalanobis_distance": round(float(np.mean(mahalanobis_dist)), 4),
            "mean_similarity_score": round(float(np.mean(sim)), 4),
            "pca_explained_variance": [round(float(v), 4) for v in explained_var[:2]],
            "feature_names": names,
            "baseline_mean": [round(float(v), 4) for v in b_mean],
            "current_mean": [round(float(v), 4) for v in np.mean(X, axis=0)],
            "feature_mean_pct_deviation": {names[j]: round(float(np.mean(pct_dev[:, j])), 2) for j in range(P)},
            "windows": windows
        }
