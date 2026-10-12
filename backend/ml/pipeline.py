"""
AcouMap - Machine Learning Pipeline
Implements:
- Primary Model: Isolation Forest
- Secondary / Comparison Model: One-Class SVM
- Dimensionality Reduction: PCA
- Baseline Profile Learning & Model Persistence
- Real Anomaly Scoring & Health Index Calculation
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from typing import Dict, Any, List, Optional, Tuple


class FanAnomalyPipeline:
    """
    Industrial acoustic anomaly detection pipeline.
    Trains on normal acoustic feature vectors to establish normal operational envelopes.
    """

    def __init__(self,
                 model_dir: Optional[str] = None,
                 warning_threshold: float = 0.50,
                 critical_threshold: float = 0.75):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.model_dir = os.path.join(base_dir, "models")
        else:
            self.model_dir = model_dir

        os.makedirs(self.model_dir, exist_ok=True)

        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        # Active models
        self.isolation_forest: Optional[IsolationForest] = None
        self.oc_svm: Optional[OneClassSVM] = None
        self.pca: Optional[PCA] = None

        # Baseline statistics
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None
        self.baseline_cov: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.model_metadata: Dict[str, Any] = {}

    def fit_normal_baseline(self,
                            normal_feature_matrix: np.ndarray,
                            feature_names: List[str],
                            fan_id: str = "fan_00",
                            version: str = "v1.0.0") -> Dict[str, Any]:
        """
        Learn normal acoustic behavior profile from verified normal recordings.
        """
        X = normal_feature_matrix.astype(np.float64)
        N, P = X.shape

        self.feature_names = feature_names
        self.baseline_mean = np.mean(X, axis=0)
        self.baseline_std = np.std(X, axis=0)
        self.baseline_std[self.baseline_std < 1e-6] = 1e-4

        # Regularized covariance matrix
        cov = np.cov(X, rowvar=False) if N > 1 else np.eye(P)
        if cov.ndim == 0:
            cov = np.eye(P)
        lambda_reg = 0.1
        self.baseline_cov = (1.0 - lambda_reg) * cov + lambda_reg * np.diag(np.diag(cov)) + np.eye(P) * 1e-4

        # 1. Normalize normal features for model training
        X_norm = (X - self.baseline_mean) / self.baseline_std

        # 2. Train Primary Model: Isolation Forest
        self.isolation_forest = IsolationForest(
            n_estimators=150,
            contamination=0.03,  # Expect low anomaly rate in normal baseline
            random_state=42,
            bootstrap=True,
            n_jobs=-1
        )
        self.isolation_forest.fit(X_norm)

        # 3. Train Comparison Model: One-Class SVM
        self.oc_svm = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.05
        )
        self.oc_svm.fit(X_norm)

        # 4. Fit PCA (2 components for 2D Acoustic Map)
        self.pca = PCA(n_components=min(2, P), random_state=42)
        self.pca.fit(X_norm)

        # Store metadata
        self.model_metadata = {
            "fan_id": fan_id,
            "version": version,
            "training_samples": N,
            "feature_dimension": P,
            "feature_names": feature_names,
            "algorithms": {
                "primary": "Isolation Forest (n_estimators=150, contamination=0.03)",
                "comparison": "One-Class SVM (kernel=rbf, nu=0.05)",
                "projection": "PCA (2 components)"
            },
            "pca_explained_variance_ratio": [round(float(v), 4) for v in self.pca.explained_variance_ratio_],
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold
        }

        # Save to disk
        self.save_models(fan_id)
        return self.model_metadata

    def compute_anomaly_scores(self, X_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute true anomaly scores using Isolation Forest decision function.
        In scikit-learn:
          decision_function > 0 => normal
          decision_function < 0 => anomaly
        We map this to [0, 1] anomaly score:
          score = 1.0 / (1.0 + exp(alpha * decision_function))
        where score near 0 = healthy normal, score near 1 = critical anomaly.
        """
        if self.isolation_forest is None:
            raise RuntimeError("Isolation Forest model is not trained/loaded.")

        # Real decision function from model
        if_raw = self.isolation_forest.decision_function(X_norm)
        # Smooth logistic mapping calibrated around decision boundary
        if_scores = 1.0 / (1.0 + np.exp(4.0 * if_raw))

        # Comparison One-Class SVM score
        if self.oc_svm is not None:
            svm_raw = self.oc_svm.decision_function(X_norm)
            svm_scores = 1.0 / (1.0 + np.exp(3.0 * svm_raw))
        else:
            svm_scores = np.zeros_like(if_scores)

        # Classify severity
        severities = np.empty(len(if_scores), dtype=object)
        for i, score in enumerate(if_scores):
            if score < self.warning_threshold:
                severities[i] = "NORMAL"
            elif score < self.critical_threshold:
                severities[i] = "WARNING"
            else:
                severities[i] = "CRITICAL"

        return if_scores, svm_scores, severities

    def evaluate_models(self, X_norm_test: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Calculates REAL classification metrics when ground truth labels exist.
        Never fabricates metrics. If labels do not exist or are invalid, returns explicit notice.
        """
        if len(np.unique(y_true)) < 2:
            return {
                "available": False,
                "reason": "Insufficient labeled evaluation data (all samples have the same class)."
            }

        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

        if_scores, svm_scores, _ = self.compute_anomaly_scores(X_norm_test)
        if_pred = (if_scores >= self.warning_threshold).astype(int)
        svm_pred = (svm_scores >= self.warning_threshold).astype(int)

        return {
            "available": True,
            "isolation_forest": {
                "precision": round(float(precision_score(y_true, if_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, if_pred, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_true, if_pred, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_true, if_scores)), 4)
            },
            "one_class_svm": {
                "precision": round(float(precision_score(y_true, svm_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, svm_pred, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_true, svm_pred, zero_division=0)), 4),
                "roc_auc": round(float(roc_auc_score(y_true, svm_scores)), 4)
            }
        }

    def save_models(self, fan_id: str):
        """Persist trained pipeline artifacts."""
        path = os.path.join(self.model_dir, f"{fan_id}_pipeline.joblib")
        data = {
            "isolation_forest": self.isolation_forest,
            "oc_svm": self.oc_svm,
            "pca": self.pca,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "baseline_cov": self.baseline_cov,
            "feature_names": self.feature_names,
            "metadata": self.model_metadata
        }
        joblib.dump(data, path)

    def load_models(self, fan_id: str) -> bool:
        """Load persisted models for a specific fan."""
        path = os.path.join(self.model_dir, f"{fan_id}_pipeline.joblib")
        if not os.path.exists(path):
            return False

        data = joblib.load(path)
        self.isolation_forest = data["isolation_forest"]
        self.oc_svm = data["oc_svm"]
        self.pca = data["pca"]
        self.baseline_mean = data["baseline_mean"]
        self.baseline_std = data["baseline_std"]
        self.baseline_cov = data.get("baseline_cov")
        self.feature_names = data["feature_names"]
        self.model_metadata = data.get("metadata", {})
        return True
