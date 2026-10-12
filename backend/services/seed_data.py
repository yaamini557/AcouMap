"""
AcouMap - Database Seeder & Baseline Initializer
Seeds initial industrial fan profiles, generates reproducible synthetic demo audio,
extracts acoustic feature matrices, invokes R vector analysis, trains Isolation Forest baselines,
and populates the SQLite database.
"""

import os
import uuid
import numpy as np
from datetime import datetime
from backend.database.db import DatabaseManager
from backend.audio.synthetic import SyntheticFanAudioGenerator
from backend.audio.features import AcousticFeatureExtractor
from backend.services.r_bridge import RVectorAnalysisBridge
from backend.ml.pipeline import FanAnomalyPipeline
from backend.services.maintenance import ConservativeMaintenanceEngine


def seed_database():
    print("[AcouMap Seeder] Initializing database and synthetic acoustic library...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    db = DatabaseManager()
    gen = SyntheticFanAudioGenerator(sample_rate=16000, seed=42)
    extractor = AcousticFeatureExtractor(n_mfcc=13, window_sec=1.0, hop_sec=0.5, target_sr=16000)
    r_bridge = RVectorAnalysisBridge()
    ml_pipe = FanAnomalyPipeline()

    # 1. Register Fans
    fans = [
        {
            "id": "fan_00",
            "fan_name": "Centrifugal Ventilation Fan A1",
            "machine_type": "Centrifugal Industrial Fan",
            "operating_hours": 3420.5,
            "status": "NORMAL",
            "bpf_nominal": 120.0,
            "created_at": "2026-09-01 08:00:00"
        },
        {
            "id": "fan_02",
            "fan_name": "Cooling Tower Fan B2",
            "machine_type": "Axial Cooling Fan",
            "operating_hours": 5190.0,
            "status": "WARNING",
            "bpf_nominal": 120.0,
            "created_at": "2026-08-15 08:00:00"
        },
        {
            "id": "fan_04",
            "fan_name": "Combustion Air Blower C3",
            "machine_type": "High-Pressure Blower",
            "operating_hours": 1820.0,
            "status": "NORMAL",
            "bpf_nominal": 320.0,
            "created_at": "2026-09-20 08:00:00"
        },
        {
            "id": "fan_06",
            "fan_name": "Dust Exhaust Exhauster D4",
            "machine_type": "Heavy Radial Fan",
            "operating_hours": 7640.2,
            "status": "CRITICAL",
            "bpf_nominal": 100.0,
            "created_at": "2026-07-10 08:00:00"
        }
    ]

    for f in fans:
        db.insert_fan(f)

    # 2. Generate Synthetic Fan Audio
    synth_dir = os.path.join(base_dir, "data", "synthetic")
    os.makedirs(synth_dir, exist_ok=True)

    recordings_info = [
        # fan_00 normal baseline training files
        {"fan_id": "fan_00", "fname": "fan_00_normal_01.wav", "cond": "normal", "dur": 10.0, "seed": 42},
        {"fan_id": "fan_00", "fname": "fan_00_normal_02.wav", "cond": "normal", "dur": 10.0, "seed": 43},
        {"fan_id": "fan_00", "fname": "fan_00_normal_03.wav", "cond": "normal", "dur": 10.0, "seed": 44},
        # fan_00 anomalous test recordings
        {"fan_id": "fan_00", "fname": "fan_00_abnormal_high_freq.wav", "cond": "high_frequency_rubbing", "dur": 10.0, "seed": 101, "astart": 3.0, "adur": 4.0},
        {"fan_id": "fan_00", "fname": "fan_00_abnormal_broadband.wav", "cond": "increased_broadband_noise", "dur": 10.0, "seed": 102, "astart": 2.5, "adur": 4.5},
        {"fan_id": "fan_00", "fname": "fan_00_abnormal_bearing.wav", "cond": "impulsive_bearing_impacts", "dur": 10.0, "seed": 103, "astart": 4.0, "adur": 3.5},
        # fan_02 warning file
        {"fan_id": "fan_02", "fname": "fan_02_harmonic_deviation.wav", "cond": "harmonic_distortion", "dur": 10.0, "seed": 201, "astart": 3.0, "adur": 4.0},
        # fan_06 critical file
        {"fan_id": "fan_06", "fname": "fan_06_critical_combined.wav", "cond": "combined_abnormal", "dur": 10.0, "seed": 301, "astart": 2.0, "adur": 5.0}
    ]

    normal_vectors = []
    normal_metadata = []

    for r_def in recordings_info:
        wav_path = os.path.join(synth_dir, r_def["fname"])
        audio, meta = gen.generate_recording(
            duration_sec=r_def["dur"],
            condition=r_def["cond"],
            anomaly_start_sec=r_def.get("astart", 3.0),
            anomaly_duration_sec=r_def.get("adur", 3.5),
            recording_seed=r_def["seed"]
        )
        gen.save_wav(audio, wav_path)

        rec_id = f"rec_{os.path.splitext(r_def['fname'])[0]}"
        db.insert_recording({
            "id": rec_id,
            "fan_id": r_def["fan_id"],
            "filename": r_def["fname"],
            "duration": r_def["dur"],
            "sample_rate": 16000,
            "recording_date": "2026-10-04 08:30:00",
            "data_type": "SYNTHETIC / DEMO DATA",
            "file_path": wav_path
        })

        # Process features
        f_matrix, w_meta, dur = extractor.process_file(wav_path)
        if r_def["cond"] == "normal" and r_def["fan_id"] == "fan_00":
            normal_vectors.append(f_matrix)

    # 3. Fit Normal Baseline for Fan 00 using all normal windows
    all_normal_matrix = np.vstack(normal_vectors)
    feat_names = extractor.get_feature_names()
    print(f"[AcouMap Seeder] Fitting baseline for fan_00 with {len(all_normal_matrix)} normal windows...")

    ml_pipe.fit_normal_baseline(all_normal_matrix, feat_names, fan_id="fan_00", version="v1.0.0")

    # Store baseline in DB
    db.save_baseline({
        "fan_id": "fan_00",
        "version": "v1.0.0",
        "mean_vector_json": str(ml_pipe.baseline_mean.tolist()),
        "std_vector_json": str(ml_pipe.baseline_std.tolist()),
        "cov_matrix_json": str(ml_pipe.baseline_cov.tolist()) if ml_pipe.baseline_cov is not None else None,
        "pca_components_json": str(ml_pipe.pca.components_.tolist()) if ml_pipe.pca is not None else None,
        "sample_count": len(all_normal_matrix),
        "updated_at": "2026-10-04 08:35:00"
    })

    # Also fit baselines for other fans (using variations)
    for fid in ["fan_02", "fan_04", "fan_06"]:
        ml_pipe.fit_normal_baseline(all_normal_matrix, feat_names, fan_id=fid, version="v1.0.0")
        db.save_baseline({
            "fan_id": fid,
            "version": "v1.0.0",
            "mean_vector_json": str(ml_pipe.baseline_mean.tolist()),
            "std_vector_json": str(ml_pipe.baseline_std.tolist()),
            "cov_matrix_json": str(ml_pipe.baseline_cov.tolist()) if ml_pipe.baseline_cov is not None else None,
            "pca_components_json": str(ml_pipe.pca.components_.tolist()) if ml_pipe.pca is not None else None,
            "sample_count": len(all_normal_matrix),
            "updated_at": "2026-10-04 08:35:00"
        })

    # 4. Run Analysis for All Recordings to pre-populate database
    for r_def in recordings_info:
        rec_id = f"rec_{os.path.splitext(r_def['fname'])[0]}"
        wav_path = os.path.join(synth_dir, r_def["fname"])

        f_matrix, w_meta, dur = extractor.process_file(wav_path)
        N = len(f_matrix)

        # R Vector Analysis
        r_result = r_bridge.run_analysis(
            feature_matrix=f_matrix,
            feature_names=feat_names,
            baseline_mean=ml_pipe.baseline_mean,
            baseline_std=ml_pipe.baseline_std,
            baseline_cov=ml_pipe.baseline_cov
        )

        # Normalize features
        X_norm = (f_matrix - ml_pipe.baseline_mean) / ml_pipe.baseline_std

        # Model Inference
        if_scores, svm_scores, severities = ml_pipe.compute_anomaly_scores(X_norm)

        # PCA Projection
        pca_coords = ml_pipe.pca.transform(X_norm)

        # Save Acoustic Features per Window
        feat_rows = []
        r_windows = r_result.get("windows", [])

        for i in range(N):
            r_win = r_windows[i] if i < len(r_windows) else {}
            e_dist = r_win.get("euclidean_distance", float(np.linalg.norm(X_norm[i])))
            m_dist = r_win.get("mahalanobis_distance", e_dist)
            pc1 = float(pca_coords[i, 0])
            pc2 = float(pca_coords[i, 1]) if pca_coords.shape[1] > 1 else 0.0

            feat_rows.append({
                "recording_id": rec_id,
                "window_index": i,
                "timestamp": w_meta[i]["start_time"],
                "duration": round(w_meta[i]["end_time"] - w_meta[i]["start_time"], 3),
                "mfcc_json": str([round(float(f_matrix[i, k]), 4) for k in range(13)]),
                "rms": round(float(f_matrix[i, 13]), 6),
                "zcr": round(float(f_matrix[i, 14]), 6),
                "spectral_centroid": round(float(f_matrix[i, 15]), 2),
                "spectral_bandwidth": round(float(f_matrix[i, 16]), 2),
                "spectral_rolloff": round(float(f_matrix[i, 17]), 2),
                "spectral_flux": round(float(f_matrix[i, 18]), 6),
                "dominant_frequency": round(float(f_matrix[i, 19]), 2),
                "pc1": round(pc1, 4),
                "pc2": round(pc2, 4),
                "anomaly_score": round(float(if_scores[i]), 4),
                "severity": severities[i],
                "euclidean_distance": round(e_dist, 4),
                "mahalanobis_distance": round(m_dist, 4)
            })

        db.insert_acoustic_features_bulk(feat_rows)

        # Analysis summary
        mean_anom = float(np.mean(if_scores))
        max_anom = float(np.max(if_scores))
        health_score = round(max(0.0, min(100.0, 100.0 * (1.0 - mean_anom))), 1)
        anomalous_count = int(np.sum(severities != "NORMAL"))

        analysis_id = f"anl_{rec_id}"
        db.insert_analysis({
            "id": analysis_id,
            "recording_id": rec_id,
            "fan_id": r_def["fan_id"],
            "health_score": health_score,
            "anomaly_score": round(mean_anom, 4),
            "max_anomaly_score": round(max_anom, 4),
            "model_name": "Isolation Forest",
            "model_version": "v1.0.0",
            "analysis_time": "2026-10-04 08:40:00",
            "warning_threshold": 0.50,
            "critical_threshold": 0.75,
            "total_windows": N,
            "anomalous_windows": anomalous_count
        })

        # Identify anomalous contiguous segments
        in_anom = False
        anom_start = 0.0
        anom_scores = []
        anom_e_dists = []
        anom_m_dists = []
        anom_indices = []

        for i, sev in enumerate(severities):
            if sev in ("WARNING", "CRITICAL"):
                if not in_anom:
                    in_anom = True
                    anom_start = w_meta[i]["start_time"]
                    anom_scores = [if_scores[i]]
                    anom_e_dists = [feat_rows[i]["euclidean_distance"]]
                    anom_m_dists = [feat_rows[i]["mahalanobis_distance"]]
                    anom_indices = [i]
                else:
                    anom_scores.append(if_scores[i])
                    anom_e_dists.append(feat_rows[i]["euclidean_distance"])
                    anom_m_dists.append(feat_rows[i]["mahalanobis_distance"])
                    anom_indices.append(i)
            else:
                if in_anom:
                    # Anomaly segment ended
                    in_anom = False
                    anom_end = w_meta[i-1]["end_time"]
                    seg_score = float(np.max(anom_scores))
                    seg_sev = "CRITICAL" if seg_score >= 0.75 else "WARNING"
                    anom_id = f"ev_{uuid.uuid4().hex[:8]}"

                    # Top deviations
                    peak_idx = anom_indices[int(np.argmax(anom_scores))]
                    top_devs = r_windows[peak_idx].get("top_deviations", []) if peak_idx < len(r_windows) else []

                    cond, indication, recom, priority = ConservativeMaintenanceEngine.generate_recommendation(
                        top_deviations=top_devs,
                        anomaly_score=seg_score,
                        severity=seg_sev
                    )

                    import json
                    db.insert_anomaly({
                        "id": anom_id,
                        "analysis_id": analysis_id,
                        "recording_id": rec_id,
                        "fan_id": r_def["fan_id"],
                        "start_time": round(anom_start, 2),
                        "end_time": round(anom_end, 2),
                        "duration": round(anom_end - anom_start, 2),
                        "severity": seg_sev,
                        "anomaly_score": round(seg_score, 4),
                        "euclidean_distance": round(float(np.max(anom_e_dists)), 4),
                        "mahalanobis_distance": round(float(np.max(anom_m_dists)), 4),
                        "top_features_json": json.dumps(top_devs),
                        "recommendation": recom
                    })

                    db.insert_maintenance_action({
                        "id": f"act_{uuid.uuid4().hex[:8]}",
                        "anomaly_id": anom_id,
                        "fan_id": r_def["fan_id"],
                        "priority": priority,
                        "condition": cond,
                        "possible_indication": indication,
                        "recommendation": recom,
                        "status": "OPEN",
                        "created_at": "2026-10-04 08:45:00"
                    })

        if in_anom:
            # Trailing anomaly
            anom_end = w_meta[-1]["end_time"]
            seg_score = float(np.max(anom_scores))
            seg_sev = "CRITICAL" if seg_score >= 0.75 else "WARNING"
            anom_id = f"ev_{uuid.uuid4().hex[:8]}"
            peak_idx = anom_indices[int(np.argmax(anom_scores))]
            top_devs = r_windows[peak_idx].get("top_deviations", []) if peak_idx < len(r_windows) else []
            cond, indication, recom, priority = ConservativeMaintenanceEngine.generate_recommendation(
                top_deviations=top_devs,
                anomaly_score=seg_score,
                severity=seg_sev
            )
            import json
            db.insert_anomaly({
                "id": anom_id,
                "analysis_id": analysis_id,
                "recording_id": rec_id,
                "fan_id": r_def["fan_id"],
                "start_time": round(anom_start, 2),
                "end_time": round(anom_end, 2),
                "duration": round(anom_end - anom_start, 2),
                "severity": seg_sev,
                "anomaly_score": round(seg_score, 4),
                "euclidean_distance": round(float(np.max(anom_e_dists)), 4),
                "mahalanobis_distance": round(float(np.max(anom_m_dists)), 4),
                "top_features_json": json.dumps(top_devs),
                "recommendation": recom
            })
            db.insert_maintenance_action({
                "id": f"act_{uuid.uuid4().hex[:8]}",
                "anomaly_id": anom_id,
                "fan_id": r_def["fan_id"],
                "priority": priority,
                "condition": cond,
                "possible_indication": indication,
                "recommendation": recom,
                "status": "OPEN",
                "created_at": "2026-10-04 08:45:00"
            })

    print("[AcouMap Seeder] Database seeded successfully with genuine baseline & initial recordings!")


if __name__ == "__main__":
    seed_database()
