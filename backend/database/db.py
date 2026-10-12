"""
AcouMap - SQLite Database Management
Thread-safe database connection pool and schema management for:
- fans
- recordings
- analyses
- anomalies
- acoustic_features
- maintenance_actions
- baselines
"""

import sqlite3
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.db_path = os.path.join(base_dir, "data", "acoumap.db")
        else:
            self.db_path = db_path

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def init_schema(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Fans
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS fans (
                id TEXT PRIMARY KEY,
                fan_name TEXT NOT NULL,
                machine_type TEXT NOT NULL,
                operating_hours REAL DEFAULT 0.0,
                status TEXT DEFAULT 'NORMAL',
                bpf_nominal REAL DEFAULT 120.0,
                created_at TEXT NOT NULL
            )
            """)

            # 2. Recordings
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS recordings (
                id TEXT PRIMARY KEY,
                fan_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                duration REAL NOT NULL,
                sample_rate INTEGER NOT NULL,
                recording_date TEXT NOT NULL,
                data_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                FOREIGN KEY (fan_id) REFERENCES fans(id) ON DELETE CASCADE
            )
            """)

            # 3. Analyses
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                recording_id TEXT NOT NULL,
                fan_id TEXT NOT NULL,
                health_score REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                max_anomaly_score REAL NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                analysis_time TEXT NOT NULL,
                warning_threshold REAL NOT NULL,
                critical_threshold REAL NOT NULL,
                total_windows INTEGER NOT NULL,
                anomalous_windows INTEGER NOT NULL,
                FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE,
                FOREIGN KEY (fan_id) REFERENCES fans(id) ON DELETE CASCADE
            )
            """)

            # 4. Anomalies
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                fan_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                duration REAL NOT NULL,
                severity TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                euclidean_distance REAL NOT NULL,
                mahalanobis_distance REAL NOT NULL,
                top_features_json TEXT,
                recommendation TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
            )
            """)

            # 5. Acoustic Features per Window
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS acoustic_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recording_id TEXT NOT NULL,
                window_index INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                duration REAL NOT NULL,
                mfcc_json TEXT NOT NULL,
                rms REAL NOT NULL,
                zcr REAL NOT NULL,
                spectral_centroid REAL NOT NULL,
                spectral_bandwidth REAL NOT NULL,
                spectral_rolloff REAL NOT NULL,
                spectral_flux REAL NOT NULL,
                dominant_frequency REAL NOT NULL,
                pc1 REAL NOT NULL,
                pc2 REAL NOT NULL,
                anomaly_score REAL NOT NULL,
                severity TEXT NOT NULL,
                euclidean_distance REAL NOT NULL,
                mahalanobis_distance REAL NOT NULL,
                FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
            )
            """)

            # 6. Maintenance Actions
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_actions (
                id TEXT PRIMARY KEY,
                anomaly_id TEXT,
                fan_id TEXT NOT NULL,
                priority TEXT NOT NULL,
                condition TEXT NOT NULL,
                possible_indication TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                status TEXT DEFAULT 'OPEN',
                created_at TEXT NOT NULL,
                FOREIGN KEY (fan_id) REFERENCES fans(id) ON DELETE CASCADE
            )
            """)

            # 7. Learned Normal Baselines
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS baselines (
                fan_id TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                mean_vector_json TEXT NOT NULL,
                std_vector_json TEXT NOT NULL,
                cov_matrix_json TEXT,
                pca_components_json TEXT,
                sample_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (fan_id) REFERENCES fans(id) ON DELETE CASCADE
            )
            """)

            conn.commit()

    # --- Query & Mutation Helpers ---

    def insert_fan(self, fan_dict: Dict[str, Any]):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO fans (id, fan_name, machine_type, operating_hours, status, bpf_nominal, created_at)
            VALUES (:id, :fan_name, :machine_type, :operating_hours, :status, :bpf_nominal, :created_at)
            """, fan_dict)
            conn.commit()

    def get_all_fans(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM fans ORDER BY id").fetchall()
            return [dict(r) for r in rows]

    def get_fan_by_id(self, fan_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM fans WHERE id = ?", (fan_id,)).fetchone()
            return dict(row) if row else None

    def insert_recording(self, rec: Dict[str, Any]):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO recordings (id, fan_id, filename, duration, sample_rate, recording_date, data_type, file_path)
            VALUES (:id, :fan_id, :filename, :duration, :sample_rate, :recording_date, :data_type, :file_path)
            """, rec)
            conn.commit()

    def get_recordings(self, fan_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            if fan_id:
                rows = conn.execute("SELECT * FROM recordings WHERE fan_id = ? ORDER BY recording_date DESC", (fan_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM recordings ORDER BY recording_date DESC").fetchall()
            return [dict(r) for r in rows]

    def get_recording_by_id(self, recording_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
            return dict(row) if row else None

    def insert_analysis(self, analysis: Dict[str, Any]):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT INTO analyses (
                id, recording_id, fan_id, health_score, anomaly_score, max_anomaly_score,
                model_name, model_version, analysis_time, warning_threshold, critical_threshold,
                total_windows, anomalous_windows
            ) VALUES (
                :id, :recording_id, :fan_id, :health_score, :anomaly_score, :max_anomaly_score,
                :model_name, :model_version, :analysis_time, :warning_threshold, :critical_threshold,
                :total_windows, :anomalous_windows
            )
            """, analysis)
            # Update fan overall status
            new_status = "NORMAL"
            if analysis["max_anomaly_score"] >= analysis["critical_threshold"]:
                new_status = "CRITICAL"
            elif analysis["max_anomaly_score"] >= analysis["warning_threshold"]:
                new_status = "WARNING"

            conn.execute("UPDATE fans SET status = ? WHERE id = ?", (new_status, analysis["fan_id"]))
            conn.commit()

    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
            return dict(row) if row else None

    def get_latest_analysis_for_fan(self, fan_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM analyses WHERE fan_id = ? ORDER BY analysis_time DESC LIMIT 1", (fan_id,)).fetchone()
            return dict(row) if row else None

    def insert_anomaly(self, anomaly: Dict[str, Any]):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT INTO anomalies (
                id, analysis_id, recording_id, fan_id, start_time, end_time, duration,
                severity, anomaly_score, euclidean_distance, mahalanobis_distance,
                top_features_json, recommendation
            ) VALUES (
                :id, :analysis_id, :recording_id, :fan_id, :start_time, :end_time, :duration,
                :severity, :anomaly_score, :euclidean_distance, :mahalanobis_distance,
                :top_features_json, :recommendation
            )
            """, anomaly)
            conn.commit()

    def get_anomalies_for_analysis(self, analysis_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM anomalies WHERE analysis_id = ? ORDER BY start_time ASC", (analysis_id,)).fetchall()
            return [dict(r) for r in rows]

    def insert_acoustic_features_bulk(self, feature_records: List[Dict[str, Any]]):
        with self.get_connection() as conn:
            conn.executemany("""
            INSERT INTO acoustic_features (
                recording_id, window_index, timestamp, duration, mfcc_json,
                rms, zcr, spectral_centroid, spectral_bandwidth, spectral_rolloff,
                spectral_flux, dominant_frequency, pc1, pc2, anomaly_score,
                severity, euclidean_distance, mahalanobis_distance
            ) VALUES (
                :recording_id, :window_index, :timestamp, :duration, :mfcc_json,
                :rms, :zcr, :spectral_centroid, :spectral_bandwidth, :spectral_rolloff,
                :spectral_flux, :dominant_frequency, :pc1, :pc2, :anomaly_score,
                :severity, :euclidean_distance, :mahalanobis_distance
            )
            """, feature_records)
            conn.commit()

    def get_acoustic_features(self, recording_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM acoustic_features WHERE recording_id = ? ORDER BY window_index ASC", (recording_id,)).fetchall()
            return [dict(r) for r in rows]

    def insert_maintenance_action(self, action: Dict[str, Any]):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT INTO maintenance_actions (
                id, anomaly_id, fan_id, priority, condition, possible_indication, recommendation, status, created_at
            ) VALUES (
                :id, :anomaly_id, :fan_id, :priority, :condition, :possible_indication, :recommendation, :status, :created_at
            )
            """, action)
            conn.commit()

    def get_maintenance_actions(self, fan_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            if fan_id:
                rows = conn.execute("SELECT * FROM maintenance_actions WHERE fan_id = ? ORDER BY created_at DESC", (fan_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM maintenance_actions ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def update_maintenance_status(self, action_id: str, new_status: str):
        with self.get_connection() as conn:
            conn.execute("UPDATE maintenance_actions SET status = ? WHERE id = ?", (new_status, action_id))
            conn.commit()

    def save_baseline(self, baseline: Dict[str, Any]):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO baselines (
                fan_id, version, mean_vector_json, std_vector_json, cov_matrix_json, pca_components_json, sample_count, updated_at
            ) VALUES (
                :fan_id, :version, :mean_vector_json, :std_vector_json, :cov_matrix_json, :pca_components_json, :sample_count, :updated_at
            )
            """, baseline)
            conn.commit()

    def get_baseline(self, fan_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM baselines WHERE fan_id = ?", (fan_id,)).fetchone()
            return dict(row) if row else None
