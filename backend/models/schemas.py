"""
AcouMap - Pydantic Data Schemas
Defines request and response schemas for FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class FanSchema(BaseModel):
    id: str
    fan_name: str
    machine_type: str
    operating_hours: float
    status: str
    bpf_nominal: float
    created_at: str


class RecordingSchema(BaseModel):
    id: str
    fan_id: str
    filename: str
    duration: float
    sample_rate: int
    recording_date: str
    data_type: str
    file_path: str


class AnalysisRequest(BaseModel):
    recording_id: Optional[str] = None
    fan_id: str = "fan_00"
    model_name: str = "Isolation Forest"
    warning_threshold: float = 0.50
    critical_threshold: float = 0.75


class TopDeviation(BaseModel):
    feature: str
    pct_change: float
    z_score: float
    current_value: float
    baseline_value: float


class AnomalyEventSchema(BaseModel):
    id: str
    analysis_id: str
    recording_id: str
    fan_id: str
    start_time: float
    end_time: float
    duration: float
    severity: str
    anomaly_score: float
    euclidean_distance: float
    mahalanobis_distance: float
    top_features: List[TopDeviation]
    recommendation: str


class AcousticPoint(BaseModel):
    id: int
    window_index: int
    timestamp: float
    duration: float
    rms: float
    zcr: float
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    spectral_flux: float
    dominant_frequency: float
    pc1: float
    pc2: float
    anomaly_score: float
    severity: str
    euclidean_distance: float
    mahalanobis_distance: float


class AnalysisResponse(BaseModel):
    analysis_id: str
    recording_id: str
    fan_id: str
    health_score: float
    mean_anomaly_score: float
    max_anomaly_score: float
    overall_status: str
    total_windows: int
    anomalous_windows: int
    analysis_time: str
    anomalies: List[AnomalyEventSchema]
    r_vector_summary: Dict[str, Any]


class MaintenanceActionSchema(BaseModel):
    id: str
    anomaly_id: Optional[str] = None
    fan_id: str
    priority: str
    condition: str
    possible_indication: str
    recommendation: str
    status: str
    created_at: str


class DatasetInfoSchema(BaseModel):
    dataset_name: str
    dataset_source: str
    dataset_version: str
    total_recordings: int
    normal_recordings: int
    anomalous_recordings: int
    machine_types: List[str]
    sample_rate: int
    nominal_duration_sec: float
    is_real_data: bool
    status_label: str
    official_url: str
