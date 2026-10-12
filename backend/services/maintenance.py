"""
AcouMap - Conservative Maintenance Recommendation Engine
Translates measurable acoustic feature deviations into objective engineering observations
and conservative inspection guidelines without unsubstantiated diagnostic claims.
"""

from typing import Dict, Any, List, Tuple


class ConservativeMaintenanceEngine:
    """
    Rule-based explainable maintenance advisor.
    Generates conservative recommendations grounded in measurable physical acoustic shifts.
    """

    @staticmethod
    def generate_recommendation(top_deviations: List[Dict[str, Any]],
                                anomaly_score: float,
                                severity: str) -> Tuple[str, str, str, str]:
        """
        Returns:
            condition: Objective summary of acoustic condition
            possible_indication: Physical acoustic observation (not definitive fault diagnosis)
            recommendation: Conservative physical inspection checklist
            priority: Priority level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        # Determine priority
        if severity == "CRITICAL" or anomaly_score >= 0.80:
            priority = "CRITICAL"
        elif severity == "WARNING" or anomaly_score >= 0.50:
            priority = "HIGH" if anomaly_score >= 0.65 else "MEDIUM"
        else:
            priority = "LOW"

        # Analyze top deviations
        dev_features = {d["feature"]: d["pct_change"] for d in top_deviations if "feature" in d}

        high_freq_shift = dev_features.get("spectral_centroid", 0.0) > 20.0 or dev_features.get("spectral_rolloff", 0.0) > 20.0
        energy_surge = dev_features.get("rms", 0.0) > 25.0
        tonal_shift = abs(dev_features.get("dominant_frequency", 0.0)) > 15.0
        flux_increase = dev_features.get("spectral_flux", 0.0) > 30.0

        if severity == "NORMAL":
            return (
                "Acoustic behavior within learned baseline envelope",
                "Stable rotational frequency and normal aerodynamic broadband sound",
                "Continue routine scheduled monitoring. No immediate physical intervention required.",
                "LOW"
            )

        condition = "Significant acoustic deviation detected"

        # Compose possible indication based strictly on measurable metrics
        indications = []
        if high_freq_shift:
            centroid_change = dev_features.get('spectral_centroid', 0.0)
            indications.append(f"Increased high-frequency activity (+{round(centroid_change, 1)}% spectral shift)")
        if energy_surge:
            rms_change = dev_features.get('rms', 0.0)
            indications.append(f"Sudden acoustic energy increase (+{round(rms_change, 1)}% RMS)")
        if tonal_shift:
            freq_change = dev_features.get('dominant_frequency', 0.0)
            indications.append(f"Unusual tonal component detected ({'+' if freq_change > 0 else ''}{round(freq_change, 1)}% frequency shift)")
        if flux_increase:
            indications.append("Acoustic behavior differs from baseline with high transient spectral fluctuation")

        if not indications:
            possible_indication = "Multivariate acoustic vector deviation from normal baseline cluster"
        else:
            possible_indication = "; ".join(indications)

        # Conservative recommendations (checklist of physical components to inspect)
        inspections = [
            "1. Inspect fan bearing assemblies for abnormal acoustic friction or lubrication depletion.",
            "2. Check fan blades for physical deformation, dirt accumulation, or casing clearance rubbing.",
            "3. Verify fan motor mounting bolts and vibration isolation dampers for loose hardware.",
            "4. Inspect intake/exhaust ducting for flow obstruction or acoustic resonance.",
            "5. Compare current recording with historical baseline recordings to evaluate trend drift."
        ]

        recommendation = "\n".join(inspections)

        return condition, possible_indication, recommendation, priority
