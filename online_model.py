"""
Online learning model with ADWIN drift detection.
"""

import numpy as np
import pandas as pd
from river import linear_model, drift, compose, preprocessing

class OnlineLearner:
    def __init__(self, model_type="PARegressor"):
        self.model_type = model_type
        self.drift_detector = drift.ADWIN(clock=config.ADWIN_CLOCK, delta=config.ADWIN_DELTA)
        self.model = self._build_model()
        self.scaler = preprocessing.StandardScaler()
        self.is_warm = False
        self.samples_seen = 0
        self.last_drift_at = None

    def _build_model(self):
        if self.model_type == "PARegressor":
            return linear_model.PARegressor(C=1.0, mode=1, eps=0.1)
        else:
            # Hoeffding Tree Regressor fallback
            from river import tree
            return tree.HoeffdingTreeRegressor(grace_period=200)

    def learn_one(self, x: dict, y: float):
        """Update model with one sample; detect drift on prediction error."""
        if not self.is_warm:
            self.scaler.learn_one(x)
            self.samples_seen += 1
            if self.samples_seen >= config.WINDOW_SIZE:
                self.is_warm = True
            return

        x_scaled = self.scaler.transform_one(x)
        y_pred = self.model.predict_one(x_scaled)
        error = abs(y - y_pred)
        self.drift_detector.update(error)

        if self.drift_detector.drift_detected:
            # Reset model on drift
            self.model = self._build_model()
            self.last_drift_at = self.samples_seen

        self.model.learn_one(x_scaled, y)
        self.samples_seen += 1

    def predict_one(self, x: dict) -> float:
        if not self.is_warm:
            return 0.0
        x_scaled = self.scaler.transform_one(x)
        return self.model.predict_one(x_scaled)

    def get_state(self) -> dict:
        return {
            "samples_seen": self.samples_seen,
            "is_warm": self.is_warm,
            "last_drift_at": self.last_drift_at,
            "drift_detected": self.drift_detector.drift_detected,
            "window_width": self.drift_detector.width
        }
