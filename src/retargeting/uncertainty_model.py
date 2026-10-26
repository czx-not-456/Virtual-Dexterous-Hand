"""Explicit empirical uncertainty proxies for the 15-sensor to 6-control encoder.

Covariances from repeated *pose* captures mix sensor noise and intra-pose motion.
They are not independently calibrated metrology or distribution-free confidence
bounds on a real physical device.  We propagate their units transparently.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def symmetric_psd(matrix: np.ndarray, size: int) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    if x.shape != (size, size) or not np.isfinite(x).all():
        raise ValueError(f'expected finite {size}x{size} covariance')
    if not np.allclose(x, x.T, atol=1e-8):
        raise ValueError('covariance must be symmetric')
    if np.linalg.eigvalsh(x).min() < -1e-7:
        raise ValueError('covariance must be positive semidefinite')
    return (x + x.T) / 2


class EncoderUncertainty:
    def __init__(self, mapper, path: str | Path | None = None,
                 control_sigma_deg: float = 2.0):
        if control_sigma_deg < 0:
            raise ValueError('control_sigma_deg must be >= 0')
        self.mapper = mapper
        self.control_cov = np.diag((control_sigma_deg / mapper.limits_deg)**2)
        self.sensor_cov = np.diag(np.full(15, 2.0**2))
        self.model_cov = np.zeros((6, 6))
        self.source = 'uncalibrated_2deg_diagonal_proxy'
        if path:
            p = Path(path)
            config = json.loads(p.read_text(encoding='utf-8'))
            if config.get('channel_count') != 15 or config.get('control_count') != 6:
                raise ValueError('incompatible uncertainty dimensions')
            self.sensor_cov = symmetric_psd(config['within_pose_sensor_cov_deg2'], 15)
            self.model_cov = symmetric_psd(config['encoder_label_residual_cov_normalized'], 6)
            self.source = str(p)

    def covariance(self, sensors_deg=None, controls_deg=None, source='sensor',
                   blend_weight: float = 0.8) -> np.ndarray:
        if source not in ('sensor', 'control', 'blend'):
            raise ValueError('unknown encoder uncertainty source')
        if source in ('sensor', 'blend'):
            sensors = np.asarray(sensors_deg, float)
            if sensors.shape != (15,) or not np.isfinite(sensors).all():
                raise ValueError('sensor covariance needs fifteen finite sensor angles')
            normalized = (sensors - self.mapper.sensor_min) / self.mapper.sensor_span
            # Saturated channels supply no local derivative, but their clipping
            # also adds unmodelled bias; the residual model remains as a proxy.
            gate = ((normalized > 0) & (normalized < 1)).astype(float)
            jac = (gate / self.mapper.sensor_span)[:, None] * self.mapper.W
            sensor_u = jac.T @ self.sensor_cov @ jac + self.model_cov
        if source == 'sensor':
            return symmetric_psd(sensor_u, 6)
        if source == 'control':
            return self.control_cov.copy()
        if not 0 <= blend_weight <= 1:
            raise ValueError('blend weight must be between 0 and 1')
        # Conservative covariance envelope under *unknown* cross correlation:
        # Cov(w X + (1-w)Y) <= 2[w^2 Cov(X)+(1-w)^2 Cov(Y)] in PSD order.
        cov = 2 * (blend_weight**2 * sensor_u +
                   (1 - blend_weight)**2 * self.control_cov)
        return symmetric_psd(cov, 6)


def fit_uncertainty_proxy(samples, encoder, min_variance_deg2: float = 0.25,
                          residual_floor_normalized: float = 0.001):
    """Estimate *proxy*, not true stochastic noise, from repeated labeled poses.

    Fits within-group covariance without pooling between different poses.
    Residual covariance is evaluated on the SAME capture used to train the
    existing encoder (optimistic, NOT held out); uses only 6D teacher labels.
    """
    groups = {}
    rows = [r for r in samples if r.targets_deg is not None]
    for r in rows:
        groups.setdefault((r.pose, r.pose_sequence_index), []).append(r)
    if len(groups) < 2:
        raise ValueError('need repeated examples for at least two pose groups')
    residual = []
    sensor_centered = []
    for group in groups.values():
        if len(group) < 3:
            continue
        s = np.stack([r.sensors_deg for r in group])
        sensor_centered.extend(s - s.mean(axis=0))
        for r in group:
            u = encoder.controls_from_observation(r.sensors_deg, r.targets_deg,
                                                  source='sensor')
            residual.append(u - np.clip(r.targets_deg / encoder.limits_deg, 0., 1.))
    if len(residual) < 12:
        raise ValueError('not enough repeated rows')
    sc = np.cov(np.stack(sensor_centered), rowvar=False)
    rc = np.cov(np.stack(residual), rowvar=False)
    # shrinkage ensures PSD even when some channels are constant or collinear
    sc = .8 * sc + .2 * np.diag(np.diag(sc)) + np.eye(15) * min_variance_deg2
    rc = .8 * rc + .2 * np.diag(np.diag(rc)) + np.eye(6) * residual_floor_normalized**2
    return {
        'version': 1, 'channel_count': 15, 'control_count': 6,
        'source_rows': len(rows), 'pose_groups': len(groups),
        'within_pose_sensor_cov_deg2': sc.tolist(),
        'encoder_label_residual_cov_normalized': rc.tolist(),
        'method': 'within-pose centered variability and supplied-label residuals; NOT independent noise calibration',
        'assumptions': 'proxy covariances; independence, stationarity and coverage unverified',
    }
