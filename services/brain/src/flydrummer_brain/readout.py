"""Train a small multi-label linear readout from MaleCNS descending activity."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .score import ScoreFragment


INSTRUMENTS = (
    "kick",
    "snare",
    "closed_hat",
    "open_hat",
    "crash",
    "ride",
    "tom",
)
READOUT_VERSION = 2
DEFAULT_WINDOW_MS = 10.0
DEFAULT_LATENCY_MS = 5.0
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class NeuralTrace:
    timestamps_ms: np.ndarray
    rates_hz: np.ndarray
    descending_body_ids: tuple[int, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrainingPair:
    trace: NeuralTrace
    score: ScoreFragment


@dataclass(frozen=True)
class ReadoutScaler:
    mean: np.ndarray
    scale: np.ndarray
    feature_ids_sha256: str
    artifact_pair_sha256: str | None = None


@dataclass(frozen=True)
class ReadoutModel:
    coefficients: np.ndarray
    intercepts: np.ndarray
    thresholds: np.ndarray
    window_ms: float
    latency_ms: float
    feature_ids_sha256: str
    class_support: dict[str, int]
    training_provenance: tuple[dict[str, Any], ...]
    artifact_pair_sha256: str | None = None


@dataclass(frozen=True)
class DrumCommand:
    time_seconds: float
    instrument: str
    probability: float


def feature_ids_sha256(body_ids: Sequence[int]) -> str:
    """Hash the ordered descending IDs used as the readout's feature columns."""
    return hashlib.sha256(np.asarray(body_ids, dtype="<i8").tobytes()).hexdigest()


def _trace_content_sha256(
    timestamps_ms: np.ndarray,
    rates_hz: np.ndarray,
    descending_body_ids: Sequence[int],
) -> str:
    """Hash the canonical arrays used by the persisted descending trace."""
    digest = hashlib.sha256()
    digest.update(np.asarray(descending_body_ids, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(timestamps_ms, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(rates_hz, dtype="<f4").tobytes())
    return digest.hexdigest()


def _validate_trace(trace: NeuralTrace) -> float:
    timestamps = np.asarray(trace.timestamps_ms, dtype=np.float64)
    rates = np.asarray(trace.rates_hz)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("trace timestamps must contain at least two samples")
    if rates.ndim != 2 or rates.shape != (timestamps.size, len(trace.descending_body_ids)):
        raise ValueError("trace rate shape must match timestamps and descending body IDs")
    if not np.isfinite(timestamps).all() or not np.isfinite(rates).all():
        raise ValueError("trace timestamps and rates must be finite")
    differences = np.diff(timestamps)
    if np.any(differences <= 0):
        raise ValueError("trace timestamps must be strictly increasing")
    dt_ms = float(trace.metadata.get("dt_ms", np.median(differences)))
    if not math.isfinite(dt_ms) or dt_ms <= 0 or not np.allclose(
        differences, dt_ms, rtol=0.0, atol=1e-6
    ):
        raise ValueError("trace timestamps must use one fixed simulator step")
    if len(set(trace.descending_body_ids)) != len(trace.descending_body_ids):
        raise ValueError("descending body IDs must be unique")
    return dt_ms


def load_neural_trace(path: str | Path) -> NeuralTrace:
    """Load only the time-stamped descending activity and its provenance."""
    trace_path = Path(path).expanduser().resolve()
    with np.load(trace_path, allow_pickle=False) as archive:
        timestamps = np.asarray(archive["timestamps_ms"], dtype=np.float64)
        rates = np.asarray(archive["descending_rates_hz"], dtype=np.float32)
        body_ids = tuple(int(value) for value in archive["descending_body_ids"])
        metadata = json.loads(str(archive["metadata_json"]))
    trace = NeuralTrace(
        timestamps_ms=timestamps,
        rates_hz=rates,
        descending_body_ids=body_ids,
        metadata=metadata,
    )
    _validate_trace(trace)
    if metadata.get("dataset_id") != "male-cns:v1.0":
        raise ValueError("readout requires a real male-cns:v1.0 descending trace")
    stored_digest = metadata.get("trace_sha256")
    actual_digest = _trace_content_sha256(timestamps, rates, body_ids)
    if not isinstance(stored_digest, str) or stored_digest != actual_digest:
        raise ValueError("trace content does not match its SHA-256")
    if not np.count_nonzero(rates):
        raise ValueError("descending trace contains no neural activity")
    return trace


def _window_features(
    trace: NeuralTrace, *, window_ms: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt_ms = _validate_trace(trace)
    if not math.isfinite(window_ms) or window_ms <= 0:
        raise ValueError("window_ms must be positive and finite")
    samples_per_window = int(round(window_ms / dt_ms))
    if samples_per_window < 1 or not math.isclose(
        samples_per_window * dt_ms, window_ms, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("window_ms must be an integer multiple of the simulator step")

    rates = np.asarray(trace.rates_hz, dtype=np.float32)
    timestamps = np.asarray(trace.timestamps_ms, dtype=np.float64)
    windows: list[np.ndarray] = []
    centers: list[float] = []
    starts: list[float] = []
    ends: list[float] = []
    for start in range(0, len(timestamps), samples_per_window):
        stop = min(start + samples_per_window, len(timestamps))
        windows.append(rates[start:stop].mean(axis=0, dtype=np.float64))
        centers.append(float(timestamps[start:stop].mean()))
        starts.append(float(timestamps[start]))
        ends.append(float(timestamps[stop - 1]))
    return (
        np.asarray(windows, dtype=np.float64),
        np.asarray(centers, dtype=np.float64),
        np.asarray(starts, dtype=np.float64),
        np.asarray(ends, dtype=np.float64),
    )


def _training_labels(
    score: ScoreFragment, centers_ms: np.ndarray, *, window_ms: float
) -> tuple[np.ndarray, dict[str, int]]:
    labels = np.zeros((len(centers_ms), len(INSTRUMENTS)), dtype=np.uint8)
    instrument_indices = {instrument: index for index, instrument in enumerate(INSTRUMENTS)}
    support = {instrument: 0 for instrument in INSTRUMENTS}
    for event in score.events:
        if event.instrument not in instrument_indices:
            raise ValueError(f"unsupported drum label {event.instrument!r}")
        if not math.isfinite(event.time_seconds):
            raise ValueError("score event time must be finite")
        support[event.instrument] += 1
        event_ms = event.time_seconds * 1000.0
        # Each target uses only the first causal window after the sensory onset.
        selected = (centers_ms >= event_ms) & (centers_ms < event_ms + window_ms)
        labels[selected, instrument_indices[event.instrument]] = 1
    return labels, support


def _power_iteration_lipschitz(
    features: np.ndarray, sample_weights: np.ndarray, *, regularization: float
) -> float:
    vector = np.full(features.shape[1], 1.0 / math.sqrt(features.shape[1]), dtype=np.float64)
    for _ in range(16):
        projected = features @ vector
        vector = features.T @ (sample_weights * projected)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-15:
            return regularization + 1.0
        vector /= norm
    eigenvalue = float(np.dot(features @ vector, sample_weights * (features @ vector)))
    bound = 0.25 * (eigenvalue + float(sample_weights.sum())) / float(sample_weights.sum())
    return max(bound + regularization, 1e-6)


def _fit_binary_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int = 300,
    regularization: float = 0.01,
) -> tuple[np.ndarray, float]:
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        raise ValueError("every readout class needs positive and negative training windows")
    weights = np.where(
        labels != 0,
        len(labels) / (2.0 * positive),
        len(labels) / (2.0 * negative),
    ).astype(np.float64)
    total_weight = float(weights.sum())
    step_size = 1.0 / _power_iteration_lipschitz(
        features, weights, regularization=regularization
    )
    coefficients = np.zeros(features.shape[1], dtype=np.float64)
    intercept = 0.0

    for _ in range(iterations):
        logits = np.clip(features @ coefficients + intercept, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        residual = (probabilities - labels) * weights
        coefficient_gradient = (features.T @ residual) / total_weight
        coefficient_gradient += regularization * coefficients
        intercept_gradient = float(residual.sum() / total_weight)
        coefficients -= step_size * coefficient_gradient
        intercept -= step_size * intercept_gradient
        if max(
            float(np.max(np.abs(coefficient_gradient))), abs(intercept_gradient)
        ) < 1e-6:
            break
    return coefficients, intercept


def fit_readout(
    training_pairs: Sequence[TrainingPair],
    *,
    window_ms: float = DEFAULT_WINDOW_MS,
    latency_ms: float = DEFAULT_LATENCY_MS,
) -> tuple[ReadoutModel, ReadoutScaler]:
    """Fit all seven binary outputs using only explicitly supplied score fragments."""
    if not training_pairs:
        raise ValueError("at least one labeled training source is required")
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError("latency_ms must be non-negative and finite")

    feature_hash: str | None = None
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    provenance: list[dict[str, Any]] = []
    class_support = {instrument: 0 for instrument in INSTRUMENTS}

    for pair in training_pairs:
        trace = pair.trace
        score = pair.score
        if trace.metadata.get("dataset_id") != "male-cns:v1.0":
            raise ValueError("training requires real male-cns:v1.0 descending activity")
        if score.source_sha256 != trace.metadata.get("source_score_sha256"):
            raise ValueError("training score hash does not match its sensory trace")
        ids_hash = feature_ids_sha256(trace.descending_body_ids)
        if feature_hash is None:
            feature_hash = ids_hash
        elif feature_hash != ids_hash:
            raise ValueError("training traces use different descending feature IDs or order")

        features, centers_ms, window_starts_ms, window_ends_ms = _window_features(
            trace, window_ms=window_ms
        )
        start_ms = score.start_time_seconds * 1000.0
        end_ms = score.end_time_seconds * 1000.0
        if end_ms <= start_ms or start_ms < centers_ms[0] - window_ms:
            raise ValueError("training score fragment has an invalid time range")
        if end_ms > trace.timestamps_ms[-1] + trace.metadata.get("dt_ms", 0.5) + 1e-6:
            raise ValueError("training score fragment extends beyond its neural trace")
        # A training example is eligible only when every sample in its feature
        # window lies inside the half-open training interval. In particular,
        # the window crossing into the contiguous validation segment is dropped.
        selected = (window_starts_ms >= start_ms) & (window_ends_ms < end_ms)
        if not selected.any():
            raise ValueError("training score fragment contains no neural windows")
        labels, source_support = _training_labels(
            score, centers_ms, window_ms=window_ms
        )
        features = features[selected]
        labels = labels[selected]
        if not np.any(features):
            raise ValueError("training segment contains no descending activity")
        feature_parts.append(features)
        label_parts.append(labels)
        for instrument in INSTRUMENTS:
            class_support[instrument] += source_support[instrument]
        provenance.append(
            {
                "source_filename": score.source_filename,
                "source_score_sha256": score.source_sha256,
                "trace_sha256": trace.metadata.get("trace_sha256"),
                "dataset_id": trace.metadata.get("dataset_id"),
                "simulator_repository": trace.metadata.get("simulator_repository"),
                "simulator_commit": trace.metadata.get("simulator_commit"),
                "seed": trace.metadata.get("seed"),
                "start_bar": score.start_bar,
                "bars": score.bars,
                "train_start_ms": float(start_ms),
                "train_end_ms": float(end_ms),
                "label_support": source_support,
            }
        )

    missing = [instrument for instrument, count in class_support.items() if count == 0]
    if missing:
        raise ValueError(
            "positive training support is missing for required instrument(s): "
            + ", ".join(missing)
        )
    assert feature_hash is not None

    training_features = np.concatenate(feature_parts, axis=0)
    training_labels = np.concatenate(label_parts, axis=0)
    mean = training_features.mean(axis=0, dtype=np.float64)
    scale = training_features.std(axis=0, dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    scaled_features = (training_features - mean) / scale
    if not np.isfinite(scaled_features).all():
        raise ValueError("scaled training activity contains non-finite values")

    coefficients = np.empty((len(INSTRUMENTS), scaled_features.shape[1]), dtype=np.float64)
    intercepts = np.empty(len(INSTRUMENTS), dtype=np.float64)
    for index, instrument in enumerate(INSTRUMENTS):
        coefficients[index], intercepts[index] = _fit_binary_logistic(
            scaled_features, training_labels[:, index]
        )

    model = ReadoutModel(
        coefficients=coefficients,
        intercepts=intercepts,
        thresholds=np.full(len(INSTRUMENTS), DEFAULT_THRESHOLD, dtype=np.float64),
        window_ms=window_ms,
        latency_ms=latency_ms,
        feature_ids_sha256=feature_hash,
        class_support=class_support,
        training_provenance=tuple(provenance),
    )
    scaler = ReadoutScaler(mean=mean, scale=scale, feature_ids_sha256=feature_hash)
    return model, scaler


def predict_events(
    trace: NeuralTrace,
    model: ReadoutModel,
    scaler: ReadoutScaler,
) -> tuple[DrumCommand, ...]:
    """Decode neural activity only; no score, MIDI or target labels are accepted."""
    _validate_trace(trace)
    current_hash = feature_ids_sha256(trace.descending_body_ids)
    if current_hash != model.feature_ids_sha256 or current_hash != scaler.feature_ids_sha256:
        raise ValueError("readout model/scaler descending feature IDs do not match the trace")
    features, centers_ms, _, _ = _window_features(trace, window_ms=model.window_ms)
    _validate_artifact_pair_identity(model, scaler)
    if features.shape[1] != model.coefficients.shape[1] or features.shape[1] != scaler.mean.size:
        raise ValueError("readout model/scaler feature count does not match the trace")
    scaled = (features - scaler.mean) / scaler.scale
    logits = np.clip(scaled @ model.coefficients.T + model.intercepts, -40.0, 40.0)
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    commands: list[DrumCommand] = []
    for index, instrument in enumerate(INSTRUMENTS):
        active = probabilities[:, index] >= model.thresholds[index]
        onsets = np.flatnonzero(active & np.r_[True, ~active[:-1]])
        for onset in onsets:
            command_time_ms = max(0.0, float(centers_ms[onset]) - model.latency_ms)
            commands.append(
                DrumCommand(
                    time_seconds=command_time_ms / 1000.0,
                    instrument=instrument,
                    probability=float(probabilities[onset, index]),
                )
            )
    commands.sort(key=lambda command: (command.time_seconds, INSTRUMENTS.index(command.instrument)))
    return tuple(commands)


def save_readout_artifacts(
    model_path: str | Path,
    scaler_path: str | Path,
    model: ReadoutModel,
    scaler: ReadoutScaler,
) -> None:
    """Save small readout and scaler files separately from the connectome."""
    model_path = Path(model_path).expanduser().resolve()
    scaler_path = Path(scaler_path).expanduser().resolve()
    if model_path == scaler_path:
        raise ValueError("model and scaler must be saved to separate files")
    if model.feature_ids_sha256 != scaler.feature_ids_sha256:
        raise ValueError("model and scaler feature IDs do not match")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    model_metadata = {
        "version": READOUT_VERSION,
        "instruments": list(INSTRUMENTS),
        "window_ms": model.window_ms,
        "latency_ms": model.latency_ms,
        "feature_ids_sha256": model.feature_ids_sha256,
        "class_support": model.class_support,
        "training_provenance": list(model.training_provenance),
    }
    scaler_metadata = {
        "version": READOUT_VERSION,
        "feature_ids_sha256": scaler.feature_ids_sha256,
        "fitted_on": "training descending-activity windows only",
    }
    pair_sha256 = _readout_artifact_pair_sha256(
        model.coefficients,
        model.intercepts,
        model.thresholds,
        model_metadata,
        scaler.mean,
        scaler.scale,
        scaler_metadata,
    )
    for existing_pair_sha256 in (
        model.artifact_pair_sha256,
        scaler.artifact_pair_sha256,
    ):
        if existing_pair_sha256 is not None and existing_pair_sha256 != pair_sha256:
            raise ValueError("readout model and scaler pair identity does not match contents")
    model_metadata["artifact_pair_sha256"] = pair_sha256
    scaler_metadata["artifact_pair_sha256"] = pair_sha256
    np.savez_compressed(
        model_path,
        coefficients=np.asarray(model.coefficients, dtype="<f8"),
        intercepts=np.asarray(model.intercepts, dtype="<f8"),
        thresholds=np.asarray(model.thresholds, dtype="<f8"),
        metadata_json=np.asarray(json.dumps(model_metadata, sort_keys=True)),
    )
    np.savez_compressed(
        scaler_path,
        mean=np.asarray(scaler.mean, dtype="<f8"),
        scale=np.asarray(scaler.scale, dtype="<f8"),
        metadata_json=np.asarray(json.dumps(scaler_metadata, sort_keys=True)),
    )


def load_readout_artifacts(
    model_path: str | Path, scaler_path: str | Path
) -> tuple[ReadoutModel, ReadoutScaler]:
    model_path = Path(model_path).expanduser().resolve()
    scaler_path = Path(scaler_path).expanduser().resolve()
    with np.load(model_path, allow_pickle=False) as archive:
        coefficients = np.asarray(archive["coefficients"], dtype=np.float64)
        intercepts = np.asarray(archive["intercepts"], dtype=np.float64)
        thresholds = np.asarray(archive["thresholds"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata_json"]))
    with np.load(scaler_path, allow_pickle=False) as archive:
        mean = np.asarray(archive["mean"], dtype=np.float64)
        scale = np.asarray(archive["scale"], dtype=np.float64)
        scaler_metadata = json.loads(str(archive["metadata_json"]))
    if metadata.get("version") != READOUT_VERSION or metadata.get("instruments") != list(INSTRUMENTS):
        raise ValueError("unsupported or incompatible readout model artifact")
    if scaler_metadata.get("version") != READOUT_VERSION:
        raise ValueError("unsupported readout scaler artifact")
    feature_hash = metadata.get("feature_ids_sha256")
    if feature_hash != scaler_metadata.get("feature_ids_sha256"):
        raise ValueError("readout model and scaler artifacts do not match")
    model_pair_sha256 = metadata.pop("artifact_pair_sha256", None)
    scaler_pair_sha256 = scaler_metadata.pop("artifact_pair_sha256", None)
    actual_pair_sha256 = _readout_artifact_pair_sha256(
        coefficients,
        intercepts,
        thresholds,
        metadata,
        mean,
        scale,
        scaler_metadata,
    )
    if (
        not isinstance(model_pair_sha256, str)
        or model_pair_sha256 != scaler_pair_sha256
        or model_pair_sha256 != actual_pair_sha256
    ):
        raise ValueError("readout model and scaler pair identity does not match contents")
    model = ReadoutModel(
        coefficients=coefficients,
        intercepts=intercepts,
        thresholds=thresholds,
        window_ms=float(metadata["window_ms"]),
        latency_ms=float(metadata["latency_ms"]),
        feature_ids_sha256=feature_hash,
        class_support={key: int(value) for key, value in metadata["class_support"].items()},
        training_provenance=tuple(metadata["training_provenance"]),
        artifact_pair_sha256=actual_pair_sha256,
    )
    scaler = ReadoutScaler(
        mean=mean,
        scale=scale,
        feature_ids_sha256=feature_hash,
        artifact_pair_sha256=actual_pair_sha256,
    )
    if (
        model.coefficients.shape != (len(INSTRUMENTS), scaler.mean.size)
        or model.intercepts.shape != (len(INSTRUMENTS),)
        or model.thresholds.shape != (len(INSTRUMENTS),)
        or scaler.scale.shape != scaler.mean.shape
        or np.any(scaler.scale <= 0)
    ):
        raise ValueError("readout artifact arrays have incompatible shapes or values")
    return model, scaler


def _readout_artifact_pair_sha256(
    coefficients: np.ndarray,
    intercepts: np.ndarray,
    thresholds: np.ndarray,
    model_metadata: dict[str, Any],
    mean: np.ndarray,
    scale: np.ndarray,
    scaler_metadata: dict[str, Any],
) -> str:
    """Return a stable identity for the exact persisted model/scaler contents."""
    digest = hashlib.sha256()
    digest.update(b"flydrummer-readout-pair-v1\0")
    metadata_payload = json.dumps(
        {"model": model_metadata, "scaler": scaler_metadata},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest.update(metadata_payload)
    for name, value in (
        ("coefficients", coefficients),
        ("intercepts", intercepts),
        ("thresholds", thresholds),
        ("mean", mean),
        ("scale", scale),
    ):
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_artifact_pair_identity(
    model: ReadoutModel, scaler: ReadoutScaler
) -> None:
    model_pair_sha256 = model.artifact_pair_sha256
    scaler_pair_sha256 = scaler.artifact_pair_sha256
    if (model_pair_sha256 is None) != (scaler_pair_sha256 is None) or (
        model_pair_sha256 is not None and model_pair_sha256 != scaler_pair_sha256
    ):
        raise ValueError("readout model and scaler pair identity does not match")
