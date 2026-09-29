"""Evaluate readout events on a continuous, previously held-out score segment."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from .readout import (
    INSTRUMENTS,
    DrumCommand,
    NeuralTrace,
    ReadoutModel,
    ReadoutScaler,
    _validate_trace,
    _window_features,
    load_neural_trace,
    load_readout_artifacts,
    predict_events,
)
from .score import ScoreFragment, import_midi


_METRIC_TOLERANCES_MS = (50.0, 100.0)
_SPLIT_EPSILON_MS = 1e-6


def _best_state(
    left: tuple[int, float], right: tuple[int, float]
) -> tuple[int, float]:
    if right[0] > left[0] or (
        right[0] == left[0] and right[1] < left[1] - 1e-9
    ):
        return right
    return left


def _match_times(
    truth_ms: Sequence[float], predicted_ms: Sequence[float], tolerance_ms: float
) -> tuple[tuple[float, float], ...]:
    """Find maximum-cardinality, minimum-error non-crossing event matches."""
    if not math.isfinite(tolerance_ms) or tolerance_ms < 0:
        raise ValueError("tolerance_ms must be non-negative and finite")
    truth = sorted(float(value) for value in truth_ms)
    predicted = sorted(float(value) for value in predicted_ms)
    if any(not math.isfinite(value) for value in (*truth, *predicted)):
        raise ValueError("event times must be finite")

    rows, columns = len(truth), len(predicted)
    counts = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    errors = np.zeros((rows + 1, columns + 1), dtype=np.float64)
    actions = np.zeros((rows + 1, columns + 1), dtype=np.uint8)
    skip_truth, skip_prediction, match = 1, 2, 3

    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            best = (int(counts[i - 1, j]), float(errors[i - 1, j]))
            action = skip_truth
            candidate = (int(counts[i, j - 1]), float(errors[i, j - 1]))
            selected = _best_state(best, candidate)
            if selected[0] > best[0] or selected[1] < best[1] - 1e-9:
                best, action = candidate, skip_prediction

            difference = abs(truth[i - 1] - predicted[j - 1])
            if difference <= tolerance_ms:
                candidate = (
                    int(counts[i - 1, j - 1]) + 1,
                    float(errors[i - 1, j - 1]) + difference,
                )
                selected = _best_state(best, candidate)
                if selected[0] > best[0] or selected[1] < best[1] - 1e-9:
                    best, action = candidate, match

            counts[i, j], errors[i, j] = best
            actions[i, j] = action

    pairs: list[tuple[float, float]] = []
    i, j = rows, columns
    while i and j:
        action = actions[i, j]
        if action == match:
            pairs.append((truth[i - 1], predicted[j - 1]))
            i -= 1
            j -= 1
        elif action == skip_truth:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return tuple(pairs)


def evaluate_predictions(
    score: ScoreFragment,
    predictions: Sequence[DrumCommand],
    *,
    tolerance_ms: float,
) -> dict[str, Any]:
    """Calculate event metrics; score labels are read only in this scorer."""
    start_ms = score.start_time_seconds * 1000.0
    end_ms = score.end_time_seconds * 1000.0
    if not math.isfinite(start_ms) or not math.isfinite(end_ms) or end_ms <= start_ms:
        raise ValueError("held-out score fragment has an invalid time range")
    unsupported = sorted(
        {event.instrument for event in score.events} - set(INSTRUMENTS)
    )
    if unsupported:
        raise ValueError("unsupported held-out drum label(s): " + ", ".join(unsupported))

    instruments: dict[str, dict[str, Any]] = {}
    supported_f1: list[float] = []
    all_errors: list[float] = []
    total_tp = total_fp = total_fn = total_support = 0
    for instrument in INSTRUMENTS:
        truth_ms = sorted(
            event.time_seconds * 1000.0
            for event in score.events
            if event.instrument == instrument
        )
        predicted_ms = sorted(
            command.time_seconds * 1000.0
            for command in predictions
            if command.instrument == instrument
            and start_ms <= command.time_seconds * 1000.0 < end_ms
        )
        pairs = _match_times(truth_ms, predicted_ms, tolerance_ms)
        true_positive = len(pairs)
        false_positive = len(predicted_ms) - true_positive
        false_negative = len(truth_ms) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = (2.0 * true_positive / denominator) if denominator else None
        errors = [abs(actual - predicted) for actual, predicted in pairs]
        timing_mae = float(np.mean(errors)) if errors else None
        instruments[instrument] = {
            "support": len(truth_ms),
            "predicted": len(predicted_ms),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "f1": f1,
            "matched_pairs": true_positive,
            "timing_mae_ms": timing_mae,
        }
        if truth_ms:
            assert f1 is not None
            supported_f1.append(f1)
        all_errors.extend(errors)
        total_tp += true_positive
        total_fp += false_positive
        total_fn += false_negative
        total_support += len(truth_ms)

    macro_f1 = float(np.mean(supported_f1)) if supported_f1 else None
    return {
        "tolerance_ms": tolerance_ms,
        "support": total_support,
        "predicted": sum(item["predicted"] for item in instruments.values()),
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "macro_f1": macro_f1,
        "macro_supported_instruments": len(supported_f1),
        "timing_mae_ms": float(np.mean(all_errors)) if all_errors else None,
        "matched_pairs": len(all_errors),
        "instruments": instruments,
    }


def validate_evaluation_split(model: ReadoutModel, score: ScoreFragment) -> None:
    """Reject models whose recorded training interval intersects the holdout."""
    if not model.training_provenance:
        raise ValueError("readout has no auditable training provenance")
    holdout_start = score.start_time_seconds * 1000.0
    holdout_end = score.end_time_seconds * 1000.0
    if not math.isfinite(holdout_start) or not math.isfinite(holdout_end) or holdout_end <= holdout_start:
        raise ValueError("held-out score fragment has an invalid time range")

    for source in model.training_provenance:
        source_hash = source.get("source_score_sha256")
        train_start = source.get("train_start_ms")
        train_end = source.get("train_end_ms")
        if (
            not isinstance(source_hash, str)
            or not isinstance(train_start, (int, float))
            or not isinstance(train_end, (int, float))
            or not math.isfinite(float(train_start))
            or not math.isfinite(float(train_end))
            or float(train_end) <= float(train_start)
        ):
            raise ValueError("readout training provenance lacks a valid score interval")
        if source_hash != score.source_sha256:
            continue
        overlaps = (
            float(train_start) < holdout_end - _SPLIT_EPSILON_MS
            and float(train_end) > holdout_start + _SPLIT_EPSILON_MS
        )
        if overlaps:
            raise ValueError("readout training provenance overlaps held-out score interval")


def ablate_trace(
    trace: NeuralTrace,
    *,
    start_seconds: float,
    end_seconds: float,
    window_ms: float,
    mode: str,
    seed: int = 7,
) -> NeuralTrace:
    """Zero or time-shuffle neural windows only inside the evaluation interval."""
    dt_ms = _validate_trace(trace)
    if (
        not math.isfinite(start_seconds)
        or not math.isfinite(end_seconds)
        or end_seconds <= start_seconds
    ):
        raise ValueError("ablation interval must be finite and non-empty")
    if mode not in {"zero", "shuffle"}:
        raise ValueError("ablation mode must be 'zero' or 'shuffle'")
    if not math.isfinite(window_ms) or window_ms <= 0:
        raise ValueError("window_ms must be positive and finite")
    samples_per_window = int(round(window_ms / dt_ms))
    if samples_per_window < 1 or not math.isclose(
        samples_per_window * dt_ms, window_ms, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError("window_ms must be an integer multiple of the simulator step")

    rates = np.asarray(trace.rates_hz, dtype=np.float32).copy()
    timestamps = np.asarray(trace.timestamps_ms, dtype=np.float64)
    start_ms, end_ms = start_seconds * 1000.0, end_seconds * 1000.0
    if mode == "zero":
        inside = (timestamps >= start_ms) & (timestamps < end_ms)
        if not inside.any():
            raise ValueError("ablation interval contains no neural samples")
        rates[inside] = 0.0
    else:
        _, centers, _, _ = _window_features(trace, window_ms=window_ms)
        starts = np.arange(len(centers)) * samples_per_window
        stops = np.minimum(starts + samples_per_window, len(timestamps))
        starts_ms = timestamps[starts]
        ends_ms = timestamps[stops - 1]
        selected_windows = np.flatnonzero(
            (centers >= start_ms)
            & (centers < end_ms)
            & (starts_ms >= start_ms)
            & (ends_ms < end_ms)
        )
        if len(selected_windows) < 2:
            raise ValueError("shuffle interval must contain at least two readout windows")
        chunks = [
            rates[index * samples_per_window : min((index + 1) * samples_per_window, len(rates))].copy()
            for index in selected_windows
        ]
        order = np.random.default_rng(seed).permutation(len(chunks))
        for index, shuffled_index in zip(selected_windows, order, strict=True):
            begin = index * samples_per_window
            stop = min(begin + samples_per_window, len(rates))
            rates[begin:stop] = chunks[int(shuffled_index)][: stop - begin]

    metadata = dict(trace.metadata)
    metadata["ablation"] = mode
    metadata["parent_trace_sha256"] = trace.metadata.get("trace_sha256")
    return NeuralTrace(
        timestamps_ms=trace.timestamps_ms,
        rates_hz=rates,
        descending_body_ids=trace.descending_body_ids,
        metadata=metadata,
    )


def _commands_in_holdout(
    commands: Sequence[DrumCommand], score: ScoreFragment
) -> tuple[DrumCommand, ...]:
    return tuple(
        command
        for command in commands
        if score.start_time_seconds <= command.time_seconds < score.end_time_seconds
    )


def _event_records(commands: Sequence[DrumCommand]) -> list[dict[str, Any]]:
    return [
        {
            "time_seconds": round(command.time_seconds, 6),
            "instrument": command.instrument,
            "probability": round(command.probability, 8),
        }
        for command in commands
    ]


def _signature(commands: Sequence[DrumCommand]) -> tuple[tuple[str, float], ...]:
    return tuple(
        (command.instrument, round(command.time_seconds * 1000.0, 6))
        for command in commands
    )


def evaluate_holdout(
    trace: NeuralTrace,
    score: ScoreFragment,
    model: ReadoutModel,
    scaler: ReadoutScaler,
    *,
    shuffle_seed: int = 7,
) -> dict[str, Any]:
    """Compare neural predictions, causal ablations and a no-brain baseline."""
    if score.source_sha256 != trace.metadata.get("source_score_sha256"):
        raise ValueError("held-out score hash does not match the neural trace source")
    validate_evaluation_split(model, score)

    # Labels enter only after inference, in evaluate_predictions below.
    brain = _commands_in_holdout(predict_events(trace, model, scaler), score)
    zero_trace = ablate_trace(
        trace,
        start_seconds=score.start_time_seconds,
        end_seconds=score.end_time_seconds,
        window_ms=model.window_ms,
        mode="zero",
    )
    zero = _commands_in_holdout(predict_events(zero_trace, model, scaler), score)
    shuffled_trace = ablate_trace(
        trace,
        start_seconds=score.start_time_seconds,
        end_seconds=score.end_time_seconds,
        window_ms=model.window_ms,
        mode="shuffle",
        seed=shuffle_seed,
    )
    shuffled = _commands_in_holdout(predict_events(shuffled_trace, model, scaler), score)
    baseline: tuple[DrumCommand, ...] = ()
    conditions = {
        "brain": brain,
        "zero_ablation": zero,
        "shuffled_activity": shuffled,
        "no_brain_empty_output_baseline": baseline,
    }
    metrics: dict[str, dict[str, Any]] = {}
    for condition, commands in conditions.items():
        metrics[condition] = {
            f"{int(tolerance)}ms": evaluate_predictions(
                score, commands, tolerance_ms=tolerance
            )
            for tolerance in _METRIC_TOLERANCES_MS
        }

    brain_f1 = metrics["brain"]["50ms"]["macro_f1"]
    causal_results: dict[str, dict[str, Any]] = {}
    for name, commands in (("zero_ablation", zero), ("shuffled_activity", shuffled)):
        ablated_f1 = metrics[name]["50ms"]["macro_f1"]
        changed = _signature(commands) != _signature(brain)
        disappeared = len(commands) == 0 and len(brain) > 0
        degraded = disappeared or (
            brain_f1 is not None and ablated_f1 is not None and ablated_f1 < brain_f1
        )
        causal_results[name] = {
            "changed": changed,
            "degraded_at_50ms": degraded,
            "brain_event_count": len(brain),
            "ablation_event_count": len(commands),
            "brain_macro_f1_50ms": brain_f1,
            "ablation_macro_f1_50ms": ablated_f1,
            "passes_causal_gate": changed and degraded,
        }

    return {
        "status": "CAUSAL_PASS"
        if all(item["passes_causal_gate"] for item in causal_results.values())
        else "CAUSAL_FAIL",
        "split": {
            "source_filename": score.source_filename,
            "source_score_sha256": score.source_sha256,
            "start_bar": score.start_bar,
            "bars": score.bars,
            "start_time_seconds": score.start_time_seconds,
            "end_time_seconds": score.end_time_seconds,
            "training_provenance": list(model.training_provenance),
        },
        "boundary_policy": {
            "train_feature_windows": "all sample timestamps inside [train_start, train_end)",
            "heldout_score_interval": "[start, end)",
            "scored_prediction_times": "corrected command times inside the held-out interval",
            "window_ms": model.window_ms,
        },
        "model": {
            "readout_version": 2,
            "window_ms": model.window_ms,
            "latency_ms": model.latency_ms,
            "artifact_pair_sha256": model.artifact_pair_sha256,
            "thresholds": {
                instrument: float(model.thresholds[index])
                for index, instrument in enumerate(INSTRUMENTS)
            },
            "class_support": model.class_support,
        },
        "inference_contract": list(inspect.signature(predict_events).parameters),
        "ablation_seed": shuffle_seed,
        "predicted_events": _event_records(brain),
        "metrics": metrics,
        "causal_verdict": {
            "passed": all(item["passes_causal_gate"] for item in causal_results.values()),
            "ablations": causal_results,
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _scaler_fit_declaration(path: Path) -> str:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
    fitted_on = metadata.get("fitted_on")
    if fitted_on != "training descending-activity windows only":
        raise ValueError("scaler artifact does not declare train-only fitting")
    return str(fitted_on)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate readout events on a continuous held-out score segment."
    )
    parser.add_argument("--trace", type=Path, required=True, help="Nirvana descending trace NPZ")
    parser.add_argument("--score-midi", type=Path, required=True, help="score file used only for held-out labels")
    parser.add_argument("--model", type=Path, required=True, help="saved readout model NPZ")
    parser.add_argument("--scaler", type=Path, required=True, help="saved train-only scaler NPZ")
    parser.add_argument("--start-bar", type=int, default=13)
    parser.add_argument("--bars", type=int, default=4)
    parser.add_argument("--shuffle-seed", type=int, default=7)
    parser.add_argument("--report-out", type=Path, help="optional local JSON report path")
    args = parser.parse_args()

    paths = {
        "trace": args.trace.expanduser().resolve(),
        "score_midi": args.score_midi.expanduser().resolve(),
        "model": args.model.expanduser().resolve(),
        "scaler": args.scaler.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            parser.error(f"{name} file does not exist: {path}")

    trace = load_neural_trace(paths["trace"])
    score = import_midi(paths["score_midi"], start_bar=args.start_bar, bars=args.bars)
    model, scaler = load_readout_artifacts(paths["model"], paths["scaler"])
    result = evaluate_holdout(
        trace, score, model, scaler, shuffle_seed=args.shuffle_seed
    )
    result.update(
        {
            "timestamp_msk": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(timespec="seconds"),
            "inputs": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in paths.items()
            },
            "trace_provenance": {
                key: trace.metadata.get(key)
                for key in (
                    "dataset_id",
                    "simulator_repository",
                    "simulator_commit",
                    "seed",
                    "trace_sha256",
                    "dt_ms",
                )
            },
            "artifact_integrity": {
                "trace_content_sha256_verified": trace.metadata.get("trace_sha256"),
                "readout_model_scaler_pair_sha256": model.artifact_pair_sha256,
            },
            "scaler_fit_declaration": _scaler_fit_declaration(paths["scaler"]),
            "rights": "MIDI processing/distribution rights are not established by local evaluation.",
        }
    )
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.report_out:
        report_path = args.report_out.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if result["status"] != "CAUSAL_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
