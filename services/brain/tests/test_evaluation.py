from __future__ import annotations

import importlib.util
from dataclasses import replace

import numpy as np
import pytest

from flydrummer_brain.score import DrumEvent, ScoreFragment


def _evaluation_api():
    spec = importlib.util.find_spec("flydrummer_brain.evaluation")
    assert spec is not None, "flydrummer_brain.evaluation is not implemented"
    from flydrummer_brain import evaluation

    return evaluation


def _score(events: tuple[DrumEvent, ...], *, start: float = 0.0, end: float = 2.0):
    return ScoreFragment(
        source_filename="Nirvana-Smells Like Teen Spirit.mid",
        source_sha256="a" * 64,
        source_size_bytes=128,
        midi_format=1,
        midi_track_count=1,
        ticks_per_quarter=480,
        start_bar=13,
        bars=4,
        start_tick=0,
        end_tick=3840,
        start_time_seconds=start,
        end_time_seconds=end,
        time_signature_numerator=4,
        time_signature_denominator=4,
        source_tracks=(),
        events=events,
        tempo_map=(),
        unknown_notes=(),
    )


def _event(time_seconds: float, instrument: str) -> DrumEvent:
    return DrumEvent(
        time_seconds=time_seconds,
        instrument=instrument,
        velocity=0.8,
        source_tick=0,
        note=36,
        track_index=0,
        track_name="Drums",
        channel=9,
    )


def test_event_metrics_match_one_to_one_and_time_error_uses_only_matches() -> None:
    evaluation = _evaluation_api()
    readout = pytest.importorskip("flydrummer_brain.readout")
    score = _score((_event(1.0, "kick"), _event(1.2, "kick"), _event(1.0, "snare")))
    commands = (
        readout.DrumCommand(0.96, "kick", 0.8),
        readout.DrumCommand(1.02, "kick", 0.8),
        readout.DrumCommand(1.28, "kick", 0.8),
    )

    at_50_ms = evaluation.evaluate_predictions(score, commands, tolerance_ms=50.0)
    at_100_ms = evaluation.evaluate_predictions(score, commands, tolerance_ms=100.0)

    assert at_50_ms["instruments"]["kick"]["support"] == 2
    assert at_50_ms["instruments"]["kick"]["true_positive"] == 1
    assert at_50_ms["instruments"]["kick"]["false_positive"] == 2
    assert at_50_ms["instruments"]["kick"]["false_negative"] == 1
    assert at_50_ms["instruments"]["kick"]["f1"] == pytest.approx(0.4)
    assert at_50_ms["instruments"]["kick"]["timing_mae_ms"] == pytest.approx(20.0)
    assert at_50_ms["instruments"]["snare"]["timing_mae_ms"] is None
    assert at_100_ms["instruments"]["kick"]["true_positive"] == 2
    assert at_100_ms["instruments"]["kick"]["false_positive"] == 1
    assert at_100_ms["instruments"]["kick"]["timing_mae_ms"] == pytest.approx(50.0)
    assert at_100_ms["macro_f1"] == pytest.approx(0.4)


def test_holdout_rejects_overlapping_training_labels_but_allows_touching_boundary() -> None:
    evaluation = _evaluation_api()
    readout = pytest.importorskip("flydrummer_brain.readout")
    model = readout.ReadoutModel(
        coefficients=np.zeros((7, 1)),
        intercepts=np.zeros(7),
        thresholds=np.full(7, 0.5),
        window_ms=10.0,
        latency_ms=5.0,
        feature_ids_sha256="b" * 64,
        class_support={instrument: 1 for instrument in readout.INSTRUMENTS},
        training_provenance=(
            {
                "source_score_sha256": "a" * 64,
                "train_start_ms": 0.0,
                "train_end_ms": 1000.0,
            },
        ),
    )
    heldout = _score((_event(1.0, "kick"),), start=1.0, end=2.0)

    evaluation.validate_evaluation_split(model, heldout)
    overlapping = replace(
        model,
        training_provenance=(
            {
                "source_score_sha256": "a" * 64,
                "train_start_ms": 0.0,
                "train_end_ms": 1000.5,
            },
        ),
    )
    with pytest.raises(ValueError, match="overlaps held-out score interval"):
        evaluation.validate_evaluation_split(overlapping, heldout)


def test_ablation_changes_only_holdout_activity_and_shuffle_is_reproducible() -> None:
    evaluation = _evaluation_api()
    readout = pytest.importorskip("flydrummer_brain.readout")
    timestamps = np.arange(0.5, 2000.5, 0.5, dtype=np.float64)
    rates = np.zeros((len(timestamps), 2), dtype=np.float32)
    rates[::4, 0] = np.arange(1000, dtype=np.float32)
    rates[1::4, 1] = np.arange(1000, dtype=np.float32) * 0.5
    trace = readout.NeuralTrace(
        timestamps_ms=timestamps,
        rates_hz=rates,
        descending_body_ids=(101, 202),
        metadata={"dataset_id": "male-cns:v1.0", "dt_ms": 0.5},
    )

    zeroed = evaluation.ablate_trace(
        trace, start_seconds=1.0, end_seconds=2.0, window_ms=10.0, mode="zero"
    )
    shuffled = evaluation.ablate_trace(
        trace, start_seconds=1.0, end_seconds=2.0, window_ms=10.0,
        mode="shuffle", seed=17,
    )
    shuffled_again = evaluation.ablate_trace(
        trace, start_seconds=1.0, end_seconds=2.0, window_ms=10.0,
        mode="shuffle", seed=17,
    )

    assert np.count_nonzero(zeroed.rates_hz[(timestamps >= 1000) & (timestamps < 2000)]) == 0
    np.testing.assert_array_equal(shuffled.rates_hz, shuffled_again.rates_hz)
    assert not np.array_equal(shuffled.rates_hz, rates)
    np.testing.assert_array_equal(shuffled.rates_hz[timestamps < 1000], rates[timestamps < 1000])
    np.testing.assert_array_equal(shuffled.rates_hz[timestamps >= 2000], rates[timestamps >= 2000])


def test_holdout_inference_predictions_do_not_depend_on_target_labels() -> None:
    evaluation = _evaluation_api()
    readout = pytest.importorskip("flydrummer_brain.readout")
    timestamps = np.arange(0.5, 2000.5, 0.5, dtype=np.float64)
    rates = np.zeros((len(timestamps), 1), dtype=np.float32)
    rates[2000:2400, 0] = 2.0
    trace = readout.NeuralTrace(
        timestamps_ms=timestamps,
        rates_hz=rates,
        descending_body_ids=(101,),
        metadata={"dataset_id": "male-cns:v1.0", "source_score_sha256": "a" * 64, "dt_ms": 0.5},
    )
    feature_hash = readout.feature_ids_sha256(trace.descending_body_ids)
    model = readout.ReadoutModel(
        coefficients=np.array([[5.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]),
        intercepts=np.array([-2.5, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]),
        thresholds=np.full(7, 0.5),
        window_ms=10.0,
        latency_ms=5.0,
        feature_ids_sha256=feature_hash,
        class_support={instrument: 1 for instrument in readout.INSTRUMENTS},
        training_provenance=({"source_score_sha256": "a" * 64, "train_start_ms": 0.0, "train_end_ms": 1000.0},),
    )
    scaler = readout.ReadoutScaler(np.zeros(1), np.ones(1), feature_hash)
    first_score = _score((_event(1.2, "kick"),), start=1.0, end=2.0)
    second_score = _score((_event(1.8, "snare"),), start=1.0, end=2.0)
    thresholds_before = model.thresholds.copy()

    first = evaluation.evaluate_holdout(trace, first_score, model, scaler, shuffle_seed=3)
    second = evaluation.evaluate_holdout(trace, second_score, model, scaler, shuffle_seed=3)

    assert first["predicted_events"] == second["predicted_events"]
    assert first["metrics"]["brain"]["50ms"]["instruments"]["kick"]["support"] == 1
    assert second["metrics"]["brain"]["50ms"]["instruments"]["snare"]["support"] == 1
    assert first["metrics"]["no_brain_empty_output_baseline"]["50ms"]["predicted"] == 0
    np.testing.assert_array_equal(model.thresholds, thresholds_before)


def test_holdout_rejects_mismatched_model_scaler_pair_identity() -> None:
    evaluation = _evaluation_api()
    readout = pytest.importorskip("flydrummer_brain.readout")
    timestamps = np.arange(0.5, 2000.5, 0.5, dtype=np.float64)
    rates = np.zeros((len(timestamps), 1), dtype=np.float32)
    rates[2000:2400, 0] = 2.0
    score = _score((_event(1.2, "kick"),), start=1.0, end=2.0)
    trace = readout.NeuralTrace(
        timestamps_ms=timestamps,
        rates_hz=rates,
        descending_body_ids=(101,),
        metadata={
            "dataset_id": "male-cns:v1.0",
            "source_score_sha256": score.source_sha256,
            "dt_ms": 0.5,
        },
    )
    feature_hash = readout.feature_ids_sha256(trace.descending_body_ids)
    model = readout.ReadoutModel(
        coefficients=np.zeros((7, 1)),
        intercepts=np.zeros(7),
        thresholds=np.full(7, 0.5),
        window_ms=10.0,
        latency_ms=5.0,
        feature_ids_sha256=feature_hash,
        class_support={instrument: 1 for instrument in readout.INSTRUMENTS},
        training_provenance=(
            {
                "source_score_sha256": score.source_sha256,
                "train_start_ms": 0.0,
                "train_end_ms": 1000.0,
            },
        ),
        artifact_pair_sha256="a" * 64,
    )
    scaler = readout.ReadoutScaler(
        np.zeros(1), np.ones(1), feature_hash, artifact_pair_sha256="b" * 64
    )

    with pytest.raises(ValueError, match="readout model and scaler pair identity"):
        evaluation.evaluate_holdout(trace, score, model, scaler, shuffle_seed=3)


def test_heldout_boundary_event_is_scored_in_half_open_interval() -> None:
    evaluation = _evaluation_api()
    readout = pytest.importorskip("flydrummer_brain.readout")
    timestamps = np.arange(0.5, 2000.5, 0.5, dtype=np.float64)
    rates = np.zeros((len(timestamps), 1), dtype=np.float32)
    rates[2000:2400, 0] = 2.0
    trace = readout.NeuralTrace(
        timestamps_ms=timestamps,
        rates_hz=rates,
        descending_body_ids=(101,),
        metadata={
            "dataset_id": "male-cns:v1.0",
            "source_score_sha256": "a" * 64,
            "dt_ms": 0.5,
        },
    )
    feature_hash = readout.feature_ids_sha256(trace.descending_body_ids)
    model = readout.ReadoutModel(
        coefficients=np.array([[5.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]),
        intercepts=np.array([-2.5, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0]),
        thresholds=np.full(7, 0.5),
        window_ms=10.0,
        latency_ms=5.0,
        feature_ids_sha256=feature_hash,
        class_support={instrument: 1 for instrument in readout.INSTRUMENTS},
        training_provenance=(
            {
                "source_score_sha256": "a" * 64,
                "train_start_ms": 0.0,
                "train_end_ms": 1000.0,
            },
        ),
    )
    scaler = readout.ReadoutScaler(np.zeros(1), np.ones(1), feature_hash)
    score = _score((_event(1.0, "kick"),), start=1.0, end=2.0)

    result = evaluation.evaluate_holdout(trace, score, model, scaler, shuffle_seed=3)

    assert result["boundary_policy"]["heldout_score_interval"] == "[start, end)"
    assert result["metrics"]["brain"]["50ms"]["instruments"]["kick"]["true_positive"] == 1
    assert all(1.0 <= event["time_seconds"] < 2.0 for event in result["predicted_events"])
