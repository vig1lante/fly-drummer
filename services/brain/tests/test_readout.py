from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flydrummer_brain.score import DrumEvent, ScoreFragment


def _readout_api():
    spec = importlib.util.find_spec("flydrummer_brain.readout")
    assert spec is not None, "flydrummer_brain.readout is not implemented"
    from flydrummer_brain import readout

    return readout


def test_no_score_inference_thresholds_simultaneous_events_and_latency() -> None:
    readout = _readout_api()
    assert list(inspect.signature(readout.predict_events).parameters) == [
        "trace",
        "model",
        "scaler",
    ]

    timestamps = np.arange(0.5, 20.5, 0.5, dtype=np.float64)
    rates = np.zeros((40, 2), dtype=np.float32)
    rates[:20] = 1.0
    trace = readout.NeuralTrace(
        timestamps_ms=timestamps,
        rates_hz=rates,
        descending_body_ids=(101, 202),
        metadata={"dataset_id": "male-cns:v1.0"},
    )
    feature_hash = readout.feature_ids_sha256(trace.descending_body_ids)
    model = readout.ReadoutModel(
        coefficients=np.array(
            [
                [5.0, 0.0],
                [0.0, 5.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        intercepts=np.array([-2.5, -2.5, -10, -10, -10, -10, -10], dtype=float),
        thresholds=np.full(7, 0.5),
        window_ms=10.0,
        latency_ms=5.0,
        feature_ids_sha256=feature_hash,
        class_support={instrument: 1 for instrument in readout.INSTRUMENTS},
        training_provenance=(),
    )
    scaler = readout.ReadoutScaler(
        mean=np.zeros(2),
        scale=np.ones(2),
        feature_ids_sha256=feature_hash,
    )

    commands = readout.predict_events(trace, model, scaler)

    assert [(command.instrument, command.time_seconds) for command in commands] == [
        ("kick", 0.00025),
        ("snare", 0.00025),
    ]
    assert all(command.probability > 0.5 for command in commands)

    empty_trace = readout.NeuralTrace(
        timestamps_ms=timestamps,
        rates_hz=np.zeros((40, 2), dtype=np.float32),
        descending_body_ids=(101, 202),
        metadata={"dataset_id": "male-cns:v1.0"},
    )
    assert readout.predict_events(empty_trace, model, scaler) == ()


def _score_fragment(source_hash: str, end_seconds: float = 2.0) -> ScoreFragment:
    instruments = ("kick", "snare", "closed_hat", "open_hat", "crash", "ride", "tom")
    events = tuple(
        DrumEvent(
            time_seconds=event_time,
            instrument=instrument,
            velocity=0.8,
            source_tick=index,
            note=36 + index,
            track_index=0,
            track_name="Drums",
            channel=9,
        )
        for event_time, index in ((0.1, 0), (1.1, 1))
        for instrument in instruments
    )
    return ScoreFragment(
        source_filename="training.mid",
        source_sha256=source_hash,
        source_size_bytes=128,
        midi_format=1,
        midi_track_count=1,
        ticks_per_quarter=480,
        start_bar=1,
        bars=1,
        start_tick=0,
        end_tick=int(end_seconds * 480),
        start_time_seconds=0.0,
        end_time_seconds=end_seconds,
        time_signature_numerator=4,
        time_signature_denominator=4,
        source_tracks=(),
        events=events,
        tempo_map=(),
        unknown_notes=(),
    )


def _training_pair(readout, *, missing_instrument: str | None = None):
    timestamps = np.arange(0.5, 4000.5, 0.5, dtype=np.float64)
    rates = np.zeros((len(timestamps), 2), dtype=np.float32)
    rates[:4000, 0] = np.linspace(0.0, 2.0, 4000, dtype=np.float32)
    rates[:4000, 1] = 1.0
    rates[4000:, :] = 1000.0
    trace = readout.NeuralTrace(
        timestamps_ms=timestamps,
        rates_hz=rates,
        descending_body_ids=(101, 202),
        metadata={
            "dataset_id": "male-cns:v1.0",
            "source_filename": "training.mid",
            "source_score_sha256": "a" * 64,
            "trace_sha256": "b" * 64,
            "simulator_commit": "c" * 40,
            "seed": 7,
            "dt_ms": 0.5,
        },
    )
    score = _score_fragment("a" * 64)
    if missing_instrument is not None:
        score = replace(
            score,
            events=tuple(event for event in score.events if event.instrument != missing_instrument),
        )
    return readout.TrainingPair(trace=trace, score=score)


def test_training_fits_scaler_on_training_range_and_requires_all_seven_labels() -> None:
    readout = _readout_api()
    model, scaler = readout.fit_readout([_training_pair(readout)])

    assert tuple(readout.INSTRUMENTS) == (
        "kick",
        "snare",
        "closed_hat",
        "open_hat",
        "crash",
        "ride",
        "tom",
    )
    assert model.class_support == {instrument: 2 for instrument in readout.INSTRUMENTS}
    assert scaler.mean[0] < 10.0
    assert scaler.mean[1] == pytest.approx(1.0)
    assert all(source["train_end_ms"] == 2000.0 for source in model.training_provenance)

    with pytest.raises(ValueError, match="(?i)positive training support.*ride"):
        readout.fit_readout([_training_pair(readout, missing_instrument="ride")])


def test_readout_model_and_scaler_round_trip_as_separate_artifacts(tmp_path: Path) -> None:
    readout = _readout_api()
    model, scaler = readout.fit_readout([_training_pair(readout)])
    model_path = tmp_path / "readout-model.npz"
    scaler_path = tmp_path / "readout-scaler.npz"

    readout.save_readout_artifacts(model_path, scaler_path, model, scaler)
    predictions_before_save = readout.predict_events(
        _training_pair(readout).trace, model, scaler
    )
    loaded_model, loaded_scaler = readout.load_readout_artifacts(model_path, scaler_path)
    predictions_after_load = readout.predict_events(
        _training_pair(readout).trace, loaded_model, loaded_scaler
    )

    assert model_path.is_file() and scaler_path.is_file() and model_path != scaler_path
    np.testing.assert_array_equal(loaded_model.coefficients, model.coefficients)
    np.testing.assert_array_equal(loaded_scaler.mean, scaler.mean)
    assert loaded_model.training_provenance == model.training_provenance
    assert predictions_after_load == predictions_before_save
    with np.load(model_path, allow_pickle=False) as model_file:
        assert "coefficients" in model_file.files
        assert not any("connectome" in name.casefold() for name in model_file.files)


def test_training_excludes_window_crossing_train_end_with_event_at_boundary() -> None:
    readout = _readout_api()
    end_ms = 1005.84
    timestamps = np.arange(0.5, 4000.5, 0.5, dtype=np.float64)
    rates = np.ones((len(timestamps), 2), dtype=np.float32)
    rates[timestamps >= end_ms] = 1000.0
    trace = readout.NeuralTrace(
        timestamps_ms=timestamps,
        rates_hz=rates,
        descending_body_ids=(101, 202),
        metadata={
            "dataset_id": "male-cns:v1.0",
            "source_filename": "training.mid",
            "source_score_sha256": "a" * 64,
            "trace_sha256": "b" * 64,
            "simulator_commit": "c" * 40,
            "seed": 7,
            "dt_ms": 0.5,
        },
    )
    original = _score_fragment("a" * 64, end_seconds=end_ms / 1000.0)
    train_events = tuple(event for event in original.events if event.time_seconds == 0.1)
    boundary_event = replace(original.events[0], time_seconds=end_ms / 1000.0)
    score = replace(original, events=train_events + (boundary_event,))

    _, scaler = readout.fit_readout([readout.TrainingPair(trace, score)])

    # The 10 ms window centered at 1005.25 ms straddles this 1005.84 ms
    # boundary and contains nine holdout samples. None may affect train fit.
    np.testing.assert_allclose(scaler.mean, [1.0, 1.0])


def test_neural_trace_loader_rejects_changed_content_with_stale_digest(
    tmp_path: Path,
) -> None:
    readout = _readout_api()
    timestamps = np.arange(0.5, 20.5, 0.5, dtype=np.float64)
    rates = np.ones((len(timestamps), 2), dtype=np.float32)
    body_ids = np.asarray((101, 202), dtype="<i8")
    digest = hashlib.sha256(
        body_ids.tobytes()
        + np.asarray(timestamps, dtype="<f8").tobytes()
        + np.asarray(rates, dtype="<f4").tobytes()
    ).hexdigest()
    metadata = {
        "dataset_id": "male-cns:v1.0",
        "dt_ms": 0.5,
        "trace_sha256": digest,
    }
    changed_rates = rates.copy()
    changed_rates[3, 1] = 2.0
    trace_path = tmp_path / "changed-trace.npz"
    np.savez_compressed(
        trace_path,
        timestamps_ms=timestamps,
        descending_rates_hz=changed_rates,
        descending_body_ids=body_ids,
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    with pytest.raises(ValueError, match="trace content does not match its SHA-256"):
        readout.load_neural_trace(trace_path)


def test_readout_loader_rejects_scaler_from_a_different_fit(tmp_path: Path) -> None:
    readout = _readout_api()
    first_model, first_scaler = readout.fit_readout(
        [_training_pair(readout)]
    )
    second_pair = _training_pair(readout)
    changed_rates = second_pair.trace.rates_hz.copy()
    changed_rates[:4000] += 0.25
    second_trace = replace(second_pair.trace, rates_hz=changed_rates)
    second_model, second_scaler = readout.fit_readout(
        [replace(second_pair, trace=second_trace)]
    )
    first_model_path = tmp_path / "first-model.npz"
    first_scaler_path = tmp_path / "first-scaler.npz"
    second_model_path = tmp_path / "second-model.npz"
    second_scaler_path = tmp_path / "second-scaler.npz"
    readout.save_readout_artifacts(
        first_model_path, first_scaler_path, first_model, first_scaler
    )
    readout.save_readout_artifacts(
        second_model_path, second_scaler_path, second_model, second_scaler
    )

    with pytest.raises(ValueError, match="readout model and scaler pair identity"):
        readout.load_readout_artifacts(first_model_path, second_scaler_path)
