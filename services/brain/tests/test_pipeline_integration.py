from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from flydrummer_brain.evaluation import evaluate_holdout
from flydrummer_brain.readout import (
    INSTRUMENTS,
    NeuralTrace,
    TrainingPair,
    fit_readout,
    load_neural_trace,
    load_readout_artifacts,
    predict_events,
    save_readout_artifacts,
)
from flydrummer_brain.score import import_midi
from flydrummer_brain.sensory import run_malecns_sensory_trace, save_sensory_trace


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
NIRVANA_MIDI = REPOSITORY_ROOT / "music/midi/Nirvana-Smells Like Teen Spirit.mid"
FREE_BIRD_MIDI = REPOSITORY_ROOT / "music/midi/Lynyrd Skynyrd-Free Bird-09-26-2026.mid"


def _vlq(value: int) -> bytes:
    encoded = [value & 0x7F]
    value >>= 7
    while value:
        encoded.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(encoded)


def _meta(kind: int, payload: bytes) -> bytes:
    return b"\x00\xff" + bytes((kind,)) + _vlq(len(payload)) + payload


def _small_score_midi() -> bytes:
    notes = (36, 38, 42, 46, 49, 51, 45)
    track = b"".join(
        (
            _meta(0x03, b"Drums"),
            _meta(0x58, bytes((4, 2, 24, 8))),
            _meta(0x51, (500_000).to_bytes(3, "big")),
            *(
                b"\x00\x99" + bytes((note, 96))
                for note in notes
            ),
            _meta(0x2F, b""),
        )
    )
    return (
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, 480)
        + b"MTrk"
        + struct.pack(">I", len(track))
        + track
    )


def _required_file(variable: str) -> Path:
    raw = os.environ.get(variable)
    if not raw:
        pytest.skip(f"локальный артефакт задаётся переменной {variable}")
    path = Path(raw).expanduser()
    if not path.is_file():
        pytest.skip(f"локальный артефакт из {variable} не найден")
    return path


def _required_directory(variable: str) -> Path:
    raw = os.environ.get(variable)
    if not raw:
        pytest.skip(f"локальный каталог MaleCNS задаётся переменной {variable}")
    path = Path(raw).expanduser()
    if not path.is_dir():
        pytest.skip(f"локальный каталог из {variable} не найден")
    return path


def test_small_midi_reaches_repeatable_real_malecns_and_trained_readout(
    tmp_path: Path,
) -> None:
    data_dir = _required_directory("FLYBRAIN_DATA")
    cache_dir = _required_directory("FLYBRAIN_CACHE_DIR")
    midi_path = tmp_path / "seven-lane-fixture.mid"
    midi_path.write_bytes(_small_score_midi())
    score = import_midi(midi_path, start_bar=1, bars=1)

    first = run_malecns_sensory_trace(
        score=score,
        data_dir=data_dir,
        cache_dir=cache_dir,
        duration_ms=2_000.0,
        seed=7,
    )
    repeated = run_malecns_sensory_trace(
        score=score,
        data_dir=data_dir,
        cache_dir=cache_dir,
        duration_ms=2_000.0,
        seed=7,
    )

    assert first.dataset_id == "male-cns:v1.0"
    assert first.simulator_commit == "c7878b9ee273564186acf8cbdef96289335efa5e"
    assert first.descending_body_ids
    assert first.trace_sha256 == repeated.trace_sha256
    assert first.input_frames_sha256 == repeated.input_frames_sha256
    assert np.count_nonzero(first.trace_rates_hz) > 0
    assert {frame.instrument for frame in first.input_frames} == set(INSTRUMENTS)
    assert all(frame.score_time_ms == 0.0 for frame in first.input_frames)
    assert all(frame.simulator_time_ms == 0.0 for frame in first.input_frames)
    assert first.timestamps_ms[-1] == pytest.approx(2_000.0)
    assert not hasattr(first, "drum_events")

    raw_trace_path = tmp_path / "small-real-malecns-trace.npz"
    save_sensory_trace(raw_trace_path, first)
    trace = load_neural_trace(raw_trace_path)
    # Keep the smoke readout small while retaining actual descending cells and rates.
    active_columns = np.flatnonzero(np.count_nonzero(trace.rates_hz, axis=0))
    selected_columns = active_columns[:24]
    assert selected_columns.size > 0
    compact_trace = NeuralTrace(
        timestamps_ms=trace.timestamps_ms,
        rates_hz=trace.rates_hz[:, selected_columns],
        descending_body_ids=tuple(trace.descending_body_ids[index] for index in selected_columns),
        metadata=trace.metadata,
    )
    model, scaler = fit_readout([TrainingPair(compact_trace, score)])
    predicted = predict_events(compact_trace, model, scaler)

    model_path = tmp_path / "small-readout-model.npz"
    scaler_path = tmp_path / "small-readout-scaler.npz"
    save_readout_artifacts(model_path, scaler_path, model, scaler)
    loaded_model, loaded_scaler = load_readout_artifacts(model_path, scaler_path)
    assert predict_events(compact_trace, loaded_model, loaded_scaler) == predicted
    assert predicted
    assert {event.instrument for event in predicted} <= set(INSTRUMENTS)
    assert all(0.0 <= event.time_seconds < score.end_time_seconds for event in predicted)
    with np.load(scaler_path, allow_pickle=False) as saved_scaler:
        scaler_metadata = json.loads(str(saved_scaler["metadata_json"]))
    assert scaler_metadata["fitted_on"] == "training descending-activity windows only"


def test_real_nirvana_holdout_pipeline_keeps_free_bird_auxiliary_and_causal() -> None:
    if not NIRVANA_MIDI.is_file() or not FREE_BIRD_MIDI.is_file():
        pytest.skip("пользовательские MIDI доступны только в локальном checkout")
    trace_path = _required_file("FLYDRUMMER_NIRVANA_TRACE")
    model_path = _required_file("FLYDRUMMER_READOUT_MODEL")
    scaler_path = _required_file("FLYDRUMMER_READOUT_SCALER")

    nirvana_train = import_midi(NIRVANA_MIDI, start_bar=1, bars=12)
    nirvana_holdout = import_midi(NIRVANA_MIDI, start_bar=13, bars=4)
    auxiliary_training_score = import_midi(FREE_BIRD_MIDI, start_bar=1, bars=8)
    trace = load_neural_trace(trace_path)
    model, scaler = load_readout_artifacts(model_path, scaler_path)

    assert trace.metadata["trace_sha256"] == (
        "6fd26435515c89ed74206f0085caf850dfdb5b37b782ebe8a5a2a47ac03f0c4f"
    )
    assert model.artifact_pair_sha256 == (
        "a0d51c79af633d95b8b65eb6c8d342ea4566df7f93720ca95276e379cf22a875"
    )
    assert model.artifact_pair_sha256 == scaler.artifact_pair_sha256
    assert trace.metadata["source_score_sha256"] == nirvana_holdout.source_sha256
    assert trace.metadata["dataset_id"] == "male-cns:v1.0"
    assert trace.metadata["simulator_commit"] == "c7878b9ee273564186acf8cbdef96289335efa5e"
    nirvana_train_entries = [
        source
        for source in model.training_provenance
        if source["source_score_sha256"] == nirvana_train.source_sha256
    ]
    auxiliary_entries = [
        source
        for source in model.training_provenance
        if source["source_score_sha256"] == auxiliary_training_score.source_sha256
    ]
    assert len(nirvana_train_entries) == 1
    assert nirvana_train_entries[0]["train_start_ms"] == pytest.approx(
        nirvana_train.start_time_seconds * 1_000.0
    )
    assert nirvana_train_entries[0]["train_end_ms"] <= (
        nirvana_holdout.start_time_seconds * 1_000.0 + 1e-6
    )
    assert len(auxiliary_entries) == 1
    assert auxiliary_entries[0]["start_bar"] == 1
    assert auxiliary_entries[0]["bars"] == 8
    assert auxiliary_training_score.source_sha256 != nirvana_holdout.source_sha256

    result = evaluate_holdout(
        trace, nirvana_holdout, model, scaler, shuffle_seed=7
    )
    repeated_result = evaluate_holdout(
        trace, nirvana_holdout, model, scaler, shuffle_seed=7
    )
    assert result["status"] == "CAUSAL_PASS"
    assert repeated_result == result
    assert result["split"]["source_filename"].startswith("Nirvana")
    assert result["boundary_policy"] == {
        "train_feature_windows": "all sample timestamps inside [train_start, train_end)",
        "heldout_score_interval": "[start, end)",
        "scored_prediction_times": "corrected command times inside the held-out interval",
        "window_ms": model.window_ms,
    }
    assert result["predicted_events"]
    assert all(
        nirvana_holdout.start_time_seconds
        <= event["time_seconds"]
        < nirvana_holdout.end_time_seconds
        for event in result["predicted_events"]
    )
    assert all(
        evidence["passes_causal_gate"]
        for evidence in result["causal_verdict"]["ablations"].values()
    )
    assert result["metrics"]["brain"]["50ms"]["support"] == 57
    assert result["inference_contract"] == ["trace", "model", "scaler"]
