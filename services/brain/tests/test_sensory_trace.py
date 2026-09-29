from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from flydrummer_brain.score import DrumEvent, ScoreFragment


def _sensory_api():
    spec = importlib.util.find_spec("flydrummer_brain.sensory")
    assert spec is not None, "flydrummer_brain.sensory is not implemented"
    return importlib.import_module("flydrummer_brain.sensory")


@pytest.fixture(scope="module")
def malecns_connectome():
    from flybrain import connectome

    return connectome.load(
        cache_dir=Path(os.environ["FLYBRAIN_CACHE_DIR"]),
        rebuild=False,
        verbose=False,
    )


def _score(events: tuple[DrumEvent, ...]) -> ScoreFragment:
    return ScoreFragment(
        source_filename="local-test-fixture.mid",
        source_sha256="a" * 64,
        source_size_bytes=128,
        midi_format=1,
        midi_track_count=1,
        ticks_per_quarter=480,
        start_bar=1,
        bars=1,
        start_tick=0,
        end_tick=1920,
        start_time_seconds=0.0,
        end_time_seconds=2.0,
        time_signature_numerator=4,
        time_signature_denominator=4,
        source_tracks=(),
        events=events,
        tempo_map=(),
        unknown_notes=(),
    )


def _event(instrument: str, note: int, velocity: float = 1.0) -> DrumEvent:
    return DrumEvent(
        time_seconds=0.0,
        instrument=instrument,
        velocity=velocity,
        source_tick=0,
        note=note,
        track_index=0,
        track_name="Drums",
        channel=9,
    )


def test_drum_lanes_map_to_distinct_real_malecns_sensory_populations(malecns_connectome) -> None:
    sensory = _sensory_api()
    lane_notes = (
        ("kick", 36),
        ("snare", 38),
        ("closed_hat", 42),
        ("open_hat", 46),
        ("crash", 49),
        ("ride", 51),
        ("tom", 45),
    )
    score = _score(tuple(_event(lane, note) for lane, note in lane_notes))

    encoding = sensory.encode_score(score, malecns_connectome, dt_ms=0.5)

    assert set(encoding.populations) == {lane for lane, _ in lane_notes}
    assert {
        lane: population.cell_type_prefix
        for lane, population in encoding.populations.items()
    } == {
        "kick": "JO-A",
        "snare": "JO-B",
        "closed_hat": "JO-C",
        "open_hat": "JO-D",
        "crash": "JO-E",
        "ride": "JO-F",
        "tom": "JO-unclear",
    }
    all_ids: set[int] = set()
    for lane, _ in lane_notes:
        population = encoding.populations[lane]
        assert population.body_ids
        assert population.superclass == "cb_sensory"
        assert population.entry_nerve == "AN"
        assert all_ids.isdisjoint(population.body_ids)
        all_ids.update(population.body_ids)
    assert {frame.instrument for frame in encoding.frames} == set(encoding.populations)


def test_same_seed_reproduces_real_descending_trace_and_score_is_not_output(malecns_connectome) -> None:
    sensory = _sensory_api()
    score = _score((_event("kick", 36), _event("snare", 38)))
    encoding = sensory.encode_score(score, malecns_connectome, dt_ms=0.5)

    first = sensory.simulate_malecns_sensory_trace(
        malecns_connectome, encoding, duration_ms=50.0, seed=7
    )
    repeated = sensory.simulate_malecns_sensory_trace(
        malecns_connectome, encoding, duration_ms=50.0, seed=7
    )
    silent = sensory.simulate_malecns_sensory_trace(
        malecns_connectome,
        sensory.encode_score(replace(score, events=()), malecns_connectome, dt_ms=0.5),
        duration_ms=50.0,
        seed=7,
    )

    assert first.descending_body_ids
    assert first.trace_rates_hz.shape == (100, len(first.descending_body_ids))
    assert np.count_nonzero(first.trace_rates_hz) > 0
    assert first.descending_spike_count > 0
    assert first.trace_sha256 == repeated.trace_sha256
    assert np.count_nonzero(silent.trace_rates_hz) == 0
    assert silent.descending_spike_count == 0
    assert not hasattr(first, "drum_events")


def test_expired_score_input_leaves_no_forced_signal_in_real_simulator(malecns_connectome) -> None:
    sensory = _sensory_api()
    score = _score((_event("kick", 36, velocity=0.5),))
    encoding = sensory.encode_score(
        score, malecns_connectome, dt_ms=0.5, pulse_duration_ms=5.0
    )

    trace = sensory.simulate_malecns_sensory_trace(
        malecns_connectome, encoding, duration_ms=50.0, seed=3
    )

    assert trace.remaining_forced_input_cells == 0
    assert trace.remaining_forced_input_rate_hz == 0.0
    assert trace.input_frames
    assert trace.input_frames[0].rate_hz == pytest.approx(100.0)
    assert trace.timestamps_ms[-1] == pytest.approx(50.0)
