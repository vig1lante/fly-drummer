"""Encode score hits as sensory inputs to the pinned MaleCNS v1.0 simulator.

The lane-to-Johnston's-organ assignment is an explicit computational channel map,
not a claim that the fly naturally classifies drum-kit instruments this way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from .malecns import (
    DATASET_ID,
    FLYBRAIN_COMMIT,
    FLYBRAIN_REPOSITORY,
    _cache_matches,
    _verified_inputs,
    _write_cache_provenance,
)
from .score import ScoreFragment, import_midi


SENSORY_LANE_POPULATION_PREFIX: dict[str, str] = {
    "kick": "JO-A",
    "snare": "JO-B",
    "closed_hat": "JO-C",
    "open_hat": "JO-D",
    "crash": "JO-E",
    "ride": "JO-F",
    "tom": "JO-unclear",
}
SENSORY_SUPERCLASS = "cb_sensory"
SENSORY_ENTRY_NERVE = "AN"


@dataclass(frozen=True)
class SensoryPopulation:
    instrument: str
    cell_type_prefix: str
    cell_types: tuple[str, ...]
    body_ids: tuple[int, ...]
    indices: tuple[int, ...]
    superclass: str
    entry_nerve: str


@dataclass(frozen=True)
class SensoryInputFrame:
    frame_index: int
    instrument: str
    cell_type_prefix: str
    source_tick: int
    score_time_ms: float
    simulator_step: int
    simulator_time_ms: float
    velocity: float
    rate_hz: float
    pulse_steps: int


@dataclass(frozen=True)
class SensoryEncoding:
    source_filename: str
    source_sha256: str
    dt_ms: float
    pulse_duration_ms: float
    rate_scale_hz: float
    end_time_seconds: float
    populations: dict[str, SensoryPopulation]
    frames: tuple[SensoryInputFrame, ...]


@dataclass(frozen=True)
class SensoryTraceResult:
    timestamp_msk: str
    dataset_id: str
    simulator_repository: str
    simulator_commit: str
    male_cns_input_sha256: dict[str, str]
    source_filename: str
    source_score_sha256: str
    seed: int
    duration_ms: float
    dt_ms: float
    pulse_duration_ms: float
    rate_scale_hz: float
    timestamps_ms: np.ndarray
    descending_body_ids: tuple[int, ...]
    trace_rates_hz: np.ndarray
    trace_sha256: str
    input_frames_sha256: str
    input_frames: tuple[SensoryInputFrame, ...]
    populations: dict[str, SensoryPopulation]
    sensory_spike_count: int
    descending_spike_count: int
    remaining_forced_input_cells: int
    remaining_forced_input_rate_hz: float

    def summary(self) -> dict[str, Any]:
        return {
            "timestamp_msk": self.timestamp_msk,
            "dataset_id": self.dataset_id,
            "simulator_repository": self.simulator_repository,
            "simulator_commit": self.simulator_commit,
            "male_cns_input_sha256": self.male_cns_input_sha256,
            "source_filename": self.source_filename,
            "source_score_sha256": self.source_score_sha256,
            "seed": self.seed,
            "duration_ms": self.duration_ms,
            "dt_ms": self.dt_ms,
            "samples": len(self.timestamps_ms),
            "descending_neurons": len(self.descending_body_ids),
            "descending_body_ids_sha256": _array_sha256(
                np.asarray(self.descending_body_ids, dtype="<i8")
            ),
            "trace_sha256": self.trace_sha256,
            "input_frames_sha256": self.input_frames_sha256,
            "input_frame_count": len(self.input_frames),
            "input_encoding": {
                "simulator_api": "flybrain.Brain.set_poisson",
                "cell_population_superclass": SENSORY_SUPERCLASS,
                "cell_entry_nerve": SENSORY_ENTRY_NERVE,
                "rate_scale_hz": self.rate_scale_hz,
                "pulse_duration_ms": self.pulse_duration_ms,
                "lane_population_prefixes": {
                    instrument: population.cell_type_prefix
                    for instrument, population in sorted(self.populations.items())
                },
                "populations": {
                    instrument: {
                        "cell_types": list(population.cell_types),
                        "cell_count": len(population.body_ids),
                        "body_ids_sha256": _array_sha256(
                            np.asarray(population.body_ids, dtype="<i8")
                        ),
                    }
                    for instrument, population in sorted(self.populations.items())
                },
                "note": (
                    "Lane-to-population assignment is a computational channel map; "
                    "it is not a measured acoustic transduction or biological drum classifier."
                ),
            },
            "sensory_spike_count": self.sensory_spike_count,
            "descending_spike_count": self.descending_spike_count,
            "remaining_forced_input_cells": self.remaining_forced_input_cells,
            "remaining_forced_input_rate_hz": self.remaining_forced_input_rate_hz,
            "output_contract": "time-stamped descending activity only; no drum events are emitted",
        }

def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _resolve_population(network_data: Any, instrument: str, prefix: str) -> SensoryPopulation:
    neurons = network_data.neurons
    type_names = neurons["type"].fillna("").astype(str)
    mask = (
        type_names.str.startswith(prefix)
        & (neurons["superclass"] == SENSORY_SUPERCLASS)
        & (neurons["entryNerve"] == SENSORY_ENTRY_NERVE)
    )
    indices_array = np.flatnonzero(mask.to_numpy())
    if not len(indices_array):
        raise ValueError(
            f"MaleCNS v1.0 has no {prefix} population with "
            f"superclass={SENSORY_SUPERCLASS} and entryNerve={SENSORY_ENTRY_NERVE}"
        )
    selected = neurons.iloc[indices_array]
    ordered = np.argsort(selected.bodyId.to_numpy(dtype=np.int64), kind="stable")
    indices = indices_array[ordered]
    body_ids = selected.bodyId.to_numpy(dtype=np.int64)[ordered]
    cell_types = tuple(sorted(set(selected.type.astype(str))))
    return SensoryPopulation(
        instrument=instrument,
        cell_type_prefix=prefix,
        cell_types=cell_types,
        body_ids=tuple(int(value) for value in body_ids),
        indices=tuple(int(value) for value in indices),
        superclass=SENSORY_SUPERCLASS,
        entry_nerve=SENSORY_ENTRY_NERVE,
    )


def encode_score(
    score: ScoreFragment,
    network_data: Any,
    *,
    dt_ms: float,
    pulse_duration_ms: float = 5.0,
    rate_scale_hz: float = 200.0,
) -> SensoryEncoding:
    """Map score hits to real MaleCNS JO cell IDs and sparse time-stamped input frames."""
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be positive and finite")
    if not math.isfinite(pulse_duration_ms) or pulse_duration_ms <= 0:
        raise ValueError("pulse_duration_ms must be positive and finite")
    if not math.isfinite(rate_scale_hz) or rate_scale_hz <= 0:
        raise ValueError("rate_scale_hz must be positive and finite")
    pulse_steps = max(1, int(round(pulse_duration_ms / dt_ms)))
    actual_pulse_duration_ms = pulse_steps * dt_ms

    used_instruments = sorted({event.instrument for event in score.events})
    populations: dict[str, SensoryPopulation] = {}
    for instrument in used_instruments:
        prefix = SENSORY_LANE_POPULATION_PREFIX.get(instrument)
        if prefix is None:
            raise ValueError(f"no sensory population mapping for score lane {instrument!r}")
        populations[instrument] = _resolve_population(network_data, instrument, prefix)

    frames: list[SensoryInputFrame] = []
    for event in sorted(score.events, key=lambda item: (item.time_seconds, item.source_tick, item.instrument)):
        if not math.isfinite(event.time_seconds) or event.time_seconds < 0:
            raise ValueError("score event time must be non-negative and finite")
        if event.time_seconds >= score.end_time_seconds:
            raise ValueError("score event falls outside the selected score fragment")
        if not math.isfinite(event.velocity) or not 0.0 <= event.velocity <= 1.0:
            raise ValueError("score event velocity must be between 0 and 1")
        score_time_ms = event.time_seconds * 1000.0
        step_index = int(round(score_time_ms / dt_ms))
        frame_index = len(frames)
        frames.append(
            SensoryInputFrame(
                frame_index=frame_index,
                instrument=event.instrument,
                cell_type_prefix=populations[event.instrument].cell_type_prefix,
                source_tick=event.source_tick,
                score_time_ms=score_time_ms,
                simulator_step=step_index,
                simulator_time_ms=step_index * dt_ms,
                velocity=event.velocity,
                rate_hz=event.velocity * rate_scale_hz,
                pulse_steps=pulse_steps,
            )
        )

    return SensoryEncoding(
        source_filename=score.source_filename,
        source_sha256=score.source_sha256,
        dt_ms=dt_ms,
        pulse_duration_ms=actual_pulse_duration_ms,
        rate_scale_hz=rate_scale_hz,
        end_time_seconds=score.end_time_seconds,
        populations=populations,
        frames=tuple(frames),
    )


def simulate_malecns_sensory_trace(
    network_data: Any,
    encoding: SensoryEncoding,
    *,
    duration_ms: float,
    seed: int,
    male_cns_input_sha256: dict[str, str] | None = None,
) -> SensoryTraceResult:
    """Run the pinned whole-CNS LIF network with score frames injected into JO cells."""
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("duration_ms must be positive and finite")

    from flybrain import brain

    model = brain.Brain(network_data, device="cpu", seed=seed)
    dt_ms = float(model.p.dt)
    if not math.isclose(encoding.dt_ms, dt_ms, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"encoding dt_ms={encoding.dt_ms} does not match simulator dt_ms={dt_ms}")
    steps = int(round(duration_ms / dt_ms))
    if steps <= 0 or not math.isclose(steps * dt_ms, duration_ms, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"duration_ms must be a positive multiple of simulator dt={dt_ms} ms")

    descending_indices = network_data.select(superclass="descending_neuron")
    if not descending_indices.size:
        raise RuntimeError("pinned MaleCNS graph has no descending_neuron cells")
    descending_ids = tuple(
        int(value)
        for value in network_data.neurons.iloc[descending_indices].bodyId.to_numpy(dtype=np.int64)
    )
    sensory_indices = np.asarray(
        sorted({index for population in encoding.populations.values() for index in population.indices}),
        dtype=np.int64,
    )
    boundaries: dict[int, dict[str, list[SensoryInputFrame]]] = {}
    for frame in encoding.frames:
        starts = boundaries.setdefault(frame.simulator_step, {"starts": [], "ends": []})
        starts["starts"].append(frame)
        end_step = frame.simulator_step + frame.pulse_steps
        ends = boundaries.setdefault(end_step, {"starts": [], "ends": []})
        ends["ends"].append(frame)

    timestamps = np.empty(steps, dtype="<f8")
    rates = np.empty((steps, len(descending_indices)), dtype="<f4")
    model.record_activity = True
    active_frames: dict[int, SensoryInputFrame] = {}
    for step_index in range(steps):
        boundary = boundaries.get(step_index)
        if boundary is not None:
            for frame in boundary["ends"]:
                active_frames.pop(frame.frame_index, None)
            for frame in boundary["starts"]:
                active_frames[frame.frame_index] = frame
            # Refresh at each onset/offset so an expired score frame cannot leave forced input behind.
            model.set_poisson(sensory_indices, 0.0)
            active_rates: dict[str, float] = {}
            for frame in active_frames.values():
                active_rates[frame.instrument] = max(
                    active_rates.get(frame.instrument, 0.0), frame.rate_hz
                )
            for instrument, rate_hz in active_rates.items():
                model.set_poisson(encoding.populations[instrument].indices, rate_hz)
        model.step()
        timestamps[step_index] = float(model.t)
        rates[step_index] = np.asarray(model.rates(descending_indices), dtype="<f4")

    remaining_probabilities = (
        model.poisson_p[0, sensory_indices.tolist()]
        if len(sensory_indices)
        else model.poisson_p.new_zeros((0,))
    )
    remaining_cells = int(remaining_probabilities.count_nonzero().item())
    remaining_rate_hz = (
        float(remaining_probabilities.max().item()) * 1000.0 / dt_ms
        if remaining_probabilities.numel()
        else 0.0
    )
    trace_digest = hashlib.sha256()
    trace_digest.update(np.asarray(descending_ids, dtype="<i8").tobytes())
    trace_digest.update(timestamps.tobytes())
    trace_digest.update(rates.tobytes())
    frame_payload = [asdict(frame) for frame in encoding.frames]
    frame_digest = hashlib.sha256(
        json.dumps(frame_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return SensoryTraceResult(
        timestamp_msk=datetime.now(ZoneInfo("Europe/Moscow")).isoformat(timespec="seconds"),
        dataset_id=DATASET_ID,
        simulator_repository=FLYBRAIN_REPOSITORY,
        simulator_commit=FLYBRAIN_COMMIT,
        male_cns_input_sha256=male_cns_input_sha256 or {},
        source_filename=encoding.source_filename,
        source_score_sha256=encoding.source_sha256,
        seed=seed,
        duration_ms=float(steps * dt_ms),
        dt_ms=dt_ms,
        pulse_duration_ms=encoding.pulse_duration_ms,
        rate_scale_hz=encoding.rate_scale_hz,
        timestamps_ms=timestamps,
        descending_body_ids=descending_ids,
        trace_rates_hz=rates,
        trace_sha256=trace_digest.hexdigest(),
        input_frames_sha256=frame_digest,
        input_frames=encoding.frames,
        populations=encoding.populations,
        sensory_spike_count=int(model.spike_counts[0, sensory_indices].sum().item())
        if len(sensory_indices)
        else 0,
        descending_spike_count=int(model.spike_counts[0, descending_indices].sum().item()),
        remaining_forced_input_cells=remaining_cells,
        remaining_forced_input_rate_hz=remaining_rate_hz,
    )


def run_malecns_sensory_trace(
    *,
    score: ScoreFragment,
    data_dir: Path,
    cache_dir: Path,
    duration_ms: float | None = None,
    seed: int = 7,
    pulse_duration_ms: float = 5.0,
    rate_scale_hz: float = 200.0,
) -> SensoryTraceResult:
    """Load the pinned official graph, encode the score, and return its real descending trace."""
    data_dir = Path(data_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"MaleCNS data directory does not exist: {data_dir}")
    os.environ["FLYBRAIN_DATA"] = str(data_dir)
    from flybrain import brain, connectome

    if Path(connectome.DATA_DIR).expanduser().resolve() != data_dir:
        raise RuntimeError("flybrain.connectome was imported with a different FLYBRAIN_DATA directory")
    input_sha256 = _verified_inputs(data_dir, connectome)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rebuild = not _cache_matches(cache_dir, input_sha256)
    network_data = connectome.load(
        cache_dir=cache_dir,
        rebuild=rebuild,
        verbose=False,
    )
    if rebuild:
        _write_cache_provenance(cache_dir, input_sha256)
    dt_ms = float(brain.LIFParams().dt)
    encoding = encode_score(
        score,
        network_data,
        dt_ms=dt_ms,
        pulse_duration_ms=pulse_duration_ms,
        rate_scale_hz=rate_scale_hz,
    )
    if duration_ms is None:
        duration_ms = math.ceil(score.end_time_seconds * 1000.0 / dt_ms) * dt_ms
    return simulate_malecns_sensory_trace(
        network_data,
        encoding,
        duration_ms=duration_ms,
        seed=seed,
        male_cns_input_sha256=input_sha256,
    )


def save_sensory_trace(path: Path, result: SensoryTraceResult) -> None:
    """Persist score frames, verified neuron IDs, and the real timestamped neural trace."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_dtype = np.dtype(
        [
            ("frame_index", "<i4"),
            ("instrument", "U16"),
            ("cell_type_prefix", "U16"),
            ("source_tick", "<i8"),
            ("score_time_ms", "<f8"),
            ("simulator_step", "<i8"),
            ("simulator_time_ms", "<f8"),
            ("velocity", "<f4"),
            ("rate_hz", "<f4"),
            ("pulse_steps", "<i4"),
        ]
    )
    frames = np.empty(len(result.input_frames), dtype=frame_dtype)
    for row, frame in enumerate(result.input_frames):
        frames[row] = tuple(asdict(frame).values())
    populations_json = {
        instrument: {
            "cell_type_prefix": population.cell_type_prefix,
            "cell_types": list(population.cell_types),
            "body_ids": list(population.body_ids),
            "superclass": population.superclass,
            "entry_nerve": population.entry_nerve,
        }
        for instrument, population in sorted(result.populations.items())
    }
    np.savez_compressed(
        path,
        timestamps_ms=result.timestamps_ms,
        descending_rates_hz=result.trace_rates_hz,
        descending_body_ids=np.asarray(result.descending_body_ids, dtype="<i8"),
        input_frames=frames,
        populations_json=json.dumps(populations_json, sort_keys=True),
        metadata_json=json.dumps(result.summary(), sort_keys=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode a MIDI drum score as MaleCNS sensory input and save its descending trace."
    )
    parser.add_argument("--score-midi", type=Path, required=True)
    parser.add_argument("--start-bar", type=int, default=1)
    parser.add_argument("--bars", type=int, default=16)
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("FLYBRAIN_DATA"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--duration-ms", type=float)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pulse-duration-ms", type=float, default=5.0)
    parser.add_argument("--rate-scale-hz", type=float, default=200.0)
    parser.add_argument("--trace-out", type=Path, required=True)
    args = parser.parse_args()
    if args.data_dir is None:
        parser.error("pass --data-dir or set FLYBRAIN_DATA")
    score = import_midi(args.score_midi, start_bar=args.start_bar, bars=args.bars)
    result = run_malecns_sensory_trace(
        score=score,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        duration_ms=args.duration_ms,
        seed=args.seed,
        pulse_duration_ms=args.pulse_duration_ms,
        rate_scale_hz=args.rate_scale_hz,
    )
    save_sensory_trace(args.trace_out, result)
    print(json.dumps(result.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
