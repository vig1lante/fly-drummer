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

DATASET_ID = "male-cns:v1.0"
FLYBRAIN_REPOSITORY = "https://github.com/HEREISCB/flybrain"
FLYBRAIN_COMMIT = "c7878b9ee273564186acf8cbdef96289335efa5e"
POPULATION_TYPE = "DNp01"
COMPILED_CACHE_FILES = ("W_post_pre.npz", "neurons.parquet")


@dataclass(frozen=True)
class TraceResult:
    timestamp_msk: str
    dataset_id: str
    simulator_repository: str
    simulator_commit: str
    input_sha256: dict[str, str]
    neurons: int
    connections: int
    population_type: str
    body_ids: tuple[int, ...]
    stimulus: str
    duration_ms: float
    dt_ms: float
    timestamps_ms: tuple[float, ...]
    rates_hz: tuple[tuple[float, ...], ...]
    spike_count: int
    nonzero_samples: int
    trace_sha256: str

    def summary(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("timestamps_ms")
        result.pop("rates_hz")
        result["samples"] = len(self.timestamps_ms)
        result["first_sample"] = {
            "t_ms": self.timestamps_ms[0],
            "rates_hz": list(self.rates_hz[0]),
        }
        result["last_sample"] = {
            "t_ms": self.timestamps_ms[-1],
            "rates_hz": list(self.rates_hz[-1]),
        }
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_inputs(data_dir: Path, connectome_module: Any) -> dict[str, str]:
    manifest_path = Path(connectome_module.__file__).parent / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = [
        item
        for item in manifest["malecns"]["files"]
        if "connectome.load" in item.get("used_by", "")
    ]
    if len(required) != 3:
        raise RuntimeError("upstream MaleCNS manifest no longer lists the expected three simulation inputs")

    hashes: dict[str, str] = {}
    for item in required:
        path = data_dir / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing official MaleCNS input: {path}")
        actual = _sha256(path)
        expected = item.get("sha256")
        if not expected or actual != expected:
            raise ValueError(f"MaleCNS SHA-256 does not match the pinned manifest: {item['path']}")
        hashes[item["path"]] = actual
    return hashes


def _cache_matches(cache_dir: Path, input_sha256: dict[str, str]) -> bool:
    cache_files = {name: cache_dir / name for name in COMPILED_CACHE_FILES}
    metadata_path = cache_dir / "flydrummer-cache-provenance.json"
    if not metadata_path.is_file() or any(not path.is_file() for path in cache_files.values()):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return False
        compiled_sha256 = {
            name: _sha256(path)
            for name, path in cache_files.items()
        }
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("dataset_id") == DATASET_ID
        and metadata.get("simulator_commit") == FLYBRAIN_COMMIT
        and metadata.get("input_sha256") == input_sha256
        and metadata.get("compiled_sha256") == compiled_sha256
    )


def _write_cache_provenance(cache_dir: Path, input_sha256: dict[str, str]) -> None:
    """Record pinned inputs and the compiled files produced from them after a rebuild."""
    compiled_sha256 = {
        name: _sha256(cache_dir / name)
        for name in COMPILED_CACHE_FILES
    }
    metadata = {
        "dataset_id": DATASET_ID,
        "simulator_repository": FLYBRAIN_REPOSITORY,
        "simulator_commit": FLYBRAIN_COMMIT,
        "input_sha256": input_sha256,
        "compiled_sha256": compiled_sha256,
    }
    metadata_path = cache_dir / "flydrummer-cache-provenance.json"
    temporary_path = metadata_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(metadata_path)


def run_malecns_trace(
    *,
    data_dir: Path,
    cache_dir: Path,
    duration_ms: float = 50,
    seed: int = 7,
    rate_hz: float = 200,
) -> TraceResult:
    """Run the pinned LIF simulator and record the real MaleCNS DNp01 activity.

    The smoke applies a seeded Poisson stimulus directly to the real DNp01 cells. It
    verifies simulator/data provenance and trace reproducibility; it is not a song
    encoder or evidence that score input has reached the brain.
    """
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("duration_ms must be positive and finite")
    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be positive and finite")

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
    network_data = connectome.load(cache_dir=cache_dir, rebuild=rebuild, verbose=False)
    if rebuild:
        _write_cache_provenance(cache_dir, input_sha256)

    dn_indices = network_data.select(type=POPULATION_TYPE)
    if dn_indices.size == 0:
        raise RuntimeError(f"pinned MaleCNS graph contains no {POPULATION_TYPE} cells")
    body_ids = tuple(int(value) for value in network_data.neurons.iloc[dn_indices]["bodyId"])

    model = brain.Brain(network_data, device="cpu", seed=seed)
    steps = int(round(duration_ms / model.p.dt))
    if steps <= 0 or not math.isclose(steps * model.p.dt, duration_ms, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"duration_ms must be a positive multiple of simulator dt={model.p.dt} ms")
    model.record_activity = True
    model.set_poisson(dn_indices, rate_hz)

    timestamps: list[float] = []
    rates: list[tuple[float, ...]] = []
    for _ in range(steps):
        model.step()
        timestamps.append(float(model.t))
        sample = np.asarray(model.rates(dn_indices), dtype=np.float32)
        rates.append(tuple(float(value) for value in sample))

    trace_times = np.asarray(timestamps, dtype="<f8")
    trace_rates = np.asarray(rates, dtype="<f4")
    trace_ids = np.asarray(body_ids, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(trace_ids.tobytes())
    digest.update(trace_times.tobytes())
    digest.update(trace_rates.tobytes())

    return TraceResult(
        timestamp_msk=datetime.now(ZoneInfo("Europe/Moscow")).isoformat(timespec="seconds"),
        dataset_id=DATASET_ID,
        simulator_repository=FLYBRAIN_REPOSITORY,
        simulator_commit=FLYBRAIN_COMMIT,
        input_sha256=input_sha256,
        neurons=network_data.n,
        connections=int(network_data.W.nnz),
        population_type=POPULATION_TYPE,
        body_ids=body_ids,
        stimulus=f"seeded Poisson {rate_hz:g} Hz applied directly to {POPULATION_TYPE}",
        duration_ms=float(duration_ms),
        dt_ms=float(model.p.dt),
        timestamps_ms=tuple(timestamps),
        rates_hz=tuple(rates),
        spike_count=int(model.spike_counts[0, dn_indices].sum().item()),
        nonzero_samples=int(np.count_nonzero(trace_rates)),
        trace_sha256=digest.hexdigest(),
    )


def save_trace(path: Path, result: TraceResult) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps_ms=np.asarray(result.timestamps_ms, dtype="<f8"),
        rates_hz=np.asarray(result.rates_hz, dtype="<f4"),
        body_ids=np.asarray(result.body_ids, dtype="<i8"),
        metadata_json=json.dumps(result.summary(), sort_keys=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real MaleCNS v1.0 DNp01 trace smoke.")
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("FLYBRAIN_DATA"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--duration-ms", type=float, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rate-hz", type=float, default=200)
    parser.add_argument("--trace-out", type=Path)
    args = parser.parse_args()
    if args.data_dir is None:
        parser.error("pass --data-dir or set FLYBRAIN_DATA")

    result = run_malecns_trace(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        duration_ms=args.duration_ms,
        seed=args.seed,
        rate_hz=args.rate_hz,
    )
    if args.trace_out:
        save_trace(args.trace_out, result)
    print(json.dumps(result.summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
