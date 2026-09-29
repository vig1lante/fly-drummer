from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from flydrummer_brain import malecns, sensory


INPUT_SHA256 = {"verified-raw.feather": "a" * 64}
CACHE_FILES = {
    "W_post_pre.npz": b"compiled sparse graph",
    "neurons.parquet": b"compiled neuron table",
}


def _write_cache(cache_dir: Path, *, include_compiled_sha256: bool = True) -> None:
    cache_dir.mkdir(parents=True)
    for name, content in CACHE_FILES.items():
        (cache_dir / name).write_bytes(content)
    metadata = {
        "dataset_id": malecns.DATASET_ID,
        "simulator_commit": malecns.FLYBRAIN_COMMIT,
        "input_sha256": INPUT_SHA256,
    }
    if include_compiled_sha256:
        metadata["compiled_sha256"] = {
            name: hashlib.sha256(content).hexdigest()
            for name, content in CACHE_FILES.items()
        }
    (cache_dir / "flydrummer-cache-provenance.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def test_cache_without_compiled_content_digests_is_not_reusable(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, include_compiled_sha256=False)

    assert not malecns._cache_matches(cache_dir, INPUT_SHA256)


@pytest.mark.parametrize("filename", ["W_post_pre.npz", "neurons.parquet"])
def test_cache_with_tampered_compiled_file_is_not_reusable(
    tmp_path: Path, filename: str
) -> None:
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir)
    (cache_dir / filename).write_bytes(b"tampered compiled content")

    assert not malecns._cache_matches(cache_dir, INPUT_SHA256)


class _FakeGraph:
    n = 1
    W = SimpleNamespace(nnz=1)
    neurons = pd.DataFrame({"bodyId": [4815162342]})

    def select(self, *, type: str) -> np.ndarray:
        assert type == malecns.POPULATION_TYPE
        return np.asarray([0], dtype=np.int64)


class _FakeBrainModel:
    def __init__(self, *_args, **_kwargs) -> None:
        self.p = SimpleNamespace(dt=0.5)
        self.t = 0.0
        self.record_activity = False
        self.spike_counts = np.asarray([[1]], dtype=np.int64)

    def set_poisson(self, _indices: np.ndarray, _rate_hz: float) -> None:
        pass

    def step(self) -> None:
        self.t += self.p.dt

    def rates(self, indices: np.ndarray) -> np.ndarray:
        return np.ones(len(indices), dtype=np.float32)


def _install_fake_flybrain(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path, cache_dir: Path
) -> list[bool]:
    data_dir.mkdir()
    monkeypatch.setenv("FLYBRAIN_DATA", str(data_dir))
    rebuild_values: list[bool] = []

    def load(*, cache_dir: Path, rebuild: bool, verbose: bool) -> _FakeGraph:
        assert verbose is False
        rebuild_values.append(rebuild)
        if rebuild:
            for name in CACHE_FILES:
                (cache_dir / name).write_bytes(b"rebuilt " + name.encode())
        return _FakeGraph()

    connectome = SimpleNamespace(DATA_DIR=data_dir, load=load)
    brain = SimpleNamespace(
        Brain=_FakeBrainModel,
        LIFParams=lambda: SimpleNamespace(dt=0.5),
    )
    flybrain = ModuleType("flybrain")
    flybrain.connectome = connectome
    flybrain.brain = brain
    monkeypatch.setitem(sys.modules, "flybrain", flybrain)
    monkeypatch.setattr(malecns, "_verified_inputs", lambda *_args: INPUT_SHA256)
    monkeypatch.setattr(sensory, "_verified_inputs", lambda *_args: INPUT_SHA256)
    return rebuild_values


@pytest.mark.parametrize("trace_path", ["malecns", "sensory"])
@pytest.mark.parametrize(
    ("legacy_metadata", "tampered_filename"),
    [
        (True, None),
        (False, "W_post_pre.npz"),
        (False, "neurons.parquet"),
    ],
    ids=["legacy-metadata", "tampered-graph", "tampered-neurons"],
)
def test_trace_paths_rebuild_untrusted_compiled_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trace_path: str,
    legacy_metadata: bool,
    tampered_filename: str | None,
) -> None:
    data_dir = tmp_path / "data"
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, include_compiled_sha256=not legacy_metadata)
    if tampered_filename is not None:
        (cache_dir / tampered_filename).write_bytes(b"tampered compiled content")
    rebuild_values = _install_fake_flybrain(monkeypatch, data_dir, cache_dir)

    if trace_path == "malecns":
        malecns.run_malecns_trace(
            data_dir=data_dir,
            cache_dir=cache_dir,
            duration_ms=0.5,
            seed=7,
            rate_hz=200,
        )
    else:
        monkeypatch.setattr(sensory, "encode_score", lambda *_args, **_kwargs: object())
        monkeypatch.setattr(
            sensory,
            "simulate_malecns_sensory_trace",
            lambda *_args, **_kwargs: "trace",
        )
        result = sensory.run_malecns_sensory_trace(
            score=SimpleNamespace(),
            data_dir=data_dir,
            cache_dir=cache_dir,
            duration_ms=0.5,
            seed=7,
        )
        assert result == "trace"

    assert rebuild_values == [True]
    metadata = json.loads(
        (cache_dir / "flydrummer-cache-provenance.json").read_text(encoding="utf-8")
    )
    assert metadata["dataset_id"] == "male-cns:v1.0"
    assert metadata["simulator_repository"] == malecns.FLYBRAIN_REPOSITORY
    assert metadata["simulator_commit"] == malecns.FLYBRAIN_COMMIT
    assert metadata["input_sha256"] == INPUT_SHA256
    assert metadata["compiled_sha256"] == {
        name: hashlib.sha256((cache_dir / name).read_bytes()).hexdigest()
        for name in CACHE_FILES
    }
