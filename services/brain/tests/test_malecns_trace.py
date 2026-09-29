from __future__ import annotations

import os
from pathlib import Path

from flydrummer_brain.malecns import run_malecns_trace


def test_real_malecns_descending_trace_is_nonzero_and_reproducible() -> None:
    data_dir = Path(os.environ["FLYBRAIN_DATA"])
    cache_dir = Path(os.environ["FLYBRAIN_CACHE_DIR"])

    first = run_malecns_trace(
        data_dir=data_dir,
        cache_dir=cache_dir,
        duration_ms=50,
        seed=7,
        rate_hz=200,
    )
    second = run_malecns_trace(
        data_dir=data_dir,
        cache_dir=cache_dir,
        duration_ms=50,
        seed=7,
        rate_hz=200,
    )

    assert first.dataset_id == "male-cns:v1.0"
    assert first.neurons > 0
    assert first.connections > 0
    assert first.population_type == "DNp01"
    assert first.body_ids
    assert first.timestamps_ms[-1] == 50
    assert first.nonzero_samples > 0
    assert first.trace_sha256 == second.trace_sha256
