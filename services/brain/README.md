# Brain service

The `services/brain` Python 3.12 project pins the FlyBrain simulator and runs a real MaleCNS v1.0 trace smoke. Raw connectome tables and compiled caches are local-only; keep them outside this repository.

## Provenance

- Simulator: [`HEREISCB/flybrain`](https://github.com/HEREISCB/flybrain), commit `c7878b9ee273564186acf8cbdef96289335efa5e` (MIT). `pyproject.toml` and `uv.lock` pin this exact revision.
- Dataset: [`male-cns:v1.0`](https://male-cns.janelia.org/download/), published by the MaleCNS/Janelia project under CC-BY. Simulator code and connectome data have separate licenses; retain dataset attribution if using the data.
- Python: `>=3.12,<3.13` for this service; the pinned simulator declares `>=3.12` and supports 3.12/3.13.

For the `connectome.load` graph build, the pinned simulator manifest requires these three files. Download them from the [official MaleCNS download page](https://male-cns.janelia.org/download/) into a local directory and keep that directory out of Git:

| File | SHA-256 |
| --- | --- |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2` |
| `body-neurotransmitters-male-cns-v1.0.feather` | `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621` |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1` |

The simulator manifest lists a separate 2.65 GB `tbar-neurotransmitters` file as used only by `scripts/audit_nt.py`; this trace service does not require it. The smoke verifies the three files against the pinned manifest before loading the connectome. The compiled graph cache also belongs outside the repository.

After a cache rebuild, `flydrummer-cache-provenance.json` records the pinned dataset, simulator repository/commit, verified raw-input hashes, and SHA-256 digests for both `W_post_pre.npz` and `neurons.parquet`. Each trace path checks those two file digests before reusing the compiled cache. A cache without the digests or with changed content is rebuilt from the verified raw inputs.

## Install, test, and smoke

Run these commands from the repository root. Set the two paths to local directories; the data directory must contain the three files above.

```sh
export FLYBRAIN_DATA=/absolute/path/to/malecns-v1.0
export FLYBRAIN_CACHE_DIR=/absolute/path/to/flybrain-cache

uv sync --project services/brain --python 3.12 --locked
uv run --project services/brain --python 3.12 --locked pytest
uv run --project services/brain --python 3.12 --locked flydrummer-brain-trace \
  --data-dir "$FLYBRAIN_DATA" \
  --cache-dir "$FLYBRAIN_CACHE_DIR" \
  --duration-ms 50 --seed 7 --rate-hz 200 \
  --trace-out /tmp/flydrummer-malecns-trace.npz
```

The first run compiles the connectome cache and may take several minutes and multiple gigabytes of disk. The CLI prints a compact JSON summary with the dataset ID, node/connection counts, Moscow timestamp, population, nonzero activity, and deterministic trace hash. The optional NPZ is a local generated trace; do not commit it.

The `flydrummer-brain-trace` smoke applies a seeded Poisson stimulus directly to the real descending population `DNp01`, runs the pinned LIF simulator, and records each simulator step. Repeated runs with the same data, simulator revision, seed, duration, and rate produce the same trace hash. This verifies the simulator/data path and a real nonzero population trace. The separate `flydrummer-brain-sensory-trace` command below maps imported score events into brain sensory input and records descending activity; it does not itself produce a predicted drum performance.

## MIDI score import

`flydrummer_brain.score.import_midi(path, start_bar=1, bars=16)` reads Standard MIDI format 0 or 1 with PPQN timing and changing time signatures. Bar numbers count from the start of the song; each meter change must occur on a bar boundary under the preceding meter so the fragment range is unambiguous. It selects tracks named as drums; if none are named, it falls back to tracks using MIDI channel 10 (channel index `9`). The selected interval is one-based by bar, start-inclusive and end-exclusive. Event times are absolute song seconds, velocity is normalized to `0..1`, simultaneous events are retained, and tempo changes are integrated into both event times and the tempo map.

Drum notes use the explicit General MIDI mapping in `flydrummer_brain.score.DRUM_NOTE_MAP`. Notes outside that mapping are returned in `ScoreFragment.unknown_notes` and included in `ScoreFragment.to_dict()["mapping_report"]`; they are not silently converted or dropped. A meter change is accepted when its event falls on a bar boundary under the preceding meter. Changes inside a bar or conflicting meter markers are rejected, as are malformed/truncated MIDI files.

The input MIDI's origin and processing rights must be documented separately. Keep the input and any derived score fragments local until those conditions are established; a locally available score is not evidence that it may be committed or distributed.

## Sensory encoding and descending trace

`flydrummer-brain-sensory-trace` passes imported score hits into the pinned FlyBrain `Brain` through `set_poisson`, then records the rates of every graph neuron whose `superclass` is `descending_neuron`. It emits no `DrumEvent` or predicted performance; score-derived events are used only as timed sensory input frames. Each simulator tick first clears forced input on all selected sensory cells, then applies only the populations with an active frame. The saved result contains the sparse input frames, their exact MaleCNS body IDs, all descending body IDs, simulator timestamps, rates and provenance hashes.

This phase uses the official MaleCNS v1.0 annotation fields to resolve real Johnston's organ neurons under `superclass=cb_sensory` and `entryNerve=AN`. The explicit computational channel map is:

| Score lane | MaleCNS type prefix |
| --- | --- |
| `kick` | `JO-A` |
| `snare` | `JO-B` |
| `closed_hat` | `JO-C` |
| `open_hat` | `JO-D` |
| `crash` | `JO-E` |
| `ride` | `JO-F` |
| `tom` | `JO-unclear` |

Every body ID is resolved from the verified graph at runtime and stored in the local trace. `JO-unclear` is an explicitly uncertain annotation group. The lane assignment is a reproducible channelization choice, not evidence that individual Johnston's organ subtypes identify kick, snare or cymbal sounds. Velocity sets the injected Poisson rate (`velocity × --rate-scale-hz`, default 200 Hz); each hit is held for 5 ms by default. Neither rate nor pulse duration is calibrated against acoustic recordings. This is a computational sensory-input experiment, not a validated model of hearing, behavior or biological drum recognition. The fixed connectome is not trained.

Run the full imported 16-bar fragment with official data and local-only output:

```sh
uv run --project services/brain --python 3.12 --locked flydrummer-brain-sensory-trace \
  --score-midi "music/midi/Nirvana-Smells Like Teen Spirit.mid" \
  --start-bar 1 --bars 16 \
  --data-dir "$FLYBRAIN_DATA" \
  --cache-dir "$FLYBRAIN_CACHE_DIR" \
  --seed 7 \
  --trace-out /private/tmp/flydrummer-malecns-sensory-trace.npz
```

The default duration rounds the selected fragment's end time up to the next 0.5 ms simulator step. The NPZ is a local-only neural artifact; do not place it in the web bundle or commit it. Score origin and processing rights remain separate gates and are not established by this command or by computing a hash.

## Descending-activity readout

`flydrummer_brain.readout` fits seven independent binary logistic outputs from fixed-width windows of real descending-neuron activity. The connectome is not modified. The scaler is fitted from training windows only, and training labels must match each trace's recorded source-score hash. `predict_events(trace, model, scaler)` accepts no score, MIDI, or target labels; it emits thresholded, simultaneous `DrumCommand` values from neural activity alone.

The default readout window is 10 ms, the fixed threshold is `0.5` for each instrument, and the 5 ms timestamp correction moves each command from the center of its activity window to that window's start. This is a time-bin alignment convention, not a measured biological conduction delay. Consecutive positive windows for the same instrument produce one command until the output becomes inactive. The 10 ms window bounds event timing resolution.

Training requires positive examples for `kick`, `snare`, `closed_hat`, `open_hat`, `crash`, `ride`, and `tom`. Pass only explicitly selected training fragments to `fit_readout`; reserve a separate continuous song segment before fitting if that segment will later be used for evaluation. A training window is selected only when its first and last simulator samples are both inside the half-open training interval; a window crossing into the validation interval is discarded before fitting either scaler or model. Validation labels use the half-open score interval, and predicted commands are scored only when their corrected command time falls inside it. This keeps the boundary policy explicit while retaining each input's song time axis and score/trace provenance. Save the small model and scaler as separate `.npz` artifacts with `save_readout_artifacts`; these contain readout parameters and normalization statistics, not connectome data. Trace loading recomputes the digest over descending IDs, timestamps and rates. Model and scaler artifacts carry one content-derived pair SHA-256, which is checked on load and before inference. Keep both artifacts local until score-processing rights are confirmed.

Example API usage for a local Nirvana trace, with its first 12 bars used for training and bars 13–16 kept aside for later evaluation, plus a separate auxiliary training trace:

```python
from flydrummer_brain.readout import (
    TrainingPair,
    fit_readout,
    load_neural_trace,
    predict_events,
    save_readout_artifacts,
)
from flydrummer_brain.score import import_midi

nirvana_trace = load_neural_trace("/private/tmp/nirvana-trace.npz")
freebird_trace = load_neural_trace("/private/tmp/freebird-trace.npz")
model, scaler = fit_readout([
    TrainingPair(
        nirvana_trace,
        import_midi("music/midi/Nirvana-Smells Like Teen Spirit.mid", start_bar=1, bars=12),
    ),
    TrainingPair(
        freebird_trace,
        import_midi("music/midi/Lynyrd Skynyrd-Free Bird-09-26-2026.mid", start_bar=1, bars=8),
    ),
])
save_readout_artifacts("/private/tmp/readout-model.npz", "/private/tmp/readout-scaler.npz", model, scaler)
nirvana_commands = predict_events(nirvana_trace, model, scaler)
```

Call `predict_events` only with the intended product song's descending trace. Do not pass or post-process source score events as predicted commands. MIDI provenance and permission to process or distribute the source remain separate gates.

## Held-out readout evaluation

`flydrummer-brain-evaluate` evaluates a saved model on a continuous score interval. The score labels are imported only for the scorer; inference receives the neural trace, model and scaler. The command rejects overlapping training intervals recorded for the same score hash, reports per-instrument support/F1 and matched-pair Timing MAE at ±50 ms and ±100 ms, and compares the neural predictions with zero activity, shuffled 10 ms neural windows and an explicit empty-output no-brain baseline. A causal gate fails if either neural ablation leaves the event output unchanged or does not reduce macro F1 at ±50 ms (or remove all output).

Example evaluation of Nirvana held-out bars 13–16 after fitting on bars 1–12, with the separate Free Bird file used only as auxiliary C4 training input:

```sh
uv run --project services/brain --python 3.12 --locked flydrummer-brain-evaluate \
  --trace /private/tmp/2026-09-29T18-29-50+03-00-fly-drummer-mvp-FD-1.C3-nirvana-trace.npz \
  --score-midi "music/midi/Nirvana-Smells Like Teen Spirit.mid" \
  --model /private/tmp/2026-09-29T19-36-38+03-00-fly-drummer-mvp-FD-1.C4-model.npz \
  --scaler /private/tmp/2026-09-29T19-36-38+03-00-fly-drummer-mvp-FD-1.C4-scaler.npz \
  --start-bar 13 --bars 4 --shuffle-seed 7 \
  --report-out /private/tmp/flydrummer-nirvana-heldout-evaluation.json
```

The JSON on stdout and optional local report contain predicted Nirvana events, input hashes, train/holdout intervals, the half-open boundary policy, per-instrument and macro metrics, zero/shuffle/baseline results, the verified trace-content digest and the content-derived model/scaler pair identity. An empty output is the named no-brain baseline; it does not use held-out score labels to generate events. A failed causal gate still emits measured metrics and returns exit code `1`. Keep the evaluation report and all source-derived artifacts local until score-processing and distribution rights are established.
