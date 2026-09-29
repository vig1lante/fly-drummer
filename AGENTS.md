# Fly Drummer service instructions

This repository is one service for loop-engineering. Run all shell, Git, Python and web commands from `/Users/vigilante/PycharmProjects/fly-drummer`.

## Project boundaries

- `services/brain` owns offline score import, MaleCNS simulation, trace, readout, metrics and export.
- `apps/web` owns static replay, Web Audio, TARGET/FLY UI and 3D.
- The score may feed TARGET and brain training, but browser drum execution must read only exported BrainOutput.
- A single song time controls sound, trace and animation. No independent visual clock may create drum events.
- Preserve dataset, model, audio and asset provenance. Do not commit raw large connectome tables, full traces, credentials or audio without distribution rights. Store local-only data outside web public directories; document its location and hash without treating local availability as a deployable asset.

## Commands

The Brain service uses Python 3.12. From the repository root, install it with `uv sync --project services/brain --python 3.12 --locked`, run its tests with `uv run --project services/brain --python 3.12 --locked pytest`, and run its real MaleCNS smoke with `uv run --project services/brain --python 3.12 --locked flydrummer-brain-trace --data-dir "$FLYBRAIN_DATA" --cache-dir "$FLYBRAIN_CACHE_DIR"`. Set both data and cache paths to local directories outside the repository. See `services/brain/README.md` for required data, provenance, checksums, and the full smoke command. Later web phases must document their build, test and start commands when created. Read `pyproject.toml`, `package.json` and local README files before running commands. Do not assume a server on localhost belongs to this task; start the supported service and check readiness.

## Loop conduct

Follow `loop-engineering/LOOP.md`, the active task-map and the assigned role prompt. Preserve existing user files and unrelated dirty changes. Record only measured results. Use Europe/Moscow (UTC+03:00) for run timestamps and artifact filenames.
