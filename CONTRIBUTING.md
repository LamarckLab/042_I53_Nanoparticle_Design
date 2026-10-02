# Contributing

## Development setup

```bash
git clone <repository-url>
cd 042_I53_Nanoparticle_Design
python -m pip install -e ".[dev,parquet]"
```

## Running the tests

```bash
pytest                     # full suite
python tests/run_all.py    # same suite, no pytest required
```

The suite runs entirely on synthetic structures and needs no GPU, no model weights and
no external tools. It must stay that way: anything requiring RFdiffusion, ProteinMPNN
or AlphaFold2 belongs behind a command-construction test, not an execution test.

## Conventions

- Every metric added to `cyclicnano.geometry` must be deterministic and depend only on numpy.
- Filtering decisions belong in `configs/`, never hard-coded in a stage module.
- A stage must never delete a candidate. Mark it failed and record the reason.
- Stages must be idempotent: re-running one may overwrite its own columns only.
- New metrics need a test that shows the metric separating a good fixture from a bad one.

## Adding a stage

1. Add `src/cyclicnano/sNN_<name>.py` exposing `run(cfg, state, ...) -> dict`.
2. Read inputs from the manifest, write outputs back with `upsert`.
3. Guard with `state.is_done(STAGE)` and call `state.mark_done(STAGE)` on success.
4. Register it in `cyclicnano/cli.py` and add its config block to `configs/default.yaml`.
