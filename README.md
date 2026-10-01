# I53 Nanoparticle Design

Automated de novo design pipeline for I53 icosahedral protein nanoparticles:
RFdiffusion backbones to objective geometric filtering to ProteinMPNN to AlphaFold2
self-consistency validation.

[![CI](https://github.com/LamarckLab/042_I53_Nanoparticle_Design/actions/workflows/ci.yml/badge.svg)](https://github.com/LamarckLab/042_I53_Nanoparticle_Design/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What this is

An I53 icosahedral nanoparticle is built from two cyclic building blocks: a C5 pentamer
on the five-fold axes and a C3 trimer on the three-fold axes. Designing one therefore
starts by designing and validating those two oligomers.

That front half of the protocol is a fixed sequence of four tool invocations with a
human making judgement calls in between. This package removes the human from the loop
by turning every one of those judgement calls into a threshold in a config file.

Nothing in stages 00-04 is specific to I53, or even to C5 and C3. The symmetry order is
a parameter, so the same pipeline designs any cyclic homo-oligomer from C2 to C12. I53
is the application; cyclic oligomer design is the tool.

## Pipeline

```
00  generate    RFdiffusion, symmetric mode          ->  N backbones
01  measure     29 geometric metrics per backbone    ->  metrics table (no filtering by default)
02  design      ProteinMPNN, tied positions          ->  M sequences per backbone
03  fold        AlphaFold2 monomer, single-sequence  ->  predicted structures
04  validate    CA RMSD vs design, success rate      ->  ranked backbones
```

Stages 05-08 (fusion search, icosahedral assembly, junction redesign, reporting) are
not implemented yet. See [Status](#status).

## Design principles

Three rules make the pipeline reproducible and the thresholds tunable after the fact.

**Compute everything.** Every metric is computed for every candidate regardless of
whether any rule references it. Metrics that only the future fusion stage will need,
such as terminal helix lengths, are already recorded.

**Filter declaratively, and only when asked.** No rules ship enabled: every backbone
reaches sequence design, and stage 01 records rather than rejects. Thresholds, when
wanted, live in YAML as expressions over metric names, never in code:

```yaml
backbone_filter:
  rules:
    - "n_clash == 0"
    - "helix_frac >= 0.35"
    - "rg_ratio <= 1.30"
```

Because the metrics are recorded either way, a filter can be applied after a run
finishes by adding rules and replaying stage 01 alone.

**Never delete.** A rejected candidate keeps all of its metrics and gains a pass flag
plus the rules it failed. Retuning a threshold replays one CPU-bound stage instead of
a GPU-day, and the rejected population stays available for analysis.

Together these mean the thresholds are not buried assumptions. `filter_summary.json`
reports how many candidates each individual rule rejected, so it is always visible
which threshold is doing the work.

## Installation

```bash
git clone https://github.com/LamarckLab/042_I53_Nanoparticle_Design.git
cd 042_I53_Nanoparticle_Design
python -m pip install -e ".[parquet]"
```

The package itself depends only on numpy, pandas and PyYAML. The external tools are
called as subprocesses and are not Python dependencies:

| Stage | Tool | Notes |
|-------|------|-------|
| 00 | [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) | symmetric oligomer mode |
| 02 | [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | tied positions required |
| 03 | [localcolabfold](https://github.com/YoshitakaMo/localcolabfold) | AlphaFold2 weights |

Point the config at them via `paths.*` and name their environments under
`backend.envs`. AlphaFold2 parameters are downloaded by localcolabfold and are subject
to DeepMind's own licence terms; check those before any non-academic use.

## Usage

```bash
# validate the config and the filter rules without running anything
cyclo check --config configs/default.yaml

# print the constructed tool commands without executing them
cyclo run --profile configs/profiles/amax.yaml --dry-run

# full run
cyclo run --profile configs/profiles/amax.yaml

# a single stage, or a re-run of one after changing a threshold
cyclo run --stages 01 --force

# inspect one backbone without the pipeline
cyclo metrics path/to/backbone.pdb --symmetry C5
```

Design a C3 trimer instead of a C5 pentamer by overriding two values:

```bash
cyclo run --set target.symmetry=C3 --set run.name=c3_60aa_v1
```

## Output

```
runs/<name>/
├── tables/
│   ├── backbones.csv        one row per backbone, all metrics, pass flags, rankings
│   └── sequences.csv        one row per designed sequence, RMSD and pLDDT
├── 00_backbones/            RFdiffusion output
├── 01_backbone_filter/      filter_summary.json: per-rule rejection counts
├── 02_sequences/            ProteinMPNN output
├── 03_predictions/          AlphaFold2 output
├── 04_validate/             ranked_backbones.csv
└── logs/commands.jsonl      every subprocess, its exit code and its runtime
```

`backbones.csv` is the artifact to keep. It holds every metric for every candidate,
including the rejected ones, which is what makes a filter ablation possible after the
run rather than requiring a second one.

## Two traps this package handles for you

**RFdiffusion contig length is the total, not per chain.** A C5 of 60-residue subunits
needs `contigmap.contigs=[300-300]`. Passing 60 silently produces a much smaller
oligomer. The conversion lives in `Config.total_length`.

**ProteinMPNN must tie symmetric positions.** Without `--homooligomer 1` each chain of
a Cn backbone is designed independently and the sequences cannot form a homo-oligomer.
`design.tie_chains: false` is rejected at startup.

A third is a matter of interpretation rather than a bug: AlphaFold2 is run in
single-sequence mode. A de novo design has no homologues, so an MSA search returns
either nothing or unrelated hits, and the resulting confidence is not meaningful.
`fold.msa_mode` is flagged at startup if it is set to anything else.

## Testing

```bash
pytest                     # 86 checks
python tests/run_all.py    # same suite without pytest
```

The suite runs against synthetic Cn oligomers built from ideal helices, so the
symmetry order, chain RMSD, helix fraction and clash count are all known in advance.
It needs no GPU, no model weights and no external tools, and it covers the stage 01
and stage 04 logic end to end by synthesising the outputs of the GPU stages.

## Status

Implemented and tested: stages 00-04, the filter engine, the run manifest, and the
conda backend.

Not yet implemented:

- **Stages 05-08** - fusion search, icosahedral assembly with clash and closure
  checks, junction redesign, HTML reporting.
- **Container images** - the backend abstraction already covers Docker and Apptainer;
  the images themselves are not built yet.
- **Threshold calibration** - the shipped thresholds are conservative starting points
  derived from general design practice, not values validated against experimental
  outcomes. Treat them as a starting point and retune against your own run.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
