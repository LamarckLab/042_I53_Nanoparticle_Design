# Configuration

Configuration is layered, later sources winning:

```
configs/default.yaml  <-  --config my_run.yaml  <-  --profile <machine>.yaml  <-  --set k=v
```

`configs/default.yaml` is the complete parameter surface and is always loaded. A user
config and a machine profile each override a subset. `--set` takes dotted keys and
parses values as YAML, so `--set compute.gpus=[2,3]` gives a list.

Run `cyclo check` after editing anything. It validates the keys, verifies that every
filter rule parses against a representative metric set, and reports the derived
RFdiffusion contig length.

## Sections

### `run`
| Key | Meaning |
|---|---|
| `name` | Run label. |
| `outdir` | Output directory; relative paths resolve against the repository root. |
| `seed` | Base seed. Stage 00 derives per-batch seeds from it; stages 02 and 03 use it directly. |

### `target`
| Key | Meaning |
|---|---|
| `symmetry` | `C2` through `C12`. |
| `monomer_length` | Residues **per chain**. The total contig length handed to RFdiffusion is this times the symmetry order. |
| `n_backbones` | Total backbones to generate. |

### `generate`
Batch size, diffusion steps, and the oligomer-contact guiding potential. Anything in
`extra_args` is appended verbatim to `run_inference.py`, which is the escape hatch for
RFdiffusion options this package does not model.

### `backbone_filter`
A list of expression strings under `rules`. The grammar is comparisons, boolean
operators, arithmetic, and the functions `abs`, `min`, `max`, `len`, `round`, over the
metric names in [metrics.md](metrics.md). Attribute access, imports and calls to
anything else are rejected by the parser.

Each rule must evaluate to a boolean, so write `n_clash == 0` rather than `n_clash`.
The exception is a metric that is already boolean, such as `sym_order_ok`.

An empty rule list passes everything, which is the right setting for a first
exploratory run where you want the metric distributions before choosing thresholds.

### `design`
| Key | Meaning |
|---|---|
| `n_seq_per_backbone` | Sequences per backbone. The stage-04 success rate is a fraction of this, so small values give a coarse, noisy estimate. |
| `sampling_temp` | ProteinMPNN sampling temperature. |
| `use_soluble_model` | Use the soluble-protein weights. |
| `omit_aas` | Residues to exclude. The default `CX` avoids free cysteines. |
| `tie_chains` | Must be true. Rejected at startup otherwise. |

### `fold`
| Key | Meaning |
|---|---|
| `msa_mode` | Must be `single_sequence` for de novo designs; anything else is flagged. |
| `num_models`, `num_recycle` | One model and three recycles is the usual filtering setting. |
| `model_type` | `alphafold2_ptm` for monomers. |

### `validate`
| Key | Meaning |
|---|---|
| `rmsd_cutoff` | CA RMSD below which a sequence counts as folding back. |
| `plddt_cutoff` | Minimum mean pLDDT for the same. |
| `also_record` | Extra RMSD cutoffs stored as additional columns for later re-analysis. |
| `rules` | Expressions over the per-backbone rollup: `success_rate`, `best_rmsd`, `median_rmsd`, `n_seqs`, `n_seqs_folded`, `best_plddt`. |

### `compute`
| Key | Meaning |
|---|---|
| `gpus` | Device list. On a shared machine, set this to your share and no more. |
| `cuda_device_order` | Leave as `PCI_BUS_ID`. The CUDA default reorders devices by capability, so `CUDA_VISIBLE_DEVICES` otherwise selects a different card than intended. |

### `paths` and `backend`
`paths` locates the three external tools. `backend.kind` selects `conda`, `docker`,
`apptainer` or `local`; `backend.envs` names the conda environment per tool and
`backend.images` the container image per tool.

## On the shipped thresholds

They are conservative starting points derived from general design practice. They have
not been calibrated against experimental outcomes.

The intended workflow is to run once with an empty or permissive rule list, look at
the metric distributions in `tables/backbones.csv`, and set thresholds from what you
see. Because nothing is deleted, this can be done after the fact on a completed run:
change the rules and re-run stage 01 alone.
