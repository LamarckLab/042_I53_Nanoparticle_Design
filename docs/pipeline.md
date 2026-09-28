# Pipeline stages

Each stage reads the run manifest, does its work, writes results back, and drops a
marker in `.done/`. A stage with a marker is skipped on the next invocation unless
`--force` is passed, so an interrupted run resumes where it stopped.

```
00_backbones ──> 01_backbone_filter ──> 02_sequences ──> 03_predictions ──> 04_validate
   GPU                   CPU                 GPU              GPU              CPU
```

## 00 - Backbone generation

RFdiffusion in symmetric mode, invoked in batches of `generate.batch_size` with a seed
derived from `run.seed` plus the batch index.

The only subtlety is the contig length. The symmetric config of RFdiffusion expects
the **total** residue count across all chains, so a C5 of 60-residue subunits is
`[300-300]`. `Config.total_length` performs `sym_order x monomer_length`; the stage
never sees the per-chain number.

Output: `00_backbones/batch*_*.pdb`, one row per backbone in `tables/backbones.csv`.

## 01 - Backbone filtering

Computes 24 metrics per backbone (see [metrics.md](metrics.md)) and evaluates the
`backbone_filter.rules` expressions against them. Sets `bb_pass` and records
`bb_fail_reason` listing every rule that failed.

A backbone whose PDB cannot be parsed is marked failed rather than crashing the run.

Output: metrics on every backbone row, plus `01_backbone_filter/filter_summary.json`
giving the rejection count attributable to each individual rule.

This is the only stage worth re-running on its own. It is CPU-bound and takes seconds,
so changing a threshold and running `cyclo run --stages 01 --force` is cheap.

## 02 - Sequence design

ProteinMPNN over the backbones with `bb_pass` true. Three calls:
`parse_multiple_chains.py`, then `make_tied_positions_dict.py --homooligomer 1`, then
`protein_mpnn_run.py`.

Tied positions are mandatory. Without them each chain of a Cn backbone is designed
independently and the resulting sequences cannot assemble; `design.tie_chains: false`
is rejected during `cyclo check`.

The output FASTA carries the chains joined by `/`. The parser drops the first entry
(the input sequence), splits on `/`, and records whether the chains came back
identical as a sanity check that tying actually took effect.

Output: `tables/sequences.csv`, one row per designed sequence.

## 03 - Structure prediction

localcolabfold in `--msa-mode single_sequence`, monomer only, templates off.

An MSA is not merely unnecessary for a de novo design, it is actively misleading: the
sequence has no homologues, so the search returns nothing useful and any confidence
derived from it is not interpretable. `cyclo check` flags any other `fold.msa_mode`.

Only the monomer is predicted. This validates that the designed sequence encodes the
designed fold; it does not validate that the subunits assemble into the ring. That
limitation is inherent to the protocol, not to this implementation.

Output: rank-1 PDB and score JSON per sequence; `pred_path`, `mean_plddt`, `min_plddt`
and `ptm` written back to the sequence table.

## 04 - Self-consistency

Per sequence: CA RMSD between the predicted monomer and chain A of the design
backbone, via Kabsch superposition. A sequence passes when the RMSD is at or below
`validate.rmsd_cutoff` and its mean pLDDT is at or above `validate.plddt_cutoff`.

Per backbone: `success_rate` is the fraction of its sequences that pass, and is the
quantity the ranking is built on. `best_rmsd` and `median_rmsd` are recorded alongside.

Additional cutoffs listed in `validate.also_record` are stored as separate columns, so
a stricter or looser definition of success can be applied later without re-predicting
anything.

Output: `04_validate/ranked_backbones.csv`, sorted by success rate then best RMSD.

## Not yet implemented

| Stage | Purpose |
|---|---|
| 05 | Fusion search: scan building-block pairs and helical fusion registers for geometries whose C5 and C3 axes meet at the icosahedral 37.3774 degree angle. |
| 06 | Assembly: expand the fused subunit by the 60 icosahedral rotations, then check for clashes and confirm the shell actually closes. |
| 07 | Junction redesign: redesign the fusion helix and the new inter-component contacts with the building-block bodies held fixed. |
| 08 | Reporting. |
