# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Stage 00: symmetric backbone generation with RFdiffusion, with the per-chain to
  total contig-length conversion handled internally.
- Stage 01: objective backbone filtering. Twenty-nine geometric metrics are computed
  for every backbone and the pass/fail decision is expressed as config rules.
- Stage 02: ProteinMPNN sequence design with tied positions enforced.
- Stage 03: AlphaFold2 monomer prediction via localcolabfold in single-sequence mode.
- Stage 04: self-consistency RMSD and per-backbone success-rate rollup.
- Declarative filter engine with an AST whitelist, per-rule rejection accounting and
  startup syntax validation.
- Two-grain run manifest (backbones, sequences) with partial-upsert semantics and
  per-stage resume markers.
- Backend abstraction covering conda, Docker and Apptainer.
- Test suite of 86 checks against synthetic oligomers with known ground truth.

### Validated against real tools
- Stages 00 and 02 were executed against RFdiffusion and ProteinMPNN on a real GPU
  host. Symmetric output parses as five chains of 60 residues with continuous
  residue numbering, and tied-position design returns identical sequences on every
  chain. Both are now pinned by fixtures in `tests/data/`.
- Filter thresholds were recalibrated against 24 real backbones. The shipped
  defaults pass 71% of them and `configs/presets/compact_subunit.yaml` passes 33%.

### Changed
- The emitted RFdiffusion command now reproduces a verified hand-run invocation
  exactly: lowercase symmetry token, explicit `hydra.run.dir`, and no speculative
  keys. `inference.seed` does not exist in either shipped config and would have
  aborted the run; it is opt-in through `generate.seed_key`.
- Guiding potentials are off by default, matching that baseline invocation.
- GPU pinning is optional: `compute.gpus: null` leaves `CUDA_VISIBLE_DEVICES` unset.

### Added
- Gyration-tensor shape descriptors (`shape_anisotropy`, `asphericity`,
  `acylindricity`, `axis_ratio`) for the subunit, and the same set over the whole
  assembly, since the design target is a globular ring rather than a globular chain.
- `n_helices`, separating helix count from total secondary-structure element count.
- Optional `r_0` and `d_0` parameters on the oligomer-contact potential.

### Removed
- `configs/presets/globular_3helix.yaml`. It passed 1 backbone in 24 on real output
  and demanded a three-helix all-alpha fold that RFdiffusion rarely produces at these
  chain lengths. Replaced by `compact_subunit.yaml`.

### Not yet implemented
- Stage 03 execution: AlphaFold2 is not deployed yet, so only its command
  construction is covered.
- Fusion search and icosahedral assembly (stages 05-08).
- Container images.
