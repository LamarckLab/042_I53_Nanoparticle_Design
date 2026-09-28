# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Stage 00: symmetric backbone generation with RFdiffusion, with the per-chain to
  total contig-length conversion handled internally.
- Stage 01: objective backbone filtering. Twenty-four geometric metrics are computed
  for every backbone and the pass/fail decision is expressed as config rules.
- Stage 02: ProteinMPNN sequence design with tied positions enforced.
- Stage 03: AlphaFold2 monomer prediction via localcolabfold in single-sequence mode.
- Stage 04: self-consistency RMSD and per-backbone success-rate rollup.
- Declarative filter engine with an AST whitelist, per-rule rejection accounting and
  startup syntax validation.
- Two-grain run manifest (backbones, sequences) with partial-upsert semantics and
  per-stage resume markers.
- Backend abstraction covering conda, Docker and Apptainer.
- Test suite of 73 checks against synthetic oligomers with known ground truth.

### Not yet implemented
- Fusion search and icosahedral assembly (stages 05-08).
- Container images.
