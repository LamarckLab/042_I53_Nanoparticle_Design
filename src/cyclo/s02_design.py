"""Stage 02 - sequence design with ProteinMPNN.

The one non-negotiable here is tied positions. Without `--homooligomer 1` each chain
of a Cn backbone is designed independently and the resulting sequences cannot form a
homo-oligomer at all, so the config value is asserted rather than trusted.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .config import Config
from .manifest import RunState
from .runner import Runner

STAGE = "02_sequences"


def _mpnn(cfg: Config, script: str) -> str:
    root = cfg.get("paths.proteinmpnn")
    return f"{root}/{script}" if root else script


def build_commands(cfg: Config, pdb_dir: Path, work: Path) -> list[tuple[str, list[str]]]:
    parsed, tied = work / "parsed_chains.jsonl", work / "tied_positions.jsonl"
    n_seq = int(cfg.get("design.n_seq_per_backbone", 10))
    temp = str(cfg.get("design.sampling_temp", 0.1))

    cmds = [
        ("proteinmpnn", ["python", _mpnn(cfg, "helper_scripts/parse_multiple_chains.py"),
                         f"--input_path={pdb_dir}", f"--output_path={parsed}"]),
    ]
    if cfg.get("design.tie_chains", True):
        cmds.append(("proteinmpnn", ["python", _mpnn(cfg, "helper_scripts/make_tied_positions_dict.py"),
                                     f"--input_path={parsed}", f"--output_path={tied}",
                                     "--homooligomer", "1"]))

    run_cmd = ["python", _mpnn(cfg, "protein_mpnn_run.py"),
               "--jsonl_path", str(parsed),
               "--out_folder", str(work),
               "--num_seq_per_target", str(n_seq),
               "--sampling_temp", temp,
               "--seed", str(int(cfg.get("run.seed", 0))),
               "--batch_size", "1"]
    if cfg.get("design.tie_chains", True):
        run_cmd += ["--tied_positions_jsonl", str(tied)]
    if cfg.get("design.use_soluble_model", False):
        run_cmd += ["--use_soluble_model"]
    if cfg.get("design.omit_aas"):
        run_cmd += ["--omit_AAs", str(cfg.get("design.omit_aas"))]

    cmds.append(("proteinmpnn", run_cmd))
    return cmds


def parse_fasta(path: Path) -> list[dict]:
    """Read a ProteinMPNN output FASTA.

    Entry 0 is the input sequence and is skipped. Tied designs are emitted as
    chains joined by '/', which collapse back to a single monomer sequence.
    """
    entries, header, chunks = [], None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if header is not None:
                entries.append((header, "".join(chunks)))
            header, chunks = line[1:], []
        elif line.strip():
            chunks.append(line.strip())
    if header is not None:
        entries.append((header, "".join(chunks)))

    out = []
    for i, (head, seq) in enumerate(entries[1:]):           # entry 0 is the input sequence
        parts = [p for p in seq.split("/") if p]
        meta = {}
        for field in head.split(","):
            if "=" in field:
                k, v = field.split("=", 1)
                try:
                    meta[k.strip()] = float(v)
                except ValueError:
                    meta[k.strip()] = v.strip()
        out.append({
            "seq_index": i,
            "sequence": parts[0],
            "n_chains_in_fasta": len(parts),
            "chains_identical": len(set(parts)) == 1,
            "mpnn_score": meta.get("score"),
            "mpnn_global_score": meta.get("global_score"),
            "mpnn_seq_recovery": meta.get("seq_recovery"),
        })
    return out


def run(cfg: Config, state: RunState, runner: Runner, force: bool = False) -> list[dict]:
    if state.is_done(STAGE) and not force:
        print(f"[{STAGE}] already done, skipping")
        return state.sequences.load().to_dict("records")

    if not cfg.get("design.tie_chains", True):
        raise ValueError("design.tie_chains=false cannot produce a homo-oligomer sequence")

    df = state.backbones.load()
    if "bb_pass" in df.columns:
        df = df[df["bb_pass"].astype(bool)]
    if df.empty:
        raise RuntimeError("no backbones passed stage 01; loosen backbone_filter.rules")

    work = state.stage_dir(STAGE)
    staged = work / "input_pdbs"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True)
    for record in df.to_dict("records"):                    # only passing backbones go to the GPU
        shutil.copy2(record["backbone_path"], staged / f"{record['backbone_id']}.pdb")

    for tool, argv in build_commands(cfg, staged, work):
        runner.run(tool, argv)

    rows = []
    for fasta in sorted((work / "seqs").glob("*.fa")):
        bid = fasta.stem
        for entry in parse_fasta(fasta):
            rows.append({
                "sequence_id": f"{bid}__seq{entry['seq_index']:03d}",
                "backbone_id": bid,
                "stage": STAGE,
                **entry,
            })
    state.sequences.upsert(rows)
    if not runner.dry_run:                                   # a dry run must leave no trace
        state.mark_done(STAGE)
    print(f"[{STAGE}] {len(rows)} sequences from {len(df)} backbones")
    return rows
