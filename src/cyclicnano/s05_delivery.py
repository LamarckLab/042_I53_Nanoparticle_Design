"""Stage 05 - assemble the designs that passed into a handover folder.

Stage 04 decides which backbones pass; this stage only collects them. For each one
it writes the designed pentamer, the best of its sequences, and a row in a summary
table. The split matters: the PDB is what you look at, the FASTA is what gets
ordered, and neither substitutes for the other.

The structure files are poly-glycine backbones, as RFdiffusion wrote them. The
designed sequence is not threaded onto them.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .config import Config
from .manifest import RunState

STAGE = "05_delivery"


def best_sequence(sequences: pd.DataFrame, backbone_id: str) -> dict | None:
    """The sequence of this backbone that folded back closest to the design."""
    group = sequences[sequences["backbone_id"] == backbone_id]
    group = group[group["sc_rmsd"].notna()]
    return group.nsmallest(1, "sc_rmsd").iloc[0].to_dict() if len(group) else None


def run(cfg: Config, state: RunState, force: bool = False) -> dict:
    if state.is_done(STAGE) and not force:
        print(f"[{STAGE}] already done, skipping")
        return {}

    backbones = state.backbones.load()
    sequences = state.sequences.load()
    if backbones.empty:
        raise RuntimeError("no backbones in the manifest; run the earlier stages first")
    if "final_pass" not in backbones.columns:
        raise RuntimeError("no stage 04 verdict in the manifest; run stage 04 first")

    outdir = state.stage_dir(STAGE)
    for stale in outdir.iterdir():                          # a rerun must not leave old hits behind
        stale.unlink() if stale.is_file() else shutil.rmtree(stale, ignore_errors=True)

    passing = backbones[backbones["final_pass"].fillna(False).astype(bool)]
    passing = passing.sort_values(["success_rate", "best_rmsd"], ascending=[False, True])

    rows = []
    for record in passing.to_dict("records"):
        bid = record["backbone_id"]
        best = best_sequence(sequences, bid)
        if best is None:
            continue

        source = Path(str(record["backbone_path"]))
        if source.exists():
            shutil.copy2(source, outdir / f"{bid}.pdb")

        (outdir / f"{bid}.fasta").write_text(
            f">{bid} success_rate={record['success_rate']:.2f} "
            f"sc_rmsd={best['sc_rmsd']:.2f} plddt={best.get('mean_plddt', float('nan')):.1f}\n"
            f"{best['sequence']}\n",
            encoding="utf-8",
        )
        rows.append({
            "backbone_id": bid,
            "symmetry": record.get("symmetry"),
            "n_res_monomer": record.get("n_res_monomer"),
            "success_rate": record.get("success_rate"),
            "best_rmsd": best["sc_rmsd"],
            "best_plddt": best.get("mean_plddt"),
            "best_sequence_id": best["sequence_id"],
            "sequence": best["sequence"],
            "pdb": f"{bid}.pdb",
            "fasta": f"{bid}.fasta",
        })

    summary = outdir / "delivery.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)

    print(f"[{STAGE}] {len(rows)} designs delivered to {outdir}")
    if rows:
        print(f"[{STAGE}] best: {rows[0]['backbone_id']} "
              f"(success_rate {rows[0]['success_rate']:.2f}, rmsd {rows[0]['best_rmsd']:.2f} A)")
    state.mark_done(STAGE)
    return {"n_delivered": len(rows), "summary": str(summary)}
