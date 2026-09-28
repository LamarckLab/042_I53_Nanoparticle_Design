"""Stage 00 - symmetric backbone generation with RFdiffusion.

Two things this stage exists to get right, both of which are easy to get wrong by hand:

  * RFdiffusion's symmetric config takes the TOTAL contig length across all chains,
    not the per-chain length. `Config.total_length` does that conversion.
  * Every batch gets an explicit, derived seed, so a run is reproducible end to end.
"""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .manifest import RunState
from .runner import Runner

STAGE = "00_backbones"


def _contig(cfg: Config) -> str:
    n = cfg.total_length
    return f"[{n}-{n}]"


def build_command(cfg: Config, out_prefix: Path, n_designs: int, seed: int) -> list[str]:
    root = cfg.get("paths.rfdiffusion")
    script = f"{root}/scripts/run_inference.py" if root else "run_inference.py"
    pot = cfg.get("generate.potentials.olig_contacts", {}) or {}
    intra = pot.get("weight_intra", 1.0)
    inter = pot.get("weight_inter", 0.1)

    argv = [
        "python", script,
        "--config-name", "symmetry",
        f"inference.symmetry={cfg.symmetry}",
        f"inference.num_designs={n_designs}",
        f"inference.output_prefix={out_prefix}",
        f"inference.seed={seed}",
        f"contigmap.contigs={_contig(cfg)}",
        f"diffuser.T={cfg.get('generate.num_steps', 50)}",
        f'potentials.guiding_potentials=["type:olig_contacts,'
        f'weight_intra:{intra},weight_inter:{inter}"]',
        "potentials.olig_intra_all=True",
        "potentials.olig_inter_all=True",
        f"potentials.guide_scale={cfg.get('generate.potentials.guide_scale', 2.0)}",
        f"potentials.guide_decay={cfg.get('generate.potentials.guide_decay', 'quadratic')}",
    ]
    return argv + list(cfg.get("generate.extra_args", []) or [])


def run(cfg: Config, state: RunState, runner: Runner, force: bool = False) -> list[dict]:
    if state.is_done(STAGE) and not force:
        print(f"[{STAGE}] already done, skipping")
        return state.backbones.load().to_dict("records")

    outdir = state.stage_dir(STAGE)
    total = int(cfg.require("target.n_backbones"))
    batch = int(cfg.get("generate.batch_size", 10))
    base_seed = int(cfg.get("run.seed", 0))

    n_batches = (total + batch - 1) // batch
    for b in range(n_batches):
        n = min(batch, total - b * batch)
        prefix = outdir / f"batch{b:03d}"
        if len(sorted(outdir.glob(f"batch{b:03d}_*.pdb"))) >= n and not force:
            continue                                        # resume: this batch already landed
        runner.run("rfdiffusion", build_command(cfg, prefix, n, base_seed + b))

    rows = []
    for path in sorted(outdir.glob("*.pdb")):
        rows.append({
            "backbone_id": path.stem,
            "backbone_path": str(path),
            "symmetry": cfg.symmetry,
            "monomer_length_requested": cfg.monomer_length,
            "stage": STAGE,
        })
    state.backbones.upsert(rows)
    state.mark_done(STAGE)
    print(f"[{STAGE}] {len(rows)} backbones in {outdir}")
    return rows
