"""Stage 00 - symmetric backbone generation with RFdiffusion.

The emitted command mirrors a hand-run invocation that is known to work, and adds
nothing to it by default. Extra hydra keys are opt-in, because hydra rejects keys the
installed config does not define and an unsupported one fails the whole run.

The one conversion this stage performs is the contig length: the symmetric config
takes the TOTAL residue count across all chains, so a C5 of 60-residue subunits is
[300-300], not [60-60]. Passing the per-chain number silently produces a much smaller
oligomer, so `Config.total_length` computes it instead.
"""
from __future__ import annotations

from pathlib import Path

from .config import Config
from .manifest import RunState
from .runner import Runner

STAGE = "00_backbones"


def _num(value) -> str:
    """Format 1.0 as 1 and 0.1 as 0.1, matching how the flags are written by hand."""
    f = float(value)
    return str(int(f)) if f == int(f) else str(f)


def build_command(cfg: Config, out_prefix: Path, n_designs: int,
                  hydra_dir: Path | None = None, seed: int | None = None) -> list[str]:
    root = cfg.get("paths.rfdiffusion")
    script = f"{root}/scripts/run_inference.py" if root else "run_inference.py"

    argv = [
        "python", script,
        "--config-name", "symmetry",
        f"inference.symmetry={cfg.symmetry.lower()}",       # RFdiffusion expects c5, not C5
        f"contigmap.contigs=[{cfg.total_length}-{cfg.total_length}]",
    ]

    if cfg.get("generate.potentials.enabled", True):
        pot = cfg.get("generate.potentials.olig_contacts", {}) or {}
        terms = [
            "type:olig_contacts",
            f"weight_intra:{_num(pot.get('weight_intra', 1))}",
            f"weight_inter:{_num(pot.get('weight_inter', 0.1))}",
        ]
        # r_0 / d_0 exist on the potential but are left at the RFdiffusion defaults
        # unless set, so the emitted command stays identical to the verified one.
        for name in ("r_0", "d_0"):
            value = cfg.get(f"generate.potentials.{name}")
            if value is not None:
                terms.append(f"{name}:{_num(value)}")
        argv += [
            'potentials.guiding_potentials=["' + ",".join(terms) + '"]',
            "potentials.olig_intra_all=True",
            "potentials.olig_inter_all=True",
            f"potentials.guide_scale={_num(cfg.get('generate.potentials.guide_scale', 2))}",
            f"potentials.guide_decay={cfg.get('generate.potentials.guide_decay', 'quadratic')}",
        ]

    argv += [f"inference.output_prefix={out_prefix}"]
    if hydra_dir is not None:
        argv += [f"hydra.run.dir={hydra_dir}"]              # keep hydra logs out of the cwd
    argv += [f"inference.num_designs={n_designs}"]

    # Opt-in only: the seed key differs between RFdiffusion builds and an unknown
    # hydra key aborts the run, so it is emitted solely when configured explicitly.
    seed_key = cfg.get("generate.seed_key")
    if seed_key and seed is not None:
        argv += [f"{seed_key}={seed}"]

    return argv + list(cfg.get("generate.extra_args", []) or [])


def run(cfg: Config, state: RunState, runner: Runner, force: bool = False) -> list[dict]:
    if state.is_done(STAGE) and not force:
        print(f"[{STAGE}] already done, skipping")
        return state.backbones.load().to_dict("records")

    outdir = state.stage_dir(STAGE)
    configured = cfg.get("generate.hydra_run_dir")
    hydra_dir = Path(configured) if configured else outdir / "hydra"

    total = int(cfg.require("target.n_backbones"))
    batch = int(cfg.get("generate.batch_size", 10))
    base_seed = int(cfg.get("run.seed", 0))

    n_batches = (total + batch - 1) // batch
    for b in range(n_batches):
        n = min(batch, total - b * batch)
        prefix = outdir / f"batch{b:03d}"
        if len(sorted(outdir.glob(f"batch{b:03d}_*.pdb"))) >= n and not force:
            continue                                        # resume: this batch already landed
        runner.run("rfdiffusion",
                   build_command(cfg, prefix, n, hydra_dir / f"batch{b:03d}", base_seed + b))

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
