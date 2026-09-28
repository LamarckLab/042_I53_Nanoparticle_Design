"""Config loading: default.yaml, optionally overlaid with a machine profile and CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

SYMMETRY_ORDERS = {f"C{i}": i for i in range(2, 13)}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """Dotted read access over the merged config dict."""

    def __init__(self, data: dict):
        self.data = data

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted, None)
        if value is None:
            raise KeyError(f"config is missing required key: {dotted}")
        return value

    # -- derived values -----------------------------------------------------
    @property
    def symmetry(self) -> str:
        return str(self.require("target.symmetry")).upper()

    @property
    def sym_order(self) -> int:
        sym = self.symmetry
        if sym not in SYMMETRY_ORDERS:
            raise ValueError(f"unsupported symmetry {sym!r}; expected one of {sorted(SYMMETRY_ORDERS)}")
        return SYMMETRY_ORDERS[sym]

    @property
    def monomer_length(self) -> int:
        return int(self.require("target.monomer_length"))

    @property
    def total_length(self) -> int:
        """RFdiffusion symmetric mode takes the TOTAL contig length, not per-chain.

        Getting this wrong silently produces oligomers of the wrong size, so the
        conversion lives here rather than in the caller.
        """
        return self.sym_order * self.monomer_length

    @property
    def outdir(self) -> Path:
        return (REPO_ROOT / str(self.require("run.outdir"))).resolve() \
            if not Path(str(self.require("run.outdir"))).is_absolute() \
            else Path(str(self.require("run.outdir")))

    def validate(self) -> list[str]:
        """Cheap startup checks; returns a list of human-readable problems."""
        problems = []
        try:
            self.sym_order
        except ValueError as exc:
            problems.append(str(exc))
        if self.monomer_length < 20:
            problems.append(f"target.monomer_length={self.monomer_length} is implausibly short")
        if int(self.get("design.n_seq_per_backbone", 0)) < 1:
            problems.append("design.n_seq_per_backbone must be >= 1")
        if not self.get("design.tie_chains", True):
            problems.append("design.tie_chains=false will give each chain a different "
                            "sequence, which cannot form a homo-oligomer")
        if self.get("fold.msa_mode") != "single_sequence":
            problems.append("fold.msa_mode is not single_sequence; de novo designs have no "
                            "homologues and an MSA yields misleading confidence")
        if not self.get("compute.gpus"):
            problems.append("compute.gpus is empty")
        return problems


def load_config(path: str | Path | None = None,
                profile: str | Path | None = None,
                overrides: dict | None = None) -> Config:
    """default.yaml <- user config <- machine profile <- CLI overrides."""
    data = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")) or {}
    for extra in (path, profile):
        if extra:
            p = Path(extra)
            if not p.is_absolute():
                p = REPO_ROOT / p
            data = _deep_merge(data, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data)


def parse_overrides(pairs: list[str]) -> dict:
    """Turn ['target.symmetry=C3', 'target.monomer_length=80'] into a nested dict."""
    out: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"override must be key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out
