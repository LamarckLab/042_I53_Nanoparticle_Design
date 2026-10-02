"""Minimal dependency-free PDB backbone reader.

RFdiffusion writes N/CA/C/O backbone-only PDBs; that is all any metric here needs.
Keeping this self-contained means the metric layer has zero external binaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BACKBONE_ATOMS = ("N", "CA", "C", "O")


@dataclass
class Chain:
    chain_id: str
    resnums: np.ndarray                      # (L,) int
    coords: dict[str, np.ndarray] = field(default_factory=dict)  # atom -> (L, 3)

    @property
    def n_res(self) -> int:
        return len(self.resnums)

    @property
    def ca(self) -> np.ndarray:
        return self.coords["CA"]


@dataclass
class Structure:
    chains: dict[str, Chain]
    path: str | None = None

    @property
    def chain_ids(self) -> list[str]:
        return list(self.chains.keys())

    @property
    def n_chains(self) -> int:
        return len(self.chains)

    def all_ca(self) -> np.ndarray:
        return np.concatenate([c.ca for c in self.chains.values()], axis=0)

    def all_backbone(self) -> np.ndarray:
        """(N, 3) stack of every backbone atom present, for clash detection."""
        out = []
        for c in self.chains.values():
            for name in BACKBONE_ATOMS:
                if name in c.coords:
                    out.append(c.coords[name])
        return np.concatenate(out, axis=0)


def read_pdb(path: str, atoms: tuple[str, ...] = BACKBONE_ATOMS) -> Structure:
    """Parse ATOM records into per-chain coordinate arrays.

    Only the first altloc and first MODEL are kept; residues missing CA are dropped.
    """
    per_chain: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    order: dict[str, list[int]] = {}

    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            name = line[12:16].strip()
            if name not in atoms:
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            cid = line[21]
            resnum = int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            res_map = per_chain.setdefault(cid, {})
            if resnum not in res_map:
                res_map[resnum] = {}
                order.setdefault(cid, []).append(resnum)
            res_map[resnum][name] = xyz

    chains: dict[str, Chain] = {}
    for cid, res_map in per_chain.items():
        resnums = [r for r in order[cid] if "CA" in res_map[r]]
        if not resnums:
            continue
        coords = {}
        for name in atoms:
            present = [r for r in resnums if name in res_map[r]]
            if len(present) == len(resnums):                     # only keep complete atom sets
                coords[name] = np.stack([res_map[r][name] for r in resnums])
        chains[cid] = Chain(chain_id=cid, resnums=np.array(resnums, dtype=int), coords=coords)

    if not chains:
        raise ValueError(f"no backbone atoms parsed from {path}")
    return Structure(chains=chains, path=path)


def write_pdb(path: str, structure: Structure) -> None:
    """Write backbone atoms back out; used by selftest fixtures and monomer extraction."""
    serial = 1
    with open(path, "w") as fh:
        for cid, chain in structure.chains.items():
            for i, resnum in enumerate(chain.resnums):
                for name in BACKBONE_ATOMS:
                    if name not in chain.coords:
                        continue
                    x, y, z = chain.coords[name][i]
                    fh.write(
                        f"ATOM  {serial:5d}  {name:<3s} GLY {cid}{resnum:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {name[0]}\n"
                    )
                    serial += 1
            fh.write("TER\n")
        fh.write("END\n")


def extract_monomer(structure: Structure, chain_id: str | None = None) -> Structure:
    """Return a single-chain Structure, relabelled to chain A."""
    cid = chain_id or structure.chain_ids[0]
    src = structure.chains[cid]
    return Structure(chains={"A": Chain("A", src.resnums.copy(), {k: v.copy() for k, v in src.coords.items()})})
