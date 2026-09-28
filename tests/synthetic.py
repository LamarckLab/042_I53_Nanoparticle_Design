"""Build synthetic Cn oligomers with known ground truth.

Used by the self-test so the metric layer can be checked without RFdiffusion. Each
monomer is an ideal antiparallel helical hairpin; copies are placed by exact rotation
about z, so the symmetry order, chain RMSD and helix fraction are all known in advance.
Loop bond lengths are approximate on purpose - only the helices need ideal geometry.
"""
from __future__ import annotations

import numpy as np

from cyclo.pdbio import Chain, Structure

HELIX_RADIUS = 2.3          # ideal alpha-helix CA radius, angstrom
HELIX_RISE = 1.5            # rise per residue along the axis
HELIX_TWIST = 100.0         # degrees per residue


def ideal_helix(n: int, origin=(0.0, 0.0, 0.0), direction: int = 1, phase: float = 0.0) -> np.ndarray:
    """CA trace of an ideal alpha helix running along +z (direction=1) or -z."""
    i = np.arange(n)
    ang = np.radians(HELIX_TWIST * i * direction + phase)
    return np.stack([
        origin[0] + HELIX_RADIUS * np.cos(ang),
        origin[1] + HELIX_RADIUS * np.sin(ang),
        origin[2] + HELIX_RISE * i * direction,
    ], axis=1)


def _bridge(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    """Arcing loop between two helix termini; bond lengths are approximate by design."""
    ts = np.linspace(0.0, 1.0, n + 2)[1:-1]
    arc = np.stack([start + t * (end - start) for t in ts])
    arc[:, 2] += 4.0 * np.sin(np.pi * ts)
    return arc


def helix_bundle(n_helices: int = 3, per_helix: int = 18, loop: int = 3,
                 bundle_radius: float = 6.0) -> np.ndarray:
    """Up-down helix bundle: the compact topology a real designed subunit usually has.

    A two-helix hairpin is legitimately elongated and trips the compactness filter, so
    the fixture uses a bundle instead.
    """
    segments, prev_end = [], None
    for k in range(n_helices):
        ang = 2.0 * np.pi * k / n_helices
        centre = (bundle_radius * np.cos(ang), bundle_radius * np.sin(ang), 0.0)
        direction = 1 if k % 2 == 0 else -1
        z0 = 0.0 if direction == 1 else per_helix * HELIX_RISE
        h = ideal_helix(per_helix, origin=(centre[0], centre[1], z0),
                        direction=direction, phase=180.0 * k)
        if prev_end is not None:
            segments.append(_bridge(prev_end, h[0], loop))
        segments.append(h)
        prev_end = h[-1]
    return np.concatenate(segments, axis=0)


def helical_hairpin(h1: int = 28, loop: int = 4, h2: int = 28, separation: float = 14.0) -> np.ndarray:
    """Two antiparallel ideal helices joined by an arcing loop (deliberately elongated)."""
    a = ideal_helix(h1, origin=(0.0, 0.0, 0.0), direction=1)
    b = ideal_helix(h2, origin=(separation, 0.0, a[-1, 2]), direction=-1, phase=180.0)
    return np.concatenate([a, _bridge(a[-1], b[0], loop), b], axis=0)


def _rotz(deg: float) -> np.ndarray:
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def cyclic_oligomer(n_sym: int = 5, ring_radius: float = 16.0, monomer=None, **kwargs) -> Structure:
    """Place n_sym copies of a subunit on a ring, related by exact Cn rotation."""
    mono = helix_bundle(**kwargs) if monomer is None else monomer
    mono = mono - mono.mean(0) + np.array([ring_radius, 0.0, 0.0])

    chains = {}
    for k in range(n_sym):
        coords = mono @ _rotz(360.0 * k / n_sym).T
        cid = "ABCDEFGHIJKL"[k]
        chains[cid] = Chain(cid, np.arange(1, len(coords) + 1), {"CA": coords})
    return Structure(chains=chains)


def clashing_oligomer(n_sym: int = 5) -> Structure:
    """Same construction with the ring tightened until subunits interpenetrate."""
    return cyclic_oligomer(n_sym=n_sym, ring_radius=12.0)


def extended_monomer(n: int = 60) -> Structure:
    """A single straight helix: compact enough locally, but far too extended overall."""
    coords = ideal_helix(n)
    return Structure(chains={"A": Chain("A", np.arange(1, n + 1), {"CA": coords})})
