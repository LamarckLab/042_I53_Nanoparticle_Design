"""Metric-layer tests against synthetic oligomers with known ground truth."""
from __future__ import annotations

import numpy as np

from cyclo.geometry import (backbone_metrics, count_clashes, rmsd, secondary_structure,
                            symmetry_frame, terminal_run)
from synthetic import clashing_oligomer, cyclic_oligomer, extended_monomer


# ------------------------------------------------------------------ symmetry
def test_symmetry_order_recovered():
    for n in (3, 5, 6, 8):
        detected = symmetry_frame(cyclic_oligomer(n_sym=n))["sym_order_detected"]
        assert abs(detected - n) < 0.01, f"C{n}: detected {detected}"


def test_symmetry_chains_superpose_exactly():
    for n in (3, 5, 6):
        assert symmetry_frame(cyclic_oligomer(n_sym=n))["sym_rmsd"] < 1e-6


def test_symmetry_axis_is_z():
    axis = symmetry_frame(cyclic_oligomer(n_sym=5))["axis"]
    assert abs(abs(float(np.dot(axis, [0, 0, 1]))) - 1.0) < 1e-6


def test_expected_symmetry_mismatch_is_flagged():
    m = backbone_metrics(cyclic_oligomer(n_sym=5), expected_sym=3)
    assert m["sym_order_ok"] is False
    assert backbone_metrics(cyclic_oligomer(n_sym=5), expected_sym=5)["sym_order_ok"] is True


# ------------------------------------------------------------------ rmsd
def test_rmsd_is_zero_for_identical_coordinates():
    ca = cyclic_oligomer(n_sym=5).chains["A"].ca
    assert rmsd(ca, ca) < 1e-9


def test_rmsd_invariant_under_rigid_motion():
    struct = cyclic_oligomer(n_sym=5)
    a = struct.chains["A"].ca
    assert rmsd(struct.chains["B"].ca, a) < 1e-6           # symmetry mate is a pure rotation
    assert rmsd(a + np.array([3.0, -2.0, 7.0]), a) < 1e-6


def test_rmsd_rejects_length_mismatch():
    ca = cyclic_oligomer(n_sym=5).chains["A"].ca
    try:
        rmsd(ca, ca[:-5])
        raise AssertionError("expected a shape mismatch error")
    except ValueError:
        pass


# ------------------------------------------------------------------ secondary structure
def test_helix_bundle_is_mostly_helix():
    ss = secondary_structure(cyclic_oligomer(n_sym=5).chains["A"].ca)
    assert ss.count("H") / len(ss) > 0.70
    assert ss.count("E") == 0                              # an all-alpha bundle has no strand
    assert "L" in ss                                       # loops must not be called helix


def test_terminal_run():
    assert terminal_run("HHHLLLHH", "H", "N") == 3
    assert terminal_run("HHHLLLHH", "H", "C") == 2
    assert terminal_run("LHHH", "H", "N") == 0


# ------------------------------------------------------------------ clashes
def test_separated_ring_is_clash_free():
    m = backbone_metrics(cyclic_oligomer(n_sym=5), expected_sym=5)
    assert m["n_clash"] == 0
    assert m["iface_contacts_per_chain"] > 0               # but the subunits do touch


def test_interpenetrating_ring_is_flagged():
    m = backbone_metrics(clashing_oligomer(n_sym=5), expected_sym=5)
    assert m["n_clash_inter"] > 0
    assert m["min_interchain_dist"] < 2.5


def test_interface_contacts_are_not_counted_as_clashes():
    """Regression: a 4 A CA-CA proxy flagged ordinary interface contacts as overlap."""
    struct = cyclic_oligomer(n_sym=5, ring_radius=16.0)
    clashes = count_clashes(struct)
    assert clashes["n_clash_inter"] == 0
    assert clashes["min_interchain_dist"] > 2.5


# ------------------------------------------------------------------ shape
def test_extended_backbone_is_flagged_by_compactness():
    assert backbone_metrics(extended_monomer(60))["rg_ratio"] > 1.5


def test_compact_bundle_passes_compactness():
    assert backbone_metrics(cyclic_oligomer(n_sym=5))["rg_ratio"] < 1.3


def test_ring_geometry_is_self_consistent():
    m = backbone_metrics(cyclic_oligomer(n_sym=5), expected_sym=5)
    assert 0 < m["pore_radius"] < m["max_radius"]
    assert m["height"] > 0


def test_metric_set_is_complete():
    m = backbone_metrics(cyclic_oligomer(n_sym=5), expected_sym=5)
    required = {
        "n_chains", "n_res_monomer", "helix_frac", "strand_frac", "loop_frac", "n_sse",
        "loop_max_len", "nterm_helix_len", "cterm_helix_len", "rg", "rg_ratio",
        "contact_order", "iface_contacts_per_chain", "pore_radius", "max_radius",
        "height", "sym_angle_deg", "sym_order_detected", "sym_rmsd", "n_clash",
        "n_clash_intra", "n_clash_inter", "min_interchain_dist", "sym_order_ok",
    }
    assert required <= set(m), f"missing: {sorted(required - set(m))}"
