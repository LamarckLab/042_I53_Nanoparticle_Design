"""PDB parsing and writing."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from cyclicnano.pdbio import extract_monomer, read_pdb, write_pdb
from synthetic import cyclic_oligomer


def _roundtrip(struct):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.pdb"
        write_pdb(str(path), struct)
        return read_pdb(str(path))


def test_roundtrip_preserves_chains_and_residues():
    src = cyclic_oligomer(n_sym=5)
    back = _roundtrip(src)
    assert back.n_chains == 5
    assert back.chains["A"].n_res == src.chains["A"].n_res


def test_roundtrip_preserves_coordinates_to_pdb_precision():
    src = cyclic_oligomer(n_sym=5)
    back = _roundtrip(src)
    assert float(np.abs(back.chains["A"].ca - src.chains["A"].ca).max()) < 1e-3


def test_extract_monomer_relabels_to_chain_a():
    mono = extract_monomer(cyclic_oligomer(n_sym=5), chain_id="C")
    assert mono.chain_ids == ["A"]
    assert mono.chains["A"].n_res == 60


def test_extract_monomer_copies_rather_than_aliases():
    src = cyclic_oligomer(n_sym=5)
    mono = extract_monomer(src)
    mono.chains["A"].coords["CA"][0] += 100.0
    assert float(np.abs(src.chains["A"].ca[0] - mono.chains["A"].ca[0]).max()) > 50.0


def test_missing_file_raises():
    try:
        read_pdb("does_not_exist.pdb")
        raise AssertionError("expected an error")
    except (FileNotFoundError, OSError):
        pass


def test_empty_pdb_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "empty.pdb"
        path.write_text("HEADER nothing here\nEND\n", encoding="utf-8")
        try:
            read_pdb(str(path))
            raise AssertionError("expected a ValueError")
        except ValueError:
            pass
