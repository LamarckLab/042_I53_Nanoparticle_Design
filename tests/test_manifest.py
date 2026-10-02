"""Run-state tables: partial upsert semantics and stage resume markers."""
from __future__ import annotations

import tempfile
from pathlib import Path

from cyclicnano.manifest import RunState


def _state():
    return RunState(Path(tempfile.mkdtemp(prefix="cyclicnano_manifest_")))


def test_empty_table_loads_as_empty_frame():
    assert _state().backbones.load().empty


def test_insert_then_read_back():
    state = _state()
    state.backbones.upsert([{"backbone_id": "a", "x": 1}, {"backbone_id": "b", "x": 2}])
    df = state.backbones.load()
    assert len(df) == 2
    assert sorted(df["backbone_id"]) == ["a", "b"]


def test_upsert_adds_columns_without_touching_others():
    state = _state()
    state.backbones.upsert([{"backbone_id": "a", "x": 1}, {"backbone_id": "b", "x": 2}])
    state.backbones.upsert([{"backbone_id": "a", "y": 9}])
    df = state.backbones.load().set_index("backbone_id")
    assert df.loc["a", "x"] == 1                            # earlier stage survives
    assert df.loc["a", "y"] == 9
    assert len(df) == 2                                     # no rows added


def test_upsert_overwrites_existing_values():
    state = _state()
    state.backbones.upsert([{"backbone_id": "a", "x": 1}])
    state.backbones.upsert([{"backbone_id": "a", "x": 42}])
    assert state.backbones.load().set_index("backbone_id").loc["a", "x"] == 42


def test_upsert_adds_new_rows():
    state = _state()
    state.backbones.upsert([{"backbone_id": "a", "x": 1}])
    state.backbones.upsert([{"backbone_id": "b", "x": 2}])
    assert len(state.backbones.load()) == 2


def test_upsert_deduplicates_within_one_call():
    state = _state()
    state.backbones.upsert([{"backbone_id": "a", "x": 1}, {"backbone_id": "a", "x": 2}])
    df = state.backbones.load()
    assert len(df) == 1 and df.iloc[0]["x"] == 2            # last write wins


def test_upsert_requires_the_key_column():
    state = _state()
    try:
        state.backbones.upsert([{"x": 1}])
        raise AssertionError("expected a ValueError")
    except ValueError:
        pass


def test_empty_upsert_is_a_noop():
    state = _state()
    state.backbones.upsert([{"backbone_id": "a", "x": 1}])
    state.backbones.upsert([])
    assert len(state.backbones.load()) == 1


def test_stage_markers():
    state = _state()
    assert not state.is_done("01")
    state.mark_done("01")
    assert state.is_done("01")
    state.clear_done("01")
    assert not state.is_done("01")


def test_tables_are_independent():
    state = _state()
    state.backbones.upsert([{"backbone_id": "a"}])
    state.sequences.upsert([{"sequence_id": "a__seq000", "backbone_id": "a"}])
    assert len(state.backbones.load()) == 1
    assert len(state.sequences.load()) == 1
