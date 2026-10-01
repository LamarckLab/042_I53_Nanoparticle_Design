"""Config loading, derived values and startup validation."""
from __future__ import annotations

from cyclo.config import load_config, parse_overrides
from cyclo.filters import FilterSet
from cyclo.geometry import backbone_metrics
from synthetic import cyclic_oligomer


def _cfg(*overrides):
    return load_config(overrides=parse_overrides(list(overrides)))


def test_symmetry_order_parsed():
    assert _cfg("target.symmetry=C5").sym_order == 5
    assert _cfg("target.symmetry=c3").sym_order == 3


def test_unsupported_symmetry_rejected():
    try:
        _cfg("target.symmetry=D2").sym_order
        raise AssertionError("expected a ValueError")
    except ValueError:
        pass


def test_contig_length_is_total_not_per_chain():
    """RFdiffusion's symmetric config takes the total across all chains."""
    assert _cfg("target.symmetry=C5", "target.monomer_length=60").total_length == 300
    assert _cfg("target.symmetry=C3", "target.monomer_length=80").total_length == 240


def test_overrides_are_nested_and_typed():
    cfg = _cfg("design.n_seq_per_backbone=32", "compute.gpus=[2,3]")
    assert cfg.get("design.n_seq_per_backbone") == 32
    assert cfg.get("compute.gpus") == [2, 3]


def test_malformed_override_rejected():
    try:
        parse_overrides(["target.symmetry"])
        raise AssertionError("expected a ValueError")
    except ValueError:
        pass


def test_default_config_validates_cleanly():
    assert _cfg("target.symmetry=C5", "target.monomer_length=60").validate() == []


def test_untied_chains_rejected():
    problems = _cfg("design.tie_chains=false").validate()
    assert any("tie_chains" in p for p in problems)


def test_msa_mode_other_than_single_sequence_is_flagged():
    problems = _cfg("fold.msa_mode=mmseqs2_uniref_env").validate()
    assert any("msa_mode" in p for p in problems)


def test_null_gpus_is_allowed_and_leaves_the_card_choice_alone():
    from cyclo.runner import Runner
    assert _cfg("compute.gpus=null").validate() == []
    assert "CUDA_VISIBLE_DEVICES" not in Runner(kind="local", gpus=[]).env_vars()


def test_explicit_gpu_list_still_pins():
    from cyclo.runner import Runner
    env = Runner(kind="local", gpus=[2, 3]).env_vars()
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_implausible_monomer_length_rejected():
    assert any("monomer_length" in p for p in _cfg("target.monomer_length=5").validate())


def test_require_raises_for_missing_key():
    try:
        _cfg().require("no.such.key")
        raise AssertionError("expected a KeyError")
    except KeyError:
        pass


def test_shipped_default_rules_parse_against_real_metrics():
    cfg = _cfg("target.symmetry=C5")
    metrics = backbone_metrics(cyclic_oligomer(n_sym=5), expected_sym=5)
    FilterSet("backbone", cfg.get("backbone_filter.rules")).check_syntax(metrics)


def test_shipped_validate_rules_parse():
    cfg = _cfg()
    sample = {"success_rate": 0.5, "best_rmsd": 0.8, "median_rmsd": 1.1,
              "n_seqs": 10, "n_seqs_folded": 10, "best_plddt": 90.0}
    FilterSet("validate", cfg.get("validate.rules")).check_syntax(sample)


def test_compact_subunit_preset_parses():
    from synthetic import cyclic_oligomer
    cfg = load_config(profile="configs/presets/compact_subunit.yaml",
                      overrides=parse_overrides(["target.symmetry=C5"]))
    FilterSet("backbone", cfg.get("backbone_filter.rules")).check_syntax(
        backbone_metrics(cyclic_oligomer(n_sym=5), expected_sym=5))


def test_zero_seed_is_rejected():
    """ProteinMPNN reads --seed 0 as 'pick a random one', which breaks reproducibility."""
    assert any("run.seed" in p for p in _cfg("run.seed=0").validate())
    assert _cfg("run.seed=1").validate() == []
