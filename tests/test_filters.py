"""Declarative filter engine: correctness and sandboxing."""
from __future__ import annotations

from cyclo.filters import FilterSet, RuleError, evaluate
from cyclo.geometry import backbone_metrics
from synthetic import cyclic_oligomer

METRICS = backbone_metrics(cyclic_oligomer(n_sym=5), expected_sym=5)


def test_simple_comparisons():
    assert evaluate("n_clash == 0", METRICS)
    assert not evaluate("helix_frac > 0.99", METRICS)


def test_bare_boolean_metric():
    assert evaluate("sym_order_ok", METRICS)


def test_chained_comparison():
    assert evaluate("0 < helix_frac < 1", METRICS)


def test_boolean_operators():
    assert evaluate("n_clash == 0 and helix_frac > 0.5", METRICS)
    assert evaluate("n_clash > 100 or helix_frac > 0.5", METRICS)
    assert evaluate("not (helix_frac > 0.99)", METRICS)


def test_arithmetic_and_whitelisted_functions():
    assert evaluate("abs(sym_order_detected - 5) < 0.1", METRICS)
    assert evaluate("max(helix_frac, 0.1) > 0.1", METRICS)
    assert evaluate("n_clash_intra + n_clash_inter == n_clash", METRICS)


def test_rejects_statements_and_imports():
    for bad in ("import os", "__import__('os')", "open('x')", "n_clash.__class__"):
        try:
            evaluate(bad, METRICS)
            raise AssertionError(f"{bad!r} should have been rejected")
        except RuleError:
            pass


def test_rejects_unknown_metric():
    try:
        evaluate("no_such_metric > 1", METRICS)
        raise AssertionError("expected a RuleError")
    except RuleError:
        pass


def test_rejects_non_boolean_result():
    try:
        evaluate("n_clash", METRICS)
        raise AssertionError("expected a RuleError")
    except RuleError:
        pass


def test_filterset_reports_failed_rules():
    verdict = FilterSet("t", ["n_clash == 0", "helix_frac > 0.99"]).apply(METRICS)
    assert not verdict.passed
    assert verdict.failed_rules == ["helix_frac > 0.99"]
    assert "helix_frac" in verdict.reason


def test_filterset_passes_when_all_rules_hold():
    verdict = FilterSet("t", ["n_clash == 0", "helix_frac > 0.5"]).apply(METRICS)
    assert verdict.passed and verdict.failed_rules == []


def test_empty_filterset_passes_everything():
    assert FilterSet("t", []).apply(METRICS).passed


def test_summary_attributes_rejections_per_rule():
    fs = FilterSet("t", ["n_clash == 0", "helix_frac > 0.99"])
    summary = fs.summary([METRICS, METRICS])
    assert summary["n_input"] == 2
    assert summary["n_pass"] == 0
    assert summary["rejected_by_rule"]["helix_frac > 0.99"] == 2
    assert summary["rejected_by_rule"]["n_clash == 0"] == 0


def test_check_syntax_raises_on_bad_rule():
    try:
        FilterSet("t", ["helix_frac >>> 1"]).check_syntax(METRICS)
        raise AssertionError("expected a RuleError")
    except RuleError:
        pass
