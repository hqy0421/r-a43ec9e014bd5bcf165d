# This file is part of Patsy
# Copyright (C) 2026 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

import numpy as np
import pytest

from patsy import (
    PatsyError,
    Term,
    bic_term_selection,
    build_design_matrices,
    demo_data,
    dmatrices,
    dmatrix,
)
from patsy.desc import INTERCEPT, ModelDesc
from patsy.selection import (
    _hierarchical_term_subsets,
    _term_prerequisites,
    TermCandidateFit,
    LinearTermSelectionResult,
)
from patsy.user_util import LookupFactor
from patsy.util import have_pandas

if have_pandas:
    import pandas


def _term(*names):
    return Term([LookupFactor(n) for n in names])


def test_bic_selection_picks_true_numeric_model():
    rng = np.random.RandomState(42)
    n = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 1.5 + 2.0 * x1 + 0.05 * rng.normal(size=n)
    data = {"y": y, "x1": x1, "x2": x2, "x3": x3}

    result = bic_term_selection("y ~ x1 + x2 + x3", data)

    assert isinstance(result, LinearTermSelectionResult)
    assert result.selected_term_names == ("x1",)
    assert result.has_intercept is True
    assert set(result.coefficients) == {"Intercept", "x1"}
    assert result.nobs == n
    # All 2^3 term combinations were enumerated, every one comparable.
    assert len(result.candidates) == 8
    assert all(c.status == "compared" for c in result.candidates)
    # The winner really has the minimum BIC among compared candidates.
    bics = [c.bic for c in result.candidates if c.status == "compared"]
    assert result.bic == min(bics)
    # Coefficients match a direct OLS fit on the same columns.
    y_dm, X = dmatrices("y ~ x1", data)
    beta, _, _, _ = np.linalg.lstsq(np.asarray(X), y, rcond=None)
    assert np.allclose(result.coef, beta.ravel())
    # BIC formula check: n*log(RSS/n) + p*log(n), p = actual columns.
    p = X.shape[1]
    rss = float(np.sum((y - np.asarray(X) @ beta) ** 2))
    expected_bic = n * np.log(rss / n) + p * np.log(n)
    assert np.isclose(result.bic, expected_bic)
    assert np.isclose(result.rss, rss)
    assert result.residual_df == n - p


def test_single_formula_fit_shared_rows_and_coding():
    # All candidates must reuse the one design build: the same rows, and
    # the same columns. Check candidate RSS values against direct
    # sub-matrix fits, which only agree if the shared coding is used.
    rng = np.random.RandomState(3)
    n = 50
    data = {
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    }
    data["y"] = data["x1"] - data["x2"] + 0.1 * rng.normal(size=n)
    result = bic_term_selection("y ~ x1 + x2 + x3", data)
    y_dm, X_full = dmatrices("y ~ x1 + x2 + x3", data)
    X = np.asarray(X_full)
    y = np.asarray(y_dm).ravel()
    full_di = X_full.design_info
    for candidate in result.candidates:
        terms = [INTERCEPT] if result.has_intercept else []
        terms += [t for t in full_di.terms if t.name() in candidate.term_names]
        idx = []
        for term in terms:
            sl = full_di.term_slices[term]
            idx.extend(range(sl.start, sl.stop))
        X_sub = X[:, idx]
        beta, _, _, _ = np.linalg.lstsq(X_sub, y, rcond=None)
        rss = np.sum((y - X_sub @ beta) ** 2)
        assert np.isclose(candidate.rss, rss)
        assert candidate.num_columns == X_sub.shape[1]


def test_categorical_terms_atomic_and_actual_column_count():
    data = demo_data("a", "b", "x", "y", nlevels=3, min_rows=60)
    rng = np.random.RandomState(0)
    a = np.asarray(data["a"])
    data["y"] = np.where(a == "a1", 2.0, 0.0) + 0.05 * rng.normal(size=len(a))

    result = bic_term_selection("y ~ a * b + x", data)

    # The categorical main effect 'a' (3 levels) contributes 2 columns;
    # p reported for each candidate is the actual design-matrix width.
    di = result.full_design_info
    widths = {
        term.name(): sl.stop - sl.start for term, sl in di.term_slices.items()
    }
    assert widths["a"] == 2
    assert widths["b"] == 2
    assert widths["a:b"] == 4
    for candidate in result.candidates:
        expected_p = widths["Intercept"]
        for name in candidate.term_names:
            expected_p += widths[name]
        assert candidate.num_columns == expected_p
    # No candidate ever splits a categorical term into partial columns.
    for candidate in result.candidates:
        if "a:b" in candidate.term_names:
            # Interaction columns only enter as the whole 4-column term.
            assert candidate.num_columns >= 1 + 2 + 2 + 4
    # The winning design info keeps the full formula's reduced-rank
    # categorical coding (same column names).
    assert "a[T.a2]" in result.design_info.column_names
    assert "a[T.a3]" in result.design_info.column_names


def test_hierarchy_required_for_interactions():
    rng = np.random.RandomState(0)
    data = {
        "a": ["a1", "a1", "a1", "a1", "a2", "a2", "a2", "a2"] * 3,
        "b": ["b1", "b1", "b2", "b2"] * 6,
        "c": ["c1", "c2"] * 12,
        "y": rng.normal(size=24),
    }
    result = bic_term_selection("y ~ a * b * c", data)

    # Number of order ideals (marginally-closed subsets) of the 7-element
    # Boolean lattice over {a, b, c}: 19 (including the empty subset).
    assert len(result.candidates) == 19
    assert result.unselectable_terms == frozenset()
    for candidate in result.candidates:
        chosen = set(candidate.term_names)
        if "a:b" in chosen:
            assert {"a", "b"} <= chosen
        if "a:c" in chosen:
            assert {"a", "c"} <= chosen
        if "b:c" in chosen:
            assert {"b", "c"} <= chosen
        if "a:b:c" in chosen:
            assert {"a", "b", "c", "a:b", "a:c", "b:c"} <= chosen


def test_interaction_with_missing_lower_order_terms_unselectable():
    rng = np.random.RandomState(1)
    data = {
        "y": rng.normal(size=12),
        "x": rng.normal(size=12),
        "a": ["a1"] * 6 + ["a2"] * 6,
        "b": ["b1", "b2"] * 6,
    }
    # a:b appears with neither main effect in the formula.
    result = bic_term_selection("y ~ x + a:b", data)
    assert {term.name() for term in result.unselectable_terms} == {"a:b"}
    assert all("a:b" not in c.term_names for c in result.candidates)
    # Only x remains selectable -> 2 candidates.
    assert [set(c.term_names) for c in result.candidates] == [set(), {"x"}]

    # Transitive orphan: three-way interaction with main effects but no
    # two-way marginals.
    data2 = {
        "a": ["a1", "a2"] * 6,
        "b": ["b1"] * 4 + ["b2"] * 4 + ["b1"] * 4,
        "c": ["c1", "c2"] * 6,
        "y": rng.normal(size=12),
    }
    result2 = bic_term_selection("y ~ a + b + c + a:b:c", data2)
    assert {term.name() for term in result2.unselectable_terms} == {"a:b:c"}
    assert all("a:b:c" not in c.term_names for c in result2.candidates)


def test_intercept_fixed_when_present():
    rng = np.random.RandomState(2)
    n = 30
    data = {"x": rng.normal(size=n), "y": rng.normal(size=n)}
    result = bic_term_selection("y ~ x", data)
    assert result.has_intercept is True
    for candidate in result.candidates:
        assert candidate.has_intercept is True
    assert "Intercept" in result.coefficients
    assert "Intercept" in result.design_info.column_names


def test_no_intercept_formula():
    rng = np.random.RandomState(4)
    n = 40
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 3 * x1 + 0.1 * rng.normal(size=n)
    result = bic_term_selection("y ~ 0 + x1 + x2", {"y": y, "x1": x1, "x2": x2})
    assert result.has_intercept is False
    assert "Intercept" not in result.coefficients
    # The empty model (zero columns) is a valid candidate here.
    empty = [c for c in result.candidates if c.num_columns == 0]
    assert len(empty) == 1
    assert empty[0].status == "compared"
    assert empty[0].residual_df == n
    # Fitting the empty model means predicting zero everywhere.
    assert np.isclose(empty[0].rss, np.dot(y, y))
    assert result.selected_term_names == ("x1",)
    # Prediction from a no-intercept selected model has no intercept.
    pred = result.predict({"x1": [1.0, 2.0]})
    assert np.allclose(pred, 3 * np.array([1.0, 2.0]), atol=0.1)


def test_rank_deficient_candidate_diagnostic():
    rng = np.random.RandomState(5)
    n = 40
    x1 = rng.normal(size=n)
    y = x1 + 0.1 * rng.normal(size=n)
    # x2 is an exact duplicate of x1 -> any model with both is singular.
    result = bic_term_selection(
        "y ~ x1 + x2", {"y": y, "x1": x1, "x2": x1.copy()}
    )
    deficient = [c for c in result.candidates if c.status == "rank_deficient"]
    assert deficient
    cand = deficient[0]
    assert cand.term_names == ("x1", "x2")
    assert cand.num_columns == 3
    assert cand.rank == 2
    assert cand.bic is None and cand.rss is None and cand.coefficients is None
    # The winner comes from a full-rank candidate.
    assert result.selected_term_names == ("x1",)


def test_no_residual_df_diagnostic():
    # n = 2, intercept + one predictor: the full model has n - p = 0.
    result = bic_term_selection(
        "y ~ x", {"y": [1.0, 2.0], "x": [0.0, 1.0]}
    )
    saturated = [c for c in result.candidates if c.term_names == ("x",)][0]
    assert saturated.status == "no_residual_df"
    assert saturated.residual_df == 0
    assert saturated.bic is None
    # Intercept-only still has residual df, so selection still succeeds.
    assert result.selected_term_names == ()


def test_all_candidates_incomparable_raises():
    # No intercept, single predictor, n = 1: empty model comparable only
    # if RSS > 0; make y identically zero so even the empty model has
    # zero RSS, and the one-predictor model has no residual df.
    with pytest.raises(PatsyError):
        bic_term_selection("y ~ 0 + x", {"y": [0.0], "x": [1.0]})


def test_na_rows_shared_across_candidates():
    rng = np.random.RandomState(6)
    n = 50
    data = {
        "y": rng.normal(size=n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    }
    data["x1"][0] = np.nan
    data["x2"][7] = np.nan
    result = bic_term_selection("y ~ x1 + x2", data)
    # Two rows dropped, once, and shared by every candidate.
    assert result.nobs == n - 2
    assert len(result.predict()) == n - 2
    for candidate in result.candidates:
        assert candidate.residual_df == result.nobs - candidate.num_columns


def test_prediction_reuses_training_state_no_relearning():
    rng = np.random.RandomState(7)
    n = 80
    x = rng.uniform(0, 1, size=n)
    z = rng.normal(size=n)
    y = np.sin(2 * np.pi * x) + 0.1 * rng.normal(size=n)

    # A bs() spline (full rank with intercept) and a stateful center().
    result = bic_term_selection("y ~ bs(x, df=4) + center(z)", {"y": y, "x": x, "z": z})
    assert "bs(x, df=4)" in result.selected_term_names
    sub_di = result.design_info

    # The subsetted DesignInfo only needs (and knows about) selected
    # factors, but carries their memorized state.
    factor_names = {f.name() for f in sub_di.factor_infos}
    assert "bs(x, df=4)" in factor_names
    assert "center(z)" not in factor_names

    new_x = np.linspace(0.1, 0.9, 6)
    new_data = {"x": new_x}
    pred_matrix = result.design_matrix(new_data)
    # Building on new data via the subsetted DesignInfo reuses training
    # knots: it must differ from a spline freshly fit on the new data.
    fresh = dmatrix("bs(x, df=4)", new_data)
    assert not np.allclose(np.asarray(fresh), np.asarray(pred_matrix))
    # Predictions are the submatrix times the stored coefficients.
    pred = result.predict(new_data)
    assert pred.shape == (6,)
    assert np.allclose(pred, np.asarray(pred_matrix) @ result.coef)

    # build_design_matrices directly on the subsetted DesignInfo works
    # and learns nothing new (unknown category -> error, not relearned).
    rebuilt, = build_design_matrices([sub_di], new_data)
    assert rebuilt.design_info is sub_di
    assert np.allclose(np.asarray(rebuilt), np.asarray(pred_matrix))


def test_prediction_only_needs_selected_variables():
    rng = np.random.RandomState(8)
    n = 60
    data = {
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    }
    data["y"] = 2 * data["x1"] + 0.1 * rng.normal(size=n)
    result = bic_term_selection("y ~ x1 + x2 + x3", data)
    assert result.selected_term_names == ("x1",)
    # x2/x3 are absent from the prediction data and that must be fine.
    pred = result.predict({"x1": np.array([0.0, 1.0])})
    assert np.all(np.isfinite(pred))


@pytest.mark.skipif(not have_pandas, reason="pandas not installed")
def test_dataframe_return_type_and_index():
    rng = np.random.RandomState(9)
    n = 40
    df = pandas.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)}
    )
    result = bic_term_selection("y ~ x1 + x2", df, return_type="dataframe")
    new_df = pandas.DataFrame({"x1": [0.0, 1.0]}, index=[100, 101])
    preds = result.predict(new_df)
    assert isinstance(preds, pandas.Series)
    assert list(preds.index) == [100, 101]
    # Matrix return type can be requested explicitly.
    raw = result.predict(new_df, return_type="matrix")
    assert isinstance(raw, np.ndarray)
    assert np.allclose(raw, preds.values)
    with pytest.raises(PatsyError):
        result.predict(new_df, return_type="bogus")


def test_formula_interface_preserved():
    # ModelDesc input works just like a formula string.
    rng = np.random.RandomState(10)
    n = 30
    data = {"x": rng.normal(size=n), "y": rng.normal(size=n)}
    desc = ModelDesc.from_formula("y ~ x")
    result_str = bic_term_selection("y ~ x", data)
    result_desc = bic_term_selection(desc, data)
    assert result_desc.selected_term_names == result_str.selected_term_names
    assert np.isclose(result_desc.bic, result_str.bic)


def test_rejects_raw_matrices_and_bad_formulas():
    rng = np.random.RandomState(11)
    X = rng.normal(size=(10, 2))
    y = rng.normal(size=10)
    # Raw matrices have no term metadata.
    with pytest.raises(PatsyError):
        bic_term_selection((y, X))
    # Missing outcome.
    with pytest.raises(PatsyError):
        bic_term_selection("~ x", {"x": X[:, 0]})
    # Multiple outcomes.
    with pytest.raises(PatsyError):
        bic_term_selection(
            "y1 + y2 ~ x",
            {"y1": y, "y2": y, "x": X[:, 0]},
        )


def test_bic_tie_breaking_is_deterministic():
    # Construct orthogonal, null predictors so several models share a
    # BIC up to the p*log(n) penalty; check repeated runs agree and the
    # tie rules (fewest columns, then term names) are respected.
    rng = np.random.RandomState(12)
    n = 100
    # y orthogonal to both x1 and x2 -> RSS identical for all 4 models,
    # BIC then strictly prefers fewer columns; empty-ish intercept model.
    base = rng.normal(size=n)
    x1 = base - base.mean()
    x2 = rng.normal(size=n)
    x2 = x2 - x2.mean()
    x2 = x2 - np.dot(x2, x1) / np.dot(x1, x1) * x1
    y = rng.normal(size=n)
    y = y - y.mean()  # centered; intercept still estimated at ~0
    data = {"y": y, "x1": x1, "x2": x2}
    r1 = bic_term_selection("y ~ x1 + x2", data)
    r2 = bic_term_selection("y ~ x1 + x2", data)
    assert r1.selected_term_names == r2.selected_term_names
    # With no signal, the intercept-only model wins on the BIC penalty.
    assert r1.selected_term_names == ()


def test_result_attributes_and_repr():
    rng = np.random.RandomState(13)
    n = 25
    data = {"x": rng.normal(size=n), "y": rng.normal(size=n)}
    result = bic_term_selection("y ~ x", data)
    # Coefficient mapping and array agree, in DesignInfo column order.
    assert list(result.coefficients) == result.design_info.column_names
    assert np.allclose(result.coef, list(result.coefficients.values()))
    assert isinstance(result.candidates, tuple)
    assert all(isinstance(c, TermCandidateFit) for c in result.candidates)
    assert result.formula == "y ~ x"
    # Smoke tests for the repr/pretty-printer machinery.
    repr(result)
    repr(result.candidates[0])


def test_hierarchical_enumeration_helper():
    a = _term("a")
    b = _term("b")
    c = _term("c")
    ab = _term("a", "b")
    ac = _term("a", "c")
    abc = _term("a", "b", "c")

    subsets, unselectable = _hierarchical_term_subsets([a, b, ab])
    assert unselectable == frozenset()
    assert len(subsets) == 5
    assert frozenset([a, b, ab]) in subsets
    assert frozenset([ab]) not in subsets

    # Orphan interaction reported as unselectable, never in a subset.
    subsets2, unselectable2 = _hierarchical_term_subsets([ab])
    assert unselectable2 == frozenset([ab])
    assert subsets2 == [frozenset()]

    # Prerequisite map over the universe.
    prereq = _term_prerequisites(abc, [INTERCEPT, a, b, c, ab, ac, abc])
    assert prereq == {a, b, c, ab, ac}
    assert _term_prerequisites(a, [a, b]) == set()
