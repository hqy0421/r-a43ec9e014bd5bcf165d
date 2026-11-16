# This file is part of Patsy
# Copyright (C) 2026 Nathaniel Smith <njs@pobox.com>
# See file LICENSE.txt for license information.

# Term-wise linear model selection.
#
# The main formula is evaluated exactly once (via the ordinary
# dmatrices() machinery), so every candidate model shares the same
# valid rows and the same column coding (categorical contrasts,
# stateful-transform state, spline knots, ...). Candidate models are
# then formed by selecting whole *terms* out of the already-built
# design matrix and compared on BIC.

# These are made available in the patsy.* namespace:
__all__ = ["bic_term_selection", "LinearTermSelectionResult", "TermCandidateFit"]

import itertools

import numpy as np

from patsy import PatsyError
from patsy.compat import OrderedDict
from patsy.desc import INTERCEPT
from patsy.design_info import DesignInfo, DesignMatrix
from patsy.build import build_design_matrices
from patsy.eval import EvalEnvironment
from patsy.highlevel import _do_highlevel_design
from patsy.util import have_pandas, repr_pretty_delegate, repr_pretty_impl, no_pickling

if have_pandas:
    import pandas


def _term_sort_key(term):
    # Lower-order interactions are decided before higher-order ones, and
    # ties are broken by name so that enumeration is deterministic.
    return (len(term.factors), term.name())


def _term_prerequisites(term, universe):
    """The terms in `universe` that are strict lower-order relatives of
    `term` (the marginals needed to satisfy hierarchy / marginality).

    The empty-factors intercept term is handled separately and never
    appears here.
    """
    factors = term.factors
    prereqs = set()
    for other in universe:
        if other == INTERCEPT:
            continue
        if other == term:
            continue
        other_factors = other.factors
        # A proper marginal: every factor of `other` is a factor of
        # `term`, and `other` has strictly fewer factors.
        if len(other_factors) < len(factors) and all(
            f in factors for f in other_factors
        ):
            prereqs.add(other)
    return prereqs


def _all_proper_factor_subsets(factors):
    """Every non-empty proper subset of `factors` (as frozensets), i.e.
    the factor sets of all strict marginals of an interaction term.
    """
    factors = tuple(factors)
    subsets = []
    for size in range(1, len(factors)):
        for combo in itertools.combinations(factors, size):
            subsets.append(frozenset(combo))
    return subsets


def test__all_proper_factor_subsets():
    subsets = _all_proper_factor_subsets(("a", "b", "c"))
    assert frozenset() not in subsets
    assert frozenset(("a", "b", "c")) not in subsets
    assert frozenset(("a",)) in subsets
    assert frozenset(("b", "c")) in subsets
    assert len(subsets) == 6
    assert _all_proper_factor_subsets(("a",)) == []


def test__term_prerequisites():
    from patsy.desc import Term
    from patsy.user_util import LookupFactor

    def t(*names):
        return Term([LookupFactor(n) for n in names])

    a = t("a")
    b = t("b")
    c = t("c")
    ab = t("a", "b")
    ac = t("a", "c")
    abc = t("a", "b", "c")
    universe = [INTERCEPT, a, b, c, ab, ac, abc]

    assert _term_prerequisites(a, universe) == set()
    assert _term_prerequisites(ab, universe) == {a, b}
    assert _term_prerequisites(ac, universe) == {a, c}
    assert _term_prerequisites(abc, universe) == {a, b, c, ab, ac}


def _hierarchical_term_subsets(optional_terms):
    """Enumerate every hierarchically-valid subset of `optional_terms`.

    A subset is valid when it is marginally closed: whenever a term is
    included, all of its strict lower-order relatives (marginals) are
    included too.

    A term whose required lower-order relatives are missing from the
    term universe altogether can never satisfy hierarchy, so it is not
    selectable: such "orphan" terms never appear in any emitted subset
    (and are reported via the second return value). This covers, for
    example, an interaction term written without its main effects.

    Returns a pair ``(subsets, unselectable)`` where `subsets` is a list
    of frozensets of selectable :class:`~patsy.desc.Term` objects and
    `unselectable` is a frozenset of the orphan terms that were removed.
    """
    terms = sorted(optional_terms, key=_term_sort_key)
    # Map each factor-set that appears as a term in the universe onto
    # that term, so we can check whether each marginal exists at all.
    universe_by_factors = {frozenset(term.factors): term for term in terms}

    selectable = []
    unselectable = set()
    for term in terms:
        if len(term.factors) <= 1:
            # Main effects (and any single-factor term) have no
            # marginals, so they are always selectable.
            selectable.append(term)
            continue
        factor_sets = set(_all_proper_factor_subsets(term.factors))
        present_factor_sets = factor_sets & set(universe_by_factors)
        missing = factor_sets - set(universe_by_factors)
        orphan_marginals = {
            universe_by_factors[fs]
            for fs in present_factor_sets
            if universe_by_factors[fs] in unselectable
        }
        if missing or orphan_marginals:
            # A required lower-order relative is either absent from the
            # formula altogether, or is itself unselectable. Either way
            # this term can never satisfy hierarchy.
            unselectable.add(term)
        else:
            selectable.append(term)

    prereq_map = {term: _term_prerequisites(term, selectable) for term in selectable}

    subsets = []

    def dfs(idx, chosen):
        if idx == len(selectable):
            subsets.append(frozenset(chosen))
            return
        term = selectable[idx]
        # Option 1: leave this term out.
        dfs(idx + 1, chosen)
        # Option 2: include it, but only if hierarchy is satisfied.
        # Because terms are sorted by len(factors), every prerequisite
        # (which is strictly lower-order) has already been decided.
        if prereq_map[term] <= chosen:
            chosen.add(term)
            dfs(idx + 1, chosen)
            chosen.remove(term)

    dfs(0, set())
    return subsets, frozenset(unselectable)


def test__hierarchical_term_subsets():
    from patsy.desc import Term
    from patsy.user_util import LookupFactor

    def t(*names):
        return Term([LookupFactor(n) for n in names])

    a = t("a")
    b = t("b")
    ab = t("a", "b")

    # Without the interaction in the universe, a and b are independent:
    assert _hierarchical_term_subsets([a, b])[0] == [
        frozenset(),
        frozenset([b]),
        frozenset([a]),
        frozenset([a, b]),
    ]

    # With a:b present, the interaction can only join together with a, b.
    subsets, unselectable = _hierarchical_term_subsets([a, b, ab])
    assert unselectable == frozenset()
    assert frozenset([ab]) not in subsets
    assert frozenset([a, ab]) not in subsets
    assert frozenset([b, ab]) not in subsets
    assert frozenset([a, b, ab]) in subsets
    # The five valid models are the empty model, {a}, {b}, {a, b},
    # {a, b, a:b}.
    assert len(subsets) == 5

    # An interaction with no marginals in the universe is not
    # selectable at all: it is reported as unselectable and never
    # appears in any subset.
    orphan_subsets, orphan_unselectable = _hierarchical_term_subsets([ab])
    assert orphan_subsets == [frozenset()]
    assert orphan_unselectable == frozenset([ab])

    # Orphan-hood propagates: a three-way interaction whose two-way
    # marginals are themselves orphans is unselectable too.
    abc = t("a", "b", "c")
    c = t("c")
    prop_subsets, prop_unselectable = _hierarchical_term_subsets([a, b, c, abc])
    assert prop_unselectable == frozenset([abc])
    assert all(abc not in subset for subset in prop_subsets)


class TermCandidateFit(object):
    """The fit of one term-combination candidate considered during
    :func:`bic_term_selection`.

    Attributes:

    .. attribute:: terms

       A tuple of the :class:`~patsy.desc.Term` objects in this candidate
       (not counting the intercept, whose presence is reported by
       :attr:`has_intercept`).

    .. attribute:: term_names

       A tuple of term-name strings, in the order in which the terms
       appear in the original formula.

    .. attribute:: has_intercept

       Whether this candidate includes the intercept column.

    .. attribute:: num_columns

       The actual number of design-matrix columns used by this
       candidate (this is the *p* in the BIC formula; a single term may
       contribute several columns, e.g. categorical and interaction
       terms).

    .. attribute:: rank

       The numerical column rank of the candidate design matrix.

    .. attribute:: residual_df

       The residual degrees of freedom, i.e. ``n - num_columns``.

    .. attribute:: rss

       The residual sum of squares of the ordinary least-squares fit.

    .. attribute:: bic

       The Bayesian information criterion,
       ``n * log(RSS / n) + p * log(n)``.

    .. attribute:: coefficients

       An :class:`collections.OrderedDict` mapping column name to the
       fitted coefficient for that column. ``None`` for candidates that
       could not be compared (see :attr:`status`).

    .. attribute:: status

       One of:

       * ``"compared"`` -- the candidate had full column rank, positive
         residual degrees of freedom, and a strictly positive RSS, so
         its BIC entered the comparison;
       * ``"rank_deficient"`` -- the candidate design matrix did not have
         full column rank;
       * ``"no_residual_df"`` -- there were no degrees of freedom left
         after fitting (``n <= p``);
       * ``"zero_rss"`` -- the fit was exact (``RSS == 0``), so BIC is
         not defined.
    """

    def __init__(
        self,
        ordered_terms,
        has_intercept,
        num_columns,
        rank,
        residual_df,
        rss,
        bic,
        coefficients,
        status,
    ):
        self.terms = tuple(ordered_terms)
        self.term_names = tuple(term.name() for term in ordered_terms)
        self.has_intercept = has_intercept
        self.num_columns = num_columns
        self.rank = rank
        self.residual_df = residual_df
        self.rss = rss
        self.bic = bic
        self.coefficients = coefficients
        self.status = status

    __repr__ = repr_pretty_delegate

    def _repr_pretty_(self, p, cycle):
        assert not cycle
        repr_pretty_impl(
            p,
            self,
            [],
            [
                ("terms", list(self.term_names)),
                ("has_intercept", self.has_intercept),
                ("num_columns", self.num_columns),
                ("rank", self.rank),
                ("residual_df", self.residual_df),
                ("rss", self.rss),
                ("bic", self.bic),
                ("status", self.status),
            ],
        )

    __getstate__ = no_pickling


class LinearTermSelectionResult(object):
    """The result of a :func:`bic_term_selection` run.

    Do not construct these objects directly; they are returned by
    :func:`bic_term_selection`.

    Attributes:

    .. attribute:: selected_terms

       A tuple of the :class:`~patsy.desc.Term` objects selected into the
       winning model (not counting the intercept, whose presence is
       reported by :attr:`has_intercept`).

    .. attribute:: selected_term_names

       A tuple of term-name strings for :attr:`selected_terms`, in
       formula order.

    .. attribute:: has_intercept

       Whether the intercept was present in the main formula. If so, it
       was fixed in every candidate and appears in the winning model.

    .. attribute:: coefficients

       An :class:`collections.OrderedDict` mapping the selected model's
       column names to their fitted least-squares coefficients.

    .. attribute:: coef

       A one-dimensional :class:`numpy.ndarray` holding the same
       coefficients as :attr:`coefficients`, in the column order of
       :attr:`design_info`.

    .. attribute:: bic

       The BIC of the winning (minimum-BIC) model.

    .. attribute:: rss

       The residual sum of squares of the winning model.

    .. attribute:: residual_df

       The residual degrees of freedom of the winning model.

    .. attribute:: nobs

       The number of valid (non-missing) rows shared by every candidate.

    .. attribute:: design_info

       The :class:`DesignInfo` of the selected model. This is a
       *subset* of the training design info, so it carries over the
       memorized state of the full formula -- categorical codings,
       stateful-transform state, spline knots and so forth. It can be
       passed directly to :func:`build_design_matrices` for prediction.

    .. attribute:: full_design_info

       The :class:`DesignInfo` of the full main formula (before term
       selection).

    .. attribute:: candidates

       A tuple of :class:`TermCandidateFit` objects, one for every
       hierarchically-valid term combination, in enumeration order. Use
       these to inspect the diagnostics of candidates that could not be
       compared (rank-deficient, no residual degrees of freedom, or zero
       RSS).

    .. attribute:: unselectable_terms

       A frozenset of :class:`~patsy.desc.Term` objects that were never
       offered as choices, because including them would violate
       hierarchy: their required lower-order relatives are missing from
       the formula itself (e.g. an interaction written without its main
       effects). Empty when every term is selectable.

    .. attribute:: formula

       The formula-like object that was passed in.
    """

    def __init__(
        self,
        formula,
        full_design_info,
        selected_terms,
        has_intercept,
        coefficients,
        coef,
        bic,
        rss,
        residual_df,
        nobs,
        design_info,
        candidates,
        unselectable_terms,
        return_type,
        NA_action,
    ):
        self.formula = formula
        self.full_design_info = full_design_info
        self.selected_terms = tuple(selected_terms)
        self.selected_term_names = tuple(term.name() for term in selected_terms)
        self.has_intercept = has_intercept
        self.coefficients = coefficients
        self.coef = coef
        self.bic = bic
        self.rss = rss
        self.residual_df = residual_df
        self.nobs = nobs
        self.design_info = design_info
        self.candidates = tuple(candidates)
        self.unselectable_terms = frozenset(unselectable_terms)
        self._return_type = return_type
        self._NA_action = NA_action

    __repr__ = repr_pretty_delegate

    def _repr_pretty_(self, p, cycle):
        assert not cycle
        repr_pretty_impl(
            p,
            self,
            [],
            [
                ("selected_term_names", list(self.selected_term_names)),
                ("has_intercept", self.has_intercept),
                ("bic", self.bic),
                ("rss", self.rss),
                ("nobs", self.nobs),
                ("residual_df", self.residual_df),
            ],
        )

    def design_matrix(self, data):
        """Build the selected model's design matrix on new data.

        Only the variables needed by the selected terms have to be
        present in `data`. Categorical levels, stateful-transform state
        and spline knots are all carried over from the training data --
        nothing is re-learned from `data`.

        :arg data: A dict-like data object, as accepted by
          :func:`build_design_matrices`.
        :returns: A :class:`DesignMatrix` with :attr:`design_info` equal
          to :attr:`design_info`.
        """
        (matrix,) = build_design_matrices(
            [self.design_info], data, NA_action=self._NA_action
        )
        return matrix

    def predict(self, new_data=None, return_type=None):
        """Compute predicted values from the selected model.

        The design matrix for the selected terms is built on
        `new_data` using the :class:`DesignInfo` learned on the
        training data, so categorical encodings, stateful transforms
        (e.g. :func:`center`, :func:`standardize`) and spline knots are
        reused as-is and never re-estimated. Only the variables used by
        the selected terms have to be present in `new_data`.

        :arg new_data: A dict-like data object, as accepted by
          :func:`dmatrix`. If omitted, predictions (fitted values) are
          computed on the training data.
        :arg return_type: ``"matrix"`` (the default) to return a plain
          one-dimensional :class:`numpy.ndarray`, or ``"dataframe"`` to
          return a :class:`pandas.Series` indexed like `new_data`. If
          omitted, the ``return_type`` used for fitting is used.
        :returns: Predicted values.
        """
        if return_type is None:
            return_type = self._return_type
        if return_type not in ("matrix", "dataframe"):
            raise PatsyError(
                "unrecognized output type %r, should be "
                "'matrix' or 'dataframe'" % (return_type,)
            )
        if return_type == "dataframe" and not have_pandas:
            raise PatsyError(
                "pandas.Series was requested, but pandas is not installed"
            )

        if new_data is None:
            matrix = self._training_design_matrix
        else:
            matrix = self.design_matrix(new_data)

        if list(matrix.design_info.column_names) != list(self.coefficients):
            # Should be impossible: both come from the same DesignInfo.
            raise PatsyError(
                "internal error: prediction columns %r do not match "
                "fitted columns %r"
                % (matrix.design_info.column_names, list(self.coefficients))
            )

        predictions = np.dot(np.asarray(matrix), self.coef)

        if return_type == "dataframe":
            if have_pandas and isinstance(new_data, pandas.DataFrame):
                index = new_data.index
            else:
                index = None
            return pandas.Series(predictions, index=index)
        return predictions

    __getstate__ = no_pickling


def _fit_candidate(X_sub, y, ordered_terms, has_intercept, log_n):
    """Fit one candidate and package up its diagnostics."""
    n, p = X_sub.shape
    if p == 0:
        # The empty model (only reachable when there is no intercept):
        # predict every observation as zero. This always has full
        # column rank and leaves n residual degrees of freedom.
        beta = np.zeros(0, dtype=float)
        rss = float(np.dot(y, y))
        if rss <= 0:
            return TermCandidateFit(
                ordered_terms,
                has_intercept,
                p,
                0,
                n,
                rss,
                None,
                None,
                "zero_rss",
            )
        bic = n * np.log(rss / n)
        return TermCandidateFit(
            ordered_terms,
            has_intercept,
            p,
            0,
            n,
            rss,
            float(bic),
            beta,
            "compared",
        )
    rank = np.linalg.matrix_rank(X_sub)
    if rank != p:
        return TermCandidateFit(
            ordered_terms,
            has_intercept,
            p,
            rank,
            n - p,
            None,
            None,
            None,
            "rank_deficient",
        )
    if n - p <= 0:
        return TermCandidateFit(
            ordered_terms,
            has_intercept,
            p,
            rank,
            n - p,
            None,
            None,
            None,
            "no_residual_df",
        )
    beta, _, _, _ = np.linalg.lstsq(X_sub, y, rcond=None)
    residuals = y - np.dot(X_sub, beta)
    rss = float(np.dot(residuals, residuals))
    if rss <= 0:
        return TermCandidateFit(
            ordered_terms,
            has_intercept,
            p,
            rank,
            n - p,
            rss,
            None,
            None,
            "zero_rss",
        )
    bic = n * np.log(rss / n) + p * log_n
    return TermCandidateFit(
        ordered_terms,
        has_intercept,
        p,
        rank,
        n - p,
        rss,
        float(bic),
        beta,
        "compared",
    )


def bic_term_selection(
    formula_like,
    data={},
    eval_env=0,
    NA_action="drop",
    return_type="matrix",
):
    """Select a linear regression model term-by-term using BIC.

    The formula is evaluated **once**, using the ordinary Patsy
    machinery (:func:`dmatrices`). Every candidate model is then formed
    by selecting whole *terms* from the resulting design matrix. This
    means all candidates share the same valid rows (after NA handling)
    and the exact same column coding -- categorical contrasts,
    stateful-transform state and spline knots are learned on the full
    formula once, and never re-learned.

    Selection rules:

    * Terms are the atomic units of selection: all the columns of a
      term (e.g. all dummy columns of a categorical main effect, or all
      columns of an interaction) enter or leave together.
    * Models must respect hierarchy (marginality): a higher-order
      interaction can only be included together with all of its
      lower-order relatives that appear in the formula. An interaction
      whose required lower-order terms are absent from the formula can
      never be selected.
    * If the formula has an intercept, it is kept fixed in every
      candidate and always appears in the selected model.
    * Among all hierarchically-valid term combinations, the winner
      minimizes
      ``BIC = n * log(RSS / n) + p * log(n)`` where *n* is the number
      of valid rows, RSS is the ordinary least-squares residual sum of
      squares and *p* is the **actual number of design-matrix columns**
      in the candidate.
    * Only candidates with full column rank, at least one residual
      degree of freedom (``n > p``) and strictly positive RSS are
      compared. The rest are retained with a diagnostic `status` in the
      returned object's ``candidates`` list.

    :arg formula_like: A formula string, :class:`ModelDesc`, or any
      other formula-like object accepted by :func:`dmatrices` that
      carries both an outcome and predictors (e.g. ``"y ~ a * b + x"``).
      Raw pre-built matrices are not accepted, because term selection
      needs term metadata.
    :arg data: A dict-like object used to look up the variables in
      `formula_like`, as for :func:`dmatrices`.
    :arg eval_env: See :func:`dmatrix`.
    :arg NA_action: See :func:`dmatrix`. Rows dropped here are dropped
      for every candidate.
    :arg return_type: ``"matrix"`` (default) or ``"dataframe"``, as for
      :func:`dmatrices`. This also becomes the default return type of
      :meth:`LinearTermSelectionResult.predict`.
    :returns: A :class:`LinearTermSelectionResult`.

    Example::

      result = bic_term_selection("y ~ a * b + x", data)
      print(result.selected_term_names)
      print(result.bic)
      new_predictions = result.predict(new_data)
    """
    eval_env = EvalEnvironment.capture(eval_env, reference=1)

    # Build the full design once. This memorizes factor state (category
    # levels, spline knots, stateful transforms, ...) and fixes the set
    # of valid rows; every candidate below reuses this exact build.
    y_dm, X_dm = _do_highlevel_design(
        formula_like, data, eval_env, NA_action, "matrix"
    )
    if y_dm.shape[1] == 0:
        raise PatsyError(
            "term selection requires an outcome variable; use a two-sided "
            "formula like 'y ~ x1 + x2'"
        )
    if y_dm.shape[1] != 1:
        raise PatsyError(
            "term selection currently handles a single outcome variable, "
            "but the formula's left-hand side has %s columns" % (y_dm.shape[1],)
        )
    if not hasattr(X_dm, "design_info") or not isinstance(
        X_dm.design_info, DesignInfo
    ) or X_dm.design_info.term_slices is None:
        raise PatsyError(
            "term selection requires formula-derived term metadata; raw "
            "design matrices without term information cannot be used"
        )

    full_design_info = X_dm.design_info
    X = np.asarray(X_dm, dtype=float)
    y = np.asarray(y_dm, dtype=float).ravel()
    n = X.shape[0]
    log_n = np.log(n)

    all_terms = list(full_design_info.terms)
    has_intercept = INTERCEPT in all_terms
    optional_terms = [term for term in all_terms if term != INTERCEPT]

    subsets, unselectable_terms = _hierarchical_term_subsets(optional_terms)

    # Column slices belonging to each term, taken from the one shared
    # design matrix build.
    term_slices = dict(full_design_info.term_slices)

    candidates = []
    compared = []
    for chosen in subsets:
        # Keep formula order for stable, readable term/column ordering.
        ordered = [term for term in optional_terms if term in chosen]
        terms_for_build = (
            [INTERCEPT] + ordered if has_intercept else ordered
        )
        column_idx = []
        for term in terms_for_build:
            sl = term_slices[term]
            column_idx.extend(range(sl.start, sl.stop))
        column_idx = np.asarray(column_idx, dtype=int)
        X_sub = X[:, column_idx]
        candidate = _fit_candidate(X_sub, y, ordered, has_intercept, log_n)
        candidates.append(candidate)
        if candidate.status == "compared":
            # Tie-breaking: lowest BIC, then fewest columns, then the
            # lexicographically smallest term-name tuple. The tie rules
            # make the winner independent of enumeration order.
            key = (candidate.bic, candidate.num_columns, candidate.term_names)
            compared.append((key, candidate, column_idx, terms_for_build))

    if not compared:
        statuses = sorted(set(candidate.status for candidate in candidates))
        raise PatsyError(
            "no comparable candidate models: every hierarchically-valid "
            "term combination was rank-deficient, had no residual degrees "
            "of freedom, or had zero RSS (candidate statuses seen: %r)"
            % (statuses,)
        )

    compared.sort(key=lambda item: item[0])
    _, winner, winner_idx, winner_terms = compared[0]

    # A sub-DesignInfo carrying over all memorized factor state (category
    # codings, contrasts, spline knots, ...). build_design_matrices() on
    # this only needs the selected terms' variables and learns nothing
    # new from prediction data.
    selected_design_info = full_design_info.subset(winner_terms)

    if list(selected_design_info.column_names) != [
        full_design_info.column_names[i] for i in winner_idx
    ]:
        # Again, both are derived from the same DesignInfo, so this
        # really cannot happen -- but check rather than silently misalign.
        raise PatsyError(
            "internal error: subset DesignInfo columns do not match the "
            "selected design-matrix columns"
        )

    coefficients = OrderedDict()
    for name, value in zip(selected_design_info.column_names, winner.coefficients):
        coefficients[name] = float(value)
    coef = np.asarray(winner.coefficients, dtype=float).ravel()

    result = LinearTermSelectionResult(
        formula=formula_like,
        full_design_info=full_design_info,
        selected_terms=winner.terms,
        has_intercept=has_intercept,
        coefficients=coefficients,
        coef=coef,
        bic=winner.bic,
        rss=winner.rss,
        residual_df=winner.residual_df,
        nobs=n,
        design_info=selected_design_info,
        candidates=candidates,
        unselectable_terms=unselectable_terms,
        return_type=return_type,
        NA_action=NA_action,
    )
    # Stash the training design for fitted-value prediction; built
    # lazily-free as a view carrying the subsetted DesignInfo.
    result._training_design_matrix = DesignMatrix(
        X[:, winner_idx], selected_design_info
    )
    return result
