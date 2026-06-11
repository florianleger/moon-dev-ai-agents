"""Regression: Kelly-adaptive sizing was REMOVED from AdaptiveHybridStrategy.

History:
    - Originally, Kelly sizing could collapse position_size to 0 (deadlock);
      a 25% floor was added (this file previously tested that floor).
    - Jun 2026 (sizing simplification): with a negative trade history the floor
      locked a permanent /4 reduction on every trade, which (stacked with 5
      other multipliers) crushed the median notional to $25. Sizing is now a
      fixed risk fraction (ADAPTIVE_HYBRID_RISK_PCT, leverage-free) modulated
      only by drawdown/recovery scaling and the correlation factor.
      See tests/test_strategy_tuning.py for the new sizing tests.

This file now only guards against the Kelly block being reintroduced.
"""
import os

import pytest


def _strategy_source():
    path = os.path.join(
        os.path.dirname(__file__), '..',
        'src', 'strategies', 'custom', 'adaptive_hybrid_strategy.py'
    )
    with open(path, 'r') as f:
        return f.read()


def test_kelly_sizing_removed_from_strategy():
    """The Kelly sizing block must NOT be present in the strategy source."""
    source = _strategy_source()
    assert 'kelly_size_pct' not in source, (
        "Kelly sizing reappeared in adaptive_hybrid_strategy.py — it was removed "
        "on purpose (permanent /4 floor with negative history). See test docstring."
    )


def test_score_and_strength_multipliers_removed():
    """Sizing must not be modulated by score/strength (corr(score, pnl)=0.055)."""
    source = _strategy_source()
    assert 'score_exposure' not in source
    assert 'strength_multiplier' not in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
