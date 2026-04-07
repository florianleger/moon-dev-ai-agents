"""Tests for the Kelly sizing floor fix in AdaptiveHybridStrategy.

Bug context:
    After a streak of losing trades, the Kelly criterion can collapse to 0
    (or near-zero), which previously multiplied `position_size` by 0 and
    caused every subsequent trade to be rejected (deadlock - the bot could
    never recover because it never took another trade).

Fix:
    A floor of 25% is applied to the Kelly reduction multiplier, so even
    with a catastrophic win rate the position size is reduced to 25% of
    base risk rather than zeroed out.

Reference: src/strategies/custom/adaptive_hybrid_strategy.py
    if kelly_size_pct < base_risk_pct:
        reduction = max(0.25, kelly_size_pct / base_risk_pct)  # Floor 25%
"""
import pytest


# ---------------------------------------------------------------------------
# Pure logic helper - mirrors the Kelly sizing block in adaptive_hybrid_strategy
# ---------------------------------------------------------------------------

def compute_kelly_reduction(recent_pnls, kelly_fraction=0.5, base_risk_pct=0.015,
                            apply_floor=True):
    """Reproduce the Kelly sizing logic from AdaptiveHybridStrategy._prepare_trade.

    Returns the multiplier that will be applied to position_size.
    A return value of 1.0 means "no Kelly reduction" (Kelly suggested >= base risk).

    Args:
        recent_pnls: list of recent trade PnLs (positive = win, <=0 = loss)
        kelly_fraction: ADAPTIVE_HYBRID_KELLY_FRACTION (default 0.5 = half-Kelly)
        base_risk_pct: ADAPTIVE_HYBRID_MAX_POSITION_PCT / 100 (default 1.5%)
        apply_floor: whether to apply the 25% floor fix (False = pre-fix behavior)
    """
    # Need at least 10 samples (matches strategy code)
    if len(recent_pnls) < 10:
        return 1.0

    wins = [p for p in recent_pnls if p > 0]
    losses = [p for p in recent_pnls if p <= 0]

    # Edge case: no wins or no losses -> strategy skips Kelly entirely
    if not wins or not losses:
        return 1.0

    win_rate = len(wins) / len(recent_pnls)
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))

    if avg_loss == 0:
        return 1.0

    payoff_ratio = avg_win / avg_loss
    kelly = win_rate - (1 - win_rate) / payoff_ratio
    kelly = max(0, kelly)  # Never negative
    kelly_size_pct = kelly * kelly_fraction

    if kelly_size_pct < base_risk_pct:
        if apply_floor:
            reduction = max(0.25, kelly_size_pct / base_risk_pct)
        else:
            reduction = kelly_size_pct / base_risk_pct
        return reduction
    return 1.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKellyFloor:
    """Validate the 25% floor on Kelly reduction prevents the deadlock bug."""

    def test_zero_wins_ten_losses_does_not_zero_position(self):
        """0/10 win rate must NOT collapse position_size to zero."""
        # Strategy code: if `not wins` -> Kelly block is skipped entirely.
        # So 0 wins / 10 losses returns 1.0 (Kelly disabled, base size kept).
        recent_pnls = [-50.0] * 10
        reduction = compute_kelly_reduction(recent_pnls)
        assert reduction > 0, "position_size must never be zeroed by Kelly"
        assert reduction == 1.0, (
            "With zero wins, Kelly is skipped (no payoff ratio computable)"
        )

    def test_one_win_nine_losses_floored_at_25pct(self):
        """1/10 win rate with tiny payoff: floor must clamp at 0.25."""
        # 1 small win, 9 large losses -> Kelly will be very negative -> 0
        # kelly_size_pct = 0 -> reduction = max(0.25, 0) = 0.25
        recent_pnls = [10.0] + [-100.0] * 9
        reduction = compute_kelly_reduction(recent_pnls)
        assert reduction == 0.25, (
            f"Expected floor at 0.25, got {reduction}. Bug: deadlock would occur."
        )

    def test_floor_prevents_deadlock_vs_no_floor(self):
        """Demonstrate the bug: without the floor, reduction would be 0."""
        recent_pnls = [10.0] + [-100.0] * 9

        without_floor = compute_kelly_reduction(recent_pnls, apply_floor=False)
        with_floor = compute_kelly_reduction(recent_pnls, apply_floor=True)

        # Pre-fix behavior: reduction collapses to 0 (deadlock)
        assert without_floor == 0.0, "Pre-fix should produce zero reduction"
        # Post-fix: floored at 0.25
        assert with_floor == 0.25, "Post-fix should floor at 0.25"
        assert with_floor > without_floor

    def test_two_wins_eight_losses_still_floored(self):
        """20% win rate with 1:1 payoff -> Kelly very low -> floored at 0.25."""
        recent_pnls = [50.0, 50.0] + [-50.0] * 8
        reduction = compute_kelly_reduction(recent_pnls)
        # win_rate=0.2, payoff=1.0, kelly = 0.2 - 0.8/1.0 = -0.6 -> 0
        # kelly_size_pct = 0 -> floored at 0.25
        assert reduction == 0.25

    def test_high_win_rate_not_artificially_capped(self):
        """80% win rate with 1.5x payoff: Kelly should NOT be plafonné à 0.25."""
        recent_pnls = [150.0] * 8 + [-100.0] * 2
        # win_rate = 0.8, avg_win=150, avg_loss=100, payoff=1.5
        # kelly = 0.8 - 0.2/1.5 = 0.8 - 0.133 = 0.667
        # kelly_size_pct = 0.667 * 0.5 = 0.333 (33%)
        # base_risk_pct = 0.015 (1.5%) -> kelly_size_pct >> base_risk_pct
        # -> reduction stays at 1.0 (no reduction applied)
        reduction = compute_kelly_reduction(recent_pnls)
        assert reduction == 1.0, (
            f"Good win rate should not trigger reduction, got {reduction}"
        )
        assert reduction > 0.25, "Good win rate must exceed the floor"

    def test_marginal_kelly_uses_actual_ratio_above_floor(self):
        """When Kelly suggests something between floor and base, use the real ratio."""
        # Construct a scenario where kelly_size_pct / base_risk_pct ~ 0.5
        # base_risk_pct = 0.015, want kelly_size_pct ~ 0.0075
        # kelly_size_pct = kelly * 0.5 -> kelly = 0.015
        # win_rate=0.5, payoff=1.03 -> kelly = 0.5 - 0.5/1.03 ~ 0.0145
        recent_pnls = [103.0] * 5 + [-100.0] * 5
        reduction = compute_kelly_reduction(recent_pnls)
        # Should be > 0.25 (above floor) and < 1.0 (Kelly is reducing)
        assert 0.25 <= reduction <= 1.0
        # Specifically: kelly=0.0146, kelly_size_pct=0.0073, ratio=0.486
        # Floor check: max(0.25, 0.486) = 0.486
        assert reduction == pytest.approx(0.486, abs=0.05)

    def test_perfect_loss_streak_skipped_safely(self):
        """100% loss rate -> Kelly block skipped (no wins), no zeroing."""
        recent_pnls = [-25.0, -50.0, -10.0, -75.0, -30.0,
                       -40.0, -20.0, -60.0, -15.0, -45.0]
        reduction = compute_kelly_reduction(recent_pnls)
        # No wins -> early return 1.0 (deadlock impossible via Kelly path)
        assert reduction == 1.0

    def test_insufficient_samples_no_kelly(self):
        """< 10 trades: Kelly disabled, no position reduction."""
        recent_pnls = [10.0, -5.0, 20.0]
        reduction = compute_kelly_reduction(recent_pnls)
        assert reduction == 1.0

    def test_floor_value_is_exactly_25_percent(self):
        """Regression: ensure the floor constant remains at 0.25 (25%)."""
        # Worst possible Kelly scenario that doesn't short-circuit
        recent_pnls = [1.0] + [-1000.0] * 9
        reduction = compute_kelly_reduction(recent_pnls)
        assert reduction == 0.25, (
            f"Floor must be exactly 25%, got {reduction}. "
            "If you changed the floor, update both the strategy and this test."
        )


# ---------------------------------------------------------------------------
# Integration smoke test - verifies the actual strategy code contains the fix
# ---------------------------------------------------------------------------

def test_strategy_source_contains_floor():
    """Static check: the Kelly floor must be present in the strategy source."""
    import os
    strategy_path = os.path.join(
        os.path.dirname(__file__), '..',
        'src', 'strategies', 'custom', 'adaptive_hybrid_strategy.py'
    )
    if not os.path.exists(strategy_path):
        pytest.skip("Strategy file not found")

    with open(strategy_path, 'r') as f:
        source = f.read()

    # The fix line should be present
    assert 'max(0.25' in source and 'kelly_size_pct / base_risk_pct' in source, (
        "Kelly floor fix not found in adaptive_hybrid_strategy.py. "
        "Expected line: reduction = max(0.25, kelly_size_pct / base_risk_pct)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
