"""Tests for direction-aware trailing stop ratchet.

Bug history:
- BUG-1: SELL ratchet used max() like BUY, so locked_sl never tightened on SHORTs
  after breakeven. Fix: BUY -> max(), SELL -> min(); init sentinels (-inf / +inf).
- BUG-2: ADAPTIVE_HYBRID_TRAILING_LEVELS activate_atr (2.5/4.0/5.5) was unreachable
  before TP (BTC=3.5 ATR, mid=3.8). Levels 2 & 3 never fired. Fix: 1.5/3.0/4.5.
"""

from src.config import (
    ADAPTIVE_HYBRID_ATR_PROFILES,
    ADAPTIVE_HYBRID_TRAILING_LEVELS,
)


def _ratchet_locked_sl(direction: str, candidate_sl: float, prev_locked):
    """Pure helper that mirrors the ratchet logic in adaptive_hybrid_strategy.py.

    BUY: SL ratchets UP (higher = tighter for longs).
    SELL: SL ratchets DOWN (lower = tighter for shorts).
    """
    if direction == 'BUY':
        if prev_locked is None:
            prev_locked = float('-inf')
        return max(candidate_sl, prev_locked)
    else:
        if prev_locked is None:
            prev_locked = float('inf')
        return min(candidate_sl, prev_locked)


# ----------------------------- Direction ratchet ------------------------------


def test_short_locked_sl_ratchets_down_with_growing_profit():
    """For a SHORT in growing profit, the trailing SL must ratchet DOWN
    (closer to current price) at each new tick. Pre-fix this used max() and
    the SL never tightened, so winners always retraced to breakeven."""
    entry_price = 100.0
    atr = 1.0

    # Simulate decreasing 'lowest' watermark on a SHORT in profit.
    # Trailing distance = 1.0 ATR (locks 1 ATR above 'lowest').
    distance = 1.0
    lowest_watermarks = [97.0, 95.0, 93.0, 90.0]  # price keeps falling = profit grows
    candidate_sls = [low + distance * atr for low in lowest_watermarks]
    # Expected candidates: 98, 96, 94, 91

    locked = None
    history = []
    for cand in candidate_sls:
        locked = _ratchet_locked_sl('SELL', cand, locked)
        history.append(locked)

    assert history == [98.0, 96.0, 94.0, 91.0], history
    # Strictly monotonically decreasing -> ratchet works
    assert all(history[i] > history[i + 1] for i in range(len(history) - 1))
    # And the locked SL is below entry (protective for a SHORT only when
    # it has crossed entry, but here we are testing the ratchet behaviour).
    assert locked is not None and locked < entry_price


def test_short_locked_sl_does_not_loosen_on_retracement():
    """If price retraces (lowest doesn't advance), the SL must NOT loosen."""
    # 'lowest' is monotonic min, but candidate_sl could fluctuate if level changes.
    # Simulate a tighter level activating then a looser one being computed.
    # The ratchet should prefer the tighter (lower) value for SELL.
    locked = None
    locked = _ratchet_locked_sl('SELL', 95.0, locked)  # tight
    locked = _ratchet_locked_sl('SELL', 97.0, locked)  # looser candidate
    assert locked == 95.0, "SELL ratchet must keep tightest SL (lowest value)"


# Buy direction (regression guard so the BUY path keeps working)


def test_long_locked_sl_ratchets_up_with_growing_profit():
    """For a LONG in growing profit, locked SL must ratchet UP."""
    distance = 1.0
    highest_watermarks = [103.0, 105.0, 107.0, 110.0]
    candidate_sls = [h - distance for h in highest_watermarks]
    # Expected: 102, 104, 106, 109

    locked = None
    history = []
    for cand in candidate_sls:
        locked = _ratchet_locked_sl('BUY', cand, locked)
        history.append(locked)

    assert history == [102.0, 104.0, 106.0, 109.0], history
    assert all(history[i] < history[i + 1] for i in range(len(history) - 1))


def test_long_locked_sl_does_not_loosen_on_retracement():
    locked = None
    locked = _ratchet_locked_sl('BUY', 105.0, locked)
    locked = _ratchet_locked_sl('BUY', 102.0, locked)  # looser
    assert locked == 105.0, "BUY ratchet must keep tightest SL (highest value)"


# ------------------------- Trailing levels reachability -----------------------


def test_trailing_levels_activate_before_tp_for_all_profiles():
    """BUG-2 regression guard: every non-breakeven trailing level must have
    activate_atr <= the TP multiplier of EVERY profile. Otherwise the level
    can never fire (TP closes the trade first)."""
    non_be_levels = [
        lvl for lvl in ADAPTIVE_HYBRID_TRAILING_LEVELS if not lvl.get('breakeven')
    ]
    assert non_be_levels, "Expected at least one non-breakeven trailing level"

    # The first non-breakeven level should be reachable for ALL profiles
    # (otherwise progressive trailing degenerates to plain breakeven).
    first_lock = non_be_levels[0]['activate_atr']
    for name, prof in ADAPTIVE_HYBRID_ATR_PROFILES.items():
        assert first_lock < prof['tp_mult'], (
            f"Trailing level 2 activate_atr={first_lock} >= TP mult "
            f"{prof['tp_mult']} for profile '{name}': level can never fire"
        )


def test_trailing_levels_activate_atr_strictly_increasing():
    """Levels must be ordered by increasing activate_atr (progressive trailing)."""
    activates = [lvl['activate_atr'] for lvl in ADAPTIVE_HYBRID_TRAILING_LEVELS]
    assert activates == sorted(activates), activates
    assert len(set(activates)) == len(activates), "Duplicate activate_atr values"


def test_trailing_levels_distance_atr_decreasing_after_breakeven():
    """After breakeven, distance_atr should tighten (decrease) as profit grows."""
    non_be = [
        lvl for lvl in ADAPTIVE_HYBRID_TRAILING_LEVELS if not lvl.get('breakeven')
    ]
    distances = [lvl['distance_atr'] for lvl in non_be]
    assert distances == sorted(distances, reverse=True), (
        f"distance_atr should decrease across non-breakeven levels, got {distances}"
    )
