"""Tests for the mechanical regime gate shared by the backtest harness and live AH."""
from src.strategies.modules.regime_gate import (
    efficiency_ratio, classify_regime, regime_gate,
    ER_CHOP_THRESHOLD, RANGE_THRESHOLD_DELTA, BLOCK_THRESHOLD_DELTA,
)


class TestEfficiencyRatio:
    def test_straight_line_is_one(self):
        assert efficiency_ratio([float(i) for i in range(20)], period=10) == 1.0

    def test_perfect_chop_is_zero(self):
        saw = [100.0 + (2 if i % 2 else -2) for i in range(20)]
        assert efficiency_ratio(saw, period=10) == 0.0

    def test_insufficient_data_is_zero(self):
        assert efficiency_ratio([1.0, 2.0], period=10) == 0.0


class TestClassifyRegime:
    def test_trending_up(self):
        assert classify_regime(close=110, ema_200=100, ema_200_prev=95, er=0.8) == 'TRENDING_UP'

    def test_trending_down(self):
        assert classify_regime(close=90, ema_200=100, ema_200_prev=105, er=0.8) == 'TRENDING_DOWN'

    def test_low_er_is_range_even_if_directional(self):
        # below the chop cutoff -> RANGE regardless of EMA structure
        assert classify_regime(close=110, ema_200=100, ema_200_prev=95,
                               er=ER_CHOP_THRESHOLD - 0.01) == 'RANGE'

    def test_above_ema_but_falling_is_range(self):
        assert classify_regime(close=110, ema_200=100, ema_200_prev=105, er=0.8) == 'RANGE'

    def test_nan_ema_is_range_not_block(self):
        # EMA200 not converged (NaN) -> trend undetermined -> RANGE (never a
        # directional block on missing data). Guards the 200-bar warmup bug.
        nan = float('nan')
        assert classify_regime(close=110, ema_200=nan, ema_200_prev=95, er=0.9) == 'RANGE'
        assert classify_regime(close=110, ema_200=100, ema_200_prev=nan, er=0.9) == 'RANGE'

    def test_nan_ema_prev_does_not_block_counter_trend(self):
        from src.strategies.modules.regime_gate import regime_gate
        g = regime_gate([100.0 + i for i in range(30)], ema_200=110,
                        ema_200_prev=float('nan'), direction='SELL', hard_block=True)
        assert g['blocked'] is False


class TestRegimeGate:
    def _trend_up(self):
        return [100.0 + i for i in range(30)]

    def _trend_down(self):
        return [200.0 - i for i in range(30)]

    def test_block_buy_in_downtrend_hard(self):
        g = regime_gate(self._trend_down(), ema_200=180, ema_200_prev=190,
                        direction='BUY', hard_block=True)
        assert g['regime'] == 'TRENDING_DOWN'
        assert g['blocked'] is True

    def test_allow_sell_in_downtrend(self):
        g = regime_gate(self._trend_down(), ema_200=180, ema_200_prev=190,
                        direction='SELL', hard_block=True)
        assert g['blocked'] is False
        assert g['threshold_delta'] == 0.0

    def test_block_sell_in_uptrend_hard(self):
        g = regime_gate(self._trend_up(), ema_200=110, ema_200_prev=100,
                        direction='SELL', hard_block=True)
        assert g['regime'] == 'TRENDING_UP'
        assert g['blocked'] is True

    def test_counter_trend_soft_penalises_instead_of_blocking(self):
        g = regime_gate(self._trend_down(), ema_200=180, ema_200_prev=190,
                        direction='BUY', hard_block=False)
        assert g['blocked'] is False
        assert g['threshold_delta'] == BLOCK_THRESHOLD_DELTA

    def test_range_raises_threshold_both_ways(self):
        saw = [100.0 + (1 if i % 2 else -1) for i in range(30)]
        for direction in ('BUY', 'SELL'):
            g = regime_gate(saw, ema_200=100, ema_200_prev=100, direction=direction)
            assert g['regime'] == 'RANGE'
            assert g['blocked'] is False
            assert g['threshold_delta'] == RANGE_THRESHOLD_DELTA
