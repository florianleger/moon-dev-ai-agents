"""Tests for the Bybit liquidation feed, dashboard PnL accounting and feed healthcheck."""
import json
import os
import sys
import types
from datetime import datetime, timedelta

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# pandas_ta is not installed in test env; mock it before any strategy import
if 'pandas_ta' not in sys.modules:
    sys.modules['pandas_ta'] = MagicMock()

from src.data_providers import bybit_liquidations as bl


# ---------------------------------------------------------------------------
# Bybit feed — message parsing + persistence
# ---------------------------------------------------------------------------

def _make_stream(tmp_path, symbols=None):
    return bl.BybitLiquidationStream(
        symbols=symbols or ['BTC', 'ETH'],
        csv_path=str(tmp_path / 'bybit_liq.csv'),
    )


def _liq_message(symbol='BTCUSDT', side='Buy', qty='0.003', price='60000', ts=None):
    ts_ms = int((ts or datetime.now()).timestamp() * 1000)
    return json.dumps({
        "topic": f"allLiquidation.{symbol}",
        "type": "snapshot",
        "ts": ts_ms,
        "data": [{"T": ts_ms, "s": symbol, "S": side, "v": qty, "p": price}],
    })


class TestBybitParsing:

    def test_symbol_mapping(self):
        assert bl.bybit_symbol('BTC') == 'BTCUSDT'
        assert bl.bybit_symbol('sui') == 'SUIUSDT'
        assert bl.bybit_symbol('kPEPE') == '1000PEPEUSDT'

    def test_topics_built_from_symbols(self, tmp_path):
        s = _make_stream(tmp_path, symbols=['BTC', 'ETH', 'SOL'])
        assert s.topics == ['allLiquidation.BTCUSDT', 'allLiquidation.ETHUSDT', 'allLiquidation.SOLUSDT']

    def test_parse_long_liquidation(self, tmp_path):
        """Bybit S='Buy' = POSITION side = a long was liquidated.

        Stored as 'SELL' (Binance order-side convention used downstream:
        SELL = long liquidated).
        """
        s = _make_stream(tmp_path)
        s._on_message(None, _liq_message(side='Buy', qty='0.003', price='60000'))

        df = s.get_recent_liquidations(minutes=5)
        assert len(df) == 1
        row = df.iloc[0]
        assert row['symbol'] == 'BTCUSDT'
        assert row['side'] == 'SELL'
        assert row['quantity'] == pytest.approx(0.003)
        assert row['price'] == pytest.approx(60000.0)
        assert row['usd_value'] == pytest.approx(180.0)
        assert set(df.columns) >= {'timestamp', 'symbol', 'side', 'quantity', 'price', 'usd_value'}

    def test_parse_short_liquidation(self, tmp_path):
        """Bybit S='Sell' = a short was liquidated -> stored as 'BUY'."""
        s = _make_stream(tmp_path)
        s._on_message(None, _liq_message(symbol='ETHUSDT', side='Sell', qty='2', price='2500'))

        df = s.get_recent_liquidations(minutes=5)
        assert df.iloc[0]['side'] == 'BUY'
        assert df.iloc[0]['usd_value'] == pytest.approx(5000.0)

    def test_pong_counts_as_liveness_not_event(self, tmp_path):
        s = _make_stream(tmp_path)
        assert s.last_message_age_s() is None
        s._on_message(None, json.dumps({"op": "pong", "success": True, "ret_msg": "pong"}))
        assert s.last_message_age_s() is not None
        assert s.last_message_age_s() < 5
        assert s.buffer_count == 0
        assert s.events_today() == 0
        assert s.last_event_age_s() is None

    def test_csv_persistence_and_historical(self, tmp_path):
        s = _make_stream(tmp_path)
        s._on_message(None, _liq_message(side='Buy'))
        s._on_message(None, _liq_message(symbol='ETHUSDT', side='Sell', qty='1', price='2500'))

        hist = s.get_historical_liquidations(hours=1)
        assert len(hist) == 2
        assert list(hist.columns) == bl.LIQUIDATION_CSV_COLUMNS
        assert hist['usd_value'].sum() == pytest.approx(180.0 + 2500.0)
        assert s.events_today() == 2

    def test_old_event_excluded_from_recent(self, tmp_path):
        s = _make_stream(tmp_path)
        old = datetime.now() - timedelta(hours=3)
        s._on_message(None, _liq_message(ts=old))
        assert s.get_recent_liquidations(minutes=15).empty
        assert len(s.get_historical_liquidations(hours=24)) == 1

    def test_restart_bumps_generation(self, tmp_path):
        """restart() must invalidate previous worker threads via generation token."""
        s = _make_stream(tmp_path)
        gen = s._generation
        s.stop_stream()
        assert s._generation == gen + 1
        assert s.running is False
        assert s.connected is False

    def test_stale_generation_events_dropped(self, tmp_path):
        """A ghost connection from a superseded generation must not double-count."""
        s = _make_stream(tmp_path)
        gen = s._generation
        s._guarded_on_message(None, _liq_message(), gen)
        assert s.buffer_count == 1
        s.stop_stream()  # bumps generation
        s._guarded_on_message(None, _liq_message(), gen)
        assert s.buffer_count == 1  # stale event ignored

    def test_stale_generation_on_open_closes_ghost_ws(self, tmp_path):
        s = _make_stream(tmp_path)
        gen = s._generation
        s.stop_stream()
        ghost_ws = MagicMock()
        s._guarded_on_open(ghost_ws, gen)
        ghost_ws.close.assert_called_once()
        assert s.connected is False
        # current generation still opens normally
        live_ws = MagicMock()
        s._guarded_on_open(live_ws, s._generation)
        live_ws.send.assert_called_once()
        assert s.connected is True

    def test_csv_rotation_triggered_on_date_rollover(self, tmp_path):
        """Rotation was boot-only; it must also run when the date rolls over."""
        s = _make_stream(tmp_path)
        s._events_today_date = (datetime.now() - timedelta(days=1)).date()
        with patch.object(s, '_rotate_csv_log') as rot:
            s._on_message(None, _liq_message())
        rot.assert_called_once()


# ---------------------------------------------------------------------------
# Dashboard accounting — the 3 CSV conventions seen in prod
# ---------------------------------------------------------------------------

from src.web.api import dashboard as dash


class TestTradePnlSeries:

    def test_ah_closed_trades_convention(self):
        """AH closed_trades.csv: total_pnl already includes partials; entry fee excluded."""
        df = pd.DataFrame({
            'pnl': [0.78, -1.37],
            'partial_pnl_realized': [0.51, 0.0],
            'total_pnl': [1.29, -1.37],
            'entry_fee': [0.01, 0.02],
        })
        result = dash._trade_pnl_series(df)
        assert result.tolist() == pytest.approx([1.28, -1.39])

    def test_ah_paper_trades_convention(self):
        """AH paper_trades.csv: no total_pnl column -> pnl + partial - entry_fee."""
        df = pd.DataFrame({
            'pnl': [0.78, -1.37],
            'partial_pnl_realized': [0.51, 0.0],
            'entry_fee': [0.01, 0.02],
        })
        result = dash._trade_pnl_series(df)
        assert result.tolist() == pytest.approx([1.28, -1.39])

    def test_independent_strategy_convention(self):
        """OTE/FMR/VB/LC: pnl net of exit fee, entry fee deducted separately."""
        df = pd.DataFrame({
            'pnl': [2.0, -1.0],
            'entry_fee': [0.15, 0.15],
        })
        result = dash._trade_pnl_series(df)
        assert result.tolist() == pytest.approx([1.85, -1.15])

    def test_missing_fee_column(self):
        df = pd.DataFrame({'pnl': [1.0, -0.5]})
        assert dash._trade_pnl_series(df).tolist() == pytest.approx([1.0, -0.5])


class TestAggregateStrategyStats:

    def _write_closed(self, folder, df):
        os.makedirs(folder, exist_ok=True)
        df.to_csv(os.path.join(folder, 'closed_trades.csv'), index=False)

    def test_total_pnl_includes_partials_and_entry_fees(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, 'DATA_BASE_PATH', str(tmp_path))
        now = datetime.now().isoformat()
        self._write_closed(tmp_path / 'adaptive_hybrid', pd.DataFrame({
            'pnl': [0.78, -1.37],
            'partial_pnl_realized': [0.51, 0.0],
            'total_pnl': [1.29, -1.37],
            'entry_fee': [0.01, 0.02],
            'exit_time': [now, now],
        }))

        stats = dash._aggregate_strategy_stats('adaptive_hybrid')
        assert stats['closed_trades'] == 2
        assert stats['total_pnl'] == pytest.approx(-0.11, abs=0.01)
        assert stats['daily_pnl'] == pytest.approx(-0.11, abs=0.01)
        assert stats['win_rate'] == 50.0

    def test_win_rate_uses_corrected_pnl(self, tmp_path, monkeypatch):
        """A trade with pnl=+0.05 but entry_fee=0.15 is actually a loss."""
        monkeypatch.setattr(dash, 'DATA_BASE_PATH', str(tmp_path))
        self._write_closed(tmp_path / 'ote_scalp', pd.DataFrame({
            'pnl': [0.05, 1.0],
            'entry_fee': [0.15, 0.15],
            'exit_time': ['2026-01-01T10:00:00', '2026-01-01T11:00:00'],
        }))
        stats = dash._aggregate_strategy_stats('ote_scalp')
        assert stats['win_rate'] == 50.0
        assert stats['total_pnl'] == pytest.approx(0.75)
        assert stats['daily_pnl'] == 0.0  # not today


class TestStatsFromCsv:

    def test_balance_uses_balance_after_ledger(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, 'DATA_BASE_PATH', str(tmp_path))
        monkeypatch.setattr(dash, '_get_strategy_folder', lambda: 'adaptive_hybrid')
        monkeypatch.setattr(dash, '_get_leverage', lambda: 5)

        folder = tmp_path / 'adaptive_hybrid'
        os.makedirs(folder)
        now = datetime.now().isoformat()
        # paper_trades: 1 closed row, no open positions
        pd.DataFrame({
            'status': ['CLOSED'],
            'pnl': [0.78],
            'partial_pnl_realized': [0.51],
            'entry_fee': [0.01],
            'position_size': [100.0],
            'exit_time': [now],
        }).to_csv(folder / 'paper_trades.csv', index=False)
        # closed_trades: source of truth with total_pnl + balance_after
        pd.DataFrame({
            'pnl': [0.78],
            'partial_pnl_realized': [0.51],
            'total_pnl': [1.29],
            'entry_fee': [0.01],
            'exit_time': [now],
            'balance_after': [501.28],
        }).to_csv(folder / 'closed_trades.csv', index=False)

        stats = dash._get_stats_from_csv()
        assert stats['open_positions'] == 0
        assert stats['realized_pnl'] == pytest.approx(1.28)
        assert stats['total_pnl'] == pytest.approx(1.28)
        assert stats['daily_pnl'] == pytest.approx(1.28)
        # balance reconstructs the strategy's paper_balance (INITIAL + total_pnl
        # - entry fees); referenced off the configured initial balance
        assert stats['balance'] == pytest.approx(dash.INITIAL_BALANCE + 1.28)

    def test_balance_includes_open_scale_out_partials_and_entry_fees(self, tmp_path, monkeypatch):
        """Partials credited on a still-open position and the entry fee of a
        position opened after the last close are part of the live paper_balance
        but invisible to the last balance_after — the dashboard must add them."""
        monkeypatch.setattr(dash, 'DATA_BASE_PATH', str(tmp_path))
        monkeypatch.setattr(dash, '_get_strategy_folder', lambda: 'adaptive_hybrid')
        monkeypatch.setattr(dash, '_get_leverage', lambda: 5)
        # No live price provider in tests (unrealized PnL = 0 with entry_price=0)
        monkeypatch.setattr('src.data_providers.market_data.get_market_data_provider',
                            lambda: None, raising=False)

        folder = tmp_path / 'adaptive_hybrid'
        os.makedirs(folder)
        now = datetime.now().isoformat()
        # paper_trades: 1 closed row + 1 OPEN row that already scaled out $4.50
        # (entry fee $0.07 deducted at open). entry_price=0 -> unrealized = 0.
        pd.DataFrame({
            'status': ['CLOSED', 'OPEN'],
            'symbol': ['BTC', 'ETH'],
            'direction': ['BUY', 'BUY'],
            'entry_price': [100.0, 0.0],
            'pnl': [0.78, 0.0],
            'partial_pnl_realized': [0.51, 4.50],
            'entry_fee': [0.01, 0.07],
            'position_size': [100.0, 75.0],
            'exit_time': [now, None],
        }).to_csv(folder / 'paper_trades.csv', index=False)
        # closed_trades ledger: last balance_after predates the scale-out
        pd.DataFrame({
            'pnl': [0.78],
            'partial_pnl_realized': [0.51],
            'total_pnl': [1.29],
            'entry_fee': [0.01],
            'exit_time': [now],
            'balance_after': [501.28],
        }).to_csv(folder / 'closed_trades.csv', index=False)

        stats = dash._get_stats_from_csv()
        assert stats['open_positions'] == 1
        assert stats['realized_pnl'] == pytest.approx(1.28)
        # INITIAL + 1.28 (closed) + 4.50 (open partials) - 0.07 (open entry fee)
        assert stats['balance'] == pytest.approx(dash.INITIAL_BALANCE + 5.71)

    def test_fallback_without_closed_trades_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(dash, 'DATA_BASE_PATH', str(tmp_path))
        monkeypatch.setattr(dash, '_get_strategy_folder', lambda: 'adaptive_hybrid')
        monkeypatch.setattr(dash, '_get_leverage', lambda: 5)

        folder = tmp_path / 'adaptive_hybrid'
        os.makedirs(folder)
        pd.DataFrame({
            'status': ['CLOSED'],
            'pnl': [2.0],
            'partial_pnl_realized': [1.0],
            'entry_fee': [0.5],
            'position_size': [100.0],
            'exit_time': ['2026-01-01T10:00:00'],
        }).to_csv(folder / 'paper_trades.csv', index=False)

        stats = dash._get_stats_from_csv()
        assert stats['realized_pnl'] == pytest.approx(2.5)
        assert stats['balance'] == pytest.approx(dash.INITIAL_BALANCE + 2.5)


# ---------------------------------------------------------------------------
# Healthcheck — feed status (cross-process via CSV) + degraded logic
# ---------------------------------------------------------------------------

class TestFeedHealthcheck:

    def _write_feed_csv(self, path, last_ts):
        pd.DataFrame({
            'timestamp': [last_ts.isoformat()],
            'symbol': ['BTCUSDT'],
            'side': ['BUY'],
            'price': [60000.0],
            'quantity': [0.01],
            'usd_value': [600.0],
        }).to_csv(path, index=False)

    def test_feed_status_from_stale_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bl, '_liquidation_stream', None)
        csv_path = tmp_path / 'bybit_liq.csv'
        self._write_feed_csv(csv_path, datetime.now() - timedelta(hours=26))

        status = bl.get_feed_status(csv_path=str(csv_path))
        assert status['provider'] == 'bybit'
        assert status['connected'] is None  # no in-process stream (web process)
        assert status['last_event_age_s'] > 3600
        assert status['events_today'] == 0

    def test_feed_status_fresh_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bl, '_liquidation_stream', None)
        csv_path = tmp_path / 'bybit_liq.csv'
        self._write_feed_csv(csv_path, datetime.now() - timedelta(minutes=5))

        status = bl.get_feed_status(csv_path=str(csv_path))
        assert status['last_event_age_s'] < 3600
        assert status['events_today'] == 1

    def test_feed_status_missing_csv(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bl, '_liquidation_stream', None)
        status = bl.get_feed_status(csv_path=str(tmp_path / 'nope.csv'))
        assert status['last_event_age_s'] is None
        assert status['events_today'] == 0

    def test_health_degraded_when_feed_stale(self, monkeypatch):
        import importlib
        web_app = importlib.import_module('src.web.app')
        monkeypatch.setattr(
            bl, 'get_feed_status',
            lambda csv_path=None: {'provider': 'bybit', 'connected': False,
                                   'last_message_age_s': None,
                                   'last_event_age_s': 7200, 'events_today': 0},
        )
        snapshot = web_app._attach_liquidation_feed_status({'status': 'ok'})
        assert snapshot['status'] == 'degraded'
        assert 'liquidation feed' in snapshot['degraded_reason']
        assert snapshot['liquidation_feed']['last_event_age_s'] == 7200

    def test_health_ok_when_feed_fresh(self, monkeypatch):
        import importlib
        web_app = importlib.import_module('src.web.app')
        monkeypatch.setattr(
            bl, 'get_feed_status',
            lambda csv_path=None: {'provider': 'bybit', 'connected': True,
                                   'last_message_age_s': 3,
                                   'last_event_age_s': 120, 'events_today': 42},
        )
        snapshot = web_app._attach_liquidation_feed_status({'status': 'ok'})
        assert snapshot['status'] == 'ok'
        assert snapshot['liquidation_feed']['events_today'] == 42

    def test_health_unknown_feed_does_not_degrade(self, monkeypatch):
        """No CSV yet (fresh deploy) -> age None -> do not flag degraded."""
        import importlib
        web_app = importlib.import_module('src.web.app')
        monkeypatch.setattr(
            bl, 'get_feed_status',
            lambda csv_path=None: {'provider': 'bybit', 'connected': None,
                                   'last_message_age_s': None,
                                   'last_event_age_s': None, 'events_today': 0},
        )
        snapshot = web_app._attach_liquidation_feed_status({'status': 'ok'})
        assert snapshot['status'] == 'ok'

    def test_frozen_scheduler_not_masked_by_feed(self, monkeypatch):
        """Feed status must never upgrade a frozen/degraded scheduler status."""
        import importlib
        web_app = importlib.import_module('src.web.app')
        monkeypatch.setattr(
            bl, 'get_feed_status',
            lambda csv_path=None: {'provider': 'bybit', 'connected': True,
                                   'last_message_age_s': 3,
                                   'last_event_age_s': 120, 'events_today': 42},
        )
        snapshot = web_app._attach_liquidation_feed_status({'status': 'frozen'})
        assert snapshot['status'] == 'frozen'


# ---------------------------------------------------------------------------
# Strategy feed watchdog
# ---------------------------------------------------------------------------

class _FakeStream:
    def __init__(self, connected=True, age=None):
        self.is_connected = connected
        self._age = age
        self.restarts = 0

    def last_message_age_s(self):
        return self._age

    def restart(self):
        self.restarts += 1
        return True


def _make_watchdog_stub():
    with patch('hyperliquid.info.Info', MagicMock()):
        from src.strategies.custom.liquidation_cascade_fade import LiquidationCascadeFadeStrategy
    s = LiquidationCascadeFadeStrategy.__new__(LiquidationCascadeFadeStrategy)
    s._last_feed_restart = 0.0
    s.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None,
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    return s


class TestFeedWatchdog:

    def test_restart_when_disconnected(self):
        s = _make_watchdog_stub()
        s._liq_stream = _FakeStream(connected=False, age=30)
        s._feed_watchdog()
        assert s._liq_stream.restarts == 1

    def test_restart_when_silent_too_long(self):
        s = _make_watchdog_stub()
        s._liq_stream = _FakeStream(connected=True, age=700)
        s._feed_watchdog()
        assert s._liq_stream.restarts == 1

    def test_no_restart_when_healthy(self):
        s = _make_watchdog_stub()
        s._liq_stream = _FakeStream(connected=True, age=30)
        s._feed_watchdog()
        assert s._liq_stream.restarts == 0

    def test_restart_rate_limited(self):
        s = _make_watchdog_stub()
        s._liq_stream = _FakeStream(connected=False, age=None)
        s._feed_watchdog()
        s._feed_watchdog()  # within 10-min cooldown
        assert s._liq_stream.restarts == 1


# ---------------------------------------------------------------------------
# Bookkeeping: closing a position flips its paper_trades.csv row to CLOSED
# (without this, _load_state resurrects closed trades as live phantoms)
# ---------------------------------------------------------------------------

class TestPaperCsvBookkeeping:

    def test_update_paper_csv_flips_status_to_closed(self, tmp_path):
        s = _make_watchdog_stub()
        s.data_dir = str(tmp_path)
        f = tmp_path / 'paper_trades.csv'
        pd.DataFrame([
            {'position_id': 'LC_SOL_1', 'symbol': 'SOL', 'status': 'OPEN'},
            {'position_id': 'LC_ETH_2', 'symbol': 'ETH', 'status': 'OPEN'},
        ]).to_csv(f, index=False)

        s._update_paper_csv('LC_SOL_1')

        df = pd.read_csv(f)
        status = dict(zip(df['position_id'], df['status']))
        assert status['LC_SOL_1'] == 'CLOSED'
        assert status['LC_ETH_2'] == 'OPEN'  # untouched

    def test_update_paper_csv_no_file_is_safe(self, tmp_path):
        s = _make_watchdog_stub()
        s.data_dir = str(tmp_path)
        s._update_paper_csv('LC_SOL_1')  # must not raise when csv absent


# ---------------------------------------------------------------------------
# Cascade detection — absolute trigger calibrated for the exhaustive Bybit feed
# ---------------------------------------------------------------------------

class _FakeCascadeStream:
    """Stream stub: `recent` for short lookbacks, `baseline` for the 24h pull."""

    def __init__(self, recent_df, baseline_df=None):
        self._recent = recent_df
        self._baseline = baseline_df

    def get_recent_liquidations(self, minutes=15):
        if minutes >= 1440:
            return self._baseline if self._baseline is not None else pd.DataFrame()
        return self._recent

    def get_historical_liquidations(self, hours=24):
        return pd.DataFrame()


def _recent_btc_liqs(total_usd, n=4):
    now = datetime.now()
    return pd.DataFrame({
        'timestamp': [now - timedelta(minutes=i) for i in range(n)],
        'symbol': ['BTCUSDT'] * n,
        'side': ['SELL'] * n,
        'price': [60000.0] * n,
        'quantity': [1.0] * n,
        'usd_value': [total_usd / n] * n,
    })


def _baseline_btc_liqs(per_window_usd=1_500_000, windows=12):
    """One event every 2 hours -> `windows` distinct 15-min windows."""
    now = datetime.now()
    rows = []
    for i in range(windows):
        # slight variance so std > 0
        rows.append({
            'timestamp': now - timedelta(hours=2 * (i + 1)),
            'symbol': 'BTCUSDT',
            'side': 'SELL',
            'price': 60000.0,
            'quantity': 1.0,
            'usd_value': per_window_usd * (0.9 + 0.02 * i),
        })
    return pd.DataFrame(rows)


def _make_cascade_stub(stream):
    with patch('hyperliquid.info.Info', MagicMock()):
        from src.strategies.custom.liquidation_cascade_fade import LiquidationCascadeFadeStrategy
    s = LiquidationCascadeFadeStrategy.__new__(LiquidationCascadeFadeStrategy)
    s._liq_stream = stream
    s._empty_feed_logged = False
    s.logger = types.SimpleNamespace(
        warning=lambda *a, **k: None,
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
    )
    return s


class TestCascadeAbsoluteTrigger:

    def test_routine_bybit_flow_does_not_trigger(self):
        """$1.5M/15min on BTC is ROUTINE on the exhaustive Bybit feed (> the
        static $1M floor calibrated for the sampled Binance feed) — with a
        baseline at the same level it must NOT fire."""
        s = _make_cascade_stub(_FakeCascadeStream(
            recent_df=_recent_btc_liqs(1_500_000),
            baseline_df=_baseline_btc_liqs(per_window_usd=1_500_000),
        ))
        assert s._detect_cascade('BTC') is None

    def test_absolute_trigger_disarmed_without_baseline(self):
        """First hours post-deploy (empty baseline): the absolute trigger must
        stay disarmed instead of firing on routine flow."""
        s = _make_cascade_stub(_FakeCascadeStream(
            recent_df=_recent_btc_liqs(1_500_000),
            baseline_df=None,
        ))
        assert s._detect_cascade('BTC') is None

    def test_genuine_anomaly_still_triggers(self):
        """Volume >> K x baseline mean fires the trigger."""
        s = _make_cascade_stub(_FakeCascadeStream(
            recent_df=_recent_btc_liqs(9_000_000),
            baseline_df=_baseline_btc_liqs(per_window_usd=1_500_000),
        ))
        cascade = s._detect_cascade('BTC')
        assert cascade is not None
        assert cascade['detected'] is True
        assert cascade['cascade_side'] == 'LONG_LIQUIDATED'
        assert cascade['fade_direction'] == 'BUY'
