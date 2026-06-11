"""
Bybit Liquidations Data Provider

Real-time liquidation feed via Bybit v5 public WebSocket (allLiquidation topic).
Replaces the Binance forceOrder stream, which is blocked from the production
server IP (and whose REST fallback endpoint was removed by Binance).

Endpoint: wss://stream.bybit.com/v5/public/linear
Topic:    allLiquidation.{SYMBOL} — pushes batched events when liquidations occur.

Message format (v5):
    {"topic":"allLiquidation.BTCUSDT","type":"snapshot","ts":1739502303204,
     "data":[{"T":1739502302929,"s":"BTCUSDT","S":"Buy","v":"0.003","p":"60000"}]}

Side convention — IMPORTANT: Bybit's "S" is the POSITION side, the opposite of
Binance's forceOrder order side. Per the official docs ("Position side. Buy,
Sell. When you receive a Buy update, this means that a long position has been
liquidated"), we convert to the Binance order-side convention expected by all
downstream consumers (side='SELL' => long liquidated, side='BUY' => short):
    Bybit "Buy"  (long liquidated)  -> stored as side='SELL'
    Bybit "Sell" (short liquidated) -> stored as side='BUY'
"""

import csv
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from termcolor import cprint

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    cprint("[BybitLiq] websocket-client not installed — feed disabled", "red")

# CSV persistence (same column layout as the Binance provider so downstream
# consumers and analysis scripts keep working)
LIQUIDATION_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'liquidations')
LIQUIDATION_LOG_FILE = os.path.join(LIQUIDATION_LOG_DIR, 'bybit_liquidations_log.csv')
LIQUIDATION_CSV_COLUMNS = ['timestamp', 'symbol', 'side', 'price', 'quantity', 'usd_value']
LIQUIDATION_RETENTION_DAYS = 7

# Symbols whose Bybit contract name differs from the {TOKEN}USDT pattern
BYBIT_SYMBOL_OVERRIDES = {
    'KPEPE': '1000PEPEUSDT',
    'PEPE': '1000PEPEUSDT',
    'KBONK': '1000BONKUSDT',
    'KSHIB': '1000SHIBUSDT',
}

DEFAULT_SYMBOLS = ['BTC', 'ETH', 'SOL', 'XRP', 'AVAX', 'SUI']

PING_INTERVAL_S = 20
RECONNECT_BASE_DELAY_S = 5
RECONNECT_MAX_DELAY_S = 300


def bybit_symbol(token: str) -> str:
    """Map a strategy token (BTC, kPEPE, ...) to a Bybit linear contract symbol."""
    tok = token.upper()
    return BYBIT_SYMBOL_OVERRIDES.get(tok, f'{tok}USDT')


class BybitLiquidationStream:
    """Real-time liquidation stream from Bybit v5 public linear WebSocket.

    Drop-in replacement for BinanceLiquidationStream as used by
    liquidation_cascade_fade: same get_recent_liquidations() /
    get_historical_liquidations() DataFrame format and is_connected property,
    plus last_message_age_s() and restart() for the feed watchdog.
    """

    WS_URL = "wss://stream.bybit.com/v5/public/linear"

    def __init__(self, symbols: Optional[List[str]] = None,
                 buffer_size: int = 10000, csv_path: Optional[str] = None):
        if symbols is None:
            try:
                from src.config import LIQ_CASCADE_TOKENS
                symbols = list(LIQ_CASCADE_TOKENS)
            except ImportError:
                symbols = list(DEFAULT_SYMBOLS)
        self.symbols = symbols
        self.topics = [f"allLiquidation.{bybit_symbol(s)}" for s in symbols]
        self.csv_path = csv_path or LIQUIDATION_LOG_FILE

        self.liquidations = deque(maxlen=buffer_size)
        self.ws: Optional["websocket.WebSocketApp"] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.ping_thread: Optional[threading.Thread] = None
        self.running = False
        self.connected = False
        self._generation = 0  # bumped on stop so stale threads exit after restart()
        self._last_message_epoch: float = 0.0  # any WS message (pong/ack included)
        self._last_event_epoch: float = 0.0    # actual liquidation events only
        self._lock = threading.Lock()

        self._events_today = 0
        self._events_today_date = datetime.now().date()

        # Cache for get_historical_liquidations (full-file read): one read per
        # cycle instead of one per symbol.
        self._hist_cache = {}
        self._HIST_CACHE_TTL_S = 55

        self._init_csv_log()

    # ------------------------------------------------------------------ CSV
    def _init_csv_log(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                csv.writer(f).writerow(LIQUIDATION_CSV_COLUMNS)
        else:
            self._rotate_csv_log()
            self._seed_events_today()

    def _rotate_csv_log(self):
        """Remove entries older than LIQUIDATION_RETENTION_DAYS."""
        try:
            cutoff = datetime.now() - timedelta(days=LIQUIDATION_RETENTION_DAYS)
            df = pd.read_csv(self.csv_path, parse_dates=['timestamp'])
            before = len(df)
            df = df[df['timestamp'] >= cutoff]
            if len(df) < before:
                df.to_csv(self.csv_path, index=False)
                cprint(f"[BybitLiq] Rotated CSV: removed {before - len(df)} old entries", "yellow")
        except Exception as e:
            cprint(f"[BybitLiq] CSV rotation error: {e}", "yellow")

    def _seed_events_today(self):
        """Repopulate today's event counter from CSV after a restart."""
        try:
            df = pd.read_csv(self.csv_path, usecols=['timestamp'])
            today = datetime.now().strftime('%Y-%m-%d')
            self._events_today = int(df['timestamp'].astype(str).str.startswith(today).sum())
        except Exception:
            pass

    def _append_to_csv(self, liq: Dict):
        try:
            with open(self.csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    liq['timestamp'].isoformat(), liq['symbol'], liq['side'],
                    liq['price'], liq['quantity'], liq['usd_value'],
                ])
        except Exception as e:
            cprint(f"[BybitLiq] CSV write error: {e}", "yellow")

    def get_historical_liquidations(self, hours: int = 24) -> pd.DataFrame:
        """Read historical liquidations from the CSV log (survives restarts).

        Cached ~55s: the cascade strategy calls this once PER SYMBOL per 60s
        cycle, which would otherwise mean 6 full-file reads per minute on a
        file that grows all day.
        """
        now = time.time()
        cached = self._hist_cache.get(hours)
        if cached is not None and now - cached[0] < self._HIST_CACHE_TTL_S:
            return cached[1]
        if not os.path.exists(self.csv_path):
            return pd.DataFrame(columns=LIQUIDATION_CSV_COLUMNS)
        try:
            df = pd.read_csv(self.csv_path, parse_dates=['timestamp'])
            cutoff = datetime.now() - timedelta(hours=hours)
            df = df[df['timestamp'] >= cutoff]
            self._hist_cache[hours] = (now, df)
            return df
        except Exception as e:
            cprint(f"[BybitLiq] CSV read error: {e}", "yellow")
            return pd.DataFrame(columns=LIQUIDATION_CSV_COLUMNS)

    # ------------------------------------------------------------- lifecycle
    def start_stream(self, timeout: float = 20) -> bool:
        """Start the WebSocket feed. Returns True once connected (reconnects forever)."""
        if not WEBSOCKET_AVAILABLE:
            cprint("[BybitLiq] websocket-client missing — cannot start feed", "red")
            return False
        if self.running:
            return self.connected

        self.running = True
        gen = self._generation
        self.ws_thread = threading.Thread(
            target=self._run_forever, args=(gen,), daemon=True, name="BybitLiq_WS")
        self.ws_thread.start()
        self.ping_thread = threading.Thread(
            target=self._ping_loop, args=(gen,), daemon=True, name="BybitLiq_Ping")
        self.ping_thread.start()

        start = time.time()
        while not self.connected and (time.time() - start) < timeout:
            time.sleep(0.2)

        if self.connected:
            cprint(f"[BybitLiq] Connected — subscribed to {len(self.topics)} topics", "green")
        else:
            cprint("[BybitLiq] Connection timeout — will keep retrying in background", "yellow")
        return self.connected

    def stop_stream(self):
        self.running = False
        self._generation += 1  # stale threads see the bump and exit
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.connected = False

    def restart(self) -> bool:
        """Tear down and reconnect the feed (used by the strategy watchdog)."""
        cprint("[BybitLiq] Restarting liquidation feed...", "yellow")
        self.stop_stream()
        time.sleep(1)
        return self.start_stream()

    def _run_forever(self, gen: int):
        """Infinite reconnection loop with 5s -> 300s exponential backoff."""
        delay = RECONNECT_BASE_DELAY_S
        while self.running and self._generation == gen:
            started = time.time()
            try:
                # Generation-guarded callbacks: restart() can bump the
                # generation between the while-check and run_forever(); without
                # the guard a stale thread would open a ghost connection whose
                # events get double-counted (volumes x2 in cascade detection).
                ws = websocket.WebSocketApp(
                    self.WS_URL,
                    on_open=lambda w: self._guarded_on_open(w, gen),
                    on_message=lambda w, m: self._guarded_on_message(w, m, gen),
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # Re-check before connecting/clobbering self.ws: stop_stream()
                # already closed the previous socket and bumped the generation.
                if not self.running or self._generation != gen:
                    return
                self.ws = ws
                ws.run_forever()
            except Exception as e:
                cprint(f"[BybitLiq] WebSocket error: {e}", "red")
            self.connected = False
            if not self.running or self._generation != gen:
                break
            # Stable connection (>5 min) resets the backoff
            if time.time() - started > 300:
                delay = RECONNECT_BASE_DELAY_S
            cprint(f"[BybitLiq] Reconnecting in {delay}s...", "yellow")
            for _ in range(delay):
                if not self.running or self._generation != gen:
                    return
                time.sleep(1)
            delay = min(delay * 2, RECONNECT_MAX_DELAY_S)

    def _ping_loop(self, gen: int):
        """Bybit requires an application-level {"op":"ping"} every ~20s."""
        while self.running and self._generation == gen:
            time.sleep(PING_INTERVAL_S)
            if not self.running or self._generation != gen:
                break
            if self.connected and self.ws:
                try:
                    self.ws.send(json.dumps({"op": "ping"}))
                except Exception:
                    pass  # connection drop is handled by _run_forever

    # ------------------------------------------------------------- callbacks
    def _guarded_on_open(self, ws, gen: int):
        """Close ghost connections opened by a superseded generation."""
        if self._generation != gen:
            try:
                ws.close()
            except Exception:
                pass
            return
        self._on_open(ws)

    def _guarded_on_message(self, ws, message, gen: int):
        """Drop events from a superseded generation (avoid double counting)."""
        if self._generation != gen:
            return
        self._on_message(ws, message)

    def _on_open(self, ws):
        self.connected = True
        ws.send(json.dumps({"op": "subscribe", "args": self.topics}))
        cprint(f"[BybitLiq] WebSocket connected, subscribing: {self.topics}", "green")

    def _on_close(self, ws, close_status_code, close_msg):
        self.connected = False
        cprint(f"[BybitLiq] WebSocket closed: {close_msg}", "yellow")

    def _on_error(self, ws, error):
        cprint(f"[BybitLiq] WebSocket error: {error}", "red")

    def _on_message(self, ws, message):
        self._last_message_epoch = time.time()
        try:
            msg = json.loads(message)
        except (json.JSONDecodeError, TypeError) as e:
            cprint(f"[BybitLiq] Bad message: {e}", "yellow")
            return

        topic = msg.get('topic', '')
        if not topic.startswith('allLiquidation.'):
            # pong / subscription ack — liveness only
            if msg.get('success') is False:
                cprint(f"[BybitLiq] Subscribe failed: {msg.get('ret_msg')}", "red")
            return

        for item in msg.get('data', []):
            try:
                quantity = float(item.get('v', 0))
                price = float(item.get('p', 0))
                ts_ms = int(item.get('T') or msg.get('ts') or 0)
                # Bybit S = POSITION side (Buy = long liquidated). Convert to
                # the Binance order-side convention used downstream:
                # SELL = long liquidated, BUY = short liquidated.
                position_side = str(item.get('S', '')).upper()
                side = 'SELL' if position_side == 'BUY' else 'BUY' if position_side == 'SELL' else position_side
                liq = {
                    'timestamp': datetime.fromtimestamp(ts_ms / 1000),
                    'symbol': item.get('s', ''),
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'usd_value': quantity * price,
                }
            except (TypeError, ValueError) as e:
                cprint(f"[BybitLiq] Bad liquidation item {item}: {e}", "yellow")
                continue

            self._last_event_epoch = time.time()
            with self._lock:
                self.liquidations.append(liq)
                rolled = self._bump_events_today(liq['timestamp'])
            self._append_to_csv(liq)
            if rolled:
                # Daily rotation (was boot-only -> the CSV grew unbounded for
                # the whole container uptime, slowing every full-file read)
                self._rotate_csv_log()

    def _bump_events_today(self, ts: datetime) -> bool:
        """Update today's counter. Returns True when the date rolled over."""
        today = datetime.now().date()
        rolled = self._events_today_date != today
        if rolled:
            self._events_today_date = today
            self._events_today = 0
        if ts.date() == today:
            self._events_today += 1
        return rolled

    # --------------------------------------------------------------- queries
    def get_recent_liquidations(self, minutes: int = 15) -> pd.DataFrame:
        """Liquidations from the last N minutes (in-memory buffer)."""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        with self._lock:
            recent = [liq for liq in self.liquidations if liq['timestamp'] >= cutoff]
        if not recent:
            return pd.DataFrame(columns=['timestamp', 'symbol', 'side', 'quantity', 'price', 'usd_value'])
        return pd.DataFrame(recent)

    def get_liquidation_ratio(self, minutes: int = 15) -> float:
        """Long/short liquidation ratio (same contract as the Binance provider).

        SELL side = long positions liquidated, BUY side = short liquidated.
        Ratio > 1.0 = more longs liquidated (bearish pressure). 1.0 if no data.
        """
        df = self.get_recent_liquidations(minutes)
        if df.empty:
            return 1.0
        long_liqs = df[df['side'] == 'SELL']['usd_value'].sum()
        short_liqs = df[df['side'] == 'BUY']['usd_value'].sum()
        if short_liqs == 0:
            return 2.0 if long_liqs > 0 else 1.0
        return round(long_liqs / short_liqs, 2)

    def get_liquidation_summary(self, minutes: int = 15) -> Dict:
        """Summary of recent liquidations (same contract as the Binance provider)."""
        df = self.get_recent_liquidations(minutes)
        if df.empty:
            return {
                'total_count': 0, 'total_usd': 0.0,
                'long_usd': 0.0, 'short_usd': 0.0,
                'ratio': 1.0, 'top_symbols': [],
            }
        long_usd = df[df['side'] == 'SELL']['usd_value'].sum()
        short_usd = df[df['side'] == 'BUY']['usd_value'].sum()
        top_symbols = df.groupby('symbol')['usd_value'].sum().nlargest(5).to_dict()
        return {
            'total_count': len(df),
            'total_usd': float(df['usd_value'].sum()),
            'long_usd': float(long_usd),
            'short_usd': float(short_usd),
            'ratio': round(long_usd / short_usd, 2) if short_usd > 0 else 1.0,
            'top_symbols': top_symbols,
        }

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def buffer_count(self) -> int:
        return len(self.liquidations)

    def last_message_age_s(self) -> Optional[float]:
        """Seconds since ANY WS message (pong/ack included). None if never."""
        if self._last_message_epoch <= 0:
            return None
        return time.time() - self._last_message_epoch

    def last_event_age_s(self) -> Optional[float]:
        """Seconds since the last actual liquidation event. None if never."""
        if self._last_event_epoch <= 0:
            return None
        return time.time() - self._last_event_epoch

    def events_today(self) -> int:
        if self._events_today_date != datetime.now().date():
            return 0
        return self._events_today


# ---------------------------------------------------------------- singleton
_liquidation_stream: Optional[BybitLiquidationStream] = None


def get_liquidation_stream() -> BybitLiquidationStream:
    """Get or create the singleton Bybit liquidation stream."""
    global _liquidation_stream
    if _liquidation_stream is None:
        _liquidation_stream = BybitLiquidationStream()
    return _liquidation_stream


def get_liquidation_ratio(minutes: int = 15) -> float:
    """Convenience function (same contract as binance_futures): starts the
    stream if not already running and returns the long/short ratio."""
    stream = get_liquidation_stream()
    if not stream.is_connected:
        stream.start_stream()
        time.sleep(2)  # give it a moment to collect some data
    return stream.get_liquidation_ratio(minutes)


# /api/health is hit every 30-60s by uptime checkers; reading the whole CSV
# synchronously inside the FastAPI event loop on every hit blocks the loop.
_CSV_STATS_CACHE: Dict[str, tuple] = {}  # path -> (epoch, result)
_CSV_STATS_CACHE_TTL_S = 30


def _csv_feed_stats(csv_path: str) -> Dict:
    """Last event age + today's event count from the CSV log (cross-process).

    Result is cached _CSV_STATS_CACHE_TTL_S per path (the age is recomputed
    from the cached max timestamp so it keeps increasing between reads).
    """
    now = time.time()
    cached = _CSV_STATS_CACHE.get(csv_path)
    if cached is not None and now - cached[0] < _CSV_STATS_CACHE_TTL_S:
        last_ts, events_today = cached[1]
        return {
            'last_event_age_s': (round((datetime.now() - last_ts).total_seconds())
                                 if last_ts is not None else None),
            'events_today': events_today,
        }

    out = {'last_event_age_s': None, 'events_today': 0}
    last_ts = None
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, usecols=['timestamp'])
            if not df.empty:
                ts = pd.to_datetime(df['timestamp'], errors='coerce').dropna()
                if not ts.empty:
                    last_ts = ts.max()
                    out['last_event_age_s'] = round((datetime.now() - last_ts).total_seconds())
                    today = datetime.now().strftime('%Y-%m-%d')
                    out['events_today'] = int(df['timestamp'].astype(str).str.startswith(today).sum())
        except Exception:
            return out
    _CSV_STATS_CACHE[csv_path] = (now, (last_ts, out['events_today']))
    return out


def get_feed_status(csv_path: Optional[str] = None) -> Dict:
    """Feed status for healthchecks.

    Works cross-process: in the bot process it reports the live WS state;
    in the web process (no stream instance) it falls back to the CSV log,
    which both processes can read.
    """
    status = {
        'provider': 'bybit',
        'connected': None,
        'last_message_age_s': None,
        'last_event_age_s': None,
        'events_today': 0,
    }
    stream = _liquidation_stream
    if stream is not None:
        status['connected'] = stream.is_connected
        status['last_message_age_s'] = (
            round(stream.last_message_age_s()) if stream.last_message_age_s() is not None else None
        )

    csv_stats = _csv_feed_stats(csv_path or (stream.csv_path if stream else LIQUIDATION_LOG_FILE))
    status['last_event_age_s'] = csv_stats['last_event_age_s']
    status['events_today'] = csv_stats['events_today']

    # In-memory event age can be fresher than the CSV tail (write lag)
    if stream is not None and stream.last_event_age_s() is not None:
        mem_age = round(stream.last_event_age_s())
        if status['last_event_age_s'] is None or mem_age < status['last_event_age_s']:
            status['last_event_age_s'] = mem_age

    return status


# Standalone smoke test
if __name__ == "__main__":
    cprint("Testing Bybit liquidation stream (60s)...", "cyan")
    stream = get_liquidation_stream()
    stream.start_stream()
    time.sleep(60)
    df = stream.get_recent_liquidations(minutes=5)
    cprint(f"Collected {len(df)} liquidations | connected={stream.is_connected} "
           f"| last_msg_age={stream.last_message_age_s()}", "white")
    if not df.empty:
        print(df.tail(10))
    stream.stop_stream()
