"""
Binance Futures Data Provider

Provides real-time liquidation data via WebSocket stream.
No API key required for public market data.

WebSocket endpoint: wss://fstream.binance.com/ws/!forceOrder@arr
"""

import json
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import pandas as pd
from termcolor import cprint

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    cprint("[BinanceFutures] websocket-client not installed, using REST fallback", "yellow")

import requests


class BinanceLiquidationStream:
    """
    Real-time liquidation stream from Binance Futures.

    Liquidations are stored in a circular buffer and can be queried
    for ratio calculations (long vs short liquidations).
    """

    WS_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"
    REST_URL = "https://fapi.binance.com/fapi/v1/allForceOrders"

    def __init__(self, buffer_size: int = 10000):
        """
        Initialize the liquidation stream.

        Args:
            buffer_size: Maximum number of liquidations to keep in memory
        """
        self.buffer_size = buffer_size
        self.liquidations = deque(maxlen=buffer_size)
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.running = False
        self.connected = False
        self.last_message_time: Optional[datetime] = None
        self._lock = threading.Lock()

    def start_stream(self) -> bool:
        """
        Start the WebSocket stream for real-time liquidations.

        Returns:
            bool: True if started successfully
        """
        if not WEBSOCKET_AVAILABLE:
            cprint("[BinanceFutures] WebSocket not available, will use REST polling", "yellow")
            return False

        if self.running:
            return True

        try:
            self.ws = websocket.WebSocketApp(
                self.WS_URL,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )

            self.ws_thread = threading.Thread(target=self._run_forever, daemon=True)
            self.ws_thread.start()
            self.running = True

            # Wait for connection (longer timeout for container startup)
            timeout = 30
            start = time.time()
            while not self.connected and (time.time() - start) < timeout:
                time.sleep(0.2)

            if self.connected:
                cprint("[BinanceFutures] Connected to liquidation stream", "green")
                return True
            else:
                cprint("[BinanceFutures] Connection timeout - liquidations will use neutral ratio", "yellow")
                return False

        except Exception as e:
            cprint(f"[BinanceFutures] Failed to start stream: {e}", "red")
            return False

    def stop_stream(self):
        """Stop the WebSocket stream."""
        self.running = False
        if self.ws:
            self.ws.close()
        self.connected = False

    def _run_forever(self):
        """Run WebSocket with automatic reconnection."""
        reconnect_delay = 5
        while self.running:
            try:
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                cprint(f"[BinanceFutures] WebSocket error: {e}", "yellow")

            if self.running:
                cprint(f"[BinanceFutures] Reconnecting in {reconnect_delay}s...", "yellow")
                time.sleep(reconnect_delay)

    def _on_open(self, ws):
        """Handle WebSocket connection opened."""
        self.connected = True
        cprint("[BinanceFutures] WebSocket connected", "green")

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed."""
        self.connected = False
        cprint(f"[BinanceFutures] WebSocket closed: {close_msg}", "yellow")

    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        cprint(f"[BinanceFutures] WebSocket error: {error}", "red")

    def _on_message(self, ws, message):
        """
        Handle incoming liquidation message.

        Message format:
        {
            "e": "forceOrder",
            "E": 1568014460893,  # Event time
            "o": {
                "s": "BTCUSDT",   # Symbol
                "S": "SELL",      # Side (SELL = long liquidated, BUY = short liquidated)
                "o": "LIMIT",     # Order type
                "f": "IOC",       # Time in force
                "q": "0.001",     # Original quantity
                "p": "9910",      # Price
                "ap": "9910",     # Average price
                "X": "FILLED",    # Order status
                "l": "0.001",     # Last filled quantity
                "z": "0.001",     # Accumulated filled quantity
                "T": 1568014460893  # Order trade time
            }
        }
        """
        try:
            data = json.loads(message)
            if data.get('e') == 'forceOrder':
                order = data['o']

                # Calculate USD value
                quantity = float(order.get('q', 0))
                price = float(order.get('ap', order.get('p', 0)))
                usd_value = quantity * price

                liquidation = {
                    'timestamp': datetime.fromtimestamp(data['E'] / 1000),
                    'symbol': order['s'],
                    'side': order['S'],  # SELL = long liq, BUY = short liq
                    'quantity': quantity,
                    'price': price,
                    'usd_value': usd_value,
                    'status': order.get('X', 'UNKNOWN')
                }

                with self._lock:
                    self.liquidations.append(liquidation)
                    self.last_message_time = datetime.now()

        except Exception as e:
            cprint(f"[BinanceFutures] Error parsing message: {e}", "yellow")

    def fetch_recent_liquidations_rest(self, limit: int = 1000) -> List[Dict]:
        """
        Fetch recent liquidations via REST API (fallback).

        Note: This endpoint may require API key for full access.
        Without key, returns limited data.

        Args:
            limit: Number of records to fetch (max 1000)

        Returns:
            List of liquidation records
        """
        try:
            # Try public endpoint first (limited data)
            url = f"{self.REST_URL}?limit={min(limit, 1000)}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                liquidations = []
                for item in data:
                    liquidations.append({
                        'timestamp': datetime.fromtimestamp(item['time'] / 1000),
                        'symbol': item['symbol'],
                        'side': item['side'],
                        'quantity': float(item['origQty']),
                        'price': float(item['averagePrice']),
                        'usd_value': float(item['origQty']) * float(item['averagePrice']),
                        'status': item['status']
                    })
                return liquidations
            else:
                # REST API may require symbol parameter or authentication
                # This is non-critical - WebSocket is the primary method
                return []

        except Exception as e:
            cprint(f"[BinanceFutures] REST API error: {e}", "yellow")
            return []

    def get_recent_liquidations(self, minutes: int = 15) -> pd.DataFrame:
        """
        Get liquidations from the last N minutes.

        Args:
            minutes: Lookback period in minutes

        Returns:
            DataFrame with liquidation records
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)

        with self._lock:
            recent = [
                liq for liq in self.liquidations
                if liq['timestamp'] >= cutoff
            ]

        if not recent:
            # Try REST fallback if WebSocket has no data
            if not self.connected:
                recent = self.fetch_recent_liquidations_rest()
                recent = [
                    liq for liq in recent
                    if liq['timestamp'] >= cutoff
                ]

        if not recent:
            return pd.DataFrame(columns=[
                'timestamp', 'symbol', 'side', 'quantity', 'price', 'usd_value'
            ])

        return pd.DataFrame(recent)

    def get_liquidation_ratio(self, minutes: int = 15) -> float:
        """
        Calculate the ratio of long to short liquidations.

        SELL side = long positions liquidated (forced to sell)
        BUY side = short positions liquidated (forced to buy)

        Ratio > 1.0 = More longs liquidated (bearish pressure)
        Ratio < 1.0 = More shorts liquidated (bullish pressure)
        Ratio = 1.0 = Balanced

        Args:
            minutes: Lookback period in minutes

        Returns:
            float: Long/Short liquidation ratio (1.0 if no data)
        """
        df = self.get_recent_liquidations(minutes)

        if df.empty:
            return 1.0

        # SELL side = long liquidations, BUY side = short liquidations
        long_liqs = df[df['side'] == 'SELL']['usd_value'].sum()
        short_liqs = df[df['side'] == 'BUY']['usd_value'].sum()

        if short_liqs == 0:
            return 2.0 if long_liqs > 0 else 1.0

        ratio = long_liqs / short_liqs
        return round(ratio, 2)

    def get_liquidation_summary(self, minutes: int = 15) -> Dict:
        """
        Get a summary of recent liquidations.

        Args:
            minutes: Lookback period in minutes

        Returns:
            Dict with summary statistics
        """
        df = self.get_recent_liquidations(minutes)

        if df.empty:
            return {
                'total_count': 0,
                'total_usd': 0.0,
                'long_usd': 0.0,
                'short_usd': 0.0,
                'ratio': 1.0,
                'top_symbols': []
            }

        long_usd = df[df['side'] == 'SELL']['usd_value'].sum()
        short_usd = df[df['side'] == 'BUY']['usd_value'].sum()

        # Top symbols by liquidation volume
        top_symbols = df.groupby('symbol')['usd_value'].sum().nlargest(5).to_dict()

        return {
            'total_count': len(df),
            'total_usd': float(df['usd_value'].sum()),
            'long_usd': float(long_usd),
            'short_usd': float(short_usd),
            'ratio': round(long_usd / short_usd, 2) if short_usd > 0 else 1.0,
            'top_symbols': top_symbols
        }

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.connected

    @property
    def buffer_count(self) -> int:
        """Get current number of liquidations in buffer."""
        return len(self.liquidations)


# Singleton instance for shared use
_liquidation_stream: Optional[BinanceLiquidationStream] = None


def get_liquidation_stream() -> BinanceLiquidationStream:
    """
    Get or create the singleton liquidation stream.

    Returns:
        BinanceLiquidationStream: Shared instance
    """
    global _liquidation_stream
    if _liquidation_stream is None:
        _liquidation_stream = BinanceLiquidationStream()
    return _liquidation_stream


def get_liquidation_ratio(minutes: int = 15) -> float:
    """
    Convenience function to get liquidation ratio.

    Starts the stream if not already running.

    Args:
        minutes: Lookback period in minutes

    Returns:
        float: Long/Short liquidation ratio
    """
    stream = get_liquidation_stream()
    if not stream.is_connected:
        stream.start_stream()
        # Give it a moment to collect some data
        time.sleep(2)
    return stream.get_liquidation_ratio(minutes)


# For standalone testing
if __name__ == "__main__":
    cprint("Testing Binance Liquidation Stream...", "cyan")

    stream = BinanceLiquidationStream()

    if stream.start_stream():
        cprint("Stream started, collecting liquidations for 30 seconds...", "white")
        time.sleep(30)

        summary = stream.get_liquidation_summary(minutes=5)
        cprint(f"\nLiquidation Summary (last 5 min):", "cyan")
        cprint(f"  Total count: {summary['total_count']}", "white")
        cprint(f"  Total USD: ${summary['total_usd']:,.2f}", "white")
        cprint(f"  Long liquidations: ${summary['long_usd']:,.2f}", "red")
        cprint(f"  Short liquidations: ${summary['short_usd']:,.2f}", "green")
        cprint(f"  Ratio (long/short): {summary['ratio']}", "yellow")

        if summary['top_symbols']:
            cprint(f"\nTop symbols:", "cyan")
            for symbol, usd in summary['top_symbols'].items():
                cprint(f"  {symbol}: ${usd:,.2f}", "white")

        stream.stop_stream()
    else:
        cprint("Failed to start stream, trying REST fallback...", "yellow")
        liqs = stream.fetch_recent_liquidations_rest(limit=100)
        cprint(f"Fetched {len(liqs)} liquidations via REST", "white")
