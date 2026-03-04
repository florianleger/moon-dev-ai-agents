"""
Native order management for HyperLiquid live trading.

Replaces the 30s polling SL/TP with native exchange orders:
- Stop-market orders for stop-loss
- Take-profit market orders
- Trailing stop implementation via order modification (cancel + replace)

HOW TO FINALIZE INTEGRATION
============================
This module uses the `hyperliquid-python-sdk` (same as nice_funcs_hyperliquid.py).
The key classes are:
  - `hyperliquid.exchange.Exchange` for placing/cancelling orders
  - `hyperliquid.info.Info` for querying positions and open orders

The SDK's `exchange.order()` method supports trigger orders natively:
    exchange.order(
        coin,           # e.g. 'BTC'
        is_buy,         # True/False
        sz,             # position size (float)
        limit_px,       # limit price (for trigger market, use a very high/low price)
        order_type,     # {"trigger": {"triggerPx": str, "isMarket": True, "tpsl": "sl"|"tp"}}
        reduce_only     # True for SL/TP
    )

For cancellation, use:
    exchange.cancel(coin, oid)

The `_send_order` and `_send_cancel` methods below are fully implemented using
the SDK. No raw EIP-712 signing is needed -- the SDK handles that internally.

See: src/nice_funcs_hyperliquid.py for account initialization patterns.
See: https://github.com/hyperliquid-org/hyperliquid-python-sdk for SDK docs.

Usage:
    from src.execution.order_manager import LiveOrderManager

    manager = LiveOrderManager()
    result = manager.place_bracket_order(
        symbol='BTC', direction='BUY', size=0.001,
        entry_price=95000, sl_price=93000, tp_price=99000
    )
    # Orders are placed directly on HyperLiquid exchange
    # No more 30s polling delay

    # Update trailing stop
    manager.update_trailing_stop('BTC', 'BUY', new_sl_price=94500)

    # Cancel all bracket orders for a position
    manager.cancel_bracket('BTC', 'BUY')
"""

import os
import json
import time
from datetime import datetime

import eth_account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from termcolor import cprint
from dotenv import load_dotenv

load_dotenv()

# API URL: testnet or mainnet based on env var
HL_API_URL = (
    constants.TESTNET_API_URL
    if os.getenv('USE_TESTNET', 'false').lower() == 'true'
    else constants.MAINNET_API_URL
)


class LiveOrderManager:
    """Manages native exchange orders for SL/TP/trailing stops on HyperLiquid.

    Holds references to active bracket orders (SL + TP) per position,
    and provides methods to place, update (trailing), and cancel them.
    """

    def __init__(self):
        """Initialize with HyperLiquid credentials from environment."""
        self._private_key = os.getenv('HYPER_LIQUID_ETH_PRIVATE_KEY')
        if not self._private_key:
            cprint("[LiveOrderManager] WARNING: HYPER_LIQUID_ETH_PRIVATE_KEY not set", "yellow")
            self._account = None
            self._exchange = None
            self._info = None
        else:
            self._account = eth_account.Account.from_key(self._private_key)
            self._exchange = Exchange(self._account, HL_API_URL)
            self._info = Info(HL_API_URL, skip_ws=True)

        # Active bracket orders: position_key -> {sl_oid, tp_oid, symbol, direction, ...}
        self._active_orders = {}

        # Cache asset metadata (fetched once)
        self._asset_meta = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def place_bracket_order(self, symbol, direction, size,
                            entry_price, sl_price, tp_price):
        """Place SL + TP orders after position entry.

        Args:
            symbol: Trading pair (e.g., 'BTC')
            direction: 'BUY' or 'SELL'
            size: Position size in asset units (e.g., 0.001 BTC)
            entry_price: Entry price (for logging)
            sl_price: Stop-loss trigger price
            tp_price: Take-profit trigger price

        Returns:
            dict with sl_oid, tp_oid, and position metadata, or None on failure
        """
        if not self._exchange:
            cprint("[LiveOrderManager] No exchange connection -- cannot place orders", "red")
            return None

        # Round size to exchange precision
        sz_decimals = self._get_sz_decimals(symbol)
        size = round(size, sz_decimals)

        # SL: opposite side, stop-market, reduce-only
        sl_side_is_buy = direction == 'SELL'  # close a SELL = BUY; close a BUY = SELL
        sl_oid = self._place_trigger_order(
            symbol=symbol,
            is_buy=sl_side_is_buy,
            size=size,
            trigger_price=sl_price,
            tpsl='sl',
        )

        # TP: opposite side, take-profit market, reduce-only
        tp_oid = self._place_trigger_order(
            symbol=symbol,
            is_buy=sl_side_is_buy,
            size=size,
            trigger_price=tp_price,
            tpsl='tp',
        )

        position_key = f"{symbol}_{direction}_{int(time.time())}"
        record = {
            'sl_oid': sl_oid,
            'tp_oid': tp_oid,
            'symbol': symbol,
            'direction': direction,
            'size': size,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'created_at': datetime.utcnow().isoformat(),
        }
        self._active_orders[position_key] = record

        sl_status = "OK" if sl_oid else "FAILED"
        tp_status = "OK" if tp_oid else "FAILED"
        cprint(
            f"[LiveOrderManager] Bracket for {symbol} {direction}: "
            f"SL@{sl_price} ({sl_status}) | TP@{tp_price} ({tp_status})",
            "cyan",
        )
        return record

    def update_trailing_stop(self, symbol, direction, new_sl_price):
        """Update stop-loss for trailing stop. Only moves SL in profitable direction.

        Args:
            symbol: Trading pair
            direction: 'BUY' or 'SELL'
            new_sl_price: New stop-loss trigger price

        Returns:
            True if SL was updated, False otherwise
        """
        if not self._exchange:
            return False

        for key, orders in self._active_orders.items():
            if orders['symbol'] != symbol or orders['direction'] != direction:
                continue

            old_sl = orders['sl_price']

            # Only move SL in profitable direction
            if direction == 'BUY' and new_sl_price <= old_sl:
                return False
            if direction == 'SELL' and new_sl_price >= old_sl:
                return False

            # Cancel old SL
            if orders.get('sl_oid'):
                self._cancel_order(symbol, orders['sl_oid'])

            # Place new SL
            sl_side_is_buy = direction == 'SELL'
            new_oid = self._place_trigger_order(
                symbol=symbol,
                is_buy=sl_side_is_buy,
                size=orders['size'],
                trigger_price=new_sl_price,
                tpsl='sl',
            )

            orders['sl_oid'] = new_oid
            orders['sl_price'] = new_sl_price

            cprint(
                f"[LiveOrderManager] Trailing SL updated {symbol} {direction}: "
                f"{old_sl} -> {new_sl_price}",
                "cyan",
            )
            return True

        return False

    def cancel_bracket(self, symbol, direction):
        """Cancel both SL and TP orders for a position.

        Args:
            symbol: Trading pair
            direction: 'BUY' or 'SELL'
        """
        if not self._exchange:
            return

        keys_to_remove = []
        for key, orders in self._active_orders.items():
            if orders['symbol'] == symbol and orders['direction'] == direction:
                if orders.get('sl_oid'):
                    self._cancel_order(symbol, orders['sl_oid'])
                if orders.get('tp_oid'):
                    self._cancel_order(symbol, orders['tp_oid'])
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._active_orders[key]

        if keys_to_remove:
            cprint(f"[LiveOrderManager] Bracket cancelled for {symbol} {direction}", "yellow")

    def get_active_orders(self):
        """Get all active bracket orders.

        Returns:
            dict: Copy of active orders keyed by position_key
        """
        return dict(self._active_orders)

    def sync_with_exchange(self, symbol=None):
        """Sync local state with actual exchange open orders.

        Removes local records whose SL/TP orders are no longer open on the
        exchange (i.e., they were filled or cancelled externally).

        Args:
            symbol: Optional symbol filter. If None, syncs all.
        """
        if not self._info or not self._account:
            return

        open_orders = self._info.open_orders(self._account.address)
        open_oids = {order['oid'] for order in open_orders}

        keys_to_remove = []
        for key, orders in self._active_orders.items():
            if symbol and orders['symbol'] != symbol:
                continue

            sl_alive = orders.get('sl_oid') in open_oids
            tp_alive = orders.get('tp_oid') in open_oids

            if not sl_alive and not tp_alive:
                # Both filled/cancelled -- position is closed
                keys_to_remove.append(key)
                cprint(
                    f"[LiveOrderManager] Bracket {key} fully resolved (SL/TP filled or cancelled)",
                    "green",
                )
            elif not sl_alive and tp_alive:
                # SL filled, cancel TP
                self._cancel_order(orders['symbol'], orders['tp_oid'])
                keys_to_remove.append(key)
                cprint(f"[LiveOrderManager] SL filled for {key}, cancelled TP", "yellow")
            elif sl_alive and not tp_alive:
                # TP filled, cancel SL
                self._cancel_order(orders['symbol'], orders['sl_oid'])
                keys_to_remove.append(key)
                cprint(f"[LiveOrderManager] TP filled for {key}, cancelled SL", "yellow")

        for key in keys_to_remove:
            del self._active_orders[key]

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _place_trigger_order(self, symbol, is_buy, size, trigger_price, tpsl='sl'):
        """Place a trigger order on HyperLiquid via the SDK.

        Args:
            symbol: Trading pair (e.g. 'BTC')
            is_buy: True for buy, False for sell
            size: Position size in asset units
            trigger_price: Price at which the order triggers
            tpsl: 'sl' for stop-loss, 'tp' for take-profit

        Returns:
            Order ID (oid) string on success, None on failure
        """
        # For trigger market orders, set limit_px to a very permissive price
        # to ensure fill after trigger. The trigger price is the actual control.
        if is_buy:
            # Buying to close: set limit well above trigger to guarantee fill
            limit_px = round(trigger_price * 1.05, self._get_px_decimals(symbol))
        else:
            # Selling to close: set limit well below trigger to guarantee fill
            limit_px = round(trigger_price * 0.95, self._get_px_decimals(symbol))

        order_type = {
            "trigger": {
                "triggerPx": str(trigger_price),
                "isMarket": True,
                "tpsl": tpsl,
            }
        }

        try:
            result = self._exchange.order(
                symbol,
                is_buy,
                size,
                limit_px,
                order_type,
                reduce_only=True,
            )

            # Extract order ID from response
            oid = self._extract_oid(result)
            if oid:
                side_str = "BUY" if is_buy else "SELL"
                cprint(
                    f"[LiveOrderManager] {tpsl.upper()} order placed: "
                    f"{side_str} {size} {symbol} trigger@{trigger_price} (oid={oid})",
                    "green",
                )
            else:
                cprint(
                    f"[LiveOrderManager] Order placed but could not extract OID: {result}",
                    "yellow",
                )
            return oid

        except Exception as e:
            cprint(f"[LiveOrderManager] Failed to place {tpsl} order for {symbol}: {e}", "red")
            return None

    def _cancel_order(self, symbol, oid):
        """Cancel an order on HyperLiquid.

        Args:
            symbol: Trading pair
            oid: Order ID to cancel
        """
        if not oid or not self._exchange:
            return

        try:
            self._exchange.cancel(symbol, oid)
            cprint(f"[LiveOrderManager] Cancelled order {oid} for {symbol}", "yellow")
        except Exception as e:
            cprint(f"[LiveOrderManager] Failed to cancel order {oid}: {e}", "red")

    def _extract_oid(self, order_result):
        """Extract order ID from SDK response.

        The SDK returns a dict like:
            {'response': {'type': 'order', 'data': {'statuses': [{'resting': {'oid': 12345}}]}}}

        Args:
            order_result: Raw response from exchange.order()

        Returns:
            Order ID (int or str) or None
        """
        if not isinstance(order_result, dict):
            return None

        try:
            statuses = order_result.get('response', {}).get('data', {}).get('statuses', [])
            if not statuses:
                return None

            status = statuses[0]

            # Resting order (trigger orders typically rest until triggered)
            if isinstance(status, dict):
                if 'resting' in status:
                    return status['resting'].get('oid')
                if 'filled' in status:
                    return status['filled'].get('oid')
                # Some responses have the oid at top level
                if 'oid' in status:
                    return status['oid']

            return None
        except (KeyError, IndexError, TypeError):
            return None

    def _get_asset_meta(self):
        """Fetch and cache asset metadata from HyperLiquid.

        Returns:
            list: Universe metadata (list of asset dicts with 'name', 'szDecimals', etc.)
        """
        if self._asset_meta is not None:
            return self._asset_meta

        if not self._info:
            return []

        try:
            meta = self._info.meta()
            self._asset_meta = meta.get('universe', [])
            return self._asset_meta
        except Exception as e:
            cprint(f"[LiveOrderManager] Failed to fetch asset metadata: {e}", "red")
            return []

    def _get_sz_decimals(self, symbol):
        """Get size decimal precision for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            int: Number of decimal places for size
        """
        universe = self._get_asset_meta()
        for asset in universe:
            if asset.get('name') == symbol:
                return asset.get('szDecimals', 3)
        return 3  # default

    def _get_px_decimals(self, symbol):
        """Get price decimal precision for a symbol.

        Uses the current ask price to determine the number of decimals.

        Args:
            symbol: Trading pair

        Returns:
            int: Number of decimal places for price
        """
        if not self._info:
            return 2

        try:
            # Use L2 book to get current price format
            l2 = self._info.l2_snapshot(symbol)
            if l2 and l2.get('levels') and l2['levels'][1]:
                ask_str = l2['levels'][1][0]['px']
                if '.' in ask_str:
                    return len(ask_str.split('.')[1])
            return 1
        except Exception:
            return 1
