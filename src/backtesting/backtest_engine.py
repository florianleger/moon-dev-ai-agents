"""Generic backtesting engine for strategy evaluation."""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime


@dataclass
class Trade:
    symbol: str
    direction: str  # BUY or SELL
    entry_price: float
    entry_time: datetime
    size: float
    sl_price: float
    tp_price: float
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ''
    pnl: float = 0.0
    pnl_pct: float = 0.0
    max_adverse: float = 0.0
    max_favorable: float = 0.0


@dataclass
class BacktestResult:
    trades: List[Trade]
    initial_balance: float
    final_balance: float
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    total_trades: int = 0
    avg_hold_bars: float = 0.0
    benchmark_return_pct: float = 0.0  # buy-and-hold
    alpha: float = 0.0
    equity_curve: List[float] = field(default_factory=list)

    def summary(self):
        start = self.trades[0].entry_time if self.trades else 'N/A'
        end = self.trades[-1].exit_time if self.trades else 'N/A'
        return f"""
=== BACKTEST RESULTS ===
Period: {start} - {end}
Initial Balance: ${self.initial_balance:,.2f}
Final Balance: ${self.final_balance:,.2f}
Total Return: {self.total_return_pct:.2f}%
Benchmark Return: {self.benchmark_return_pct:.2f}%
Alpha: {self.alpha:.2f}%
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {self.max_drawdown_pct:.2f}%
Win Rate: {self.win_rate:.1f}%
Profit Factor: {self.profit_factor:.2f}
Total Trades: {self.total_trades}
Avg PnL/Trade: ${self.avg_trade_pnl:.2f}
Avg Hold (bars): {self.avg_hold_bars:.1f}
"""


class BacktestEngine:
    """Simulates trade execution with SL/TP, slippage, and fees."""

    def __init__(self, initial_balance=10000.0, fee_pct=0.035, slippage_pct=0.1):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fee_pct = fee_pct / 100
        self.slippage_pct = slippage_pct / 100
        self.positions = {}  # key -> position dict
        self.closed_trades = []
        self.equity_curve = [initial_balance]
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0

    def open_position(self, symbol, direction, price, size_usd, sl_price, tp_price, timestamp):
        """Open a new position with SL/TP."""
        # Apply slippage
        slip = price * self.slippage_pct
        entry_price = price + slip if direction == 'BUY' else price - slip
        # Apply fees
        fee = size_usd * self.fee_pct
        self.balance -= fee

        qty = size_usd / entry_price
        key = f"{symbol}_{timestamp}"
        self.positions[key] = {
            'symbol': symbol, 'direction': direction,
            'entry_price': entry_price, 'entry_time': timestamp,
            'size_usd': size_usd, 'qty': qty,
            'sl_price': sl_price, 'tp_price': tp_price,
            'max_adverse': 0.0, 'max_favorable': 0.0
        }
        return key

    def check_exits(self, candle, timestamp):
        """Check SL/TP for all open positions against candle OHLC."""
        to_close = []
        for key, pos in self.positions.items():
            high, low, close = candle['high'], candle['low'], candle['close']
            direction = pos['direction']

            # Track MAE/MFE
            if direction == 'BUY':
                adverse = (pos['entry_price'] - low) / pos['entry_price']
                favorable = (high - pos['entry_price']) / pos['entry_price']
            else:
                adverse = (high - pos['entry_price']) / pos['entry_price']
                favorable = (pos['entry_price'] - low) / pos['entry_price']

            pos['max_adverse'] = max(pos['max_adverse'], max(0, adverse))
            pos['max_favorable'] = max(pos['max_favorable'], max(0, favorable))

            # Check SL
            if direction == 'BUY' and low <= pos['sl_price']:
                to_close.append((key, pos['sl_price'], 'SL'))
            elif direction == 'SELL' and high >= pos['sl_price']:
                to_close.append((key, pos['sl_price'], 'SL'))
            # Check TP
            elif direction == 'BUY' and high >= pos['tp_price']:
                to_close.append((key, pos['tp_price'], 'TP'))
            elif direction == 'SELL' and low <= pos['tp_price']:
                to_close.append((key, pos['tp_price'], 'TP'))

        for key, exit_price, reason in to_close:
            self._close_position(key, exit_price, reason, timestamp)

    def close_all_positions(self, close_price, timestamp, reason='END'):
        """Force-close all open positions at given price."""
        keys = list(self.positions.keys())
        for key in keys:
            self._close_position(key, close_price, reason, timestamp)

    def _close_position(self, key, exit_price, reason, timestamp):
        pos = self.positions.pop(key)
        # Apply slippage + fees
        slip = exit_price * self.slippage_pct
        actual_exit = exit_price - slip if pos['direction'] == 'BUY' else exit_price + slip
        fee = pos['size_usd'] * self.fee_pct

        if pos['direction'] == 'BUY':
            pnl = (actual_exit - pos['entry_price']) * pos['qty'] - fee
        else:
            pnl = (pos['entry_price'] - actual_exit) * pos['qty'] - fee

        self.balance += pnl
        pnl_pct = pnl / pos['size_usd'] * 100

        self.closed_trades.append(Trade(
            symbol=pos['symbol'], direction=pos['direction'],
            entry_price=pos['entry_price'], entry_time=pos['entry_time'],
            size=pos['size_usd'], sl_price=pos['sl_price'], tp_price=pos['tp_price'],
            exit_price=actual_exit, exit_time=timestamp, exit_reason=reason,
            pnl=pnl, pnl_pct=pnl_pct,
            max_adverse=pos['max_adverse'], max_favorable=pos['max_favorable']
        ))

        # Update equity tracking
        self.equity_curve.append(self.balance)
        self.peak_balance = max(self.peak_balance, self.balance)
        dd = (self.peak_balance - self.balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, dd)

    def get_results(self, benchmark_start_price=None, benchmark_end_price=None):
        """Calculate backtest metrics."""
        trades = self.closed_trades
        if not trades:
            return BacktestResult(trades=[], initial_balance=self.initial_balance,
                                  final_balance=self.balance)

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1

        # Sharpe (simplified, annualized from trade returns)
        returns = [t.pnl_pct / 100 for t in trades]
        avg_ret = np.mean(returns)
        std_ret = np.std(returns) if len(returns) > 1 else 1
        sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

        benchmark_ret = 0
        if benchmark_start_price and benchmark_end_price:
            benchmark_ret = (benchmark_end_price - benchmark_start_price) / benchmark_start_price * 100

        total_ret = (self.balance - self.initial_balance) / self.initial_balance * 100

        # Average hold duration in bars
        hold_bars = []
        for t in trades:
            if t.exit_time and t.entry_time:
                delta = t.exit_time - t.entry_time
                hold_bars.append(delta.total_seconds() / 3600)  # hours
        avg_hold = np.mean(hold_bars) if hold_bars else 0

        return BacktestResult(
            trades=trades,
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            total_return_pct=total_ret,
            sharpe_ratio=sharpe,
            max_drawdown_pct=self.max_drawdown * 100,
            win_rate=len(wins) / len(trades) * 100 if trades else 0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else 0,
            avg_trade_pnl=np.mean([t.pnl for t in trades]),
            total_trades=len(trades),
            avg_hold_bars=avg_hold,
            benchmark_return_pct=benchmark_ret,
            alpha=total_ret - benchmark_ret,
            equity_curve=self.equity_curve
        )
