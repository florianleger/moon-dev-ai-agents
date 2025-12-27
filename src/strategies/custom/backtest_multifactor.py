"""
Backtest for Multifactor Strategy
Uses backtesting.py with ta library for indicators
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Enable fractional trading for crypto
import backtesting._plotting
backtesting._plotting._MAX_CANDLES = 10000
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from termcolor import cprint
import time


def fetch_historical_data(symbol: str, days: int = 30, interval: str = '1h') -> pd.DataFrame:
    """Fetch historical data from HyperLiquid."""
    try:
        from hyperliquid.info import Info

        info = Info(skip_ws=True)

        end_time = int(time.time() * 1000)
        interval_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }.get(interval, 60 * 60 * 1000)

        candles_needed = (days * 24 * 60 * 60 * 1000) // interval_ms
        start_time = end_time - (candles_needed * interval_ms)

        cprint(f"Fetching {candles_needed} candles for {symbol}...", "yellow")
        candle_data = info.candles_snapshot(symbol, interval, start_time, end_time)

        if not candle_data:
            return None

        df = pd.DataFrame(candle_data)
        df = df.rename(columns={
            't': 'timestamp',
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        })

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    except Exception as e:
        cprint(f"Error fetching data: {e}", "red")
        return None


class MultifactorBacktest(Strategy):
    """
    Multifactor Strategy for backtesting.
    Combines EMA trend, MACD momentum, RSI, and volume.
    """
    # Parameters
    ema_fast = 20
    ema_mid = 50
    ema_slow = 200
    buy_threshold = 0.5  # Conservative threshold
    sell_threshold = -0.5
    risk_per_trade = 0.04

    def init(self):
        close = pd.Series(self.data.Close)

        # EMAs
        self.ema20 = self.I(lambda x: EMAIndicator(pd.Series(x), window=20).ema_indicator(), self.data.Close)
        self.ema50 = self.I(lambda x: EMAIndicator(pd.Series(x), window=50).ema_indicator(), self.data.Close)
        self.ema200 = self.I(lambda x: EMAIndicator(pd.Series(x), window=200).ema_indicator(), self.data.Close)

        # MACD
        self.macd = self.I(lambda x: MACD(pd.Series(x)).macd(), self.data.Close)
        self.macd_signal = self.I(lambda x: MACD(pd.Series(x)).macd_signal(), self.data.Close)
        self.macd_hist = self.I(lambda x: MACD(pd.Series(x)).macd_diff(), self.data.Close)

        # RSI
        self.rsi = self.I(lambda x: RSIIndicator(pd.Series(x), window=14).rsi(), self.data.Close)

        # Volume MA
        self.vol_ma = self.I(lambda x: pd.Series(x).rolling(20).mean(), self.data.Volume)

    def calculate_score(self):
        """Calculate combined multifactor score."""
        price = self.data.Close[-1]

        # Trend score (25%)
        if price > self.ema20[-1] > self.ema50[-1] > self.ema200[-1]:
            trend_score = 1.0
        elif price > self.ema20[-1] > self.ema50[-1]:
            trend_score = 0.5
        elif price < self.ema20[-1] < self.ema50[-1] < self.ema200[-1]:
            trend_score = -1.0
        elif price < self.ema20[-1] < self.ema50[-1]:
            trend_score = -0.5
        else:
            trend_score = 0.0

        # Momentum score (20%)
        if self.macd[-1] > self.macd_signal[-1] and self.macd_hist[-1] > self.macd_hist[-2]:
            momentum_score = 1.0
        elif self.macd[-1] > self.macd_signal[-1]:
            momentum_score = 0.5
        elif self.macd[-1] < self.macd_signal[-1] and self.macd_hist[-1] < self.macd_hist[-2]:
            momentum_score = -1.0
        elif self.macd[-1] < self.macd_signal[-1]:
            momentum_score = -0.5
        else:
            momentum_score = 0.0

        # RSI score (20%)
        rsi = self.rsi[-1]
        if rsi < 30:
            rsi_score = 1.0
        elif rsi < 45:
            rsi_score = 0.5
        elif rsi > 70:
            rsi_score = -1.0
        elif rsi > 55:
            rsi_score = -0.5
        else:
            rsi_score = 0.0

        # Volume score (15%)
        vol_ratio = self.data.Volume[-1] / self.vol_ma[-1] if self.vol_ma[-1] > 0 else 1.0
        price_up = self.data.Close[-1] > self.data.Close[-2]

        if vol_ratio > 2.0:
            volume_score = 1.0 if price_up else -1.0
        elif vol_ratio > 1.5:
            volume_score = 0.5 if price_up else -0.5
        else:
            volume_score = 0.0

        # Sentiment score (20%) - simulated for backtest
        # Using RSI divergence as proxy for sentiment
        sentiment_score = 0.0  # Neutral in backtest

        # Combined score
        combined = (
            trend_score * 0.25 +
            momentum_score * 0.20 +
            rsi_score * 0.20 +
            volume_score * 0.15 +
            sentiment_score * 0.20
        )

        return combined

    def next(self):
        # Skip if not enough data
        if len(self.data) < 200:
            return

        score = self.calculate_score()

        # Position sizing
        size = self.risk_per_trade

        if score >= self.buy_threshold:
            if not self.position.is_long:
                self.position.close()
                # Use 95% of available equity
                self.buy(size=0.95)

        elif score <= self.sell_threshold:
            if self.position.is_long:
                self.position.close()


def run_backtest(symbol: str = 'BTC', days: int = 30, cash: float = 100000):
    """Run backtest for the multifactor strategy."""
    cprint(f"\n{'='*60}", "cyan")
    cprint(f"  BACKTEST: Multifactor Strategy - {symbol}", "cyan")
    cprint(f"  Period: {days} days | Initial Capital: ${cash:,.0f}", "cyan")
    cprint(f"{'='*60}\n", "cyan")

    # Fetch data
    df = fetch_historical_data(symbol, days=days, interval='1h')

    if df is None or len(df) < 200:
        cprint(f"Not enough data for {symbol}", "red")
        return None

    cprint(f"Data range: {df.index[0]} to {df.index[-1]}", "white")
    cprint(f"Total candles: {len(df)}", "white")
    cprint(f"Price range: ${df['Close'].min():,.2f} - ${df['Close'].max():,.2f}\n", "white")

    # Run backtest
    bt = Backtest(
        df,
        MultifactorBacktest,
        cash=cash,
        commission=0.0006,  # 0.06% HyperLiquid fee
        exclusive_orders=True,
        trade_on_close=True,
        hedging=False
    )

    stats = bt.run()

    # Print results
    cprint("\n" + "="*60, "cyan")
    cprint("  RESULTS", "cyan")
    cprint("="*60, "cyan")

    metrics = [
        ("Return [%]", stats['Return [%]']),
        ("Buy & Hold Return [%]", stats['Buy & Hold Return [%]']),
        ("Max Drawdown [%]", stats['Max. Drawdown [%]']),
        ("# Trades", stats['# Trades']),
        ("Win Rate [%]", stats['Win Rate [%]']),
        ("Profit Factor", stats.get('Profit Factor', 'N/A')),
        ("Sharpe Ratio", stats['Sharpe Ratio']),
        ("Sortino Ratio", stats.get('Sortino Ratio', 'N/A')),
    ]

    for name, value in metrics:
        if isinstance(value, float):
            color = "green" if value > 0 else "red"
            cprint(f"  {name:25} {value:>10.2f}", color)
        else:
            cprint(f"  {name:25} {value:>10}", "white")

    cprint("="*60 + "\n", "cyan")

    return stats, bt


if __name__ == "__main__":
    # Run backtest for multiple assets
    results = {}

    for symbol in ['BTC', 'ETH', 'SOL']:
        try:
            stats, bt = run_backtest(symbol, days=60, cash=100000)
            if stats is not None:
                results[symbol] = {
                    'return': stats['Return [%]'],
                    'max_dd': stats['Max. Drawdown [%]'],
                    'trades': stats['# Trades'],
                    'win_rate': stats['Win Rate [%]']
                }
        except Exception as e:
            cprint(f"Error backtesting {symbol}: {e}", "red")

    # Summary
    if results:
        cprint("\n" + "="*60, "cyan")
        cprint("  SUMMARY - All Assets", "cyan")
        cprint("="*60, "cyan")
        cprint(f"  {'Asset':<8} {'Return':>10} {'Max DD':>10} {'Trades':>8} {'Win Rate':>10}", "white")
        cprint("-"*60, "white")

        for symbol, data in results.items():
            color = "green" if data['return'] > 0 else "red"
            cprint(f"  {symbol:<8} {data['return']:>9.2f}% {data['max_dd']:>9.2f}% {data['trades']:>8} {data['win_rate']:>9.1f}%", color)

        cprint("="*60, "cyan")
