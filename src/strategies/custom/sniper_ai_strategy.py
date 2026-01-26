"""
Sniper AI Strategy

A precision trading strategy requiring ALL 7 checklist conditions to align.
Target: 1-2 trades/day with 80%+ win rate.
Philosophy: "Fewer trades, but quasi-certain trades"

7-Point Sniper Checklist (ALL must be true):
1. Extreme Move - Price moved >2.5 sigma from mean
2. Funding Divergence - Contrarian funding signal
3. Liquidation Cascade Complete - Fuel exhausted
4. Multi-TF Agreement - 5m, 15m, 1h, 4h all aligned
5. Volume Climax - Peak volume >3x average, now declining
6. Time Window Optimal - London/NY open sessions
7. AI Validation - Claude confirms with 85%+ confidence
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time as time_module
from termcolor import cprint
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator

from ..base_strategy import BaseStrategy

# Default volatility thresholds (used as fallback if dynamic calibration fails)
# These are conservative estimates; the strategy will calculate actual values from 30d data
# adx_trending_threshold: percentile-based default, higher = allows more trending markets
DEFAULT_VOLATILITY_THRESHOLDS = {
    'BTC': {'move_threshold': 1.7, 'sigma': 0.68, 'adx_trending_threshold': 32, 'rsi_oversold': 28, 'rsi_overbought': 72},
    'ETH': {'move_threshold': 2.2, 'sigma': 0.89, 'adx_trending_threshold': 34, 'rsi_oversold': 26, 'rsi_overbought': 74},
    'SOL': {'move_threshold': 2.7, 'sigma': 1.07, 'adx_trending_threshold': 30, 'rsi_oversold': 24, 'rsi_overbought': 76},
    'XRP': {'move_threshold': 3.0, 'sigma': 1.20, 'adx_trending_threshold': 28, 'rsi_oversold': 25, 'rsi_overbought': 75},
    'DOGE': {'move_threshold': 3.4, 'sigma': 1.33, 'adx_trending_threshold': 28, 'rsi_oversold': 22, 'rsi_overbought': 78},
    'ADA': {'move_threshold': 3.2, 'sigma': 1.28, 'adx_trending_threshold': 30, 'rsi_oversold': 25, 'rsi_overbought': 75},
    'AVAX': {'move_threshold': 3.6, 'sigma': 1.42, 'adx_trending_threshold': 32, 'rsi_oversold': 24, 'rsi_overbought': 76},
    'LINK': {'move_threshold': 2.8, 'sigma': 1.10, 'adx_trending_threshold': 32, 'rsi_oversold': 26, 'rsi_overbought': 74},
    'DOT': {'move_threshold': 3.0, 'sigma': 1.20, 'adx_trending_threshold': 30, 'rsi_oversold': 25, 'rsi_overbought': 75},
    'MATIC': {'move_threshold': 3.5, 'sigma': 1.40, 'adx_trending_threshold': 28, 'rsi_oversold': 24, 'rsi_overbought': 76},
    # L2 tokens (typically more volatile, trend strongly)
    'ARB': {'move_threshold': 3.8, 'sigma': 1.50, 'adx_trending_threshold': 35, 'rsi_oversold': 22, 'rsi_overbought': 78},
    'OP': {'move_threshold': 3.5, 'sigma': 1.40, 'adx_trending_threshold': 34, 'rsi_oversold': 23, 'rsi_overbought': 77},
    # AI tokens (high volatility, strong trends)
    'RENDER': {'move_threshold': 4.0, 'sigma': 1.60, 'adx_trending_threshold': 36, 'rsi_oversold': 20, 'rsi_overbought': 80},
    # DeFi blue chips
    'AAVE': {'move_threshold': 3.5, 'sigma': 1.40, 'adx_trending_threshold': 33, 'rsi_oversold': 25, 'rsi_overbought': 75},
    'CRV': {'move_threshold': 4.2, 'sigma': 1.65, 'adx_trending_threshold': 35, 'rsi_oversold': 22, 'rsi_overbought': 78},
    # Infrastructure
    'FIL': {'move_threshold': 3.8, 'sigma': 1.50, 'adx_trending_threshold': 34, 'rsi_oversold': 23, 'rsi_overbought': 77},
}

# Import config with defaults
try:
    from src.config import (
        SNIPER_ASSETS,
        SNIPER_LEVERAGE,
        SNIPER_STOP_LOSS_PCT,
        SNIPER_TAKE_PROFIT_PCT,
        SNIPER_MAX_DAILY_TRADES,
        SNIPER_MAX_DAILY_LOSS_USD,
        SNIPER_MAX_DAILY_GAIN_USD,
        SNIPER_SIGMA_THRESHOLD,
        SNIPER_FUNDING_EXTREME_THRESHOLD,
        SNIPER_LIQ_RATIO_THRESHOLD,
        SNIPER_VOLUME_SPIKE_THRESHOLD,
        SNIPER_OPTIMAL_HOURS,
        SNIPER_ALLOW_NORMAL_HOURS,
        SNIPER_AI_MIN_CONFIDENCE,
        SNIPER_AI_MODEL,
        SNIPER_AI_TEMPERATURE,
        SNIPER_AI_MAX_TOKENS,
        SNIPER_CAPITULATION_MIN_DROP_PCT,
        SNIPER_EUPHORIA_MIN_RISE_PCT,
        SNIPER_RSI_OVERSOLD,
        SNIPER_RSI_OVERBOUGHT,
        PAPER_TRADING,
        PAPER_TRADING_BALANCE,
        # Advanced improvements
        SNIPER_USE_TRAILING_STOP,
        SNIPER_TRAILING_ATR_MULTIPLIER,
        SNIPER_TRAILING_ACTIVATION_PCT,
        SNIPER_USE_REGIME_FILTER,
        SNIPER_ADX_TRENDING_THRESHOLD,
        SNIPER_ADX_PERIOD,
        SNIPER_USE_CORRELATION_SIZING,
        SNIPER_CORRELATION_THRESHOLD,
        SNIPER_CORRELATION_LOOKBACK_DAYS,
        SNIPER_ENABLE_FUNDING_ARBITRAGE,
        SNIPER_FUNDING_ARBITRAGE_THRESHOLD,
        SNIPER_FUNDING_ARBITRAGE_STABILITY_PCT,
        SNIPER_USE_WEIGHTED_SCORING,
        SNIPER_MIN_WEIGHTED_SCORE,
        SNIPER_WEIGHTS,
        SNIPER_USE_CONFIDENCE_SIZING,
        SNIPER_CONFIDENCE_SIZE_MAP,
        # Dynamic threshold parameters
        SNIPER_USE_DYNAMIC_THRESHOLDS,
        SNIPER_VOL_RATIO_HIGH,
        SNIPER_VOL_RATIO_LOW,
        SNIPER_MAX_THRESHOLD_ADJUSTMENT,
        SNIPER_ADX_RANGING_THRESHOLD,
        SNIPER_RECALIBRATION_HOURS,
    )
except ImportError:
    # Default values
    SNIPER_ASSETS = ['BTC', 'ETH', 'SOL']
    SNIPER_LEVERAGE = 3
    SNIPER_STOP_LOSS_PCT = 1.5
    SNIPER_TAKE_PROFIT_PCT = 3.0
    SNIPER_MAX_DAILY_TRADES = 2
    SNIPER_MAX_DAILY_LOSS_USD = 30
    SNIPER_MAX_DAILY_GAIN_USD = 60
    SNIPER_SIGMA_THRESHOLD = 2.5
    SNIPER_FUNDING_EXTREME_THRESHOLD = 2.0
    SNIPER_LIQ_RATIO_THRESHOLD = 1.5
    SNIPER_VOLUME_SPIKE_THRESHOLD = 3.0
    SNIPER_OPTIMAL_HOURS = [7, 8, 9, 13, 14, 15, 16]
    SNIPER_ALLOW_NORMAL_HOURS = False
    SNIPER_AI_MIN_CONFIDENCE = 85
    SNIPER_AI_MODEL = 'claude-3-5-sonnet-latest'
    SNIPER_AI_TEMPERATURE = 0.3
    SNIPER_AI_MAX_TOKENS = 1024
    SNIPER_CAPITULATION_MIN_DROP_PCT = 5.0
    SNIPER_EUPHORIA_MIN_RISE_PCT = 5.0
    SNIPER_RSI_OVERSOLD = 25
    SNIPER_RSI_OVERBOUGHT = 75
    PAPER_TRADING = True
    PAPER_TRADING_BALANCE = 500
    # Advanced improvements defaults
    SNIPER_USE_TRAILING_STOP = True
    SNIPER_TRAILING_ATR_MULTIPLIER = 2.0
    SNIPER_TRAILING_ACTIVATION_PCT = 1.0
    SNIPER_USE_REGIME_FILTER = True
    SNIPER_ADX_TRENDING_THRESHOLD = 25
    SNIPER_ADX_PERIOD = 14
    SNIPER_USE_CORRELATION_SIZING = True
    SNIPER_CORRELATION_THRESHOLD = 0.7
    SNIPER_CORRELATION_LOOKBACK_DAYS = 30
    SNIPER_ENABLE_FUNDING_ARBITRAGE = True
    SNIPER_FUNDING_ARBITRAGE_THRESHOLD = 0.1
    SNIPER_FUNDING_ARBITRAGE_STABILITY_PCT = 1.0
    SNIPER_USE_WEIGHTED_SCORING = True
    SNIPER_MIN_WEIGHTED_SCORE = 8.5
    SNIPER_WEIGHTS = {
        'extreme_move': 2.0, 'funding_divergence': 1.5, 'liquidation_cascade': 1.5,
        'multi_tf': 1.0, 'volume_climax': 1.0, 'time_window': 0.5, 'ai_validation': 2.5
    }
    SNIPER_USE_CONFIDENCE_SIZING = True
    SNIPER_CONFIDENCE_SIZE_MAP = {85: 0.5, 90: 0.75, 95: 1.0}
    # Dynamic threshold defaults
    SNIPER_USE_DYNAMIC_THRESHOLDS = True
    SNIPER_VOL_RATIO_HIGH = 1.3
    SNIPER_VOL_RATIO_LOW = 0.7
    SNIPER_MAX_THRESHOLD_ADJUSTMENT = 0.25
    SNIPER_ADX_RANGING_THRESHOLD = 20
    SNIPER_RECALIBRATION_HOURS = 4


# AI System Prompt for validation
SNIPER_AI_SYSTEM_PROMPT = """You are Moon Dev's Sniper AI - a precision trading validation system.

Your role is to perform final validation on high-conviction trade setups. You analyze
market data, verify the 7-point checklist, and determine if a trade should execute.

CRITICAL: You must respond with ONLY valid JSON in the exact format specified.
Do not include any text before or after the JSON.

You are CONSERVATIVE by design:
- If ANY doubt exists, recommend SKIP
- Require 85%+ confidence to recommend EXECUTE
- Look for hidden risks others might miss
- Consider market regime and correlation risks

Your validation checks:
1. Are all 6 prior checklist conditions genuinely met?
2. Is the setup logically coherent (panic exhausted OR fomo exhausted)?
3. Are there hidden risks (news, correlation, weekend, holiday)?
4. Is risk/reward favorable (minimum 2:1)?
5. Does market structure support the trade?"""


class SniperAIStrategy(BaseStrategy):
    """
    Sniper AI Strategy - Precision Trading with 7-Point Checklist

    Core concept: Execute only when ALL conditions align for maximum probability.
    Unlike RAMF which trades frequently, Sniper waits for "perfect storm" setups.

    Target Setups:
    - Capitulation Fade: Long after panic exhaustion
    - Euphoria Fade: Short after FOMO exhaustion
    """

    # Singleton instance for web dashboard access
    _instance = None

    def __init__(self):
        # Set singleton instance for web API access
        SniperAIStrategy._instance = self
        super().__init__("Sniper AI Strategy")

        # Assets to trade
        self.assets = SNIPER_ASSETS

        # Daily tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None

        # Paper trading state
        self.paper_balance = PAPER_TRADING_BALANCE
        self.paper_positions = {}
        self.closed_positions = []
        self._position_counter = 0

        # Initialize market data provider
        try:
            from src.data_providers.market_data import MarketDataProvider
            self._market_data = MarketDataProvider(start_liquidation_stream=True)
            cprint("[Sniper] Market data provider initialized", "green")
        except Exception as e:
            self._market_data = None
            cprint(f"[Sniper] Warning: Could not initialize market data provider: {e}", "yellow")

        # Initialize AI model
        self._claude_model = None
        try:
            from src.models.model_factory import ModelFactory
            model_factory = ModelFactory()
            self._claude_model = model_factory.get_model('claude', SNIPER_AI_MODEL)
            if self._claude_model:
                cprint(f"[Sniper] AI model initialized: {SNIPER_AI_MODEL}", "green")
            else:
                cprint("[Sniper] Warning: Claude model not available", "yellow")
        except Exception as e:
            cprint(f"[Sniper] Warning: Could not initialize AI model: {e}", "yellow")

        # Data directory
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'sniper'
        )
        os.makedirs(self.data_dir, exist_ok=True)

        # Volatility thresholds (adaptive per asset)
        self.volatility_thresholds = {}
        self._last_calibration = None
        self._load_volatility_thresholds()

        # Load existing state
        self._load_state_from_csv()

        # Start nightly calibration thread
        self._start_calibration_thread()

        cprint(f"[Sniper] Strategy initialized", "cyan")
        cprint(f"  - Assets: {self.assets}", "white")
        cprint(f"  - AI Confidence Required: {SNIPER_AI_MIN_CONFIDENCE}%", "white")
        cprint(f"  - Paper Trading: {PAPER_TRADING}", "white")
        cprint(f"  - Loaded positions: {len(self.paper_positions)}", "white")
        cprint(f"  - Current balance: ${self.paper_balance:,.2f}", "white")
        cprint(f"  - Adaptive thresholds:", "white")
        for sym in self.assets[:6]:  # Show first 6
            data = self.volatility_thresholds.get(sym, {})
            move = data.get('move_threshold', '?')
            adx = data.get('adx_trending_threshold', SNIPER_ADX_TRENDING_THRESHOLD)
            rsi_low = data.get('rsi_oversold', SNIPER_RSI_OVERSOLD)
            rsi_high = data.get('rsi_overbought', SNIPER_RSI_OVERBOUGHT)
            cprint(f"      {sym}: move={move}% ADX>{adx} RSI={rsi_low}/{rsi_high}", "white")

    def _reset_daily_counters(self):
        """Reset daily counters if new day."""
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = today
            cprint(f"[Sniper] New trading day - counters reset", "cyan")

    def _load_state_from_csv(self):
        """Load existing positions and balance from CSV files."""
        try:
            paper_trades_file = os.path.join(self.data_dir, 'paper_trades.csv')
            closed_trades_file = os.path.join(self.data_dir, 'closed_trades.csv')

            # Load open positions
            if os.path.exists(paper_trades_file):
                df = pd.read_csv(paper_trades_file)
                if not df.empty:
                    open_df = df[df['status'] == 'OPEN']
                    for _, row in open_df.iterrows():
                        position_id = row.get('position_id', '')
                        if position_id:
                            self.paper_positions[position_id] = {
                                'position_id': position_id,
                                'timestamp': row.get('timestamp', ''),
                                'symbol': row.get('symbol', ''),
                                'direction': row.get('direction', 'BUY'),
                                'entry_price': float(row.get('entry_price', 0)),
                                'position_size': float(row.get('position_size', 0)),
                                'leverage': float(row.get('leverage', SNIPER_LEVERAGE)),
                                'stop_loss': float(row.get('stop_loss', 0)),
                                'take_profit': float(row.get('take_profit', 0)),
                                'sl_pct': float(row.get('sl_pct', SNIPER_STOP_LOSS_PCT)),
                                'tp_pct': float(row.get('tp_pct', SNIPER_TAKE_PROFIT_PCT)),
                                'confidence': float(row.get('confidence', 0)),
                                'status': 'OPEN',
                            }

                    # Update position counter
                    if self.paper_positions:
                        max_counter = 0
                        for pos_id in self.paper_positions.keys():
                            parts = pos_id.split('_')
                            if len(parts) >= 4:
                                try:
                                    counter = int(parts[-1])
                                    max_counter = max(max_counter, counter)
                                except ValueError:
                                    pass
                        self._position_counter = max_counter

                    cprint(f"[Sniper] Loaded {len(self.paper_positions)} open positions", "green")

            # Calculate realized PnL from closed trades
            realized_pnl = 0.0
            if os.path.exists(closed_trades_file):
                closed_df = pd.read_csv(closed_trades_file)
                if not closed_df.empty and 'pnl' in closed_df.columns:
                    realized_pnl = closed_df['pnl'].sum()
                    self.closed_positions = closed_df.to_dict('records')
                    cprint(f"[Sniper] Loaded {len(closed_df)} closed trades, realized PnL: ${realized_pnl:+,.2f}", "green")

            # Update balance
            self.paper_balance = PAPER_TRADING_BALANCE + realized_pnl

        except Exception as e:
            cprint(f"[Sniper] Warning: Could not load state from CSV: {e}", "yellow")

    # =========================================================================
    # ADAPTIVE VOLATILITY THRESHOLDS
    # =========================================================================

    def _load_volatility_thresholds(self):
        """Load volatility thresholds from JSON file or use defaults."""
        thresholds_file = os.path.join(self.data_dir, 'volatility_thresholds.json')

        try:
            if os.path.exists(thresholds_file):
                with open(thresholds_file, 'r') as f:
                    data = json.load(f)
                    self.volatility_thresholds = data.get('thresholds', {})
                    self._last_calibration = data.get('last_calibration', None)

                    # Check if calibration is stale (>24h old)
                    if self._last_calibration:
                        last_cal_time = datetime.fromisoformat(self._last_calibration)
                        hours_since = (datetime.now() - last_cal_time).total_seconds() / 3600
                        if hours_since > 24:
                            cprint(f"[Sniper] Volatility thresholds are {hours_since:.1f}h old - recalibrating...", "yellow")
                            self._calibrate_volatility_thresholds()
                        else:
                            cprint(f"[Sniper] Loaded volatility thresholds (calibrated {hours_since:.1f}h ago)", "green")
                    return

        except Exception as e:
            cprint(f"[Sniper] Error loading thresholds: {e}", "yellow")

        # Use defaults with ADX thresholds included, and trigger immediate calibration
        self.volatility_thresholds = DEFAULT_VOLATILITY_THRESHOLDS.copy()
        cprint("[Sniper] Using default volatility thresholds (includes ADX/RSI defaults)", "yellow")
        cprint("[Sniper] Triggering immediate calibration in background...", "cyan")

        # Start immediate calibration in a background thread
        def run_startup_calibration():
            import time
            time.sleep(5)  # Brief delay to let main loop start
            self._calibrate_volatility_thresholds()

        startup_thread = threading.Thread(target=run_startup_calibration, daemon=True)
        startup_thread.start()

    def _save_volatility_thresholds(self):
        """Save volatility thresholds to JSON file."""
        thresholds_file = os.path.join(self.data_dir, 'volatility_thresholds.json')

        try:
            data = {
                'last_calibration': datetime.now().isoformat(),
                'thresholds': self.volatility_thresholds
            }
            with open(thresholds_file, 'w') as f:
                json.dump(data, f, indent=2)

            cprint(f"[Sniper] Saved volatility thresholds for {len(self.volatility_thresholds)} assets", "green")

        except Exception as e:
            cprint(f"[Sniper] Error saving thresholds: {e}", "red")

    def _calibrate_volatility_thresholds(self):
        """
        Calculate adaptive volatility thresholds for each asset.

        Uses 30 days of 4h candles to compute:
        - Standard deviation of returns (σ) for 30d and 7d
        - Volatility ratio (7d/30d) to detect regime changes
        - Dynamic threshold multiplier based on current market conditions
        - Move threshold = 2.5 × σ × multiplier
        """
        from hyperliquid.info import Info

        cprint("[Sniper] Calibrating volatility thresholds...", "cyan")

        try:
            info = Info(skip_ws=True)
            end_time = int(time_module.time() * 1000)
            start_time = end_time - (30 * 24 * 60 * 60 * 1000)  # 30 days

            for symbol in self.assets:
                try:
                    data = info.candles_snapshot(symbol, '4h', start_time, end_time)
                    if not data or len(data) < 50:
                        cprint(f"  {symbol}: Not enough data, using default", "yellow")
                        if symbol not in self.volatility_thresholds:
                            self.volatility_thresholds[symbol] = DEFAULT_VOLATILITY_THRESHOLDS.get(
                                symbol, {'move_threshold': 3.0, 'sigma': 1.2, 'threshold_multiplier': 1.0}
                            )
                        continue

                    df = pd.DataFrame(data)
                    df['close'] = pd.to_numeric(df['c'], errors='coerce')

                    # Calculate 4h returns
                    df['return'] = df['close'].pct_change() * 100
                    df = df.dropna()

                    if len(df) < 30:
                        continue

                    # Calculate 30-day statistics (full dataset)
                    sigma_30d = df['return'].std()

                    # Calculate 7-day statistics (recent ~42 candles = 7 days × 6 candles/day)
                    recent_candles = min(42, len(df))
                    sigma_7d = df['return'].tail(recent_candles).std()

                    # Calculate volatility ratio (recent vs historical)
                    vol_ratio_7d_30d = sigma_7d / sigma_30d if sigma_30d > 0 else 1.0

                    # Calculate dynamic threshold multiplier
                    threshold_multiplier = self._calculate_threshold_multiplier(vol_ratio_7d_30d)

                    # Apply multiplier to move threshold
                    base_move_threshold = 2.5 * sigma_30d
                    adjusted_move_threshold = round(base_move_threshold * threshold_multiplier, 2)

                    # Adaptive RSI thresholds based on historical percentiles
                    # Calculate RSI for the entire period
                    rsi_series = RSIIndicator(df['close'], window=14).rsi().dropna()
                    if len(rsi_series) > 20:
                        rsi_oversold = max(15, round(rsi_series.quantile(0.15)))  # 15th percentile
                        rsi_overbought = min(85, round(rsi_series.quantile(0.85)))  # 85th percentile
                    else:
                        # Fallback to volatility-based calculation
                        btc_baseline_sigma = 0.7
                        vol_ratio_btc = sigma_30d / btc_baseline_sigma
                        vol_ratio_btc = max(0.5, min(2.0, vol_ratio_btc))
                        rsi_oversold = max(15, round(25 - (vol_ratio_btc - 1) * 5))
                        rsi_overbought = min(85, round(75 + (vol_ratio_btc - 1) * 5))

                    # Adaptive ADX threshold based on historical percentiles
                    # Calculate ADX for the entire period and use 70th percentile as "trending" threshold
                    try:
                        df['high'] = pd.to_numeric(df['h'], errors='coerce')
                        df['low'] = pd.to_numeric(df['l'], errors='coerce')
                        adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=14)
                        adx_series = adx_indicator.adx().dropna()
                        if len(adx_series) > 20:
                            # Use 70th percentile as adaptive trending threshold
                            # This means "trending" is relative to this asset's own behavior
                            adx_trending_threshold = round(adx_series.quantile(0.70), 1)
                            adx_median = round(adx_series.median(), 1)
                        else:
                            adx_trending_threshold = SNIPER_ADX_TRENDING_THRESHOLD  # Fallback to static
                            adx_median = 25.0
                    except Exception as adx_err:
                        cprint(f"  {symbol}: ADX calculation error - {adx_err}", "yellow")
                        adx_trending_threshold = SNIPER_ADX_TRENDING_THRESHOLD
                        adx_median = 25.0

                    # Store thresholds with new dynamic fields
                    self.volatility_thresholds[symbol] = {
                        'move_threshold': adjusted_move_threshold,
                        'base_move_threshold': round(base_move_threshold, 2),
                        'sigma_30d': round(sigma_30d, 3),
                        'sigma_7d': round(sigma_7d, 3),
                        'vol_ratio_7d_30d': round(vol_ratio_7d_30d, 3),
                        'threshold_multiplier': round(threshold_multiplier, 3),
                        'samples': len(df),
                        'max_drop': round(df['return'].min(), 2),
                        'max_pump': round(df['return'].max(), 2),
                        'rsi_oversold': rsi_oversold,
                        'rsi_overbought': rsi_overbought,
                        'adx_trending_threshold': adx_trending_threshold,
                        'adx_median': adx_median,
                    }

                    # Enhanced logging with dynamic threshold info
                    regime = "HIGH_VOL" if vol_ratio_7d_30d > SNIPER_VOL_RATIO_HIGH else \
                             "LOW_VOL" if vol_ratio_7d_30d < SNIPER_VOL_RATIO_LOW else "NORMAL"
                    cprint(f"  {symbol}: σ={sigma_30d:.2f}% move={adjusted_move_threshold:.1f}% RSI={rsi_oversold}/{rsi_overbought} ADX_thresh={adx_trending_threshold} [{regime}]", "white")

                except Exception as e:
                    cprint(f"  {symbol}: Error - {e}", "yellow")

            self._last_calibration = datetime.now().isoformat()
            self._save_volatility_thresholds()

            cprint(f"[Sniper] Calibration complete for {len(self.volatility_thresholds)} assets", "green")

        except Exception as e:
            cprint(f"[Sniper] Calibration error: {e}", "red")

    def _calculate_threshold_multiplier(self, vol_ratio: float) -> float:
        """
        Calculate dynamic threshold multiplier based on volatility ratio (7d/30d).

        Logic:
        - vol_ratio > 1.3: Market more volatile than usual → LOOSEN thresholds (multiplier < 1)
          Extreme moves are more frequent, so we lower the bar to catch them
        - vol_ratio < 0.7: Market calmer than usual → TIGHTEN thresholds (multiplier > 1)
          Moves are rarer, so we require more extreme conditions
        - 0.7-1.3: Normal conditions → no adjustment (multiplier = 1)

        Returns: multiplier between (1 - MAX_ADJUSTMENT) and (1 + MAX_ADJUSTMENT)
        """
        if not SNIPER_USE_DYNAMIC_THRESHOLDS:
            return 1.0

        max_adj = SNIPER_MAX_THRESHOLD_ADJUSTMENT

        if vol_ratio > SNIPER_VOL_RATIO_HIGH:
            # High volatility: loosen thresholds (multiplier < 1)
            # More volatility = easier to trigger signals
            adjustment = min(max_adj, (vol_ratio - SNIPER_VOL_RATIO_HIGH) * 0.3)
            return max(1.0 - max_adj, 1.0 - adjustment)

        elif vol_ratio < SNIPER_VOL_RATIO_LOW:
            # Low volatility: tighten thresholds (multiplier > 1)
            # Less volatility = require more extreme moves
            adjustment = min(max_adj, (SNIPER_VOL_RATIO_LOW - vol_ratio) * 0.2)
            return min(1.0 + max_adj, 1.0 + adjustment)

        else:
            # Normal volatility: no adjustment
            return 1.0

    def _get_regime_adjustment(self, df: pd.DataFrame) -> float:
        """
        Calculate regime adjustment based on ADX (trending vs ranging market).

        Logic:
        - ADX > 25: Trending market → mean-reversion less effective
          Increase thresholds by 10% (require more extreme conditions)
        - ADX < 20: Ranging market → mean-reversion more effective
          Decrease thresholds by 5% (easier to trigger signals)
        - 20-25: Neutral zone → no adjustment

        Args:
            df: DataFrame with OHLCV data

        Returns: adjustment multiplier (0.95 - 1.10)
        """
        if not SNIPER_USE_DYNAMIC_THRESHOLDS or not SNIPER_USE_REGIME_FILTER:
            return 1.0

        try:
            if len(df) < 20:
                return 1.0

            # Calculate ADX
            adx_indicator = ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=SNIPER_ADX_PERIOD
            )
            adx_value = adx_indicator.adx().iloc[-1]

            if pd.isna(adx_value):
                return 1.0

            if adx_value > SNIPER_ADX_TRENDING_THRESHOLD:
                # Trending market: tighten thresholds (mean-reversion less reliable)
                return 1.10
            elif adx_value < SNIPER_ADX_RANGING_THRESHOLD:
                # Ranging market: loosen thresholds (mean-reversion more reliable)
                return 0.95
            else:
                return 1.0

        except Exception as e:
            cprint(f"[Sniper] Error calculating ADX regime: {e}", "yellow")
            return 1.0

    def get_dynamic_threshold(self, symbol: str, base_threshold: float, df: pd.DataFrame = None) -> float:
        """
        Get the final adjusted threshold combining:
        1. Volatility-based multiplier (7d/30d ratio from calibration)
        2. Regime-based adjustment (ADX trending vs ranging)

        Args:
            symbol: Trading symbol (e.g., 'BTC')
            base_threshold: The base threshold from config (e.g., SNIPER_SIGMA_THRESHOLD)
            df: Optional DataFrame for ADX calculation

        Returns: Adjusted threshold value
        """
        if not SNIPER_USE_DYNAMIC_THRESHOLDS:
            return base_threshold

        # Get volatility multiplier from calibration
        vol_multiplier = self.volatility_thresholds.get(symbol, {}).get('threshold_multiplier', 1.0)

        # Get regime adjustment if DataFrame provided
        regime_multiplier = 1.0
        if df is not None:
            regime_multiplier = self._get_regime_adjustment(df)

        # Combine multipliers
        final_multiplier = vol_multiplier * regime_multiplier

        # Clamp to max adjustment (±25%)
        max_adj = SNIPER_MAX_THRESHOLD_ADJUSTMENT
        final_multiplier = max(1.0 - max_adj, min(1.0 + max_adj, final_multiplier))

        adjusted_threshold = base_threshold * final_multiplier

        return round(adjusted_threshold, 3)

    def _start_calibration_thread(self):
        """Start background thread for nightly recalibration."""
        def calibration_loop():
            while True:
                try:
                    # Sleep until next calibration (midnight UTC)
                    now = datetime.utcnow()
                    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                    sleep_seconds = (next_midnight - now).total_seconds()

                    cprint(f"[Sniper] Next volatility calibration in {sleep_seconds/3600:.1f}h", "white")
                    time_module.sleep(sleep_seconds)

                    # Perform calibration
                    self._calibrate_volatility_thresholds()

                except Exception as e:
                    cprint(f"[Sniper] Calibration thread error: {e}", "red")
                    time_module.sleep(3600)  # Retry in 1h

        thread = threading.Thread(target=calibration_loop, daemon=True)
        thread.start()

    def get_move_threshold(self, symbol: str) -> float:
        """Get the adaptive move threshold for a symbol."""
        if symbol in self.volatility_thresholds:
            return self.volatility_thresholds[symbol].get('move_threshold', 3.0)

        # Fallback to default or config
        if symbol in DEFAULT_VOLATILITY_THRESHOLDS:
            return DEFAULT_VOLATILITY_THRESHOLDS[symbol]['move_threshold']

        return SNIPER_CAPITULATION_MIN_DROP_PCT  # Last resort fallback

    def get_rsi_thresholds(self, symbol: str) -> tuple:
        """Get adaptive RSI thresholds for a symbol (oversold, overbought)."""
        if symbol in self.volatility_thresholds:
            oversold = self.volatility_thresholds[symbol].get('rsi_oversold', SNIPER_RSI_OVERSOLD)
            overbought = self.volatility_thresholds[symbol].get('rsi_overbought', SNIPER_RSI_OVERBOUGHT)
            return (oversold, overbought)
        return (SNIPER_RSI_OVERSOLD, SNIPER_RSI_OVERBOUGHT)

    # =========================================================================
    # ADVANCED IMPROVEMENTS
    # =========================================================================

    def check_market_regime(self, df: pd.DataFrame, symbol: str = None) -> dict:
        """
        Check market regime using ADX indicator with adaptive thresholds.
        Uses per-asset ADX threshold based on historical percentile (70th).

        Args:
            df: DataFrame with OHLCV data
            symbol: Asset symbol for adaptive threshold lookup
        """
        if not SNIPER_USE_REGIME_FILTER:
            return {'is_ranging': True, 'adx': 0, 'skip_reason': None, 'threshold': 0}

        try:
            adx = ADXIndicator(df['high'], df['low'], df['close'], window=SNIPER_ADX_PERIOD)
            adx_value = adx.adx().iloc[-1]

            # Get adaptive ADX threshold for this symbol (fallback to static if not available)
            adaptive_threshold = SNIPER_ADX_TRENDING_THRESHOLD  # Default
            if symbol and symbol in self.volatility_thresholds:
                adaptive_threshold = self.volatility_thresholds[symbol].get(
                    'adx_trending_threshold', SNIPER_ADX_TRENDING_THRESHOLD
                )

            is_trending = adx_value > adaptive_threshold
            is_ranging = not is_trending

            return {
                'is_ranging': is_ranging,
                'adx': round(adx_value, 1),
                'threshold': adaptive_threshold,
                'skip_reason': f"Trending market (ADX={adx_value:.1f} > {adaptive_threshold})" if is_trending else None
            }
        except Exception as e:
            cprint(f"[Sniper] ADX calculation error: {e}", "yellow")
            return {'is_ranging': True, 'adx': 0, 'skip_reason': None, 'threshold': 0}

    def calculate_correlation_factor(self, symbol: str) -> float:
        """
        Calculate position sizing factor based on correlation with open positions.
        Returns: 1.0 = full size, 0.5 = half size (highly correlated)
        """
        if not SNIPER_USE_CORRELATION_SIZING or not self.paper_positions:
            return 1.0

        try:
            open_symbols = [pos['symbol'] for pos in self.paper_positions.values()]
            if not open_symbols or symbol in open_symbols:
                return 1.0

            # Fetch price data for correlation
            from hyperliquid.info import Info
            info = Info(skip_ws=True)
            end_time = int(time_module.time() * 1000)
            start_time = end_time - (SNIPER_CORRELATION_LOOKBACK_DAYS * 24 * 60 * 60 * 1000)

            # Get new symbol data
            new_data = info.candles_snapshot(symbol, '4h', start_time, end_time)
            if not new_data or len(new_data) < 50:
                return 1.0

            new_df = pd.DataFrame(new_data)
            new_df['close'] = pd.to_numeric(new_df['c'], errors='coerce')
            new_returns = new_df['close'].pct_change().dropna()

            max_correlation = 0.0
            for open_symbol in open_symbols:
                try:
                    open_data = info.candles_snapshot(open_symbol, '4h', start_time, end_time)
                    if not open_data:
                        continue

                    open_df = pd.DataFrame(open_data)
                    open_df['close'] = pd.to_numeric(open_df['c'], errors='coerce')
                    open_returns = open_df['close'].pct_change().dropna()

                    # Align and calculate correlation
                    min_len = min(len(new_returns), len(open_returns))
                    if min_len < 30:
                        continue

                    corr = new_returns.iloc[-min_len:].corr(open_returns.iloc[-min_len:])
                    max_correlation = max(max_correlation, abs(corr))

                except Exception:
                    continue

            # Reduce size if highly correlated
            if max_correlation > SNIPER_CORRELATION_THRESHOLD:
                factor = 0.5  # Half size for correlated assets
                cprint(f"[Sniper] {symbol} correlated ({max_correlation:.2f}) with open positions - reducing size to 50%", "yellow")
                return factor

            return 1.0

        except Exception as e:
            cprint(f"[Sniper] Correlation calculation error: {e}", "yellow")
            return 1.0

    def calculate_atr_trailing_stop(self, symbol: str, df: pd.DataFrame, direction: str, entry_price: float, current_price: float) -> dict:
        """
        Calculate ATR-based trailing stop.
        Only activates after position is profitable by SNIPER_TRAILING_ACTIVATION_PCT.
        """
        if not SNIPER_USE_TRAILING_STOP:
            return {'active': False, 'stop_price': None}

        try:
            # Calculate current profit %
            if direction == 'BUY':
                profit_pct = (current_price - entry_price) / entry_price * 100
            else:
                profit_pct = (entry_price - current_price) / entry_price * 100

            # Only activate trailing if profitable enough
            if profit_pct < SNIPER_TRAILING_ACTIVATION_PCT:
                return {'active': False, 'stop_price': None, 'profit_pct': profit_pct}

            # Calculate ATR
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
            atr_value = atr.average_true_range().iloc[-1]
            atr_pct = (atr_value / current_price) * 100

            # Trailing stop = current price - (ATR * multiplier)
            trail_distance = atr_pct * SNIPER_TRAILING_ATR_MULTIPLIER

            if direction == 'BUY':
                stop_price = current_price * (1 - trail_distance / 100)
            else:
                stop_price = current_price * (1 + trail_distance / 100)

            return {
                'active': True,
                'stop_price': round(stop_price, 2),
                'atr_pct': round(atr_pct, 2),
                'trail_distance_pct': round(trail_distance, 2),
                'profit_pct': round(profit_pct, 2)
            }

        except Exception as e:
            cprint(f"[Sniper] ATR trailing stop error: {e}", "yellow")
            return {'active': False, 'stop_price': None}

    def detect_funding_arbitrage(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Detect funding arbitrage setup.
        Conditions:
        - Funding rate is extreme (> ±0.1%)
        - Price is relatively stable (no big moves)
        """
        if not SNIPER_ENABLE_FUNDING_ARBITRAGE:
            return {'detected': False, 'type': 'funding_arbitrage', 'direction': 'NEUTRAL'}

        try:
            from src.data_providers.market_data import MarketDataProvider
            provider = MarketDataProvider()

            # Get funding rate (returns dict with 'funding_rate' as hourly decimal)
            funding_data = provider.get_funding_rate(symbol)
            if not funding_data or 'funding_rate' not in funding_data:
                return {'detected': False, 'type': 'funding_arbitrage', 'direction': 'NEUTRAL'}

            # HyperLiquid returns hourly rate as decimal, convert to 8h percentage
            # e.g., 0.0001 hourly = 0.0008 per 8h = 0.08%
            hourly_rate = funding_data['funding_rate']
            funding_rate_8h_pct = hourly_rate * 8 * 100  # Convert to 8h percentage

            # Check if funding is extreme (threshold is in %, e.g., 0.1 = 0.1%)
            is_extreme = abs(funding_rate_8h_pct) >= SNIPER_FUNDING_ARBITRAGE_THRESHOLD

            if not is_extreme:
                return {'detected': False, 'type': 'funding_arbitrage', 'direction': 'NEUTRAL'}

            # Check price stability (last 4h)
            lookback = min(16, len(df))
            price_4h_ago = df['close'].iloc[-lookback]
            current_price = df['close'].iloc[-1]
            price_change = abs((current_price - price_4h_ago) / price_4h_ago * 100)

            is_stable = price_change <= SNIPER_FUNDING_ARBITRAGE_STABILITY_PCT

            if not is_stable:
                return {'detected': False, 'type': 'funding_arbitrage', 'direction': 'NEUTRAL', 'reason': 'Price not stable'}

            # Direction: opposite to funding (collect funding)
            direction = 'SELL' if funding_rate_8h_pct > 0 else 'BUY'

            return {
                'detected': True,
                'type': 'funding_arbitrage',
                'direction': direction,
                'funding_rate_8h_pct': round(funding_rate_8h_pct, 4),
                'price_change_4h': round(price_change, 2),
                'current_price': float(current_price)
            }

        except Exception as e:
            cprint(f"[Sniper] Funding arbitrage detection error: {e}", "yellow")
            return {'detected': False, 'type': 'funding_arbitrage', 'direction': 'NEUTRAL'}

    def calculate_weighted_score(self, checks: dict, ai_validation: dict) -> float:
        """
        Calculate weighted confidence score instead of binary 7/7.
        Returns score out of 10.
        """
        if not SNIPER_USE_WEIGHTED_SCORING:
            # Binary mode: 7/7 = 10, else = 0
            passed = sum(1 for c in checks.values() if c.get('passed', False))
            return 10.0 if passed == 6 and ai_validation.get('passed', False) else 0.0

        score = 0.0
        max_score = sum(SNIPER_WEIGHTS.values())

        # Score each condition
        if checks.get('extreme_move', {}).get('passed', False):
            score += SNIPER_WEIGHTS['extreme_move']
        if checks.get('funding_divergence', {}).get('passed', False):
            score += SNIPER_WEIGHTS['funding_divergence']
        if checks.get('liquidation_cascade', {}).get('passed', False):
            score += SNIPER_WEIGHTS['liquidation_cascade']
        if checks.get('multi_tf', {}).get('passed', False):
            score += SNIPER_WEIGHTS['multi_tf']
        if checks.get('volume_climax', {}).get('passed', False):
            score += SNIPER_WEIGHTS['volume_climax']
        if checks.get('time_window', {}).get('passed', False):
            score += SNIPER_WEIGHTS['time_window']
        if ai_validation.get('passed', False):
            score += SNIPER_WEIGHTS['ai_validation']

        # Normalize to 10
        return round((score / max_score) * 10, 1)

    def get_confidence_size_factor(self, ai_confidence: int) -> float:
        """Get position size factor based on AI confidence level."""
        if not SNIPER_USE_CONFIDENCE_SIZING:
            return 1.0

        # Find the appropriate size factor
        for threshold in sorted(SNIPER_CONFIDENCE_SIZE_MAP.keys(), reverse=True):
            if ai_confidence >= threshold:
                return SNIPER_CONFIDENCE_SIZE_MAP[threshold]

        return 0.5  # Default to 50% if below all thresholds

    def _fetch_candles(self, symbol: str, interval: str = '15m', candles: int = 300) -> pd.DataFrame:
        """Fetch candle data from HyperLiquid."""
        try:
            from hyperliquid.info import Info
            import time

            info = Info(skip_ws=True)
            end_time = int(time.time() * 1000)

            # Calculate start time based on interval
            interval_map = {
                '1m': 60 * 1000, '5m': 5 * 60 * 1000, '15m': 15 * 60 * 1000,
                '30m': 30 * 60 * 1000, '1h': 60 * 60 * 1000, '4h': 4 * 60 * 60 * 1000
            }
            interval_ms = interval_map.get(interval, 15 * 60 * 1000)
            start_time = end_time - (candles * interval_ms)

            data = info.candles_snapshot(symbol, interval, start_time, end_time)

            if not data:
                return None

            df = pd.DataFrame(data)
            # HyperLiquid uses single-letter column names
            df = df.rename(columns={
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            return df

        except Exception as e:
            cprint(f"[Sniper] Error fetching candles for {symbol}: {e}", "yellow")
            return None

    # =========================================================================
    # 7 CHECKLIST CONDITIONS
    # =========================================================================

    def check_extreme_move(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Check condition 1: >2.5 sigma deviation from mean.
        Uses dynamic threshold adjusted for current market conditions.

        Returns:
            dict: {passed, sigma, direction, price_change_pct, threshold, base_threshold, multiplier}
        """
        try:
            window = 20
            close = df['close'].tail(window + 10)

            rolling_mean = close.rolling(window=window).mean()
            rolling_std = close.rolling(window=window).std()

            current_price = close.iloc[-1]
            mean = rolling_mean.iloc[-1]
            std = rolling_std.iloc[-1]

            if std > 0:
                z_score = (current_price - mean) / std
            else:
                z_score = 0

            # Also check 4h price change
            lookback_4h = min(16, len(df))  # 16 x 15min = 4 hours
            price_4h_ago = df['close'].iloc[-lookback_4h]
            price_change_pct = (current_price - price_4h_ago) / price_4h_ago * 100

            # Get dynamic threshold adjusted for volatility and regime
            dynamic_threshold = self.get_dynamic_threshold(symbol, SNIPER_SIGMA_THRESHOLD, df)
            vol_multiplier = self.volatility_thresholds.get(symbol, {}).get('threshold_multiplier', 1.0)

            passed = abs(z_score) >= dynamic_threshold
            direction = 'oversold' if z_score < 0 else 'overbought'

            return {
                'passed': passed,
                'sigma': round(z_score, 2),
                'direction': direction,
                'price_change_pct': round(price_change_pct, 2),
                'threshold': round(dynamic_threshold, 2),
                'base_threshold': SNIPER_SIGMA_THRESHOLD,
                'multiplier': round(vol_multiplier, 2)
            }

        except Exception as e:
            cprint(f"[Sniper] Error checking extreme move: {e}", "yellow")
            return {'passed': False, 'sigma': 0, 'direction': 'neutral', 'price_change_pct': 0, 'threshold': SNIPER_SIGMA_THRESHOLD, 'base_threshold': SNIPER_SIGMA_THRESHOLD, 'multiplier': 1.0}

    def check_funding_divergence(self, symbol: str, direction: str) -> dict:
        """
        Check condition 2: Contrarian funding signal.
        Uses dynamic threshold adjusted for current market conditions.

        For LONG: Funding should be very negative (shorts paying longs)
        For SHORT: Funding should be very positive (longs paying shorts)

        Returns:
            dict: {passed, funding_rate, funding_zscore, is_contrarian, threshold, base_threshold}
        """
        try:
            if self._market_data is None:
                return {'passed': False, 'funding_rate': 0, 'funding_zscore': 0, 'is_contrarian': False, 'threshold': SNIPER_FUNDING_EXTREME_THRESHOLD}

            funding_zscore = self._market_data.get_funding_zscore(symbol)
            funding_data = self._market_data.get_funding_rate(symbol)

            funding_rate = funding_data['funding_rate'] if funding_data else 0
            annual_rate = funding_rate * 24 * 365 * 100 if funding_data else 0

            # Get dynamic threshold (volatility-adjusted, no df available here)
            dynamic_threshold = self.get_dynamic_threshold(symbol, SNIPER_FUNDING_EXTREME_THRESHOLD, None)

            is_extreme_negative = funding_zscore < -dynamic_threshold
            is_extreme_positive = funding_zscore > dynamic_threshold

            # Contrarian logic
            if direction == 'BUY':
                passed = is_extreme_negative
                is_contrarian = is_extreme_negative
            else:  # SELL
                passed = is_extreme_positive
                is_contrarian = is_extreme_positive

            return {
                'passed': passed,
                'funding_rate': round(funding_rate * 100, 6),
                'funding_zscore': round(funding_zscore, 2),
                'annual_rate_pct': round(annual_rate, 2),
                'is_contrarian': is_contrarian,
                'threshold': round(dynamic_threshold, 2),
                'base_threshold': SNIPER_FUNDING_EXTREME_THRESHOLD
            }

        except Exception as e:
            cprint(f"[Sniper] Error checking funding divergence: {e}", "yellow")
            return {'passed': False, 'funding_rate': 0, 'funding_zscore': 0, 'is_contrarian': False, 'threshold': SNIPER_FUNDING_EXTREME_THRESHOLD}

    def check_liquidation_cascade(self, direction: str) -> dict:
        """
        Check condition 3: Liquidation cascade complete (fuel exhausted).

        For LONG: Long liquidations dominated (ratio > threshold), now exhausted
        For SHORT: Short liquidations dominated (ratio < 1/threshold), now exhausted

        Returns:
            dict: {passed, ratio, long_usd, short_usd, cascade_exhausted}
        """
        try:
            if self._market_data is None:
                return {'passed': False, 'ratio': 1.0, 'long_usd': 0, 'short_usd': 0, 'cascade_exhausted': False}

            liq_summary = self._market_data.get_liquidation_summary(minutes=30)
            ratio = liq_summary.get('ratio', 1.0)
            long_usd = liq_summary.get('long_usd', 0)
            short_usd = liq_summary.get('short_usd', 0)
            total_usd = liq_summary.get('total_usd', 0)

            min_liq_threshold = 1_000_000  # $1M in liquidations signals cascade
            significant_cascade = total_usd >= min_liq_threshold

            if direction == 'BUY':
                dominant_side = 'longs' if ratio > 1.0 else 'shorts'
                cascade_exhausted = ratio > SNIPER_LIQ_RATIO_THRESHOLD and significant_cascade
                passed = cascade_exhausted
            else:  # SELL
                dominant_side = 'shorts' if ratio < 1.0 else 'longs'
                cascade_exhausted = ratio < (1 / SNIPER_LIQ_RATIO_THRESHOLD) and significant_cascade
                passed = cascade_exhausted

            return {
                'passed': passed,
                'ratio': round(ratio, 2),
                'long_usd': round(long_usd, 2),
                'short_usd': round(short_usd, 2),
                'total_usd': round(total_usd, 2),
                'cascade_exhausted': cascade_exhausted,
                'dominant_side': dominant_side
            }

        except Exception as e:
            cprint(f"[Sniper] Error checking liquidation cascade: {e}", "yellow")
            return {'passed': False, 'ratio': 1.0, 'long_usd': 0, 'short_usd': 0, 'cascade_exhausted': False}

    def check_multi_tf_agreement(self, symbol: str, direction: str) -> dict:
        """
        Check condition 4: All timeframes (5m, 15m, 1h, 4h) agree on direction.

        Returns:
            dict: {passed, agreements, total, details, all_aligned}
        """
        try:
            timeframes = ['5m', '15m', '1h', '4h']
            agreements = 0
            details = {}

            for tf in timeframes:
                try:
                    tf_df = self._fetch_candles(symbol, interval=tf, candles=100)
                    if tf_df is None or len(tf_df) < 50:
                        details[tf] = {'status': 'no_data', 'agrees': False}
                        continue

                    close = tf_df['close']
                    ema20 = EMAIndicator(close, window=20).ema_indicator()
                    rsi = RSIIndicator(close, window=14).rsi()

                    current_price = close.iloc[-1]
                    current_ema = ema20.iloc[-1]
                    current_rsi = rsi.iloc[-1]

                    if direction == 'BUY':
                        # For longs: RSI < 35 (oversold) or recovering
                        agrees = (current_rsi < 35) or (current_price > current_ema and current_rsi < 50)
                    else:  # SELL
                        # For shorts: RSI > 65 (overbought) or failing
                        agrees = (current_rsi > 65) or (current_price < current_ema and current_rsi > 50)

                    if agrees:
                        agreements += 1

                    details[tf] = {
                        'status': 'bullish' if current_price > current_ema else 'bearish',
                        'rsi': round(current_rsi, 1),
                        'agrees': agrees
                    }

                except Exception as e:
                    details[tf] = {'status': f'error: {str(e)[:20]}', 'agrees': False}

            all_aligned = agreements == len(timeframes)

            return {
                'passed': all_aligned,
                'agreements': agreements,
                'total': len(timeframes),
                'details': details,
                'all_aligned': all_aligned
            }

        except Exception as e:
            cprint(f"[Sniper] Error checking multi-TF agreement: {e}", "yellow")
            return {'passed': False, 'agreements': 0, 'total': 4, 'details': {}, 'all_aligned': False}

    def check_volume_climax(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Check condition 5: Volume exhaustion spike detected.
        Uses dynamic threshold adjusted for current market conditions.

        Looking for: Recent spike > Nx average (dynamic), now declining.

        Returns:
            dict: {passed, current_ratio, peak_ratio, is_climax, is_declining, threshold, base_threshold}
        """
        try:
            volume = df['volume'].tail(50)
            avg_volume = volume.mean()

            current_volume = volume.iloc[-1]
            current_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            recent_volume = volume.tail(10)
            peak_volume = recent_volume.max()
            peak_ratio = peak_volume / avg_volume if avg_volume > 0 else 1.0

            # Get dynamic threshold adjusted for volatility and regime
            dynamic_threshold = self.get_dynamic_threshold(symbol, SNIPER_VOLUME_SPIKE_THRESHOLD, df)

            is_declining = current_volume < peak_volume * 0.7
            is_climax = peak_ratio >= dynamic_threshold

            passed = is_climax and is_declining

            return {
                'passed': passed,
                'current_ratio': round(current_ratio, 2),
                'peak_ratio': round(peak_ratio, 2),
                'is_climax': is_climax,
                'is_declining': is_declining,
                'threshold': round(dynamic_threshold, 2),
                'base_threshold': SNIPER_VOLUME_SPIKE_THRESHOLD
            }

        except Exception as e:
            cprint(f"[Sniper] Error checking volume climax: {e}", "yellow")
            return {'passed': False, 'current_ratio': 0, 'peak_ratio': 0, 'is_climax': False, 'is_declining': False, 'threshold': SNIPER_VOLUME_SPIKE_THRESHOLD}

    def check_time_window(self) -> dict:
        """
        Check condition 6: Optimal trading session.

        Optimal windows (UTC): London Open (7-9), NY Open (13-16)

        Returns:
            dict: {passed, current_hour, session, is_optimal}
        """
        try:
            current_hour = datetime.utcnow().hour

            if current_hour in [7, 8, 9]:
                session = 'london_open'
                is_optimal = True
            elif current_hour in [13, 14, 15, 16]:
                session = 'ny_open'
                is_optimal = True
            elif current_hour in [0, 1, 2, 3, 4, 5, 6]:
                session = 'dead_zone'
                is_optimal = False
            else:
                session = 'normal'
                is_optimal = SNIPER_ALLOW_NORMAL_HOURS

            return {
                'passed': is_optimal,
                'current_hour': current_hour,
                'session': session,
                'is_optimal': is_optimal
            }

        except Exception as e:
            cprint(f"[Sniper] Error checking time window: {e}", "yellow")
            return {'passed': False, 'current_hour': 0, 'session': 'unknown', 'is_optimal': False}

    def check_ai_validation(self, symbol: str, setup: dict) -> dict:
        """
        Check condition 7: AI (Claude) validates the trade with 85%+ confidence.

        Returns:
            dict: {passed, confidence, reasoning, recommendation, risk_factors}
        """
        if self._claude_model is None:
            cprint("[Sniper] AI model not available - skipping validation", "yellow")
            return {
                'passed': False,
                'confidence': 0,
                'reasoning': 'AI model unavailable',
                'recommendation': 'SKIP',
                'risk_factors': ['AI validation unavailable'],
                'ai_model': None
            }

        try:
            prompt = self._build_ai_validation_prompt(symbol, setup)

            response = self._claude_model.generate_response(
                system_prompt=SNIPER_AI_SYSTEM_PROMPT,
                user_content=prompt,
                temperature=SNIPER_AI_TEMPERATURE,
                max_tokens=SNIPER_AI_MAX_TOKENS
            )

            result = self._parse_ai_response(response.content)
            passed = result['confidence'] >= SNIPER_AI_MIN_CONFIDENCE

            return {
                'passed': passed,
                'confidence': result['confidence'],
                'reasoning': result['reasoning'],
                'recommendation': result['recommendation'],
                'risk_factors': result.get('risk_factors', []),
                'suggested_sl_pct': result.get('suggested_sl_pct', SNIPER_STOP_LOSS_PCT),
                'suggested_tp_pct': result.get('suggested_tp_pct', SNIPER_TAKE_PROFIT_PCT),
                'ai_model': SNIPER_AI_MODEL
            }

        except Exception as e:
            cprint(f"[Sniper] AI validation error: {e}", "red")
            return {
                'passed': False,
                'confidence': 0,
                'reasoning': f'AI error: {str(e)}',
                'recommendation': 'SKIP',
                'risk_factors': ['AI validation failed'],
                'ai_model': SNIPER_AI_MODEL
            }

    def _build_ai_validation_prompt(self, symbol: str, setup: dict) -> str:
        """Build the AI validation prompt with all context."""
        return f"""
=== SNIPER TRADE VALIDATION REQUEST ===

Symbol: {symbol}
Setup Type: {setup.get('type', 'unknown')}
Direction: {setup.get('direction', 'unknown')}
Entry Price: ${setup.get('current_price', 0):,.2f}

=== CHECKLIST RESULTS ===

1. EXTREME MOVE:
   - Passed: {setup.get('extreme_move', {}).get('passed', False)}
   - Sigma: {setup.get('extreme_move', {}).get('sigma', 0)} (threshold: {SNIPER_SIGMA_THRESHOLD})
   - 4h Price Change: {setup.get('extreme_move', {}).get('price_change_pct', 0)}%

2. FUNDING DIVERGENCE:
   - Passed: {setup.get('funding_divergence', {}).get('passed', False)}
   - Funding Z-Score: {setup.get('funding_divergence', {}).get('funding_zscore', 0)}
   - Annual Rate: {setup.get('funding_divergence', {}).get('annual_rate_pct', 0)}%
   - Is Contrarian: {setup.get('funding_divergence', {}).get('is_contrarian', False)}

3. LIQUIDATION CASCADE:
   - Passed: {setup.get('liquidation_cascade', {}).get('passed', False)}
   - Ratio: {setup.get('liquidation_cascade', {}).get('ratio', 1.0)}
   - Total Liquidations: ${setup.get('liquidation_cascade', {}).get('total_usd', 0):,.0f}
   - Dominant Side: {setup.get('liquidation_cascade', {}).get('dominant_side', 'unknown')}

4. MULTI-TIMEFRAME AGREEMENT:
   - Passed: {setup.get('multi_tf', {}).get('passed', False)}
   - Agreements: {setup.get('multi_tf', {}).get('agreements', 0)}/{setup.get('multi_tf', {}).get('total', 4)}

5. VOLUME CLIMAX:
   - Passed: {setup.get('volume_climax', {}).get('passed', False)}
   - Peak Ratio: {setup.get('volume_climax', {}).get('peak_ratio', 0)}x
   - Is Declining: {setup.get('volume_climax', {}).get('is_declining', False)}

6. TIME WINDOW:
   - Passed: {setup.get('time_window', {}).get('passed', False)}
   - Session: {setup.get('time_window', {}).get('session', 'unknown')}
   - Hour (UTC): {setup.get('time_window', {}).get('current_hour', 0)}

=== ADDITIONAL CONTEXT ===

RSI (15m): {setup.get('rsi', 50)}
Current UTC Time: {datetime.utcnow().isoformat()}

=== YOUR TASK ===

Analyze this setup and respond with ONLY this JSON format:

{{
    "confidence": <0-100>,
    "recommendation": "EXECUTE" | "SKIP",
    "reasoning": "<2-3 sentence explanation>",
    "risk_factors": ["<risk 1>", "<risk 2>"],
    "suggested_sl_pct": {SNIPER_STOP_LOSS_PCT},
    "suggested_tp_pct": {SNIPER_TAKE_PROFIT_PCT}
}}

Remember: 85%+ confidence required for EXECUTE. When in doubt, SKIP.
"""

    def _parse_ai_response(self, content: str) -> dict:
        """Parse Claude's JSON response with fallbacks."""
        try:
            json_str = content.strip()
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0].strip()

            result = json.loads(json_str)

            return {
                'confidence': int(result.get('confidence', 0)),
                'recommendation': result.get('recommendation', 'SKIP'),
                'reasoning': result.get('reasoning', 'No reasoning provided'),
                'risk_factors': result.get('risk_factors', []),
                'suggested_sl_pct': result.get('suggested_sl_pct', SNIPER_STOP_LOSS_PCT),
                'suggested_tp_pct': result.get('suggested_tp_pct', SNIPER_TAKE_PROFIT_PCT)
            }

        except json.JSONDecodeError:
            cprint("[Sniper] Failed to parse AI JSON response", "yellow")
            return {
                'confidence': 0,
                'recommendation': 'SKIP',
                'reasoning': 'JSON parse error',
                'risk_factors': ['Response parsing failed'],
                'suggested_sl_pct': SNIPER_STOP_LOSS_PCT,
                'suggested_tp_pct': SNIPER_TAKE_PROFIT_PCT
            }

    # =========================================================================
    # SETUP DETECTION
    # =========================================================================

    def detect_capitulation_fade(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Detect capitulation fade setup (LONG entry).

        Conditions:
        - Price drops more than adaptive threshold (2.5σ) in <4h
        - RSI < adaptive oversold threshold
        - Market is ranging (ADX < 25)
        """
        try:
            # Get adaptive thresholds for this symbol
            move_threshold = self.get_move_threshold(symbol)
            rsi_oversold, _ = self.get_rsi_thresholds(symbol)

            # Check market regime (skip if trending) - uses adaptive ADX threshold
            regime = self.check_market_regime(df, symbol)
            if not regime['is_ranging']:
                return {
                    'detected': False,
                    'type': 'capitulation_fade',
                    'direction': 'BUY',
                    'skip_reason': regime['skip_reason'],
                    'adx': regime['adx'],
                    'adx_threshold': regime.get('threshold', SNIPER_ADX_TRENDING_THRESHOLD)
                }

            # Check 4h price drop
            lookback = min(16, len(df))
            price_4h_ago = df['close'].iloc[-lookback]
            current_price = df['close'].iloc[-1]
            price_change = (current_price - price_4h_ago) / price_4h_ago * 100

            # Check RSI
            rsi = RSIIndicator(df['close'], window=14).rsi().iloc[-1]

            # Adaptive detection with adaptive RSI threshold
            is_capitulation = (
                price_change <= -move_threshold and
                rsi <= rsi_oversold
            )

            # Log near-misses for debugging
            if not is_capitulation and (price_change <= -move_threshold * 0.7 or rsi <= rsi_oversold + 10):
                cprint(f"  [{symbol}] Near-capitulation: {price_change:+.2f}% (need <-{move_threshold:.1f}%), RSI={rsi:.1f} (need <{rsi_oversold})", "yellow")

            return {
                'detected': is_capitulation,
                'type': 'capitulation_fade',
                'direction': 'BUY',
                'price_change_4h': round(price_change, 2),
                'rsi': round(rsi, 1),
                'rsi_threshold': rsi_oversold,
                'current_price': float(current_price),
                'threshold_used': move_threshold,
                'adx': regime['adx']
            }

        except Exception as e:
            cprint(f"[Sniper] Error detecting capitulation: {e}", "yellow")
            return {'detected': False, 'type': 'capitulation_fade', 'direction': 'BUY'}

    def detect_euphoria_fade(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Detect euphoria fade setup (SHORT entry).

        Conditions:
        - Price rises more than adaptive threshold (2.5σ) in <4h
        - RSI > adaptive overbought threshold
        - Market is ranging (ADX < 25)
        """
        try:
            # Get adaptive thresholds for this symbol
            move_threshold = self.get_move_threshold(symbol)
            _, rsi_overbought = self.get_rsi_thresholds(symbol)

            # Check market regime (skip if trending) - uses adaptive ADX threshold
            regime = self.check_market_regime(df, symbol)
            if not regime['is_ranging']:
                return {
                    'detected': False,
                    'type': 'euphoria_fade',
                    'direction': 'SELL',
                    'skip_reason': regime['skip_reason'],
                    'adx': regime['adx'],
                    'adx_threshold': regime.get('threshold', SNIPER_ADX_TRENDING_THRESHOLD)
                }

            lookback = min(16, len(df))
            price_4h_ago = df['close'].iloc[-lookback]
            current_price = df['close'].iloc[-1]
            price_change = (current_price - price_4h_ago) / price_4h_ago * 100

            rsi = RSIIndicator(df['close'], window=14).rsi().iloc[-1]

            # Adaptive detection with adaptive RSI threshold
            is_euphoria = (
                price_change >= move_threshold and
                rsi >= rsi_overbought
            )

            # Log near-misses for debugging
            if not is_euphoria and (price_change >= move_threshold * 0.7 or rsi >= rsi_overbought - 10):
                cprint(f"  [{symbol}] Near-euphoria: {price_change:+.2f}% (need >+{move_threshold:.1f}%), RSI={rsi:.1f} (need >{rsi_overbought})", "yellow")

            return {
                'detected': is_euphoria,
                'type': 'euphoria_fade',
                'direction': 'SELL',
                'price_change_4h': round(price_change, 2),
                'rsi': round(rsi, 1),
                'rsi_threshold': rsi_overbought,
                'current_price': float(current_price),
                'threshold_used': move_threshold,
                'adx': regime['adx']
            }

        except Exception as e:
            cprint(f"[Sniper] Error detecting euphoria: {e}", "yellow")
            return {'detected': False, 'type': 'euphoria_fade', 'direction': 'SELL'}

    # =========================================================================
    # MAIN SIGNAL GENERATION
    # =========================================================================

    def run_checklist(self, symbol: str, df: pd.DataFrame, setup_type: str, direction: str) -> dict:
        """Run all 7 checklist conditions and return comprehensive result."""
        cprint(f"\n[Sniper] Running 7-point checklist for {symbol} ({setup_type})...", "cyan")

        # Run all checks
        extreme_move = self.check_extreme_move(symbol, df)
        funding_divergence = self.check_funding_divergence(symbol, direction)
        liquidation_cascade = self.check_liquidation_cascade(direction)
        multi_tf = self.check_multi_tf_agreement(symbol, direction)
        volume_climax = self.check_volume_climax(symbol, df)
        time_window = self.check_time_window()

        # Count passed conditions (first 6)
        passed_count = sum([
            extreme_move['passed'],
            funding_divergence['passed'],
            liquidation_cascade['passed'],
            multi_tf['passed'],
            volume_climax['passed'],
            time_window['passed']
        ])

        # Log checklist results
        self._log_checklist_results(symbol, {
            'extreme_move': extreme_move,
            'funding_divergence': funding_divergence,
            'liquidation_cascade': liquidation_cascade,
            'multi_tf': multi_tf,
            'volume_climax': volume_climax,
            'time_window': time_window
        })

        # Display results
        def status(passed): return "[PASS]" if passed else "[FAIL]"
        def color(passed): return "green" if passed else "red"

        cprint(f"  {status(extreme_move['passed'])} 1. Extreme Move: sigma={extreme_move['sigma']}", color(extreme_move['passed']))
        cprint(f"  {status(funding_divergence['passed'])} 2. Funding Divergence: z={funding_divergence['funding_zscore']}", color(funding_divergence['passed']))
        cprint(f"  {status(liquidation_cascade['passed'])} 3. Liquidation Cascade: ratio={liquidation_cascade['ratio']}", color(liquidation_cascade['passed']))
        cprint(f"  {status(multi_tf['passed'])} 4. Multi-TF Agreement: {multi_tf['agreements']}/{multi_tf['total']}", color(multi_tf['passed']))
        cprint(f"  {status(volume_climax['passed'])} 5. Volume Climax: peak={volume_climax['peak_ratio']}x", color(volume_climax['passed']))
        cprint(f"  {status(time_window['passed'])} 6. Time Window: {time_window['session']}", color(time_window['passed']))

        # Calculate pre-AI weighted score
        checks = {
            'extreme_move': extreme_move,
            'funding_divergence': funding_divergence,
            'liquidation_cascade': liquidation_cascade,
            'multi_tf': multi_tf,
            'volume_climax': volume_climax,
            'time_window': time_window
        }

        # Run AI validation if enough conditions pass (weighted mode: score >= 5.0, binary mode: 6/6)
        ai_validation = {'passed': False, 'confidence': 0, 'reasoning': 'Preliminary conditions not met'}
        pre_ai_score = self.calculate_weighted_score(checks, {'passed': False})

        # In weighted mode, run AI if pre-score >= 5.0 (about half)
        # In binary mode, only run if all 6 pass
        min_for_ai = 5.0 if SNIPER_USE_WEIGHTED_SCORING else 6.0
        run_ai = (SNIPER_USE_WEIGHTED_SCORING and pre_ai_score >= min_for_ai) or (not SNIPER_USE_WEIGHTED_SCORING and passed_count == 6)

        if run_ai:
            cprint(f"\n  Pre-AI score: {pre_ai_score}/10 - Running AI validation...", "cyan")
            rsi = RSIIndicator(df['close'], window=14).rsi().iloc[-1]

            setup = {
                'type': setup_type,
                'direction': direction,
                'current_price': float(df['close'].iloc[-1]),
                'rsi': round(rsi, 1),
                'extreme_move': extreme_move,
                'funding_divergence': funding_divergence,
                'liquidation_cascade': liquidation_cascade,
                'multi_tf': multi_tf,
                'volume_climax': volume_climax,
                'time_window': time_window
            }

            ai_validation = self.check_ai_validation(symbol, setup)
            cprint(f"  {status(ai_validation['passed'])} 7. AI Validation: {ai_validation['confidence']}% confidence", color(ai_validation['passed']))
            cprint(f"     Reasoning: {ai_validation['reasoning'][:100]}...", "white")
        else:
            cprint(f"\n  Pre-AI score: {pre_ai_score}/10 - skipping AI validation (need >= {min_for_ai})", "yellow")

        # Calculate final weighted score
        weighted_score = self.calculate_weighted_score(checks, ai_validation)

        # Final result - use weighted scoring if enabled
        if SNIPER_USE_WEIGHTED_SCORING:
            all_passed = weighted_score >= SNIPER_MIN_WEIGHTED_SCORE
            cprint(f"\n  Final weighted score: {weighted_score}/10 (threshold: {SNIPER_MIN_WEIGHTED_SCORE})", "cyan" if all_passed else "yellow")

            # Log near-misses for monitoring (scores between 6.0 and threshold)
            near_miss_threshold = 6.0
            if near_miss_threshold <= weighted_score < SNIPER_MIN_WEIGHTED_SCORE:
                vol_info = self.volatility_thresholds.get(symbol, {})
                vol_ratio = vol_info.get('vol_ratio_7d_30d', 1.0)
                multiplier = vol_info.get('threshold_multiplier', 1.0)
                cprint(f"\n  [NEAR-MISS] {symbol} {direction}: {weighted_score:.1f}/10 (needs {SNIPER_MIN_WEIGHTED_SCORE})", "yellow")
                cprint(f"    Vol ratio: {vol_ratio:.2f} | Multiplier: {multiplier:.2f} | Extreme: {extreme_move['sigma']:.1f}σ | Funding: {funding_divergence['funding_zscore']:.1f}", "yellow")
        else:
            all_passed = passed_count == 6 and ai_validation['passed']

        return {
            'all_passed': all_passed,
            'passed_count': passed_count + (1 if ai_validation['passed'] else 0),
            'total': 7,
            'weighted_score': weighted_score,
            'type': setup_type,
            'direction': direction,
            'current_price': float(df['close'].iloc[-1]),
            'extreme_move': extreme_move,
            'funding_divergence': funding_divergence,
            'liquidation_cascade': liquidation_cascade,
            'multi_tf': multi_tf,
            'volume_climax': volume_climax,
            'time_window': time_window,
            'ai_validation': ai_validation
        }

    def _log_checklist_results(self, symbol: str, checks: dict):
        """Log checklist results to CSV for analysis."""
        try:
            log_file = os.path.join(self.data_dir, 'checklist_log.csv')

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'extreme_move_passed': checks['extreme_move']['passed'],
                'extreme_move_sigma': checks['extreme_move']['sigma'],
                'funding_passed': checks['funding_divergence']['passed'],
                'funding_zscore': checks['funding_divergence']['funding_zscore'],
                'liquidation_passed': checks['liquidation_cascade']['passed'],
                'liquidation_ratio': checks['liquidation_cascade']['ratio'],
                'mtf_passed': checks['multi_tf']['passed'],
                'mtf_agreements': checks['multi_tf']['agreements'],
                'volume_passed': checks['volume_climax']['passed'],
                'volume_peak_ratio': checks['volume_climax']['peak_ratio'],
                'time_passed': checks['time_window']['passed'],
                'time_session': checks['time_window']['session']
            }

            df = pd.DataFrame([log_entry])

            if os.path.exists(log_file):
                df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(log_file, index=False)

        except Exception as e:
            cprint(f"[Sniper] Error logging checklist: {e}", "yellow")

    def generate_signals(self, symbol: str = None, df: pd.DataFrame = None) -> dict:
        """
        Generate trading signal for the given symbol.

        This is the main entry point - implements BaseStrategy interface.
        """
        # Reset daily counters if new day
        self._reset_daily_counters()

        # Check daily limits
        if self.daily_trades >= SNIPER_MAX_DAILY_TRADES:
            cprint(f"[Sniper] Daily trade limit reached ({SNIPER_MAX_DAILY_TRADES})", "yellow")
            return None

        if self.daily_pnl <= -SNIPER_MAX_DAILY_LOSS_USD:
            cprint(f"[Sniper] Daily loss limit reached (${SNIPER_MAX_DAILY_LOSS_USD})", "red")
            return None

        if self.daily_pnl >= SNIPER_MAX_DAILY_GAIN_USD:
            cprint(f"[Sniper] Daily gain limit reached (${SNIPER_MAX_DAILY_GAIN_USD})", "green")
            return None

        # If no symbol provided, iterate through assets
        if symbol is None:
            for asset in self.assets:
                try:
                    asset_df = self._fetch_candles(asset, interval='15m', candles=300)
                    if asset_df is None or len(asset_df) < 100:
                        continue

                    signal = self._generate_signal_for_asset(asset, asset_df)
                    if signal and signal['direction'] != 'NEUTRAL':
                        return signal

                except Exception as e:
                    cprint(f"[Sniper] Error processing {asset}: {e}", "yellow")
                    continue

            return None

        # Generate signal for specific symbol
        # Skip if symbol is not in our tracked assets
        if symbol not in self.assets:
            return None

        if df is None:
            df = self._fetch_candles(symbol, interval='15m', candles=300)

        if df is None or len(df) < 100:
            return None

        return self._generate_signal_for_asset(symbol, df)

    def _generate_signal_for_asset(self, symbol: str, df: pd.DataFrame) -> dict:
        """Generate signal for a specific asset using 7-point checklist."""
        try:
            cprint(f"\n{'='*60}", "cyan")
            cprint(f"[Sniper] Analyzing {symbol}...", "cyan", attrs=['bold'])
            cprint(f"{'='*60}", "cyan")

            # First, detect potential setups
            capitulation = self.detect_capitulation_fade(symbol, df)
            euphoria = self.detect_euphoria_fade(symbol, df)
            funding_arb = self.detect_funding_arbitrage(symbol, df)

            # Track checklist results for partial data in NEUTRAL signals
            last_checklist_result = None
            detected_setup = None

            # Check for capitulation fade (LONG)
            if capitulation['detected']:
                cprint(f"[Sniper] Capitulation fade detected! Running full checklist...", "green")
                result = self.run_checklist(symbol, df, 'capitulation_fade', 'BUY')
                last_checklist_result = result
                detected_setup = 'capitulation_fade'

                if result['all_passed']:
                    return self._build_signal(symbol, result)

            # Check for euphoria fade (SHORT)
            if euphoria['detected']:
                cprint(f"[Sniper] Euphoria fade detected! Running full checklist...", "red")
                result = self.run_checklist(symbol, df, 'euphoria_fade', 'SELL')
                last_checklist_result = result
                detected_setup = 'euphoria_fade'

                if result['all_passed']:
                    return self._build_signal(symbol, result)

            # Check for funding arbitrage (Setup #3)
            if funding_arb['detected']:
                cprint(f"[Sniper] Funding arbitrage detected! Funding={funding_arb.get('funding_rate', 0)}%", "magenta")
                result = self.run_checklist(symbol, df, 'funding_arbitrage', funding_arb['direction'])
                last_checklist_result = result
                detected_setup = 'funding_arbitrage'

                if result['all_passed']:
                    return self._build_signal(symbol, result)

            # Collect diagnostic info for NEUTRAL signals
            current_price = float(df['close'].iloc[-1]) if df is not None and len(df) > 0 else 0
            skip_reasons_set = set()  # Use set to avoid duplicates
            market_state = {}

            # Collect all skip reasons and market state info (deduplicated)
            if capitulation.get('skip_reason'):
                skip_reasons_set.add(capitulation['skip_reason'])
            if euphoria.get('skip_reason'):
                skip_reasons_set.add(euphoria['skip_reason'])

            # Add near-miss info if available
            if capitulation.get('near_miss'):
                skip_reasons_set.add(f"Near-capitulation: {capitulation.get('near_miss')}")
            if euphoria.get('near_miss'):
                skip_reasons_set.add(f"Near-euphoria: {euphoria.get('near_miss')}")

            # Get current market indicators for diagnostics
            try:
                regime = self.check_market_regime(df, symbol)
                market_state['adx'] = regime.get('adx', 0)
                market_state['adx_threshold'] = regime.get('threshold', SNIPER_ADX_TRENDING_THRESHOLD)
                market_state['is_trending'] = not regime.get('is_ranging', True)

                # Get RSI
                rsi = RSIIndicator(df['close'], window=14).rsi().iloc[-1]
                market_state['rsi'] = round(rsi, 1)
                rsi_low, rsi_high = self.get_rsi_thresholds(symbol)
                market_state['rsi_oversold'] = rsi_low
                market_state['rsi_overbought'] = rsi_high

                # Get price change (24h approx with 15m candles = 96 bars)
                if len(df) >= 96:
                    price_change_24h = ((df['close'].iloc[-1] / df['close'].iloc[-96]) - 1) * 100
                    market_state['price_change_24h'] = round(price_change_24h, 2)
            except Exception:
                pass

            # Convert set to list for output
            skip_reasons = list(skip_reasons_set)

            # No valid setup detected - log and build detailed reason
            if not capitulation['detected'] and not euphoria['detected'] and not funding_arb['detected']:
                if skip_reasons:
                    cprint(f"  [{symbol}] Skipped: {skip_reasons[0]}", "yellow")
                else:
                    skip_reasons.append("No capitulation or euphoria pattern detected")
                    cprint(f"[Sniper] No setup detected for {symbol}", "white")

            # Build detailed NEUTRAL signal
            # If we ran a checklist but it didn't pass, include that data
            if last_checklist_result:
                checklist_details = self._build_checklist_details(last_checklist_result)
                ai_val = last_checklist_result.get('ai_validation', {})
                return {
                    'token': symbol,
                    'signal': round(ai_val.get('confidence', 0) / 100, 3),
                    'direction': 'NEUTRAL',
                    'metadata': {
                        'strategy_type': 'sniper_ai',
                        'setup_type': detected_setup,
                        'current_price': last_checklist_result.get('current_price', current_price),
                        'checklist_score': f"{last_checklist_result['passed_count']}/{last_checklist_result['total']}",
                        'weighted_score': round(last_checklist_result.get('weighted_score', 0), 1),
                        'ai_confidence': ai_val.get('confidence', 0),
                        'ai_reasoning': ai_val.get('reasoning', 'Checklist threshold not met'),
                        'reason': f"Score {last_checklist_result.get('weighted_score', 0):.1f}/10 < {SNIPER_MIN_WEIGHTED_SCORE} threshold",
                        'checklist_details': checklist_details,
                        'market_state': market_state,
                    }
                }

            # No setup detected - include diagnostic info
            reason = skip_reasons[0] if skip_reasons else "No setup detected"
            return {
                'token': symbol,
                'signal': 0.0,
                'direction': 'NEUTRAL',
                'metadata': {
                    'strategy_type': 'sniper_ai',
                    'current_price': current_price,
                    'reason': reason,
                    'skip_reasons': skip_reasons,
                    'market_state': market_state,
                    'setup_type': None,
                    'checklist_score': '0/7',
                    'weighted_score': 0,
                }
            }

        except Exception as e:
            cprint(f"[Sniper] Error generating signal for {symbol}: {e}", "red")
            import traceback
            traceback.print_exc()
            return None

    def _build_signal(self, symbol: str, result: dict) -> dict:
        """Build the signal dict from checklist result."""
        ai_val = result.get('ai_validation', {})

        # Build detailed checklist for frontend display
        checklist_details = self._build_checklist_details(result)

        signal = {
            'token': symbol,
            'signal': round(ai_val.get('confidence', 85) / 100, 3),
            'direction': result['direction'],
            'metadata': {
                'strategy_type': 'sniper_ai',
                'setup_type': result['type'],
                'current_price': result['current_price'],
                'checklist_score': f"{result['passed_count']}/{result['total']}",
                'weighted_score': round(result.get('weighted_score', 0), 1),
                'ai_confidence': ai_val.get('confidence', 0),
                'ai_reasoning': ai_val.get('reasoning', ''),
                'stop_loss_pct': ai_val.get('suggested_sl_pct', SNIPER_STOP_LOSS_PCT),
                'take_profit_pct': ai_val.get('suggested_tp_pct', SNIPER_TAKE_PROFIT_PCT),
                'leverage': SNIPER_LEVERAGE,
                'extreme_move': result['extreme_move'],
                'funding_divergence': result['funding_divergence'],
                'time_window': result['time_window'],
                'checklist_details': checklist_details,
            }
        }

        # Log signal
        self._log_signal(signal)

        # Display signal
        cprint("\n" + "="*60, "magenta")
        cprint(f"[SNIPER SIGNAL] {result['direction']} {symbol} @ ${result['current_price']:,.2f}", "magenta", attrs=['bold'])
        cprint(f"  Setup: {result['type']}", "white")
        cprint(f"  Checklist: {result['passed_count']}/{result['total']} passed", "white")
        cprint(f"  AI Confidence: {ai_val.get('confidence', 0)}%", "white")
        cprint(f"  AI Reasoning: {ai_val.get('reasoning', '')[:80]}...", "white")
        cprint("="*60, "magenta")

        return signal

    def _log_signal(self, signal: dict):
        """Log signal to CSV file."""
        try:
            log_file = os.path.join(self.data_dir, 'signals.csv')
            metadata = signal.get('metadata', {})

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': signal['token'],
                'direction': signal['direction'],
                'confidence': signal['signal'],
                'price': metadata.get('current_price', 0),
                'setup_type': metadata.get('setup_type', 'unknown'),
                'checklist_score': metadata.get('checklist_score', '0/7'),
                'ai_confidence': metadata.get('ai_confidence', 0),
                'sl_pct': metadata.get('stop_loss_pct', SNIPER_STOP_LOSS_PCT),
                'tp_pct': metadata.get('take_profit_pct', SNIPER_TAKE_PROFIT_PCT)
            }

            df = pd.DataFrame([log_entry])

            if os.path.exists(log_file):
                df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(log_file, index=False)

        except Exception as e:
            cprint(f"[Sniper] Error logging signal: {e}", "yellow")

    def _build_checklist_details(self, result: dict) -> dict:
        """Build checklist details dict from checklist result for frontend display."""
        ai_val = result.get('ai_validation', {})
        return {
            'extreme_move': {
                'passed': result.get('extreme_move', {}).get('passed', False),
                'sigma': round(result.get('extreme_move', {}).get('sigma', 0), 2),
                'threshold': result.get('extreme_move', {}).get('threshold', SNIPER_SIGMA_THRESHOLD),
            },
            'funding_divergence': {
                'passed': result.get('funding_divergence', {}).get('passed', False),
                'zscore': round(result.get('funding_divergence', {}).get('funding_zscore', 0), 2),
                'is_contrarian': result.get('funding_divergence', {}).get('is_contrarian', False),
            },
            'liquidation_cascade': {
                'passed': result.get('liquidation_cascade', {}).get('passed', False),
                'ratio': round(result.get('liquidation_cascade', {}).get('ratio', 0), 2),
            },
            'multi_tf': {
                'passed': result.get('multi_tf', {}).get('passed', False),
                'agreement': result.get('multi_tf', {}).get('agreement_pct', 0),
            },
            'volume_climax': {
                'passed': result.get('volume_climax', {}).get('passed', False),
                'ratio': round(result.get('volume_climax', {}).get('volume_ratio', 0), 2),
            },
            'time_window': {
                'passed': result.get('time_window', {}).get('passed', False),
                'hour_utc': result.get('time_window', {}).get('current_hour', 0),
            },
            'ai_validation': {
                'passed': ai_val.get('passed', False),
                'confidence': ai_val.get('confidence', 0),
                'reasoning': ai_val.get('reasoning', ''),
            },
        }

    # =========================================================================
    # PAPER TRADING
    # =========================================================================

    def execute_paper_trade(self, signal: dict) -> dict:
        """Execute a paper trade (simulation)."""
        if not PAPER_TRADING:
            cprint("[Sniper] Paper trading disabled", "yellow")
            return None

        try:
            symbol = signal.get('token', '')
            direction = signal.get('direction', 'NEUTRAL')

            if not symbol or direction == 'NEUTRAL':
                return None

            metadata = signal.get('metadata', {})
            price = metadata.get('current_price', 0)

            if price == 0:
                df = self._fetch_candles(symbol, interval='15m', candles=5)
                if df is not None and len(df) > 0:
                    price = float(df['close'].iloc[-1])

            if price <= 0:
                cprint(f"[Sniper] Cannot execute trade with price={price}", "red")
                return None

            sl_pct = metadata.get('stop_loss_pct', SNIPER_STOP_LOSS_PCT)
            tp_pct = metadata.get('take_profit_pct', SNIPER_TAKE_PROFIT_PCT)

            # Calculate margin
            used_margin = sum(
                pos.get('position_size', 0) / pos.get('leverage', SNIPER_LEVERAGE)
                for pos in self.paper_positions.values()
            )
            available_margin = max(0, self.paper_balance - used_margin)

            # Position sizing: 3% risk per sniper trade
            risk_amount = self.paper_balance * 0.03
            position_size = risk_amount / (sl_pct / 100) * SNIPER_LEVERAGE

            max_position = available_margin * 0.9 * SNIPER_LEVERAGE
            position_size = min(position_size, max_position)

            # Apply confidence-based sizing (7. Dynamic position sizing)
            ai_confidence = metadata.get('ai_confidence', 85)
            confidence_factor = self.get_confidence_size_factor(ai_confidence)
            position_size *= confidence_factor

            # Apply correlation-based sizing (4. Correlation filter)
            correlation_factor = self.calculate_correlation_factor(symbol)
            position_size *= correlation_factor

            # Log sizing adjustments
            if confidence_factor < 1.0 or correlation_factor < 1.0:
                cprint(f"[Sniper] Position size adjusted: confidence={confidence_factor:.0%}, correlation={correlation_factor:.0%}", "yellow")

            if position_size < 10:
                cprint(f"[Sniper] Insufficient margin", "red")
                return None

            # Generate position ID
            self._position_counter += 1
            position_id = f"SNIPER_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._position_counter}"

            # Calculate SL/TP prices
            if direction == 'BUY':
                stop_loss_price = price * (1 - sl_pct / 100)
                take_profit_price = price * (1 + tp_pct / 100)
            else:
                stop_loss_price = price * (1 + sl_pct / 100)
                take_profit_price = price * (1 - tp_pct / 100)

            trade = {
                'position_id': position_id,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'direction': direction,
                'entry_price': price,
                'position_size': round(position_size, 2),
                'leverage': SNIPER_LEVERAGE,
                'stop_loss': round(stop_loss_price, 2),
                'take_profit': round(take_profit_price, 2),
                'sl_pct': sl_pct,
                'tp_pct': tp_pct,
                'confidence': metadata.get('ai_confidence', 0),
                'status': 'OPEN',
                'setup_type': metadata.get('setup_type', 'unknown'),
                'checklist_score': metadata.get('checklist_score', '0/7'),
            }

            self.paper_positions[position_id] = trade
            self.daily_trades += 1

            # Log to file
            log_file = os.path.join(self.data_dir, 'paper_trades.csv')
            df = pd.DataFrame([trade])

            if os.path.exists(log_file):
                df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(log_file, index=False)

            cprint(f"\n[SNIPER PAPER] Opened {direction} {symbol} (ID: {position_id})", "magenta", attrs=['bold'])
            cprint(f"  Entry: ${price:,.2f} | Size: ${position_size:,.2f}", "white")
            cprint(f"  SL: ${trade['stop_loss']:,.2f} ({sl_pct:.2f}%)", "white")
            cprint(f"  TP: ${trade['take_profit']:,.2f} ({tp_pct:.2f}%)", "white")

            return trade

        except Exception as e:
            cprint(f"[Sniper] Error executing paper trade: {e}", "red")
            import traceback
            traceback.print_exc()
            return None

    def monitor_paper_positions(self) -> list:
        """Monitor all open paper positions and close those that hit SL/TP (with ATR trailing)."""
        if not PAPER_TRADING or not self.paper_positions:
            return []

        closed = []

        symbols_to_check = set(pos['symbol'] for pos in self.paper_positions.values())
        symbol_data = {}  # Store both price and df for trailing stop calculation

        for symbol in symbols_to_check:
            try:
                # Fetch more candles for ATR calculation
                df = self._fetch_candles(symbol, interval='15m', candles=50)
                if df is not None and len(df) > 0:
                    symbol_data[symbol] = {
                        'price': float(df['close'].iloc[-1]),
                        'df': df
                    }
            except Exception as e:
                cprint(f"[Sniper] Could not fetch price for {symbol}: {e}", "yellow")

        positions_to_close = []

        for position_id, trade in self.paper_positions.items():
            symbol = trade['symbol']
            if symbol not in symbol_data:
                continue

            current_price = symbol_data[symbol]['price']
            df = symbol_data[symbol]['df']
            direction = trade['direction']
            entry_price = trade['entry_price']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']

            # Calculate ATR trailing stop (1. Trailing Stop-Loss based on ATR)
            trailing = self.calculate_atr_trailing_stop(symbol, df, direction, entry_price, current_price)

            # Update stop loss if trailing is active and provides better protection
            effective_stop = stop_loss
            if trailing['active'] and trailing['stop_price'] is not None:
                if direction == 'BUY' and trailing['stop_price'] > stop_loss:
                    effective_stop = trailing['stop_price']
                    cprint(f"  [{symbol}] Trailing SL activated: ${stop_loss:.2f} → ${effective_stop:.2f} (profit: {trailing['profit_pct']:.1f}%)", "cyan")
                elif direction == 'SELL' and trailing['stop_price'] < stop_loss:
                    effective_stop = trailing['stop_price']
                    cprint(f"  [{symbol}] Trailing SL activated: ${stop_loss:.2f} → ${effective_stop:.2f} (profit: {trailing['profit_pct']:.1f}%)", "cyan")

            close_reason = None

            if direction == 'BUY':
                if current_price <= effective_stop:
                    close_reason = 'TRAILING_STOP' if effective_stop != stop_loss else 'STOP_LOSS'
                elif current_price >= take_profit:
                    close_reason = 'TAKE_PROFIT'
            else:
                if current_price >= effective_stop:
                    close_reason = 'TRAILING_STOP' if effective_stop != stop_loss else 'STOP_LOSS'
                elif current_price <= take_profit:
                    close_reason = 'TAKE_PROFIT'

            if close_reason:
                positions_to_close.append((position_id, current_price, close_reason))

        for position_id, close_price, reason in positions_to_close:
            closed_trade = self._close_paper_position(position_id, close_price, reason)
            if closed_trade:
                closed.append(closed_trade)

        return closed

    def _close_paper_position(self, position_id: str, close_price: float, reason: str) -> dict:
        """Close a paper position and update PnL."""
        if position_id not in self.paper_positions:
            return None

        try:
            trade = self.paper_positions[position_id].copy()
            entry_price = trade['entry_price']
            direction = trade['direction']
            position_size = trade['position_size']

            if direction == 'BUY':
                price_change_pct = (close_price - entry_price) / entry_price
            else:
                price_change_pct = (entry_price - close_price) / entry_price

            pnl = position_size * price_change_pct

            trade['close_price'] = close_price
            trade['close_timestamp'] = datetime.now().isoformat()
            trade['close_reason'] = reason
            trade['pnl'] = round(pnl, 2)
            trade['pnl_pct'] = round(price_change_pct * 100, 2)
            trade['status'] = 'CLOSED'

            self.daily_pnl += pnl
            self.paper_balance += pnl

            del self.paper_positions[position_id]
            self.closed_positions.append(trade)

            color = 'green' if pnl > 0 else 'red'
            cprint(f"\n[SNIPER PAPER] Closed {trade['symbol']} ({reason})", color, attrs=['bold'])
            cprint(f"  Entry: ${entry_price:,.2f} -> Exit: ${close_price:,.2f}", "white")
            cprint(f"  PnL: ${pnl:+,.2f} ({price_change_pct*100:+.2f}%)", color)
            cprint(f"  Daily PnL: ${self.daily_pnl:+,.2f} | Balance: ${self.paper_balance:,.2f}", "white")

            # Log to closed trades file
            self._log_closed_trade(trade)

            # Update status in paper_trades.csv
            self._update_position_status_in_csv(position_id, trade)

            return trade

        except Exception as e:
            cprint(f"[Sniper] Error closing position {position_id}: {e}", "red")
            return None

    def _update_position_status_in_csv(self, position_id: str, trade: dict):
        """Update a position's status in paper_trades.csv when closed."""
        try:
            paper_trades_file = os.path.join(self.data_dir, 'paper_trades.csv')
            if not os.path.exists(paper_trades_file):
                return

            df = pd.read_csv(paper_trades_file)
            if df.empty:
                return

            mask = df['position_id'] == position_id
            if mask.any():
                df.loc[mask, 'status'] = trade.get('status', 'CLOSED')
                df.loc[mask, 'exit_price'] = trade.get('close_price', 0)
                df.loc[mask, 'exit_time'] = trade.get('close_timestamp', '')
                df.loc[mask, 'pnl'] = trade.get('pnl', 0)
                df.to_csv(paper_trades_file, index=False)

        except Exception as e:
            cprint(f"[Sniper] Warning: Could not update CSV status: {e}", "yellow")

    def _log_closed_trade(self, trade: dict):
        """Log closed trade to CSV file."""
        try:
            log_file = os.path.join(self.data_dir, 'closed_trades.csv')
            df = pd.DataFrame([trade])

            if os.path.exists(log_file):
                df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(log_file, index=False)

        except Exception as e:
            cprint(f"[Sniper] Error logging closed trade: {e}", "yellow")

    def get_paper_status(self) -> dict:
        """Get current paper trading status (for dashboard)."""
        return {
            'paper_balance': round(self.paper_balance, 2),
            'initial_balance': PAPER_TRADING_BALANCE,
            'total_pnl': round(self.paper_balance - PAPER_TRADING_BALANCE, 2),
            'daily_pnl': round(self.daily_pnl, 2),
            'daily_trades': self.daily_trades,
            'open_positions': len(self.paper_positions),
            'total_closed': len(self.closed_positions),
            'positions': list(self.paper_positions.values())
        }

    def close_all_paper_positions(self) -> list:
        """Force close all open paper positions at current market price."""
        if not self.paper_positions:
            return []

        closed = []
        position_ids = list(self.paper_positions.keys())

        for position_id in position_ids:
            trade = self.paper_positions[position_id]
            symbol = trade['symbol']

            try:
                df = self._fetch_candles(symbol, interval='15m', candles=5)
                if df is not None and len(df) > 0:
                    current_price = float(df['close'].iloc[-1])
                    closed_trade = self._close_paper_position(position_id, current_price, 'MANUAL')
                    if closed_trade:
                        closed.append(closed_trade)
            except Exception as e:
                cprint(f"[Sniper] Error closing {position_id}: {e}", "red")

        return closed


# For standalone testing
if __name__ == "__main__":
    cprint("="*60, "cyan")
    cprint("  Testing Sniper AI Strategy", "cyan", attrs=['bold'])
    cprint("="*60, "cyan")

    strategy = SniperAIStrategy()

    for symbol in ['BTC', 'ETH', 'SOL']:
        try:
            cprint(f"\nAnalyzing {symbol}...", "white", attrs=['bold'])

            df = strategy._fetch_candles(symbol, interval='15m', candles=300)

            if df is not None and len(df) >= 100:
                signal = strategy.generate_signals(symbol, df)

                if signal:
                    if signal['direction'] != 'NEUTRAL':
                        cprint(f"Signal generated!", "green")
                    else:
                        cprint(f"No actionable signal", "white")
            else:
                cprint(f"Not enough data for {symbol}", "yellow")

        except Exception as e:
            cprint(f"Error testing {symbol}: {e}", "red")

    cprint("\nTest completed!", "cyan")
