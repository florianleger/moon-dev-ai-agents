"""
Adaptive Hybrid Strategy

Multi-module scoring strategy that aggregates 14 independent signal generators.
Instead of requiring ALL conditions (AND logic), it requires enough convergence
(weighted score > threshold) for a trade signal.

Modules:
1. Mean Reversion (Bollinger Bands + RSI)
2. Momentum Breakout (range breakout + volume)
3. EMA Trend (EMA alignment + ADX + MACD)
4. Funding Rate Contrarian (extreme funding Z-score)
5. RSI Divergence (pivot-based price vs RSI divergence)
6. Sniper Lite (extreme move + funding, 50-bar / 2-sigma)
7. RAMF Lite (volatility regime + exhaustion)
8. OI Delta (Open Interest positioning pressure)
9. Sentiment (Fear & Greed + Twitter contrarian)
10. Squeeze Detector (funding + OI + volatility compression)
11. Order Imbalance (HyperLiquid L2 bid/ask depth)
12. Crowd Positioning (Binance L/S ratio + Taker buy/sell volume)
13. Social Hype (CoinGecko trending + global market macro)
14. Funding Divergence (Cross-exchange HL vs Binance funding rates)

Target: 1-3 trades/day with 55%+ win rate.
"""

import os
import json
import threading
import pandas as pd
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from termcolor import cprint
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volume import VolumeWeightedAveragePrice

from ..base_strategy import BaseStrategy
from src.data.trade_memory import TradeMemory

# Scoring modules (extracted from monolithic methods)
from src.strategies.modules.mean_reversion import score_mean_reversion
from src.strategies.modules.momentum import score_momentum_breakout
from src.strategies.modules.ema_trend import score_ema_trend
from src.strategies.modules.rsi_divergence import score_rsi_divergence
from src.strategies.modules.sniper_lite import score_sniper_lite
from src.strategies.modules.ramf_lite import score_ramf_lite
from src.strategies.modules.oi_delta import score_oi_delta
from src.strategies.modules.sentiment import score_sentiment
from src.strategies.modules.squeeze import score_squeeze_detector
from src.strategies.modules.order_imbalance import score_order_imbalance
from src.strategies.modules.crowd_positioning import score_crowd_positioning
from src.strategies.modules.funding_composite import score_funding_composite

# LLM-enhanced pipeline modules
from src.strategies.modules.llm_confirmation import llm_confirm_trade
from src.strategies.modules.llm_regime import classify_regime, adjust_weights_for_regime
from src.strategies.modules.trade_learner import analyze_closed_trade
from src.strategies.modules.mtf_confluence import score_mtf_confluence

# New scoring modules (Phase 2)
from src.strategies.modules.liquidation_cascade import score_liquidation_cascade
from src.strategies.modules.cvd import score_cvd
from src.strategies.modules.vwap_deviation import score_vwap_deviation
from src.strategies.modules.anomaly_filter import observe as anomaly_observe, is_anomalous

# ML infrastructure (Phase 3)
from src.strategies.modules.adaptive_weights import BayesianWeightOptimizer
from src.strategies.modules.quantitative_feedback import QuantitativeFeedback

# Import config with defaults
try:
    from src.config import (
        PAPER_TRADING,
        PAPER_TRADING_BALANCE,
        ADAPTIVE_HYBRID_BASE_THRESHOLD,
        ADAPTIVE_HYBRID_URGENCY_START_HOURS,
        ADAPTIVE_HYBRID_URGENCY_FLOOR,
        ADAPTIVE_HYBRID_MAX_DAILY_TRADES,
        ADAPTIVE_HYBRID_MAX_DAILY_LOSS_USD,
        ADAPTIVE_HYBRID_LEVERAGE,
        ADAPTIVE_HYBRID_ATR_SL_MULT,
        ADAPTIVE_HYBRID_ATR_TP_MULT,
        ADAPTIVE_HYBRID_WEIGHTS,
        ADAPTIVE_HYBRID_MAX_POSITION_PCT,
        ADAPTIVE_HYBRID_MIN_CONVERGENT_MODULES,
        ADAPTIVE_HYBRID_MIN_RR_RATIO,
        ADAPTIVE_HYBRID_RESET_PAPER,
        ADAPTIVE_HYBRID_ATR_PROFILES,
        ADAPTIVE_HYBRID_WEIGHT_PROFILES,
        ADAPTIVE_HYBRID_RANGING_TOKENS,
        ADAPTIVE_HYBRID_TRENDING_TOKENS,
        SNIPER_ASSETS,
        ADAPTIVE_HYBRID_MAX_HOLD_HOURS,
        ADAPTIVE_HYBRID_HOLD_TP_CHECK_HOURS,
        ADAPTIVE_HYBRID_LEVERAGE_PROFILES,
        RISK_MAX_DRAWDOWN_PCT,
        CASH_PERCENTAGE,
        ADAPTIVE_HYBRID_OPTIMAL_HOURS,
        ADAPTIVE_HYBRID_AVOID_HOURS,
        REGIME_ADX_TRENDING,
        REGIME_ADX_RANGING,
        REGIME_VOL_HIGH,
        REGIME_VOL_LOW,
        ADAPTIVE_HYBRID_TRAILING_LEVELS,
        ADAPTIVE_HYBRID_SCALE_OUT_LEVELS,
        ADAPTIVE_HYBRID_VOL_TARGET_DAILY_PCT,
        ADAPTIVE_HYBRID_VOL_MIN_POSITION_USD,
        ADAPTIVE_HYBRID_USE_REALTIME_PRICE,
        PAPER_SLIPPAGE_V2,
        PAPER_TAKER_FEE_V2,
        ADAPTIVE_HYBRID_USE_BAYESIAN_WEIGHTS,
        ADAPTIVE_HYBRID_BAYESIAN_MIN_TRADES,
        ADAPTIVE_HYBRID_BAYESIAN_DECAY,
        ADAPTIVE_HYBRID_USE_ANOMALY_FILTER,
        ADAPTIVE_HYBRID_ANOMALY_SCORE_DIVISOR,
    )
except ImportError:
    PAPER_TRADING = True
    PAPER_TRADING_BALANCE = 500
    ADAPTIVE_HYBRID_BASE_THRESHOLD = 42
    ADAPTIVE_HYBRID_URGENCY_START_HOURS = 4
    ADAPTIVE_HYBRID_URGENCY_FLOOR = 35
    ADAPTIVE_HYBRID_MAX_DAILY_TRADES = 5
    ADAPTIVE_HYBRID_MAX_DAILY_LOSS_USD = 30
    ADAPTIVE_HYBRID_LEVERAGE = 3
    ADAPTIVE_HYBRID_ATR_SL_MULT = 2.8
    ADAPTIVE_HYBRID_ATR_TP_MULT = 4.2
    ADAPTIVE_HYBRID_MAX_POSITION_PCT = 25
    ADAPTIVE_HYBRID_MIN_CONVERGENT_MODULES = 2
    ADAPTIVE_HYBRID_MIN_RR_RATIO = 1.5
    ADAPTIVE_HYBRID_RESET_PAPER = True
    ADAPTIVE_HYBRID_ATR_PROFILES = {
        'btc': {'sl_mult': 2.8, 'tp_mult': 4.2, 'tokens': ['BTC']},
        'eth': {'sl_mult': 4.0, 'tp_mult': 8.0, 'tokens': ['ETH']},
        'mid': {'sl_mult': 2.2, 'tp_mult': 3.3, 'tokens': ['SOL', 'XRP', 'AVAX', 'LINK', 'ADA', 'AAVE', 'NEAR', 'SUI', 'TAO']},
        'alt': {'sl_mult': 1.8, 'tp_mult': 2.7, 'tokens': ['DOGE', 'kPEPE', 'ENA']},
    }
    ADAPTIVE_HYBRID_WEIGHTS = {
        'mean_reversion': 0.08, 'momentum_breakout': 0.06,
        'ema_trend': 0.06, 'funding_composite': 0.08,
        'rsi_divergence': 0.06, 'sniper_lite': 0.08,
        'ramf_lite': 0.05,
        'oi_delta': 0.05, 'sentiment': 0.04,
        'squeeze_detector': 0.04, 'order_imbalance': 0.04,
        'crowd_positioning': 0.06,
        'cvd': 0.05, 'vwap_deviation': 0.05,
        'liquidation_cascade': 0.04,
    }
    ADAPTIVE_HYBRID_WEIGHT_PROFILES = {
        'ranging': {
            'mean_reversion': 0.11, 'momentum_breakout': 0.04, 'ema_trend': 0.05,
            'funding_composite': 0.08, 'rsi_divergence': 0.06, 'sniper_lite': 0.09,
            'ramf_lite': 0.05,
            'oi_delta': 0.05, 'sentiment': 0.04, 'squeeze_detector': 0.04,
            'order_imbalance': 0.04, 'crowd_positioning': 0.06,
            'cvd': 0.05, 'vwap_deviation': 0.05,
            'liquidation_cascade': 0.04,
        },
        'trending': {
            'mean_reversion': 0.04, 'momentum_breakout': 0.09, 'ema_trend': 0.06,
            'funding_composite': 0.07, 'rsi_divergence': 0.04, 'sniper_lite': 0.08,
            'ramf_lite': 0.05,
            'oi_delta': 0.08, 'sentiment': 0.05, 'squeeze_detector': 0.05,
            'order_imbalance': 0.05, 'crowd_positioning': 0.06,
            'cvd': 0.06, 'vwap_deviation': 0.04,
            'liquidation_cascade': 0.04,
        },
    }
    ADAPTIVE_HYBRID_RANGING_TOKENS = ['BTC', 'ETH']
    ADAPTIVE_HYBRID_TRENDING_TOKENS = ['DOGE', 'kPEPE', 'SUI', 'TAO']
    SNIPER_ASSETS = ['BTC', 'ETH', 'SOL', 'XRP', 'AVAX', 'SUI', 'TAO', 'NEAR',
                     'AAVE', 'ENA', 'LINK', 'DOGE', 'kPEPE', 'ADA']
    ADAPTIVE_HYBRID_MAX_HOLD_HOURS = 48
    ADAPTIVE_HYBRID_HOLD_TP_CHECK_HOURS = 24
    ADAPTIVE_HYBRID_LEVERAGE_PROFILES = {'btc': 3, 'eth': 3, 'mid': 3, 'alt': 2}
    RISK_MAX_DRAWDOWN_PCT = 15
    CASH_PERCENTAGE = 20
    ADAPTIVE_HYBRID_OPTIMAL_HOURS = [7, 8, 9, 13, 14, 15, 19, 20, 21]
    ADAPTIVE_HYBRID_AVOID_HOURS = [0, 1, 2, 3, 4, 5]
    REGIME_ADX_TRENDING = 30
    REGIME_ADX_RANGING = 20
    REGIME_VOL_HIGH = 1.2
    REGIME_VOL_LOW = 0.8
    ADAPTIVE_HYBRID_TRAILING_LEVELS = [
        {'activate_atr': 1.0, 'distance_atr': None, 'breakeven': True},
        {'activate_atr': 2.0, 'distance_atr': 1.5},
        {'activate_atr': 3.5, 'distance_atr': 1.0},
    ]
    ADAPTIVE_HYBRID_SCALE_OUT_LEVELS = [
        {'tp_pct': 0.40, 'close_pct': 0.33},
        {'tp_pct': 0.70, 'close_pct': 0.50},
    ]
    ADAPTIVE_HYBRID_VOL_TARGET_DAILY_PCT = 1.0
    ADAPTIVE_HYBRID_VOL_MIN_POSITION_USD = 10
    ADAPTIVE_HYBRID_USE_REALTIME_PRICE = True
    PAPER_SLIPPAGE_V2 = {'btc': 0.0003, 'eth': 0.0005, 'mid': 0.0012, 'alt': 0.003}
    PAPER_TAKER_FEE_V2 = 0.00045
    ADAPTIVE_HYBRID_USE_BAYESIAN_WEIGHTS = True
    ADAPTIVE_HYBRID_BAYESIAN_MIN_TRADES = 15
    ADAPTIVE_HYBRID_BAYESIAN_DECAY = 0.95
    ADAPTIVE_HYBRID_USE_ANOMALY_FILTER = True
    ADAPTIVE_HYBRID_ANOMALY_SCORE_DIVISOR = 2

# LLM-enhanced pipeline config (separate try/except for graceful fallback)
try:
    from src.config import (
        ADAPTIVE_HYBRID_LLM_CONFIRMATION,
        ADAPTIVE_HYBRID_LLM_REGIME,
        ADAPTIVE_HYBRID_LLM_LEARNER,
        ADAPTIVE_HYBRID_MTF_CONFLUENCE,
        ADAPTIVE_HYBRID_LLM_PROVIDER,
        ADAPTIVE_HYBRID_LLM_TIMEOUT_S,
    )
except ImportError:
    ADAPTIVE_HYBRID_LLM_CONFIRMATION = False
    ADAPTIVE_HYBRID_LLM_REGIME = False
    ADAPTIVE_HYBRID_LLM_LEARNER = False
    ADAPTIVE_HYBRID_MTF_CONFLUENCE = False
    ADAPTIVE_HYBRID_LLM_PROVIDER = 'groq'
    ADAPTIVE_HYBRID_LLM_TIMEOUT_S = 5


# Absolute cap on any single live order (safety net)
_MAX_LIVE_ORDER_USD = 1000

# Module families for convergence gate (require signals from >= 2 families)
MODULE_FAMILIES = {
    'technical': {'mean_reversion', 'momentum_breakout', 'ema_trend', 'rsi_divergence', 'mtf_confluence'},
    'volatility': {'sniper_lite', 'ramf_lite', 'squeeze_detector'},
    'derivatives': {'funding_composite', 'oi_delta', 'order_imbalance', 'liquidation_cascade'},
    'sentiment': {'crowd_positioning', 'sentiment'},
    'structure': {'cvd', 'vwap_deviation'},
}

# Regimes where urgency-based threshold relaxation is permitted
URGENCY_RELAXATION_REGIMES = {'ACCUMULATION', 'MARKUP'}

# Regime-based adjustments to the base score threshold (direction-aware)
REGIME_THRESHOLD_ADJUSTMENTS = {
    'ACCUMULATION': {'BUY': 0, 'SELL': +2},
    'MARKUP':       {'BUY': -3, 'SELL': +5},
    'DISTRIBUTION': {'BUY': +8, 'SELL': -3},
    'MARKDOWN':     {'BUY': +5, 'SELL': -2},
    'CAPITULATION': {'BUY': -4, 'SELL': +6},
    'EUPHORIA':     {'BUY': +6, 'SELL': -4},
}

# Regime transition tracking per symbol
_regime_transition_cache = {}  # {symbol: {'prev': regime, 'current': regime, 'transition_at': timestamp}}


class AdaptiveHybridStrategy(BaseStrategy):
    """
    Adaptive Hybrid Strategy - Multi-module convergence scoring.

    Runs 8 independent signal modules, aggregates scores by weighted average,
    and triggers trades when enough modules converge on a direction.
    """

    _instance = None

    def __init__(self):
        AdaptiveHybridStrategy._instance = self
        super().__init__("Adaptive Hybrid")

        self.assets = SNIPER_ASSETS
        self.weights = ADAPTIVE_HYBRID_WEIGHTS

        # Daily tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_date = None
        self.last_trade_time = None
        self.last_trade_time_per_token = {}  # {symbol: datetime} — per-token urgency tracking
        self.consecutive_losses = 0  # Circuit breaker: pause after consecutive losses (escalating cooldowns)
        self._loss_breaker_until = None  # datetime when breaker expires
        self._post_breaker_trades = 0  # Countdown: reduce size by 50% for N trades after breaker

        # Paper trading state
        self.paper_balance = PAPER_TRADING_BALANCE
        self.paper_positions = {}
        self.closed_positions = []
        self._position_counter = 0
        self._position_lock = threading.RLock()

        # High Water Mark
        self.peak_balance = PAPER_TRADING_BALANCE

        # Risk agent reference (set externally via set_risk_agent, e.g. from main.py)
        self._risk_agent = None

        # Singleton LiveOrderManager (preserves bracket order state across calls)
        self._live_order_manager = None

        # Trailing stops state: {position_id: {'highest': float, 'lowest': float, 'trailing_active': bool}}
        self.trailing_stops = {}

        # OI history for delta calculation: {symbol: deque of (timestamp, oi_value)}
        # Loaded from disk to survive restarts
        from src.strategies.modules.oi_delta import _load_oi_history
        self._oi_history = _load_oi_history()

        # Market data provider (for funding rates)
        self._market_data = None
        try:
            from src.data_providers.market_data import MarketDataProvider
            self._market_data = MarketDataProvider(start_liquidation_stream=False)
        except Exception as e:
            cprint(f"[AdaptiveHybrid] Warning: Market data provider unavailable ({type(e).__name__}: {e})", "yellow")

        # Trade memory (persistent decision logging)
        self._trade_memory = TradeMemory.get_instance()

        # ML adaptive components
        self._weight_optimizer = BayesianWeightOptimizer(
            module_names=list(ADAPTIVE_HYBRID_WEIGHTS.keys()),
            decay=ADAPTIVE_HYBRID_BAYESIAN_DECAY,
            min_trades=ADAPTIVE_HYBRID_BAYESIAN_MIN_TRADES,
        )
        self._feedback = QuantitativeFeedback(window_size=30)

        # Current regime (updated each signal generation cycle)
        self._current_regime = None

        # Funding rate history for per-token Z-score
        self._funding_history = {}  # {symbol: deque(maxlen=200)}
        self._funding_last_rate = {}  # {symbol: float} — dedup consecutive identical rates

        # Lock for caches accessed by ThreadPoolExecutor workers
        self._cache_lock = threading.Lock()

        # Confirmation delay: pending signals awaiting bar confirmation
        self._pending_signals = {}

        # 4H trend filter cache: {symbol: {'trend': str, 'timestamp': datetime}}
        self._4h_trend_cache = {}

        # BTC trend cache (15-minute TTL)
        self._btc_trend_cache = None
        self._btc_trend_timestamp = None

        # Global regime cache (15-minute TTL)
        self._global_regime = None  # 'trending_volatile', 'trending_calm', 'ranging_volatile', 'ranging_calm'
        self._global_regime_timestamp = None

        # Benchmark tracking (BTC & ETH buy-and-hold comparison)
        self._benchmark_start_prices = {}  # {'BTC': price, 'ETH': price}
        self._benchmark_start_time = None

        # BTC correlation cache per-symbol: {symbol: (corr, timestamp)}
        self._btc_correlation_cache = {}  # {symbol: (float, datetime)}

        # Candle cache to avoid repeated API calls: {(symbol, interval): (df, timestamp)}
        self._candle_cache = {}  # {(symbol, interval): (DataFrame, datetime)}
        self._candle_cache_ttl = 300  # 5 min TTL — candles don't change faster than this

        # LLM model for confirmation/regime (lazy-loaded, fast provider)
        self._llm_model = None
        if ADAPTIVE_HYBRID_LLM_CONFIRMATION or ADAPTIVE_HYBRID_LLM_REGIME or ADAPTIVE_HYBRID_LLM_LEARNER:
            try:
                from src.models.model_factory import ModelFactory
                self._llm_model = ModelFactory.create_model_with_fallback(ADAPTIVE_HYBRID_LLM_PROVIDER)
                if self._llm_model:
                    cprint(f"[AdaptiveHybrid] LLM model loaded ({ADAPTIVE_HYBRID_LLM_PROVIDER})", "green")
                else:
                    cprint("[AdaptiveHybrid] LLM model unavailable, pipeline modules will use rule-based fallbacks", "yellow")
            except Exception as e:
                cprint(f"[AdaptiveHybrid] LLM init error: {e}", "yellow")

        # LLM regime cache per-symbol: {symbol: regime_result}
        self._llm_regime_cache = {}

        # Data directory
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'adaptive_hybrid'
        )
        os.makedirs(self.data_dir, exist_ok=True)

        # Check for paper trading reset flag
        if ADAPTIVE_HYBRID_RESET_PAPER:
            self._reset_paper_trading()
        else:
            # Load existing state
            self._load_state_from_csv()

        # Pre-load funding history to avoid cold start
        self._preload_funding_history()

        if not PAPER_TRADING:
            cprint("=" * 60, "red", attrs=['bold'])
            cprint("  WARNING: LIVE TRADING MODE ACTIVE", "red", attrs=['bold'])
            cprint("  Real money will be used for trades!", "red", attrs=['bold'])
            cprint("=" * 60, "red", attrs=['bold'])

        cprint(f"[AdaptiveHybrid] Strategy initialized", "cyan")
        cprint(f"  - Assets: {len(self.assets)} symbols", "white")
        cprint(f"  - Base threshold: {ADAPTIVE_HYBRID_BASE_THRESHOLD}/100", "white")
        cprint(f"  - Paper Trading: {PAPER_TRADING}", "white")
        cprint(f"  - Balance: ${self.paper_balance:,.2f}", "white")
        cprint(f"  - LLM Confirmation: {ADAPTIVE_HYBRID_LLM_CONFIRMATION}", "white")
        cprint(f"  - LLM Regime: {ADAPTIVE_HYBRID_LLM_REGIME}", "white")
        cprint(f"  - LLM Learner: {ADAPTIVE_HYBRID_LLM_LEARNER}", "white")
        cprint(f"  - MTF Confluence: {ADAPTIVE_HYBRID_MTF_CONFLUENCE}", "white")

    def _preload_funding_history(self):
        """Pre-load 7 days of funding history to avoid cold start."""
        if not self._market_data:
            return
        for symbol in SNIPER_ASSETS:
            try:
                rate_data = self._market_data.get_funding_rate(symbol)
                if rate_data and rate_data.get('funding_rate') is not None:
                    rate = rate_data['funding_rate']
                    if symbol not in self._funding_history:
                        self._funding_history[symbol] = deque(maxlen=200)
                    self._funding_history[symbol].append(rate)
                    self._funding_last_rate[symbol] = rate
            except Exception:
                pass

    def set_risk_agent(self, risk_agent):
        """Set risk agent reference for recovery mode sizing.
        NOTE: main.py should call strategy.set_risk_agent(risk_agent) after init.
        """
        self._risk_agent = risk_agent
        cprint(f"[AdaptiveHybrid] Risk agent linked", "cyan")

    # =========================================================================
    # PAPER TRADING RESET
    # =========================================================================

    def _reset_paper_trading(self):
        """Reset paper trading state: balance to initial, clear all positions and history."""
        cprint("[AdaptiveHybrid] RESET: Resetting paper trading state to initial values", "yellow", attrs=['bold'])

        self.paper_balance = PAPER_TRADING_BALANCE
        self.paper_positions = {}
        self.closed_positions = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self._position_counter = 0

        # Clear CSV files
        for filename in ['paper_trades.csv', 'closed_trades.csv']:
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                cprint(f"  Removed {filename}", "yellow")

        # Disable reset flag in config to prevent re-reset on next restart
        try:
            import re
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                'src', 'config.py'
            )
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    content = f.read()
                new_content = re.sub(
                    r'^(ADAPTIVE_HYBRID_RESET_PAPER\s*=\s*)True',
                    r'\g<1>False',
                    content,
                    flags=re.MULTILINE
                )
                if new_content != content:
                    with open(config_path, 'w') as f:
                        f.write(new_content)
                    cprint("  Reset flag cleared in config.py (set to False)", "yellow")
        except Exception as e:
            cprint(f"  Warning: Could not clear reset flag in config: {e}", "yellow")

        cprint(f"  Balance reset to ${PAPER_TRADING_BALANCE:,.2f}", "yellow")

    # =========================================================================
    # DATA FETCHING
    # =========================================================================

    def _fetch_candles(self, symbol: str, interval: str = '1h', candles: int = 200) -> pd.DataFrame:
        """Fetch candle data from HyperLiquid with caching to avoid 429 rate limits."""
        import time as _time

        cache_key = (symbol, interval)

        # Check cache first (only if cached frame has enough rows for this request)
        with self._cache_lock:
            cached = self._candle_cache.get(cache_key)
            if cached:
                df_cached, cached_at = cached
                age = (datetime.now() - cached_at).total_seconds()
                if age < self._candle_cache_ttl and len(df_cached) >= candles:
                    return df_cached.copy()

        try:
            from hyperliquid.info import Info

            info = Info(skip_ws=True, timeout=15)
            end_time = int(_time.time() * 1000)

            interval_map = {
                '1m': 60_000, '5m': 300_000, '15m': 900_000,
                '30m': 1_800_000, '1h': 3_600_000, '4h': 14_400_000
            }
            interval_ms = interval_map.get(interval, 3_600_000)
            start_time = end_time - (candles * interval_ms)

            # Small delay to respect rate limits (14 tokens × multiple calls)
            _time.sleep(0.15)

            data = info.candles_snapshot(symbol, interval, start_time, end_time)
            if not data:
                return None

            df = pd.DataFrame(data)
            df = df.rename(columns={
                't': 'timestamp', 'o': 'open', 'h': 'high',
                'l': 'low', 'c': 'close', 'v': 'volume'
            })
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Validate candle freshness
            if len(df) > 0 and 'timestamp' in df.columns:
                last_candle_time = pd.to_datetime(df['timestamp'].iloc[-1], unit='ms', utc=True)
                max_staleness = pd.Timedelta(hours=2)
                if pd.Timestamp.now(tz='UTC') - last_candle_time > max_staleness:
                    cprint(f"[AdaptiveHybrid] Stale candle data for {symbol}: last candle {last_candle_time}", "yellow")
                    return None

            # Store in cache
            with self._cache_lock:
                self._candle_cache[cache_key] = (df.copy(), datetime.now())

            return df

        except Exception as e:
            cprint(f"[AdaptiveHybrid] Error fetching candles for {symbol}: {e}", "yellow")
            return None

    def _compute_indicators(self, df: pd.DataFrame) -> dict:
        """Compute all technical indicators needed by modules."""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # RSI
        rsi_ind = RSIIndicator(close=close, window=14)
        df['rsi'] = rsi_ind.rsi()

        # ADX
        adx_ind = ADXIndicator(high=high, low=low, close=close, window=14)
        df['adx'] = adx_ind.adx()

        # EMAs
        df['ema_9'] = EMAIndicator(close=close, window=9).ema_indicator()
        df['ema_21'] = EMAIndicator(close=close, window=21).ema_indicator()
        df['ema_50'] = EMAIndicator(close=close, window=50).ema_indicator()
        df['ema_200'] = EMAIndicator(close=close, window=200).ema_indicator() if len(df) >= 200 else close

        # Bollinger Bands
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()

        # ATR
        atr_ind = AverageTrueRange(high=high, low=low, close=close, window=14)
        df['atr'] = atr_ind.average_true_range()

        # MACD
        macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd_ind.macd()
        df['macd_signal'] = macd_ind.macd_signal()
        df['macd_diff'] = macd_ind.macd_diff()

        # Volume metrics
        vol_avg = volume.rolling(20).mean()
        df['volume_ratio'] = volume / vol_avg.replace(0, 1)

        # VWAP rolling (14-period)
        vwap_ind = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume, window=14)
        df['vwap'] = vwap_ind.volume_weighted_average_price()
        df['vwap'] = df['vwap'].fillna((high + low + close) / 3)

        # Dynamic RSI thresholds (percentile-based)
        rsi_series = rsi_ind.rsi().dropna()
        if len(rsi_series) >= 30:
            rsi_p15 = float(np.percentile(rsi_series, 15))
            rsi_p85 = float(np.percentile(rsi_series, 85))
            # Clamp to reasonable bounds
            rsi_oversold = max(25, min(40, rsi_p15))
            rsi_overbought = max(60, min(75, rsi_p85))
        else:
            rsi_oversold = 30
            rsi_overbought = 70

        # Bollinger Band %B for squeeze detection
        bb_range = df['bb_upper'] - df['bb_lower']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / bb_range.replace(0, 1)

        # Return last row values as dict for easy access
        last = df.iloc[-1]
        return {
            'close': float(last['close']),
            'open': float(last['open']),
            'high': float(last['high']),
            'low': float(last['low']),
            'volume': float(last['volume']),
            'rsi': float(last['rsi']) if pd.notna(last['rsi']) else 50,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'adx': float(last['adx']) if pd.notna(last['adx']) else 20,
            'ema_9': float(last['ema_9']) if pd.notna(last['ema_9']) else float(last['close']),
            'ema_21': float(last['ema_21']) if pd.notna(last['ema_21']) else float(last['close']),
            'ema_50': float(last['ema_50']) if pd.notna(last['ema_50']) else float(last['close']),
            'ema_200': float(last['ema_200']) if pd.notna(last['ema_200']) else float(last['close']),
            'bb_upper': float(last['bb_upper']) if pd.notna(last['bb_upper']) else float(last['close']),
            'bb_lower': float(last['bb_lower']) if pd.notna(last['bb_lower']) else float(last['close']),
            'bb_mid': float(last['bb_mid']) if pd.notna(last['bb_mid']) else float(last['close']),
            'bb_pct': float(last['bb_pct']) if pd.notna(last['bb_pct']) else 0.5,
            'atr': float(last['atr']) if pd.notna(last['atr']) else 0,
            'macd': float(last['macd']) if pd.notna(last['macd']) else 0,
            'macd_signal': float(last['macd_signal']) if pd.notna(last['macd_signal']) else 0,
            'macd_diff': float(last['macd_diff']) if pd.notna(last['macd_diff']) else 0,
            'volume_ratio': float(last['volume_ratio']) if pd.notna(last['volume_ratio']) else 1.0,
            'vwap': float(last['vwap']) if pd.notna(last['vwap']) else float(last['close']),
            '_df': df,
        }

    # =========================================================================
    # SIGNAL MODULES (delegated to src/strategies/modules/)
    # =========================================================================

    def _get_funding_zscore_per_token(self, symbol: str) -> float:
        """Calculate per-token historical funding Z-score."""
        if not self._market_data:
            return 0.0

        try:
            rate_data = self._market_data.get_funding_rate(symbol)
            if rate_data is None or rate_data.get('funding_rate') is None:
                return 0.0

            rate = rate_data['funding_rate']
            if rate == 0:
                return 0.0

            # Initialize ring buffer and dedup tracker under lock
            with self._cache_lock:
                if symbol not in self._funding_history:
                    self._funding_history[symbol] = deque(maxlen=200)

                # Only store when rate changes (funding rates update every 8h)
                last_rate = self._funding_last_rate.get(symbol)
                if last_rate is None or abs(rate - last_rate) > 1e-10:
                    self._funding_history[symbol].append(rate)
                    self._funding_last_rate[symbol] = rate

                history = list(self._funding_history[symbol])

            if len(history) < 5:
                # Not enough history: fall back to cross-sectional Z-score
                zscore = self._market_data.get_funding_zscore(symbol)
                return zscore if zscore is not None else 0.0

            arr = np.array(history)
            mean = float(arr.mean())
            std = float(arr.std())
            if std < 1e-8:
                return 0.0

            return float((rate - mean) / std)
        except Exception:
            return 0.0

    def _score_squeeze_detector_wrapper(self, symbol: str, ind: dict) -> dict:
        """Wrapper for squeeze module (needs zscore from instance state)."""
        if not self._market_data:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': 'No market data'}
        try:
            funding_zscore = self._get_funding_zscore_per_token(symbol)
            return score_squeeze_detector(funding_zscore, ind)
        except Exception as e:
            return {'score': 0, 'direction': 'NEUTRAL', 'reason': f'Squeeze error: {e}'}

    # =========================================================================
    # BTC MACRO FILTER
    # =========================================================================

    def _check_btc_trend(self) -> float:
        """
        Check BTC macro trend. Returns float -1.0 (strong bear) to +1.0 (strong bull).
        Computed as (btc_price - ema200) / ema200, clamped to [-0.15, +0.15], scaled to [-1.0, +1.0].
        Cached for 15 minutes.
        """
        if self._btc_trend_cache is not None and self._btc_trend_timestamp:
            age = (datetime.now() - self._btc_trend_timestamp).total_seconds()
            if age < 900:  # 15 min cache
                return self._btc_trend_cache

        try:
            df = self._fetch_candles('BTC', interval='1h', candles=250)
            if df is None or len(df) < 200:
                return 0.0  # Neutral default when data unavailable

            close = df['close']
            ema200 = EMAIndicator(close, window=200).ema_indicator()

            current_price = float(close.iloc[-1])
            current_ema200 = float(ema200.iloc[-1])

            if current_ema200 <= 0:
                return 0.0

            # Compute trend strength: deviation from EMA200 as fraction
            raw_strength = (current_price - current_ema200) / current_ema200
            # Clamp to [-0.15, +0.15] then scale to [-1.0, +1.0]
            clamped = max(-0.15, min(0.15, raw_strength))
            trend_strength = round(clamped / 0.15, 2)

            self._btc_trend_cache = trend_strength
            self._btc_trend_timestamp = datetime.now()

            label = 'BULLISH' if trend_strength > 0 else 'BEARISH' if trend_strength < 0 else 'NEUTRAL'
            cprint(f"  [BTC Macro] Price=${current_price:,.0f} EMA200=${current_ema200:,.0f} "
                   f"-> {label} (strength={trend_strength:+.2f})", "cyan")

            return trend_strength

        except Exception as e:
            cprint(f"[AdaptiveHybrid] Error checking BTC trend: {e}", "yellow")
            return 0.0  # Neutral default

    def _detect_global_regime(self) -> str:
        """Detect global market regime based on BTC ADX and volatility ratio.
        Returns one of: 'trending_volatile', 'trending_calm', 'ranging_volatile', 'ranging_calm'.
        Cached for 15 minutes.
        """
        if self._global_regime is not None and self._global_regime_timestamp:
            age = (datetime.now() - self._global_regime_timestamp).total_seconds()
            if age < 900:  # 15 min cache
                return self._global_regime

        try:
            df = self._fetch_candles('BTC', interval='1h', candles=250)
            if df is None or len(df) < 50:
                return 'ranging_calm'  # Default conservative

            close = df['close']
            high = df['high']
            low = df['low']

            # ADX
            adx_ind = ADXIndicator(high=high, low=low, close=close, window=14)
            adx_val = float(adx_ind.adx().iloc[-1]) if pd.notna(adx_ind.adx().iloc[-1]) else 20

            # Volatility ratio: recent ATR(14) / historical ATR(50)
            atr_ind = AverageTrueRange(high=high, low=low, close=close, window=14)
            atr_series = atr_ind.average_true_range().dropna()
            if len(atr_series) >= 50:
                recent_atr = float(atr_series.iloc[-1])
                hist_atr = float(atr_series.tail(50).mean())
                vol_ratio = recent_atr / hist_atr if hist_atr > 0 else 1.0
            else:
                vol_ratio = 1.0

            # Classify regime
            is_trending = adx_val >= REGIME_ADX_TRENDING
            is_ranging = adx_val < REGIME_ADX_RANGING
            is_vol_high = vol_ratio > REGIME_VOL_HIGH
            is_vol_low = vol_ratio < REGIME_VOL_LOW

            if is_trending and is_vol_high:
                regime = 'trending_volatile'
            elif is_trending:
                regime = 'trending_calm'
            elif is_ranging and is_vol_high:
                regime = 'ranging_volatile'
            else:
                regime = 'ranging_calm'

            self._global_regime = regime
            self._global_regime_timestamp = datetime.now()

            cprint(f"  [Global Regime] ADX={adx_val:.1f} VolRatio={vol_ratio:.2f} -> {regime}", "cyan")
            return regime

        except Exception as e:
            cprint(f"[AdaptiveHybrid] Error detecting global regime: {e}", "yellow")
            return 'ranging_calm'

    def _get_btc_correlation(self, symbol: str):
        """Get Pearson correlation of token returns with BTC returns (4h cache per-symbol).
        Returns float or None if data unavailable."""
        now = datetime.now()
        with self._cache_lock:
            cached = self._btc_correlation_cache.get(symbol)
            if cached and (now - cached[1]).total_seconds() < 14400:
                return cached[0]

        try:
            # Fetch candles outside lock (I/O)
            btc_df = self._fetch_candles('BTC', interval='1h', candles=170)
            token_df = self._fetch_candles(symbol, interval='1h', candles=170)

            if btc_df is None or token_df is None or len(btc_df) < 50 or len(token_df) < 50:
                return None  # Skip penalty when data unavailable

            n = min(len(btc_df), len(token_df))
            btc_returns = btc_df['close'].tail(n).pct_change().dropna()
            token_returns = token_df['close'].tail(n).pct_change().dropna()

            n_common = min(len(btc_returns), len(token_returns))
            if n_common < 30:
                return None

            corr = float(btc_returns.tail(n_common).corr(token_returns.tail(n_common)))
            if np.isnan(corr):
                return None

            with self._cache_lock:
                self._btc_correlation_cache[symbol] = (corr, now)
            return corr
        except Exception:
            return None

    # =========================================================================
    # AGGREGATION
    # =========================================================================

    def _get_weights_for_symbol(self, symbol: str, ind: dict = None) -> dict:
        """Get module weights based on dynamic regime detection and global regime."""
        # Use global regime to bias weight selection
        global_regime = self._detect_global_regime()

        if global_regime in ('trending_volatile', 'trending_calm'):
            base_profile = 'trending'
        elif global_regime == 'ranging_volatile':
            base_profile = 'ranging'
        else:
            base_profile = None

        # Per-token ADX override (no longer short-circuits — falls through to LLM adjustments)
        selected_profile = None
        if ind is not None and 'adx' in ind:
            if ind['adx'] > 30:
                selected_profile = 'trending'
            elif ind['adx'] < 20:
                selected_profile = 'ranging'

        if selected_profile:
            selected = ADAPTIVE_HYBRID_WEIGHT_PROFILES.get(selected_profile, self.weights)
        elif base_profile:
            # Use global regime profile if available
            selected = ADAPTIVE_HYBRID_WEIGHT_PROFILES.get(base_profile, self.weights)
        elif symbol in ADAPTIVE_HYBRID_RANGING_TOKENS:
            selected = ADAPTIVE_HYBRID_WEIGHT_PROFILES.get('ranging', self.weights)
        elif symbol in ADAPTIVE_HYBRID_TRENDING_TOKENS:
            selected = ADAPTIVE_HYBRID_WEIGHT_PROFILES.get('trending', self.weights)
        else:
            selected = self.weights

        # Apply LLM regime weight adjustments (if enabled and cached)
        if ADAPTIVE_HYBRID_LLM_REGIME and symbol in self._llm_regime_cache:
            regime_result = self._llm_regime_cache[symbol]
            if regime_result.get('confidence', 0) >= 50:
                selected = adjust_weights_for_regime(selected, regime_result['regime'])

        # Bayesian weight adaptation (Phase 3)
        if ADAPTIVE_HYBRID_USE_BAYESIAN_WEIGHTS:
            selected = self._weight_optimizer.get_weights(selected)

        return selected

    def _aggregate_scores(self, module_results: dict, symbol: str = None, ind: dict = None) -> dict:
        """Aggregate all module scores into a final trading decision."""
        weights = self._get_weights_for_symbol(symbol, ind=ind) if symbol else self.weights

        # Separate by direction
        long_modules = {}
        short_modules = {}

        for name, result in module_results.items():
            if result['score'] <= 0 or result['direction'] == 'NEUTRAL':
                continue
            if result['direction'] == 'BUY':
                long_modules[name] = result
            elif result['direction'] == 'SELL':
                short_modules[name] = result

        # Calculate weighted sums for each direction
        long_weighted = sum(r['score'] * weights.get(n, 0) for n, r in long_modules.items())
        short_weighted = sum(r['score'] * weights.get(n, 0) for n, r in short_modules.items())
        long_weight_total = sum(weights.get(n, 0) for n in long_modules)
        short_weight_total = sum(weights.get(n, 0) for n in short_modules)

        if long_weighted == 0 and short_weighted == 0:
            return {'direction': 'NEUTRAL', 'score': 0, 'agreement': 0,
                    'details': 'No modules produced directional signals'}

        # Determine winning direction
        if long_weighted >= short_weighted:
            direction = 'BUY'
            winning_modules = long_modules
            losing_modules = short_modules
            losing_weight = short_weight_total
            winning_weight = long_weight_total
        else:
            direction = 'SELL'
            winning_modules = short_modules
            losing_modules = long_modules
            losing_weight = long_weight_total
            winning_weight = short_weight_total

        # Minimum convergent modules check
        if len(winning_modules) < ADAPTIVE_HYBRID_MIN_CONVERGENT_MODULES:
            return {'direction': 'NEUTRAL', 'score': 0, 'agreement': 0,
                    'details': f'Only {len(winning_modules)} module(s) converge (min {ADAPTIVE_HYBRID_MIN_CONVERGENT_MODULES} required)'}

        # Family-based convergence: require modules from at least 2 different families
        winning_families = set()
        for mod_name in winning_modules:
            for family, members in MODULE_FAMILIES.items():
                if mod_name in members:
                    winning_families.add(family)
                    break
        if len(winning_families) < 2:
            return {'direction': 'NEUTRAL', 'score': 0, 'agreement': 0,
                    'details': f'Only {len(winning_families)} signal family(ies) ({", ".join(winning_families)}), need 2+'}

        # Funding cluster cap: limit combined contribution of funding-based modules
        FUNDING_MODULES = {'funding_composite', 'squeeze_detector'}
        try:
            from src.config import ADAPTIVE_HYBRID_FUNDING_CLUSTER_CAP
            funding_cap = ADAPTIVE_HYBRID_FUNDING_CLUSTER_CAP
        except ImportError:
            funding_cap = 0.10

        funding_weight_in_winning = sum(weights.get(m, 0) for m in winning_modules if m in FUNDING_MODULES)
        if funding_weight_in_winning > funding_cap:
            excess_ratio = funding_cap / funding_weight_in_winning
            for m in list(winning_modules.keys()):
                if m in FUNDING_MODULES:
                    winning_modules[m] = {**winning_modules[m], 'score': round(winning_modules[m]['score'] * excess_ratio)}

        # Weighted average of ACTIVE modules in winning direction
        # Recompute weights after funding cap to avoid diluting non-funding modules
        active_weighted_sum = sum(r['score'] * weights.get(n, 0) for n, r in winning_modules.items())
        active_weight_total = sum(
            weights.get(n, 0) * (excess_ratio if (n in FUNDING_MODULES and funding_weight_in_winning > funding_cap) else 1.0)
            for n in winning_modules
        )

        if active_weight_total <= 0:
            return {'direction': 'NEUTRAL', 'score': 0, 'agreement': 0,
                    'details': 'No weight in winning direction'}

        raw_score = active_weighted_sum / active_weight_total

        # --- Additive adjustments (replaces multiplicative penalty stacking) ---
        adjustments = 0
        adjustment_details = []

        # Coverage adjustment: penalize signals with low directional agreement
        directional_count = len(long_modules) + len(short_modules)
        n_active = len(winning_modules)
        total_available = len(module_results)
        convergence_ratio = n_active / max(directional_count, total_available * 0.35, 1)

        if convergence_ratio < 0.40:
            adjustments -= 8
            adjustment_details.append("coverage:-8")
        elif convergence_ratio < 0.60:
            adjustments -= 4
            adjustment_details.append("coverage:-4")

        # BTC macro filter: graduated additive penalty proportional to trend strength and correlation
        btc_trend = self._check_btc_trend()  # float: -1.0 (bear) to +1.0 (bull)
        btc_penalty = 1.0  # kept for return dict compatibility
        details = ""
        if btc_trend is not None and symbol:
            corr = self._get_btc_correlation(symbol)
            if corr is not None:
                corr_clamped = max(0.33, min(1.0, abs(corr)))

                if (btc_trend < 0 and direction == 'BUY') or (btc_trend > 0 and direction == 'SELL'):
                    btc_penalty_amount = abs(btc_trend) * corr_clamped
                    btc_adj = -int(btc_penalty_amount * 8)  # max -8 points
                    btc_adj = max(-8, btc_adj)
                    if btc_adj != 0:
                        adjustments += btc_adj
                        btc_penalty = 1.0 + (btc_adj / 100)  # approximate for compat
                        label = "bearish" if btc_trend < 0 else "bullish"
                        details = f" | BTC {label} adj {btc_adj:+d} (trend={btc_trend:+.2f}, corr={corr:.2f})"
                        adjustment_details.append(f"btc:{btc_adj:+d}")

        # Graduated conflict penalty (contrarians count at 50% - they oppose by design)
        CONTRARIAN_MODULES = {'mean_reversion', 'sentiment', 'sniper_lite', 'funding_composite'}
        contrarian_losing_weight = sum(weights.get(m, 0) for m in losing_modules if m in CONTRARIAN_MODULES)
        non_contrarian_losing_weight = losing_weight - contrarian_losing_weight
        effective_losing_weight = non_contrarian_losing_weight + contrarian_losing_weight * 0.5
        conflict_ratio = effective_losing_weight / max(winning_weight, 0.01)
        conflict_factor = 1.0  # kept for return dict compatibility
        if conflict_ratio > 0.15:
            conflict_adj = -min(5, int(conflict_ratio * 10))  # max -5 points
            adjustments += conflict_adj
            conflict_factor = max(0.60, 1.0 - conflict_ratio * 0.50)  # compat
            adjustment_details.append(f"conflict:{conflict_adj:+d}")

        # Session filter - additive
        hour = datetime.utcnow().hour
        if hour in ADAPTIVE_HYBRID_AVOID_HOURS:
            adjustments -= 5
            adjustment_details.append("session:-5")
        elif hour in ADAPTIVE_HYBRID_OPTIMAL_HOURS:
            adjustments += 3
            adjustment_details.append("session:+3")

        # High conviction bonus: 5+ modules with score >= 50 agreeing
        winning_results = winning_modules
        high_conviction_count = sum(1 for r in winning_results.values() if r.get('score', 0) >= 50)
        if high_conviction_count >= 5:
            adjustments += 8  # conviction bonus
            adjustment_details.append("conviction:+8")

        final_score = raw_score + adjustments
        final_score = max(0, min(100, final_score))  # clamp to 0-100

        if adjustment_details:
            cprint(f"  [{symbol}] Additive adjustments: {', '.join(adjustment_details)} (raw={raw_score:.1f} -> final={final_score:.1f})", "white")

        # Build details string
        module_details = []
        for name, result in winning_modules.items():
            module_details.append(f"{name}={result['score']}")

        return {
            'direction': direction,
            'score': round(final_score, 1),
            'raw_score': round(raw_score, 1),
            'coverage': round(convergence_ratio * 100, 1),
            'conflict_factor': conflict_factor,
            'btc_penalty': btc_penalty,
            'adjustments': adjustments,
            'active_modules': len(winning_modules),
            'total_modules_fired': len(long_modules) + len(short_modules),
            'agreement': round(convergence_ratio, 2),
            'details': f"{direction} score={final_score:.1f} ({len(winning_modules)} modules: {', '.join(module_details)})"
                       + details,
            'module_scores': {n: r['score'] for n, r in module_results.items() if r['score'] > 0},
        }

    # =========================================================================
    # HELPER METHODS (neutral signal, 4H trend, pending signal cleanup)
    # =========================================================================

    def _create_neutral_signal(self, symbol, reason):
        """Create a standardized neutral/no-trade signal."""
        return {
            'token': symbol,
            'signal': 0,
            'direction': 'NEUTRAL',
            'metadata': {
                'strategy': 'Adaptive Hybrid',
                'score': 0,
                'threshold': 0,
                'reason': reason,
                'current_price': 0,
            }
        }

    def _get_4h_trend(self, symbol):
        """Determine 4H trend using EMA alignment + ADX.
        Returns 'BULLISH', 'BEARISH', or 'NEUTRAL'.
        Cached for 30 minutes per symbol.
        """
        now = datetime.now()
        cached = self._4h_trend_cache.get(symbol)
        if cached and (now - cached['timestamp']).total_seconds() < 1800:
            return cached['trend']

        try:
            ohlcv_4h = self._fetch_candles(symbol, interval='4h', candles=100)
            if ohlcv_4h is None or len(ohlcv_4h) < 50:
                return 'NEUTRAL'

            close = ohlcv_4h['close']

            # EMA 20 and 50 on 4h
            ema20 = close.ewm(span=20).mean()
            ema50 = close.ewm(span=50).mean()

            # ADX on 4h
            import pandas_ta as pta
            adx_df = pta.adx(ohlcv_4h['high'], ohlcv_4h['low'], close, length=14)
            if adx_df is not None and len(adx_df) > 0 and 'ADX_14' in adx_df.columns:
                adx_value = float(adx_df['ADX_14'].iloc[-1])
            else:
                adx_value = 20

            current_ema20 = float(ema20.iloc[-1])
            current_ema50 = float(ema50.iloc[-1])
            current_price = float(close.iloc[-1])

            # Trend determination
            if adx_value < 20:
                trend = 'NEUTRAL'  # No trend
            elif current_price > current_ema20 > current_ema50:
                trend = 'BULLISH'
            elif current_price < current_ema20 < current_ema50:
                trend = 'BEARISH'
            else:
                trend = 'NEUTRAL'

            self._4h_trend_cache[symbol] = {'trend': trend, 'timestamp': now}
            cprint(f"  [4H Trend] {symbol}: {trend} (ADX={adx_value:.1f}, EMA20={current_ema20:.2f}, EMA50={current_ema50:.2f})", "cyan")
            return trend
        except Exception as e:
            cprint(f"  [4H Trend] Error for {symbol}: {e}", "yellow")
            return 'NEUTRAL'

    def _cleanup_stale_pending_signals(self):
        """Remove pending signals older than 2 hours."""
        now = datetime.now()
        stale_keys = [
            k for k, v in self._pending_signals.items()
            if (now - v['timestamp']).total_seconds() > 7200
        ]
        for k in stale_keys:
            del self._pending_signals[k]

    # =========================================================================
    # BENCHMARK TRACKER
    # =========================================================================

    def _update_benchmark(self):
        """Track BTC and ETH buy-and-hold performance for alpha calculation."""
        benchmarks = ['BTC', 'ETH']
        for sym in benchmarks:
            try:
                df = self._fetch_candles(sym, interval='1h', candles=5)
                if df is None or len(df) == 0:
                    continue
                current_price = float(df['close'].iloc[-1])

                if sym not in self._benchmark_start_prices:
                    self._benchmark_start_prices[sym] = current_price
                    if not self._benchmark_start_time:
                        self._benchmark_start_time = datetime.now()
            except Exception:
                pass

    def _get_benchmark_alpha(self) -> dict:
        """Calculate alpha vs BTC and ETH buy-and-hold.
        Returns dict with benchmark returns and strategy alpha.
        """
        if not self._benchmark_start_prices:
            return {}

        result = {}
        strategy_return = (self.paper_balance - PAPER_TRADING_BALANCE) / PAPER_TRADING_BALANCE * 100

        for sym, start_price in self._benchmark_start_prices.items():
            try:
                df = self._fetch_candles(sym, interval='1h', candles=5)
                if df is None or len(df) == 0:
                    continue
                current_price = float(df['close'].iloc[-1])
                bench_return = (current_price - start_price) / start_price * 100
                alpha = strategy_return - bench_return
                result[sym] = {
                    'start_price': start_price,
                    'current_price': current_price,
                    'return_pct': round(bench_return, 2),
                    'alpha': round(alpha, 2),
                }
            except Exception:
                pass

        result['strategy_return_pct'] = round(strategy_return, 2)
        return result

    # =========================================================================
    # URGENCY SYSTEM
    # =========================================================================

    def _get_urgency_multiplier(self, symbol: str = None) -> float:
        """Progressive threshold relaxation if no trades recently (per-token).
        Only relaxes in favorable regimes (ACCUMULATION, MARKUP).
        """
        # Check regime first - only relax in favorable regimes
        current_regime = (self._current_regime or 'unknown').upper()
        if current_regime not in URGENCY_RELAXATION_REGIMES:
            return 1.0

        # Read trade times under lock (written by execute_paper_trade under lock)
        with self._position_lock:
            per_token_time = self.last_trade_time_per_token.get(symbol) if symbol else None
            global_time = self.last_trade_time

        if per_token_time is not None:
            last_trade = per_token_time
        elif global_time is not None:
            last_trade = global_time
        else:
            last_trade = None

        if last_trade is None:
            # Never traded = no urgency relaxation (conservative startup)
            return 1.0
        else:
            hours_since = (datetime.now() - last_trade).total_seconds() / 3600

        start = ADAPTIVE_HYBRID_URGENCY_START_HOURS
        if hours_since < start:
            return 1.0

        # Reduce by 5 points per hour after start, down to floor
        steps = hours_since - start
        reduction = min(steps * 0.02, 0.10)  # Max 10% reduction (was 30%, caused premature entries)
        return max(0.90, 1.0 - reduction)

    def _get_effective_threshold(self, symbol: str = None, ind: dict = None, direction: str = None) -> float:
        """Get current threshold adjusted by urgency, regime, direction, choppiness, transitions, and feedback."""
        base = ADAPTIVE_HYBRID_BASE_THRESHOLD

        # Regime-based threshold adjustment (direction-aware)
        regime = (self._current_regime or '').upper()
        adj = REGIME_THRESHOLD_ADJUSTMENTS.get(regime, {}).get(direction, 0) if direction else 0
        base += adj

        # Regime transition bonus/penalty
        if symbol and regime:
            transition_adj = self._detect_regime_transition(symbol, regime)
            if transition_adj != 0:
                base += transition_adj
                cprint(f"    [Regime Transition] {symbol}: adj={transition_adj:+.0f} (threshold now {base:.0f})", "cyan")

        # Choppiness index: raise threshold in choppy markets
        try:
            from src.config import ADAPTIVE_HYBRID_CHOPPINESS_THRESHOLD
            chop_threshold = ADAPTIVE_HYBRID_CHOPPINESS_THRESHOLD
        except ImportError:
            chop_threshold = 61.8

        if ind is not None:
            df = ind.get('_df')
            ci = self._get_choppiness_index(df)
            if ci > chop_threshold:
                base += 2
                cprint(f"    [Choppiness] {symbol or '?'}: CI={ci:.1f} > {chop_threshold} -> threshold +2", "yellow")

        urgency = self._get_urgency_multiplier(symbol)
        effective = base * urgency
        threshold = max(effective, ADAPTIVE_HYBRID_URGENCY_FLOOR)

        # Adaptive threshold from quantitative feedback
        if hasattr(self, '_feedback'):
            threshold = self._feedback.suggest_threshold_adjustment(threshold)

        threshold = max(35, min(75, threshold))  # Prevent extreme thresholds
        return threshold

    def _detect_regime_transition(self, symbol: str, current_regime: str) -> float:
        """Detect regime transitions and return threshold bonus/penalty."""
        prev = _regime_transition_cache.get(symbol, {}).get('current')
        if prev and prev != current_regime:
            _regime_transition_cache[symbol] = {
                'prev': prev, 'current': current_regime,
                'transition_at': datetime.utcnow()
            }
        elif not prev:
            _regime_transition_cache[symbol] = {
                'prev': None, 'current': current_regime,
                'transition_at': datetime.utcnow()
            }

        # Favorable transitions (negative = lower threshold = easier entry)
        TRANSITION_BONUSES = {
            ('ACCUMULATION', 'MARKUP'): -5,    # Best entry: accumulation complete
            ('MARKDOWN', 'ACCUMULATION'): -3,   # Bottom forming
            ('MARKUP', 'DISTRIBUTION'): +3,     # Top forming, be cautious
            ('DISTRIBUTION', 'MARKDOWN'): +3,   # Breakdown starting
        }

        transition = (prev, current_regime) if prev else None
        return TRANSITION_BONUSES.get(transition, 0)

    def _get_choppiness_index(self, df) -> float:
        """Calculate Choppiness Index (0-100). >61.8 = choppy, <38.2 = trending."""
        period = 14
        if df is None or len(df) < period + 1:
            return 50.0  # neutral default

        import math
        atr_sum = 0.0
        for i in range(len(df) - period, len(df)):
            tr = max(
                df['high'].iloc[i] - df['low'].iloc[i],
                abs(df['high'].iloc[i] - df['close'].iloc[i-1]),
                abs(df['low'].iloc[i] - df['close'].iloc[i-1])
            )
            atr_sum += tr

        highest = df['high'].iloc[-period:].max()
        lowest = df['low'].iloc[-period:].min()
        hl_range = highest - lowest

        if hl_range <= 0:
            return 50.0

        ci = 100 * math.log10(atr_sum / hl_range) / math.log10(period)
        return round(min(100, max(0, ci)), 1)

    def _get_atr_profile(self, symbol: str) -> tuple:
        """Get (sl_mult, tp_mult) for this token's class."""
        for profile in ADAPTIVE_HYBRID_ATR_PROFILES.values():
            if symbol in profile['tokens']:
                return profile['sl_mult'], profile['tp_mult']
        # Default to global config
        return ADAPTIVE_HYBRID_ATR_SL_MULT, ADAPTIVE_HYBRID_ATR_TP_MULT

    # =========================================================================
    # MAIN SIGNAL GENERATION
    # =========================================================================

    def generate_signals(self, symbol: str = None, df: pd.DataFrame = None) -> dict:
        """Generate trading signal for the given symbol."""
        self._reset_daily_counters()
        self._update_benchmark()

        # Daily limits
        if self._loss_breaker_until and datetime.utcnow() < self._loss_breaker_until:
            return None

        # Daily trade count limit (uses in-memory counter, restored from CSV on startup)
        if self.daily_trades >= ADAPTIVE_HYBRID_MAX_DAILY_TRADES:
            cprint(f"[AdaptiveHybrid] Daily trade limit reached ({self.daily_trades}/{ADAPTIVE_HYBRID_MAX_DAILY_TRADES}), skipping", "yellow")
            return None

        # Include unrealized losses in daily limit check
        with self._position_lock:
            _unrealized_pnl_daily = self._calc_unrealized_pnl()
        effective_daily_pnl = self.daily_pnl + min(0, _unrealized_pnl_daily)  # Only count unrealized losses, not gains
        if effective_daily_pnl <= -ADAPTIVE_HYBRID_MAX_DAILY_LOSS_USD:
            cprint(f"[AdaptiveHybrid] Daily loss limit reached (${abs(effective_daily_pnl):.2f} >= ${ADAPTIVE_HYBRID_MAX_DAILY_LOSS_USD}), skipping", "red")
            return None

        # If no symbol, return None (main.py iterates symbols)
        if symbol is None:
            return None

        if symbol not in self.assets:
            return None

        # Fetch data
        if df is None:
            df = self._fetch_candles(symbol, interval='1h', candles=200)

        if df is None or len(df) < 50:
            return None

        # Compute all indicators
        ind = self._compute_indicators(df)

        # Update current regime for trade memory logging
        self._current_regime = self._detect_global_regime()

        # LLM Regime Classification (cached, runs every ~15min per symbol)
        if ADAPTIVE_HYBRID_LLM_REGIME:
            try:
                funding_zscore = self._get_funding_zscore_per_token(symbol)
                # Build indicator history for LLM regime enrichment
                _indicator_history = None
                if df is not None and len(df) >= 5:
                    try:
                        _last5 = df.iloc[-5:]
                        _indicator_history = {}
                        if 'rsi' in _last5.columns:
                            _indicator_history['rsi'] = [round(v, 1) for v in _last5['rsi'].dropna().tolist()]
                        if 'adx' in _last5.columns:
                            _indicator_history['adx'] = [round(v, 1) for v in _last5['adx'].dropna().tolist()]
                        if 'volume' in _last5.columns and 'volume_sma' in _last5.columns:
                            _vol_ratio = (_last5['volume'] / _last5['volume_sma'].replace(0, float('nan'))).dropna()
                            _indicator_history['volume_ratio'] = [round(v, 2) for v in _vol_ratio.tolist()]
                    except Exception:
                        _indicator_history = None
                regime_result = classify_regime(
                    symbol=symbol,
                    indicators=ind,
                    funding_zscore=funding_zscore,
                    model=self._llm_model,
                    bypass=not ADAPTIVE_HYBRID_LLM_REGIME,
                    indicator_history=_indicator_history,
                )
                self._llm_regime_cache[symbol] = regime_result
                self._current_regime = regime_result.get('regime', self._current_regime)
            except Exception as e:
                cprint(f"  [LLM Regime] Error for {symbol}: {e}", "yellow")

        # Get historical context from trade memory
        memory_context = self._trade_memory.build_context_prompt(symbol)
        if memory_context:
            cprint(f"  [Memory] {symbol}: {memory_context[:200]}", "cyan")

        # Run scoring modules (delegated to pure functions)
        _funding_zscore = self._get_funding_zscore_per_token(symbol)
        module_results = {
            'mean_reversion': score_mean_reversion(df, ind),
            'momentum_breakout': score_momentum_breakout(df, ind),
            'ema_trend': score_ema_trend(ind),
            'funding_composite': score_funding_composite(symbol, ind, funding_zscore=_funding_zscore),
            'rsi_divergence': score_rsi_divergence(ind),
            'sniper_lite': score_sniper_lite(df, ind),
            'ramf_lite': score_ramf_lite(df, ind),
            'oi_delta': score_oi_delta(ind, self._market_data, self._oi_history, symbol, self._cache_lock),
            'sentiment': score_sentiment(symbol, ind),
            'squeeze_detector': self._score_squeeze_detector_wrapper(symbol, ind),
            'order_imbalance': score_order_imbalance(symbol, ind),
            'crowd_positioning': score_crowd_positioning(symbol, ind),
        }

        # Add resilient modules (Phase 2) - each wrapped for error handling
        _neutral = {'score': 0, 'direction': 'NEUTRAL', 'reason': 'Error'}
        for _name, _fn, _args in [
            ('cvd', score_cvd, (symbol, ind)),
            ('vwap_deviation', score_vwap_deviation, (df, ind)),
            ('liquidation_cascade', score_liquidation_cascade, (symbol, ind)),
        ]:
            try:
                module_results[_name] = _fn(*_args)
            except Exception:
                module_results[_name] = None

        # Filter out None results (modules with data unavailability)
        available_modules = {k: v for k, v in module_results.items() if v is not None}
        unavailable_modules = {k for k, v in module_results.items() if v is None}
        if unavailable_modules:
            cprint(f"  ⚠️ [{symbol}] {len(unavailable_modules)} modules unavailable: {', '.join(unavailable_modules)}", "yellow")
        availability_ratio = len(available_modules) / max(len(module_results), 1)
        if availability_ratio < 0.6:
            cprint(f"  ⚠️ [{symbol}] Only {availability_ratio:.0%} modules available, skipping signal", "red")
            return None

        # Liquidation cascade suppression: dampen breakout/momentum signals that oppose cascade direction
        liq_result = available_modules.get('liquidation_cascade')
        if liq_result and liq_result.get('suppress_breakout'):
            cascade_direction = liq_result.get('direction', 'NEUTRAL')
            for suppress_mod in ('momentum_breakout', 'ema_trend'):
                if suppress_mod in available_modules and available_modules[suppress_mod]['score'] > 0:
                    mod_direction = available_modules[suppress_mod].get('direction', 'NEUTRAL')
                    # Only suppress if module direction opposes the cascade signal
                    if mod_direction != cascade_direction and mod_direction != 'NEUTRAL':
                        old_score = available_modules[suppress_mod]['score']
                        available_modules[suppress_mod] = {**available_modules[suppress_mod], 'score': int(old_score * 0.3)}
                        cprint(f"    [Cascade] Suppressed {suppress_mod} ({mod_direction} vs cascade {cascade_direction}): {old_score} -> {available_modules[suppress_mod]['score']}", "yellow")

        # Anomaly observation (feed data to Isolation Forest)
        try:
            anomaly_observe(ind)
        except Exception:
            pass

        # Aggregate scores (regime-adjusted weights applied inside _get_weights_for_symbol)
        aggregated = self._aggregate_scores(available_modules, symbol=symbol, ind=ind)

        # Multi-Timeframe Confluence bonus/penalty
        if ADAPTIVE_HYBRID_MTF_CONFLUENCE and aggregated['direction'] != 'NEUTRAL':
            try:
                mtf_result = score_mtf_confluence(
                    symbol=symbol,
                    primary_direction=aggregated['direction'],
                    fetch_candles_fn=self._fetch_candles,
                )
                mtf_bonus = mtf_result['score']
                if mtf_bonus != 0:
                    aggregated['score'] = max(0, aggregated['score'] + mtf_bonus)
                    cprint(f"    [MTF] {mtf_result['details']} -> score {'+' if mtf_bonus > 0 else ''}{mtf_bonus}", "cyan")
                    aggregated['mtf_confluence'] = mtf_result
            except Exception as e:
                cprint(f"  [MTF] Error for {symbol}: {e}", "yellow")

        # Anomaly filter (Phase 4)
        if ADAPTIVE_HYBRID_USE_ANOMALY_FILTER:
            is_anom, anom_score = is_anomalous(ind)
            if is_anom:
                original_score = aggregated['score']
                aggregated['score'] = aggregated['score'] / ADAPTIVE_HYBRID_ANOMALY_SCORE_DIVISOR
                cprint(f"  ⚠️ [{symbol}] ANOMALY detected (score={anom_score:.2f}), score {original_score:.1f} -> {aggregated['score']:.1f}", "red")

        # Get threshold (direction-aware)
        threshold = self._get_effective_threshold(symbol, ind=ind, direction=aggregated.get('direction'))
        urgency = self._get_urgency_multiplier(symbol)

        # Log analysis
        fired_modules = {n: r for n, r in available_modules.items() if r['score'] > 0 and r['direction'] != 'NEUTRAL'}
        silent_modules = {n: r for n, r in available_modules.items() if r['score'] == 0}
        cprint(f"  [{symbol}] Score: {aggregated['score']:.1f}/{threshold:.0f} | "
               f"Modules: {len(fired_modules)}/{len(available_modules)} | Direction: {aggregated['direction']} | "
               f"Urgency: {urgency:.0%}", "white")

        for name, result in fired_modules.items():
            cprint(f"    {name}: {result['direction']} {result['score']} - {result['reason']}", "white")

        # Warn about silent/failed modules for diagnostics
        if len(silent_modules) > len(available_modules) // 2:
            silent_names = ', '.join(silent_modules.keys())
            cprint(f"  ⚠️ [{symbol}] {len(silent_modules)}/{len(available_modules)} modules returned score=0: {silent_names}", "yellow")

        # Coverage penalty is now handled via additive adjustments in _aggregate_scores

        # Graduated volume filter (replaces binary 0.3x gate)
        volume_ratio = ind.get('volume_ratio', 1.0)
        try:
            from src import config as _cfg
            volume_min = getattr(_cfg, 'ADAPTIVE_HYBRID_VOLUME_FILTER_MIN', 0.05)
        except (ImportError, AttributeError):
            volume_min = 0.05

        if volume_ratio < volume_min:
            # Truly dead market - reject
            cprint(f"  [{symbol}] REJECTED: volume ratio {volume_ratio:.2f}x < {volume_min}x minimum", "red")
            return self._create_neutral_signal(symbol, f"Volume filter: ratio {volume_ratio:.2f}x < {volume_min}x minimum")
        elif volume_ratio < 0.5:
            aggregated['score'] -= 5  # Low volume penalty
            cprint(f"  [{symbol}] Low volume penalty: -5 (ratio={volume_ratio:.2f}x)", "yellow")
        elif volume_ratio > 2.0:
            aggregated['score'] += 3  # Volume confirmation bonus
            cprint(f"  [{symbol}] Volume confirmation bonus: +3 (ratio={volume_ratio:.2f}x)", "cyan")

        # 4H Trend Filter (primary gate before trade decision)
        try:
            from src import config as _cfg
            use_4h_filter = getattr(_cfg, 'ADAPTIVE_HYBRID_4H_TREND_FILTER', False)
        except (ImportError, AttributeError):
            use_4h_filter = False

        if use_4h_filter and aggregated['direction'] != 'NEUTRAL':
            direction = aggregated['direction']
            trend_4h = self._get_4h_trend(symbol)

            if trend_4h == 'NEUTRAL':
                # 4h is neutral - reduce score
                try:
                    penalty_pct = getattr(_cfg, 'ADAPTIVE_HYBRID_4H_TREND_PENALTY', 0.30)
                except (NameError, AttributeError):
                    penalty_pct = 0.30
                aggregated['score'] *= (1 - penalty_pct)
                cprint(f"  [{symbol}] 4H neutral filter: score reduced by {penalty_pct*100:.0f}% to {aggregated['score']:.1f}", "yellow")
            elif (direction == 'BUY' and trend_4h == 'BEARISH') or (direction == 'SELL' and trend_4h == 'BULLISH'):
                # 4h opposes signal direction
                try:
                    reject_opposing = getattr(_cfg, 'ADAPTIVE_HYBRID_4H_TREND_REJECT', True)
                except (NameError, AttributeError):
                    reject_opposing = True
                if reject_opposing:
                    cprint(f"  [{symbol}] 4H trend REJECTS {direction} signal (4h={trend_4h})", "red")
                    return self._create_neutral_signal(symbol, f"4H trend filter: {trend_4h} opposes {direction}")

        # Decision
        if aggregated['direction'] != 'NEUTRAL' and aggregated['score'] >= threshold:
            direction = aggregated['direction']
            final_score = aggregated['score']

            # Clean up stale pending signals
            self._cleanup_stale_pending_signals()

            # 1-Bar Confirmation Delay (Change #2)
            try:
                from src import config as _cfg
                confirmation_bars = getattr(_cfg, 'ADAPTIVE_HYBRID_CONFIRMATION_BARS', 0)
            except (ImportError, AttributeError):
                confirmation_bars = 0

            if confirmation_bars > 0:
                pending_key = f"{symbol}_{direction}"
                if pending_key not in self._pending_signals:
                    # First time seeing this signal - store as pending
                    self._pending_signals[pending_key] = {
                        'timestamp': datetime.now(),
                        'score': final_score,
                        'direction': direction,
                        'count': 1,
                    }
                    # Clear opposite direction pending
                    opposite = 'BUY' if direction == 'SELL' else 'SELL'
                    self._pending_signals.pop(f"{symbol}_{opposite}", None)
                    cprint(f"  [{symbol}] Signal pending confirmation: {direction} score={final_score:.1f}", "yellow")
                    return self._create_neutral_signal(symbol, "Awaiting confirmation bar")
                else:
                    self._pending_signals[pending_key]['count'] += 1
                    if self._pending_signals[pending_key]['count'] <= confirmation_bars:
                        cprint(f"  [{symbol}] Signal confirmed {self._pending_signals[pending_key]['count']}/{confirmation_bars+1}: {direction}", "yellow")
                        return self._create_neutral_signal(symbol, f"Confirmation {self._pending_signals[pending_key]['count']}/{confirmation_bars+1}")
                    # Signal confirmed - proceed with trade
                    cprint(f"  [{symbol}] Signal CONFIRMED after {confirmation_bars} bars: {direction} score={final_score:.1f}", "green", attrs=['bold'])
                    del self._pending_signals[pending_key]

            # Map score linearly to signal strength (0.45 to 0.95)
            score = round(aggregated['score'], 1)
            score_range = max(100 - threshold, 1)  # Guard against division by zero
            strength = 0.45 + (score - threshold) / score_range * 0.50
            strength = max(0.45, min(0.95, strength))

            # Calculate ATR-based SL/TP
            atr = ind['atr']
            if atr > 0:
                sl_mult, tp_mult = self._get_atr_profile(symbol)
                sl_pct = (atr * sl_mult / ind['close']) * 100
                tp_pct = (atr * tp_mult / ind['close']) * 100
            else:
                cprint(f"  [{symbol}] ATR=0, rejecting signal (insufficient data)", "yellow")
                return None

            # Enforce minimum reward:risk ratio
            if tp_pct < sl_pct * ADAPTIVE_HYBRID_MIN_RR_RATIO:
                tp_pct = sl_pct * ADAPTIVE_HYBRID_MIN_RR_RATIO

            # LLM Trade Confirmation (final gate before signal emission)
            llm_decision = None
            if score >= 80:
                cprint(f"  [{symbol}] LLM bypassed: score {score:.1f} >= 80 (auto-confirmed)", "green")
            elif ADAPTIVE_HYBRID_LLM_CONFIRMATION:
                try:
                    signal_metadata = {
                        'score': score,
                        'threshold': threshold,
                        'signal_strength': strength,
                        'stop_loss_pct': sl_pct,
                        'take_profit_pct': tp_pct,
                        'module_scores': aggregated.get('module_scores', {}),
                    }
                    llm_decision = llm_confirm_trade(
                        symbol=symbol,
                        direction=aggregated['direction'],
                        aggregated=aggregated,
                        indicators=ind,
                        metadata=signal_metadata,
                        trade_memory=self._trade_memory,
                        model=self._llm_model,
                        bypass=not ADAPTIVE_HYBRID_LLM_CONFIRMATION,
                    )
                    if llm_decision['decision'] == 'REJECT':
                        cprint(f"  [{symbol}] LLM REJECTED: {llm_decision['reasoning']}", "red")
                        return {
                            'token': symbol,
                            'signal': 0,
                            'direction': 'NEUTRAL',
                            'metadata': {
                                'strategy': 'Adaptive Hybrid',
                                'score': score,
                                'threshold': threshold,
                                'reason': f"LLM rejected: {llm_decision['reasoning']}",
                                'llm_decision': llm_decision,
                                'current_price': ind['close'],
                            }
                        }
                    elif llm_decision['decision'] == 'ADJUST':
                        if llm_decision.get('adjusted_score') is not None:
                            old_score = score
                            score = llm_decision['adjusted_score']
                            aggregated['score'] = score
                            cprint(f"  [{symbol}] LLM ADJUSTED score: {old_score:.1f} -> {score:.1f}", "yellow")
                        if llm_decision.get('sl_adjustment') is not None:
                            sl_pct = llm_decision['sl_adjustment']
                        if llm_decision.get('tp_adjustment') is not None:
                            tp_pct = llm_decision['tp_adjustment']
                        # Re-enforce R:R after adjustment
                        if tp_pct < sl_pct * ADAPTIVE_HYBRID_MIN_RR_RATIO:
                            tp_pct = sl_pct * ADAPTIVE_HYBRID_MIN_RR_RATIO
                        # Recalculate strength after score adjustment
                        strength = 0.45 + (score - threshold) / score_range * 0.50
                        strength = max(0.45, min(0.95, strength))
                        if score < threshold:
                            cprint(f"  [{symbol}] LLM adjusted score below threshold, rejecting", "red")
                            return {
                                'token': symbol,
                                'signal': 0,
                                'direction': 'NEUTRAL',
                                'metadata': {
                                    'strategy': 'Adaptive Hybrid',
                                    'score': score,
                                    'threshold': threshold,
                                    'reason': f"LLM adjusted score {score:.1f} < threshold {threshold:.0f}",
                                    'current_price': ind['close'],
                                }
                            }
                except Exception as e:
                    cprint(f"  [LLM Confirm] Error: {e}, proceeding with signal", "yellow")

            cprint(f"  [{symbol}] SIGNAL: {aggregated['direction']} (score={score:.1f}, "
                   f"threshold={threshold:.0f}, strength={strength:.0%})", "green", attrs=['bold'])

            return {
                'token': symbol,
                'signal': strength,
                'direction': aggregated['direction'],
                'metadata': {
                    'strategy': 'Adaptive Hybrid',
                    'score': aggregated['score'],
                    'threshold': threshold,
                    'urgency_multiplier': urgency,
                    'active_modules': aggregated['active_modules'],
                    'total_fired': aggregated['total_modules_fired'],
                    'coverage': aggregated.get('coverage', 0),
                    'module_scores': aggregated.get('module_scores', {}),
                    'reason': aggregated['details'],
                    'current_price': ind['close'],
                    'stop_loss_pct': round(sl_pct, 2),
                    'take_profit_pct': round(tp_pct, 2),
                    'atr': atr,
                    'rsi': ind['rsi'],
                    'adx': ind['adx'],
                    'llm_decision': llm_decision,
                    'llm_regime': self._llm_regime_cache.get(symbol),
                    'mtf_confluence': aggregated.get('mtf_confluence'),
                }
            }

        # NEUTRAL signal with diagnostic info
        return {
            'token': symbol,
            'signal': 0,
            'direction': 'NEUTRAL',
            'metadata': {
                'strategy': 'Adaptive Hybrid',
                'score': aggregated['score'],
                'threshold': threshold,
                'urgency_multiplier': urgency,
                'reason': f"Score {aggregated['score']:.1f} < threshold {threshold:.0f}",
                'diagnostic': aggregated.get('details', ''),
                'module_scores': aggregated.get('module_scores', {}),
                'current_price': ind['close'],
            }
        }

    # =========================================================================
    # BATCH SIGNAL GENERATION (parallel)
    # =========================================================================

    def generate_signals_batch(self, symbols: list) -> list:
        """Generate signals for multiple symbols in parallel using ThreadPoolExecutor.

        _fetch_candles() is HTTP-based and thread-safe. Paper position mutations
        are protected by _position_lock in execute_paper_trade / _close_paper_position.
        """
        results = []

        def _analyze_symbol(symbol):
            try:
                return self.generate_signals(symbol=symbol)
            except Exception as e:
                cprint(f"[AdaptiveHybrid] Error analyzing {symbol}: {e}", "red")
                return None

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_analyze_symbol, sym): sym for sym in symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    signal = future.result()
                    if signal is not None:
                        results.append(signal)
                except Exception as e:
                    cprint(f"[AdaptiveHybrid] Thread error for {symbol}: {e}", "red")

        return results

    # =========================================================================
    # PAPER TRADING
    # =========================================================================

    def _calc_unrealized_pnl(self) -> float:
        """Calculate total unrealized PnL from open positions. Caller must hold _position_lock."""
        unrealized = 0.0
        for pos in self.paper_positions.values():
            entry_price = pos.get('entry_price', 0)
            current_price = pos.get('current_price', entry_price)
            position_size = pos.get('position_size', 0)
            direction = pos.get('direction', 'BUY')
            if entry_price > 0:
                if direction == 'BUY':
                    unrealized += position_size * (current_price - entry_price) / entry_price
                else:
                    unrealized += position_size * (entry_price - current_price) / entry_price
        return unrealized

    def _reset_daily_counters(self):
        """Reset daily counters if new day. Thread-safe."""
        today = datetime.utcnow().date()
        with self._position_lock:
            if self.last_trade_date != today:
                self.daily_trades = 0
                self.daily_pnl = 0.0
                # Note: consecutive_losses is NOT reset on date change —
                # it only resets after a winning trade (handled in _close_paper_position)
                self._loss_breaker_until = None
                self.last_trade_date = today

    @staticmethod
    def _price_decimals(price: float) -> int:
        """Return appropriate number of decimal places for rounding based on price magnitude."""
        if price >= 100:
            return 2
        if price >= 1:
            return 3
        if price >= 0.01:
            return 5
        return 7  # micro-price tokens like kPEPE

    def _prepare_trade(self, signal: dict) -> dict:
        """Validate signal and compute all trade parameters (shared by paper and live).

        Returns a dict with all trade parameters ready for execution, or None if
        the trade is rejected by any pre-trade check (duplicate, drawdown,
        correlation, margin, etc.).
        """
        symbol = signal.get('token', '')
        direction = signal.get('direction', 'NEUTRAL')
        cprint(f"[AdaptiveHybrid] _prepare_trade START: {direction} {symbol}", "cyan")

        if not symbol or direction == 'NEUTRAL':
            cprint(f"❌ [{symbol or 'UNKNOWN'}] REJECTED: invalid signal (symbol='{symbol}', direction='{direction}')", "red")
            return None

        # Check for ANY existing position on same symbol (block opposite-direction too)
        cprint(f"[AdaptiveHybrid] Acquiring _position_lock for duplicate check...", "cyan")
        with self._position_lock:
            existing = [p for p in self.paper_positions.values()
                        if p['symbol'] == symbol]
            num_positions = len(self.paper_positions)
        if existing:
            cprint(f"[AdaptiveHybrid] {symbol} already has open position ({existing[0]['direction']}), skipping", "yellow")
            cprint(f"❌ [{symbol}] REJECTED: duplicate position already open ({existing[0]['direction']})", "red")
            return None

        # Max simultaneous positions check
        try:
            from src.config import RISK_MAX_POSITIONS as _risk_max_pos
        except ImportError:
            _risk_max_pos = 4
        if num_positions >= _risk_max_pos:
            cprint(f"[AdaptiveHybrid] Max positions reached ({_risk_max_pos}), skipping", "yellow")
            cprint(f"❌ [{symbol}] REJECTED: max positions reached ({num_positions}/{_risk_max_pos})", "red")
            return None

        # HWM Drawdown check (mark-to-market: includes unrealized PnL)
        cprint(f"[AdaptiveHybrid] Passed duplicate check, checking HWM drawdown...", "cyan")
        with self._position_lock:
            unrealized_pnl = self._calc_unrealized_pnl()

        mark_to_market_equity = self.paper_balance + unrealized_pnl
        self.peak_balance = max(self.peak_balance, mark_to_market_equity)
        hwm_drawdown_pct = (self.peak_balance - mark_to_market_equity) / self.peak_balance * 100
        if hwm_drawdown_pct >= RISK_MAX_DRAWDOWN_PCT:
            cprint(f"[AdaptiveHybrid] HWM DRAWDOWN BREAKER: {hwm_drawdown_pct:.1f}% from peak ${self.peak_balance:,.2f}", "red", attrs=['bold'])
            cprint(f"❌ [{symbol}] REJECTED: HWM drawdown {hwm_drawdown_pct:.1f}% >= {RISK_MAX_DRAWDOWN_PCT}% (peak ${self.peak_balance:,.2f}, equity ${mark_to_market_equity:,.2f})", "red")
            return None

        # Correlation-aware position check
        cprint(f"[AdaptiveHybrid] Checking correlation limits ({len(self.paper_positions)} existing positions)...", "cyan")
        if len(self.paper_positions) > 0:
            same_direction_positions = [p for p in self.paper_positions.values() if p['direction'] == direction]
            if len(same_direction_positions) >= 2:
                cprint(f"[AdaptiveHybrid] Fetching BTC correlation for {symbol}...", "cyan")
                new_corr = self._get_btc_correlation(symbol)
                if new_corr is not None and abs(new_corr) > 0.7:
                    avg_corr = sum(abs(self._get_btc_correlation(p['symbol']) or 0) for p in same_direction_positions) / len(same_direction_positions)
                    if avg_corr > 0.6:
                        cprint(f"[AdaptiveHybrid] CORRELATION LIMIT: {len(same_direction_positions)} correlated {direction} positions (avg corr={avg_corr:.2f})", "yellow")
                        cprint(f"❌ [{symbol}] REJECTED: correlation block ({len(same_direction_positions)} {direction} positions, avg_corr={avg_corr:.2f}, new_corr={new_corr:.2f})", "red")
                        return None

        # Total notional exposure cap
        try:
            from src.config import ADAPTIVE_HYBRID_MAX_NOTIONAL_PCT
        except ImportError:
            ADAPTIVE_HYBRID_MAX_NOTIONAL_PCT = 500

        with self._position_lock:
            current_notional = sum(p.get('position_size', 0) for p in self.paper_positions.values())
        max_notional = self.paper_balance * ADAPTIVE_HYBRID_MAX_NOTIONAL_PCT / 100
        if current_notional >= max_notional:
            cprint(f"[AdaptiveHybrid] NOTIONAL CAP: ${current_notional:.0f} >= ${max_notional:.0f} ({ADAPTIVE_HYBRID_MAX_NOTIONAL_PCT}% of balance)", "yellow")
            cprint(f"❌ [{symbol}] REJECTED: notional cap reached (${current_notional:.0f} >= ${max_notional:.0f}, {ADAPTIVE_HYBRID_MAX_NOTIONAL_PCT}% of balance)", "red")
            return None

        metadata = signal.get('metadata', {})
        price = metadata.get('current_price', 0)
        cprint(f"[AdaptiveHybrid] Price from metadata: {price}", "cyan")

        if price == 0:
            cprint(f"[AdaptiveHybrid] No price in metadata, fetching candles...", "cyan")
            df = self._fetch_candles(symbol, interval='1h', candles=5)
            if df is not None and len(df) > 0:
                price = float(df['close'].iloc[-1])

        if price <= 0:
            cprint(f"[AdaptiveHybrid] Cannot execute trade with price={price}", "red")
            cprint(f"❌ [{symbol}] REJECTED: invalid price ({price}) - failed to fetch from metadata or candles", "red")
            return None

        sl_pct = metadata.get('stop_loss_pct', 1.5)
        tp_pct = metadata.get('take_profit_pct', 2.5)
        atr = metadata.get('atr', 0)

        # Dynamic leverage by token class
        token_class = 'mid'  # default
        for cls, profile in ADAPTIVE_HYBRID_ATR_PROFILES.items():
            if symbol in profile['tokens']:
                token_class = cls
                break
        leverage = ADAPTIVE_HYBRID_LEVERAGE_PROFILES.get(token_class, ADAPTIVE_HYBRID_LEVERAGE)

        # Apply entry slippage BEFORE SL/TP calculation so levels are relative to actual fill price
        entry_slippage = PAPER_SLIPPAGE_V2.get(token_class, 0.001)
        if direction == 'BUY':
            price = price * (1 + entry_slippage)
        else:
            price = price * (1 - entry_slippage)

        # Calculate SL/TP prices from slippage-adjusted entry price
        if direction == 'BUY':
            stop_loss_price = price * (1 - sl_pct / 100)
            take_profit_price = price * (1 + tp_pct / 100)
        else:
            stop_loss_price = price * (1 + sl_pct / 100)
            take_profit_price = price * (1 - tp_pct / 100)

        # Atomic: margin check + position sizing + ID generation
        cprint(f"[AdaptiveHybrid] Acquiring _position_lock for margin check + sizing...", "cyan")
        with self._position_lock:
            used_margin = sum(
                pos.get('position_size', 0) / pos.get('leverage', ADAPTIVE_HYBRID_LEVERAGE)
                for pos in self.paper_positions.values()
            )

            # Apply cash reserve
            cash_reserve = self.paper_balance * (CASH_PERCENTAGE / 100)
            available_margin = max(0, self.paper_balance - used_margin - cash_reserve)

            # Score-based exposure factor: stronger signals get larger positions
            # score 40 -> 1.0x, score 55 -> 1.5x, score 70+ -> 2.0x
            score_val = metadata.get('score', 40)
            score_exposure = min(2.0, 1.0 + max(0, score_val - 40) / 30)

            # Position sizing: 2% risk per trade, modulated by signal strength
            strength = signal.get('signal', 0.7)
            strength_multiplier = 0.5 + float(strength)  # Range: 0.5x to 1.5x
            try:
                from src import config as _cfg
                risk_pct = getattr(_cfg, 'ADAPTIVE_HYBRID_RISK_PCT', 0.015)
            except (ImportError, AttributeError):
                risk_pct = 0.015
            risk_amount = self.paper_balance * risk_pct * strength_multiplier
            sl_fraction = max(sl_pct / 100, 0.001)  # Guard against near-zero SL
            position_size = risk_amount / sl_fraction * leverage
            max_position_by_margin = available_margin * 0.9 * leverage
            # Cap on notional exposure, scaled by score
            max_position_by_pct = self.paper_balance * (ADAPTIVE_HYBRID_MAX_POSITION_PCT / 100) * score_exposure
            position_size = min(position_size, max_position_by_margin, max_position_by_pct)

            # Volatility targeting: cap position so daily vol contribution stays under target, scaled by score
            if atr > 0 and price > 0:
                daily_vol_pct = atr / price  # ATR as % of price ~ daily vol
                vol_target_usd = self.paper_balance * (ADAPTIVE_HYBRID_VOL_TARGET_DAILY_PCT / 100) * score_exposure
                if daily_vol_pct > 0:
                    vol_target_size = vol_target_usd / daily_vol_pct
                    position_size = min(position_size, vol_target_size)

            # Apply recovery mode size reduction from risk agent
            if self._risk_agent:
                recovery_factor = self._risk_agent.get_recovery_size_factor()
                if recovery_factor < 1.0:
                    position_size *= recovery_factor
                    cprint(f"[AdaptiveHybrid] Recovery mode: position size reduced by {(1-recovery_factor)*100:.0f}%", "yellow")

            # Weekend size reduction
            if datetime.utcnow().weekday() >= 5:  # Saturday=5, Sunday=6
                try:
                    from src.config import ADAPTIVE_HYBRID_WEEKEND_SIZE_REDUCTION
                except ImportError:
                    ADAPTIVE_HYBRID_WEEKEND_SIZE_REDUCTION = 0.30
                position_size *= (1.0 - ADAPTIVE_HYBRID_WEEKEND_SIZE_REDUCTION)
                cprint(f"  [{symbol}] Weekend reduction: size reduced by {ADAPTIVE_HYBRID_WEEKEND_SIZE_REDUCTION*100:.0f}%", "cyan")

            # Event calendar: reduce size near high-impact macro events
            try:
                from src.strategies.modules.event_calendar import check_upcoming_events
                upcoming = check_upcoming_events()
                if upcoming:
                    position_size *= (1.0 - upcoming['size_reduction'])
                    cprint(f"  [{symbol}] Event reduction: {upcoming['event']} in {upcoming['hours_until']:.1f}h, size -{upcoming['size_reduction']*100:.0f}%", "cyan")
            except ImportError:
                pass

            # Post-breaker size reduction (50% for 3 trades after loss breaker)
            if hasattr(self, '_post_breaker_trades') and self._post_breaker_trades > 0:
                position_size *= 0.5
                self._post_breaker_trades -= 1
                cprint(f"  [{symbol}] Post-breaker reduction: size *= 0.5 ({self._post_breaker_trades} trades remaining)", "yellow")

            # Kelly-adaptive sizing from trade memory
            try:
                from src.config import ADAPTIVE_HYBRID_KELLY_LOOKBACK, ADAPTIVE_HYBRID_KELLY_FRACTION
                if self._trade_memory:
                    import sqlite3
                    conn = sqlite3.connect(self._trade_memory.db_path)
                    try:
                        cursor = conn.execute(
                            "SELECT outcome_pnl FROM decisions WHERE outcome_pnl IS NOT NULL ORDER BY timestamp DESC LIMIT ?",
                            (ADAPTIVE_HYBRID_KELLY_LOOKBACK,)
                        )
                        recent_pnls = [row[0] for row in cursor.fetchall()]
                    finally:
                        conn.close()

                    if len(recent_pnls) >= 10:
                        wins = [p for p in recent_pnls if p > 0]
                        losses = [p for p in recent_pnls if p <= 0]
                        if wins and losses:
                            win_rate = len(wins) / len(recent_pnls)
                            avg_win = sum(wins) / len(wins)
                            avg_loss = abs(sum(losses) / len(losses))
                            if avg_loss > 0:
                                payoff_ratio = avg_win / avg_loss
                                kelly = win_rate - (1 - win_rate) / payoff_ratio
                                kelly = max(0, kelly)  # Never negative
                                kelly_size_pct = kelly * ADAPTIVE_HYBRID_KELLY_FRACTION

                                # Only reduce, never increase beyond base
                                base_risk_pct = ADAPTIVE_HYBRID_MAX_POSITION_PCT / 100
                                if kelly_size_pct < base_risk_pct:
                                    reduction = max(0.25, kelly_size_pct / base_risk_pct)  # Floor 25% to avoid deadlock
                                    position_size *= reduction
                                    cprint(f"  [{symbol}] Kelly adjustment: {kelly:.1%} optimal, using {kelly_size_pct:.1%} (half-Kelly), size *= {reduction:.2f}", "cyan")
            except Exception as e:
                cprint(f"  [{symbol}] Kelly sizing skipped: {e}", "yellow")

            if position_size < ADAPTIVE_HYBRID_VOL_MIN_POSITION_USD:
                cprint(f"[AdaptiveHybrid] Insufficient margin (${position_size:.0f} < ${ADAPTIVE_HYBRID_VOL_MIN_POSITION_USD})", "red")
                cprint(f"❌ [{symbol}] REJECTED: position_size ${position_size:.2f} < min ${ADAPTIVE_HYBRID_VOL_MIN_POSITION_USD:.2f} (avail_margin=${available_margin:.2f}, risk_amt=${risk_amount:.2f}, lev={leverage}x - check Kelly/recovery/vol caps)", "red")
                return None

            # Use V2 fees (more realistic)
            entry_fee = position_size * PAPER_TAKER_FEE_V2

            # Generate position ID (under lock to avoid counter race)
            self._position_counter += 1
            position_id = f"AH_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._position_counter}"

        trade = {
            'position_id': position_id,
            'timestamp': datetime.now().isoformat(),
            'entry_time': datetime.now(),
            'symbol': symbol,
            'direction': direction,
            'entry_price': price,
            'position_size': round(position_size, 2),
            'leverage': leverage,
            'stop_loss': round(stop_loss_price, self._price_decimals(price)),
            'take_profit': round(take_profit_price, self._price_decimals(price)),
            'sl_pct': sl_pct,
            'tp_pct': tp_pct,
            'atr': atr,
            'entry_fee': round(entry_fee, 4),
            'confidence': round(float(signal.get('signal', 0)) * 100, 1),
            'status': 'OPEN',
            'score': metadata.get('score', 0),
            'modules': json.dumps(metadata.get('module_scores', {})),
            'scale_out_level': 0,
            'partial_pnl_realized': 0.0,
            # Extra context for logging
            '_risk_amount': risk_amount,
            '_token_class': token_class,
        }

        return trade

    def execute_paper_trade(self, signal: dict) -> dict:
        """Execute a paper trade (simulation) with capital protection checks."""
        # Hold _position_lock through prepare + registration to prevent race conditions.
        # _position_lock is an RLock so nested acquisitions inside _prepare_trade() are safe.
        with self._position_lock:
            trade = self._prepare_trade(signal)
            if trade is None:
                return None

            position_id = trade['position_id']
            symbol = trade['symbol']
            direction = trade['direction']
            price = trade['entry_price']
            position_size = trade['position_size']
            leverage = trade['leverage']
            sl_pct = trade['sl_pct']
            tp_pct = trade['tp_pct']
            entry_fee = trade['entry_fee']
            risk_amount = trade.pop('_risk_amount', 0)
            trade.pop('_token_class', None)
            metadata = signal.get('metadata', {})

            # Balance floor check: reject trade if balance would go negative
            if self.paper_balance - entry_fee < 0:
                cprint(f"[AdaptiveHybrid] Rejecting trade: balance {self.paper_balance:.2f} < entry_fee {entry_fee:.4f}", "red")
                return None

            # Atomic insertion + balance update
            self.paper_positions[position_id] = trade
            self.paper_balance -= entry_fee
            self.daily_trades += 1
            self.last_trade_time = datetime.now()
            self.last_trade_time_per_token[symbol] = datetime.now()

        # Log to file
        log_file = os.path.join(self.data_dir, 'paper_trades.csv')
        df = pd.DataFrame([trade])
        if os.path.exists(log_file):
            df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            df.to_csv(log_file, index=False)

        _pd = self._price_decimals(price)
        cprint(f"\n[ADAPTIVE HYBRID] Opened {direction} {symbol} (ID: {position_id})", "magenta", attrs=['bold'])
        cprint(f"  Entry: ${price:,.{_pd}f} | Size: ${position_size:,.2f} | Leverage: {leverage}x | Score: {metadata.get('score', 0):.1f}", "white")
        cprint(f"  Risk: ${risk_amount:,.2f} | SL: {sl_pct:.2f}% | Fee: ${entry_fee:.4f}", "white")
        cprint(f"  SL: ${trade['stop_loss']:,.{_pd}f} ({sl_pct:.2f}%)", "white")
        cprint(f"  TP: ${trade['take_profit']:,.{_pd}f} ({tp_pct:.2f}%)", "white")

        # Log decision to trade memory
        try:
            module_scores = metadata.get('module_scores', {})
            modules_firing = [m for m in module_scores if module_scores[m] != 0] if isinstance(module_scores, dict) else None
            decision_id = self._trade_memory.log_decision(
                symbol=symbol,
                direction=direction,
                confidence=signal.get('signal', 0) * 100,
                source='adaptive_hybrid',
                reasoning=str(metadata.get('reason', '')),
                market_regime=self._current_regime,
                key_indicators={
                    'rsi': metadata.get('rsi'),
                    'adx': metadata.get('adx'),
                    'atr': trade.get('atr'),
                    'score': trade.get('score'),
                },
                modules_firing=modules_firing,
            )
            trade['memory_decision_id'] = decision_id
            with self._position_lock:
                if position_id in self.paper_positions:
                    self.paper_positions[position_id]['memory_decision_id'] = decision_id
        except Exception as e:
            cprint(f"  [Memory] Warning: could not log decision: {e}", "yellow")

        return trade

    def execute_live_trade(self, signal: dict) -> dict:
        """Execute a live trade on HyperLiquid with the same sizing/risk as paper trading.

        Uses _prepare_trade() for all validation and sizing, then places a real
        market order via HyperLiquid SDK + bracket SL/TP orders via LiveOrderManager.
        The position is also tracked in self.paper_positions so the dashboard can
        display it.
        """
        # --- Fix #3: Sync balance with real account before sizing ---
        try:
            import eth_account as _eth_account_sync
            from hyperliquid.info import Info as HLInfoSync
            _pk_sync = os.getenv('HYPER_LIQUID_ETH_PRIVATE_KEY')
            if _pk_sync:
                _acct_sync = _eth_account_sync.Account.from_key(_pk_sync)
                _info_sync = HLInfoSync('https://api.hyperliquid.xyz', skip_ws=True)
                _user_state_sync = _info_sync.user_state(_acct_sync.address)
                _real_balance = float(_user_state_sync.get('marginSummary', {}).get('accountValue', 0))
                if _real_balance > 0:
                    self.paper_balance = _real_balance
                    cprint(f"[AdaptiveHybrid] LIVE: Synced balance: ${_real_balance:,.2f}", "cyan")
        except Exception as _sync_err:
            cprint(f"[AdaptiveHybrid] LIVE: Could not sync balance: {_sync_err}", "yellow")

        trade = self._prepare_trade(signal)
        if trade is None:
            return None

        position_id = trade['position_id']
        symbol = trade['symbol']
        direction = trade['direction']
        price = trade['entry_price']
        position_size = trade['position_size']
        leverage = trade['leverage']
        sl_pct = trade['sl_pct']
        tp_pct = trade['tp_pct']
        entry_fee = trade['entry_fee']
        risk_amount = trade.pop('_risk_amount', 0)
        token_class = trade.pop('_token_class', 'mid')
        metadata = signal.get('metadata', {})

        # --- Fix #4: Absolute cap on live order size ---
        if position_size > _MAX_LIVE_ORDER_USD:
            cprint(f"[AdaptiveHybrid] LIVE: Position ${position_size:.0f} exceeds cap ${_MAX_LIVE_ORDER_USD}, clamping", "yellow")
            trade['position_size'] = _MAX_LIVE_ORDER_USD
            position_size = _MAX_LIVE_ORDER_USD

        # --- Place market order on HyperLiquid ---
        try:
            import eth_account as _eth_account
            from hyperliquid.exchange import Exchange as HLExchange
            from hyperliquid.info import Info as HLInfo
            from src.nice_funcs_hyperliquid import ask_bid, get_sz_px_decimals

            private_key = os.getenv('HYPER_LIQUID_ETH_PRIVATE_KEY')
            if not private_key:
                cprint("[AdaptiveHybrid] LIVE TRADE ABORTED: HYPER_LIQUID_ETH_PRIVATE_KEY not set", "red")
                return None

            account = _eth_account.Account.from_key(private_key)
            exchange = HLExchange(account, 'https://api.hyperliquid.xyz')

            # Set leverage on exchange before placing order
            try:
                exchange.update_leverage(leverage, symbol, is_cross=True)
                cprint(f"[AdaptiveHybrid] Set leverage to {leverage}x for {symbol}", "cyan")
            except Exception as e:
                cprint(f"[AdaptiveHybrid] Warning: could not set leverage: {e}", "yellow")

            # Get current market price for IOC fill
            ask, bid, _ = ask_bid(symbol)
            if direction == 'BUY':
                fill_price = ask * 1.001  # 0.1% above ask for fill
            else:
                fill_price = bid * 0.999  # 0.1% below bid for fill

            # Calculate size in asset units
            sz_decimals, px_decimals = get_sz_px_decimals(symbol)
            asset_size = round(position_size / fill_price, sz_decimals)

            if asset_size <= 0:
                cprint(f"[AdaptiveHybrid] LIVE TRADE ABORTED: asset_size={asset_size} after rounding", "red")
                return None

            # Round fill price
            if symbol == 'BTC':
                fill_price = round(fill_price)
            else:
                fill_price = round(fill_price, px_decimals) if px_decimals > 0 else round(fill_price, 1)

            is_buy = direction == 'BUY'
            cprint(f"[AdaptiveHybrid] LIVE: Placing IOC {'BUY' if is_buy else 'SELL'} {asset_size} {symbol} @ ${fill_price:,.2f}", "magenta")

            order_result = exchange.order(
                symbol, is_buy, asset_size, fill_price,
                {"limit": {"tif": "Ioc"}}, reduce_only=False
            )

            # Log only relevant fields, not the full response (avoid leaking sensitive info)
            log_info = {
                'type': order_result.get('response', {}).get('type', '?') if order_result else '?',
                'status': 'error' if order_result and order_result.get('response', {}).get('type') == 'error' else 'ok',
            }
            cprint(f"[AdaptiveHybrid] LIVE: Order result: {log_info}", "cyan")

            # Check for error response
            if order_result and order_result.get('response', {}).get('type') == 'error':
                error_msg = order_result.get('response', {}).get('data', {}).get('msg', 'unknown')
                cprint(f"[AdaptiveHybrid] LIVE TRADE FAILED: {error_msg}", "red")
                return None

            # --- Fix #2: Verify IOC fill status ---
            filled_size = asset_size  # default to full size
            try:
                if order_result:
                    _resp = order_result.get('response', {})
                    _data = _resp.get('data', {}) if isinstance(_resp, dict) else {}
                    _statuses = _data.get('statuses', []) if isinstance(_data, dict) else []
                    if _statuses:
                        _status = _statuses[0]
                        if 'filled' in _status:
                            filled_size = float(_status['filled'].get('totalSz', asset_size))
                        elif 'resting' in _status:
                            filled_size = 0  # Order resting, not filled
                        elif 'error' in _status:
                            cprint(f"[AdaptiveHybrid] LIVE: Order error: {_status['error']}", "red")
                            return None
            except Exception as _parse_err:
                cprint(f"[AdaptiveHybrid] LIVE: Could not parse fill: {_parse_err}", "yellow")

            if filled_size <= 0:
                cprint(f"[AdaptiveHybrid] LIVE: Order not filled (IOC expired)", "red")
                return None

            # Adjust for partial fills
            if filled_size < asset_size:
                cprint(f"[AdaptiveHybrid] LIVE: Partial fill: {filled_size}/{asset_size}", "yellow")
                fill_ratio = filled_size / asset_size
                trade['position_size'] = round(trade['position_size'] * fill_ratio, 2)
                trade['entry_fee'] = round(trade['entry_fee'] * fill_ratio, 4)
                position_size = trade['position_size']
                entry_fee = trade['entry_fee']
                asset_size = filled_size

        except Exception as e:
            cprint(f"[AdaptiveHybrid] LIVE TRADE FAILED: {e}", "red")
            import traceback
            traceback.print_exc()
            return None

        # --- Place bracket SL/TP orders ---
        bracket = None
        try:
            if self._live_order_manager is None:
                from src.execution.order_manager import LiveOrderManager
                self._live_order_manager = LiveOrderManager()
            lom = self._live_order_manager
            bracket = lom.place_bracket_order(
                symbol=symbol,
                direction=direction,
                size=asset_size,
                entry_price=price,
                sl_price=trade['stop_loss'],
                tp_price=trade['take_profit'],
            )
            if bracket:
                trade['sl_oid'] = bracket.get('sl_oid')
                trade['tp_oid'] = bracket.get('tp_oid')
                cprint(f"[AdaptiveHybrid] LIVE: Bracket orders placed (SL/TP)", "green")
        except Exception as e:
            cprint(f"[AdaptiveHybrid] LIVE: WARNING - Could not place bracket orders: {e}", "red")
            bracket = None

        # --- Fix #1: If bracket SL/TP failed, close position immediately ---
        if not bracket:
            cprint(f"[AdaptiveHybrid] LIVE: CRITICAL - SL/TP failed, closing position immediately", "red", attrs=['bold'])
            try:
                close_side = not is_buy  # reverse direction
                exchange.order(
                    symbol, close_side, asset_size, fill_price,
                    {"limit": {"tif": "Ioc"}}, reduce_only=True
                )
                cprint(f"[AdaptiveHybrid] LIVE: Emergency close sent for {asset_size} {symbol}", "yellow")
            except Exception as close_err:
                cprint(f"[AdaptiveHybrid] LIVE: EMERGENCY CLOSE FAILED for {symbol}: {close_err}", "red", attrs=['bold'])
            return None

        # --- Track position in paper_positions for dashboard visibility ---
        trade['mode'] = 'LIVE'
        trade['asset_size'] = asset_size

        with self._position_lock:
            self.paper_positions[position_id] = trade
            self.paper_balance -= entry_fee
            self.daily_trades += 1
            self.last_trade_time = datetime.now()
            self.last_trade_time_per_token[symbol] = datetime.now()

        # Log to file
        log_file = os.path.join(self.data_dir, 'paper_trades.csv')
        df = pd.DataFrame([trade])
        if os.path.exists(log_file):
            df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            df.to_csv(log_file, index=False)

        cprint(f"\n[ADAPTIVE HYBRID] LIVE Opened {direction} {symbol} (ID: {position_id})", "magenta", attrs=['bold'])
        cprint(f"  Entry: ${price:,.2f} | Size: ${position_size:,.2f} ({asset_size} {symbol}) | Leverage: {leverage}x", "white")
        cprint(f"  Risk: ${risk_amount:,.2f} | SL: {sl_pct:.2f}% | Fee: ${entry_fee:.4f}", "white")
        cprint(f"  SL: ${trade['stop_loss']:,.2f} ({sl_pct:.2f}%)", "white")
        cprint(f"  TP: ${trade['take_profit']:,.2f} ({tp_pct:.2f}%)", "white")

        # Log decision to trade memory
        try:
            module_scores = metadata.get('module_scores', {})
            modules_firing = [m for m in module_scores if module_scores[m] != 0] if isinstance(module_scores, dict) else None
            decision_id = self._trade_memory.log_decision(
                symbol=symbol,
                direction=direction,
                confidence=signal.get('signal', 0) * 100,
                source='adaptive_hybrid_live',
                reasoning=str(metadata.get('reason', '')),
                market_regime=self._current_regime,
                key_indicators={
                    'rsi': metadata.get('rsi'),
                    'adx': metadata.get('adx'),
                    'atr': trade.get('atr'),
                    'score': trade.get('score'),
                },
                modules_firing=modules_firing,
            )
            trade['memory_decision_id'] = decision_id
        except Exception as e:
            cprint(f"  [Memory] Warning: could not log decision: {e}", "yellow")

        return trade

    def sync_live_positions(self) -> list:
        """Sync internal position tracker with actual HyperLiquid positions.

        Queries the exchange for real positions and removes any tracked positions
        that have been closed on-chain (e.g. SL/TP filled).  Returns list of
        position_ids that were detected as closed.
        """
        closed_ids = []
        try:
            import eth_account as _eth_account
            from hyperliquid.info import Info as HLInfo

            private_key = os.getenv('HYPER_LIQUID_ETH_PRIVATE_KEY')
            if not private_key:
                return []

            account = _eth_account.Account.from_key(private_key)
            info = HLInfo('https://api.hyperliquid.xyz', skip_ws=True)
            user_state = info.user_state(account.address)

            # Build set of symbols with active positions on exchange
            exchange_positions = {}  # {symbol: {'size': float, 'side': str}}
            for pos in user_state.get('assetPositions', []):
                p = pos.get('position', {})
                szi = float(p.get('szi', 0))
                if szi != 0:
                    coin = p.get('coin', '')
                    exchange_positions[coin] = {
                        'size': abs(szi),
                        'side': 'BUY' if szi > 0 else 'SELL',
                        'entry_px': float(p.get('entryPx', 0)),
                        'unrealized_pnl': float(p.get('unrealizedPnl', 0)),
                    }

            # Check tracked positions against exchange state
            with self._position_lock:
                for position_id, trade in list(self.paper_positions.items()):
                    if trade.get('mode') != 'LIVE':
                        continue
                    symbol = trade['symbol']
                    direction = trade['direction']

                    ex_pos = exchange_positions.get(symbol)
                    if ex_pos is None or ex_pos['side'] != direction:
                        # Position closed on exchange (SL/TP filled or manually closed)
                        cprint(f"[AdaptiveHybrid] LIVE SYNC: Position {position_id} ({direction} {symbol}) no longer on exchange", "yellow")
                        closed_ids.append(position_id)

            # Close detected positions using last known price
            for position_id in closed_ids:
                with self._position_lock:
                    if position_id not in self.paper_positions:
                        continue
                    trade = self.paper_positions[position_id]
                    symbol = trade['symbol']

                # Fetch close price
                try:
                    from src.nice_funcs_hyperliquid import ask_bid
                    ask, bid, _ = ask_bid(symbol)
                    close_price = (ask + bid) / 2
                except Exception:
                    close_price = trade['entry_price']  # fallback

                self._close_paper_position(position_id, close_price, 'EXCHANGE_CLOSED')
                self.trailing_stops.pop(position_id, None)

            if exchange_positions:
                cprint(f"[AdaptiveHybrid] LIVE SYNC: {len(exchange_positions)} active exchange positions, {len(closed_ids)} closed", "cyan")

        except Exception as e:
            cprint(f"[AdaptiveHybrid] LIVE SYNC error: {e}", "red")

        return closed_ids

    def monitor_paper_positions(self) -> list:
        """Monitor all open positions: intra-candle SL/TP, trailing stop, time-based exits.

        In paper mode: checks prices and simulates SL/TP/trailing locally.
        In live mode: syncs with exchange to detect filled SL/TP orders.
        """
        if not self.paper_positions:
            return []

        # In live mode, SL/TP are native exchange orders -- just sync state
        if not PAPER_TRADING:
            closed_ids = self.sync_live_positions()
            return [{'position_id': pid, 'close_reason': 'EXCHANGE_CLOSED'} for pid in closed_ids]

        closed = []

        with self._position_lock:
            symbols_to_check = set(pos['symbol'] for pos in self.paper_positions.values())

        # Fetch price data with high/low for intra-candle checks
        candle_data = {}  # {symbol: {'close': float, 'high': float, 'low': float}}
        for symbol in symbols_to_check:
            try:
                if ADAPTIVE_HYBRID_USE_REALTIME_PRICE:
                    from src.nice_funcs_hyperliquid import ask_bid
                    ask, bid, _ = ask_bid(symbol)
                    if ask and bid:
                        mid = (ask + bid) / 2
                        candle_data[symbol] = {'close': mid, 'high': ask, 'low': bid}
                        continue
                # Fallback to candles
                df = self._fetch_candles(symbol, interval='15m', candles=5)
                if df is not None and len(df) > 0:
                    last = df.iloc[-1]
                    candle_data[symbol] = {
                        'close': float(last['close']),
                        'high': float(last['high']),
                        'low': float(last['low']),
                    }
            except Exception:
                pass

        positions_to_close = []

        with self._position_lock:
            for position_id, trade in self.paper_positions.items():
                symbol = trade['symbol']
                if symbol not in candle_data:
                    continue

                cd = candle_data[symbol]
                current_price = cd['close']
                trade['current_price'] = current_price
                candle_high = cd['high']
                candle_low = cd['low']
                direction = trade['direction']
                entry_price = trade['entry_price']
                token_class = 'mid'
                for cls, profile in ADAPTIVE_HYBRID_ATR_PROFILES.items():
                    if symbol in profile['tokens']:
                        token_class = cls
                        break
                stop_loss = trade['stop_loss']
                take_profit = trade['take_profit']

                close_reason = None
                close_price = current_price

                # Intra-candle SL/TP check using high/low
                if direction == 'BUY':
                    if candle_low <= stop_loss:
                        close_reason = 'STOP_LOSS'
                        close_price = stop_loss
                    elif candle_high >= take_profit:
                        close_reason = 'TAKE_PROFIT'
                        close_price = take_profit
                else:
                    if candle_high >= stop_loss:
                        close_reason = 'STOP_LOSS'
                        close_price = stop_loss
                    elif candle_low <= take_profit:
                        close_reason = 'TAKE_PROFIT'
                        close_price = take_profit

                # Scale-out partial take profit (Phase 3)
                if close_reason is None and direction in ('BUY', 'SELL'):
                    if direction == 'BUY':
                        tp_dist = take_profit - entry_price
                        current_progress = (candle_high - entry_price) / tp_dist if tp_dist > 0 else 0
                    else:
                        tp_dist = entry_price - take_profit
                        current_progress = (entry_price - candle_low) / tp_dist if tp_dist > 0 else 0

                    scale_out_level = trade.get('scale_out_level', 0)
                    for i, level in enumerate(ADAPTIVE_HYBRID_SCALE_OUT_LEVELS):
                        if i < scale_out_level:
                            continue
                        if current_progress >= level['tp_pct']:
                            close_pct = level['close_pct']
                            partial_size = trade['position_size'] * close_pct
                            if partial_size >= 5:  # Min $5 partial close
                                # Partial close: reduce position size
                                trade['position_size'] -= partial_size
                                trade['scale_out_level'] = i + 1
                                # Calculate partial PnL (with slippage + fees, matching _close_paper_position)
                                slippage = PAPER_SLIPPAGE_V2.get(token_class, 0.001)
                                if direction == 'BUY':
                                    partial_price = entry_price + tp_dist * level['tp_pct']
                                    partial_price = partial_price * (1 - slippage)  # worse fill for selling
                                    partial_pnl = partial_size * ((partial_price - entry_price) / entry_price)
                                else:
                                    partial_price = entry_price - tp_dist * level['tp_pct']
                                    partial_price = partial_price * (1 + slippage)  # worse fill for buying back
                                    partial_pnl = partial_size * ((entry_price - partial_price) / entry_price)
                                partial_fee = partial_size * PAPER_TAKER_FEE_V2
                                self.paper_balance += partial_pnl - partial_fee
                                trade['partial_pnl_realized'] = trade.get('partial_pnl_realized', 0) + partial_pnl - partial_fee
                                cprint(f"  [Scale-Out] {symbol} L{i+1}: closed {close_pct:.0%} (${partial_size:.0f}) at {level['tp_pct']:.0%} TP, PnL=${partial_pnl-partial_fee:.2f}", "green")
                                # Persist scale-out state to CSV
                                self._update_open_position_in_csv(position_id, trade)

                # Periodic ATR recalculation for trailing stop (every 4 hours)
                if close_reason is None:
                    import time as _trail_time
                    _last_atr_update = trade.get('last_atr_update', 0)
                    if isinstance(_last_atr_update, (int, float)):
                        _now_epoch = _trail_time.time()
                        if _now_epoch - _last_atr_update > 4 * 3600:
                            try:
                                _atr_df = self._fetch_candles(symbol, interval='1h', candles=20)
                                if _atr_df is not None and len(_atr_df) >= 14:
                                    _high_low = _atr_df['high'] - _atr_df['low']
                                    _high_close = (_atr_df['high'] - _atr_df['close'].shift()).abs()
                                    _low_close = (_atr_df['low'] - _atr_df['close'].shift()).abs()
                                    _true_range = pd.concat([_high_low, _high_close, _low_close], axis=1).max(axis=1)
                                    _new_atr = float(_true_range.rolling(14).mean().iloc[-1])

                                    if _new_atr > 0:
                                        _old_atr = trade.get('atr', _new_atr)
                                        # Blend: 70% new ATR, 30% old (smooth transition)
                                        _blended_atr = _new_atr * 0.7 + _old_atr * 0.3
                                        trade['entry_atr'] = trade.get('entry_atr', _old_atr)
                                        trade['atr'] = _blended_atr
                                        trade['last_atr_update'] = _now_epoch
                                        cprint(f"    [ATR Update] {symbol}: ATR {_old_atr:.6f} -> {_blended_atr:.6f}", "cyan")
                            except Exception:
                                pass

                # Progressive trailing stop (Phase 1)
                if close_reason is None:
                    if position_id not in self.trailing_stops:
                        self.trailing_stops[position_id] = {
                            'highest': entry_price, 'lowest': entry_price,
                            'trailing_active': False, 'breakeven_locked': False,
                            'current_level': -1,
                        }

                    ts = self.trailing_stops[position_id]
                    # Use candle high/low for watermark (track true extremes), close for trigger
                    reference_high = candle_high if candle_high else current_price
                    reference_low = candle_low if candle_low else current_price
                    ts['highest'] = max(ts['highest'], reference_high)
                    ts['lowest'] = min(ts['lowest'], reference_low)

                    atr = trade.get('atr', 0)
                    if atr <= 0:
                        atr = abs(entry_price * 0.015)

                    if direction == 'BUY':
                        profit_in_atr = (ts['highest'] - entry_price) / atr
                    else:
                        profit_in_atr = (entry_price - ts['lowest']) / atr

                    # Check each trailing level (highest first, never downgrade)
                    best_trailing_sl = None
                    for i, level in enumerate(reversed(ADAPTIVE_HYBRID_TRAILING_LEVELS)):
                        level_idx = len(ADAPTIVE_HYBRID_TRAILING_LEVELS) - 1 - i
                        # Ratchet: never downgrade to a lower level
                        if level_idx < ts.get('current_level', -1):
                            continue
                        if profit_in_atr >= level['activate_atr']:
                            if level.get('breakeven'):
                                if ts.get('breakeven_locked'):
                                    break  # Already locked, skip recalculation
                                # Breakeven lock: SL at entry +/- estimated round-trip fees
                                # BUY: SL below entry, exit if price drops back (lock in fee-adjusted breakeven)
                                # SELL: SL above entry, exit if price rises back (lock in fee-adjusted breakeven)
                                fee_offset = entry_price * 2 * (PAPER_TAKER_FEE_V2 + PAPER_SLIPPAGE_V2.get(token_class, 0.001))
                                if direction == 'BUY':
                                    be_sl = entry_price - fee_offset
                                else:
                                    be_sl = entry_price + fee_offset
                                best_trailing_sl = be_sl
                                ts['breakeven_locked'] = True
                            else:
                                dist = level['distance_atr']
                                if direction == 'BUY':
                                    best_trailing_sl = ts['highest'] - dist * atr
                                else:
                                    best_trailing_sl = ts['lowest'] + dist * atr
                            ts['current_level'] = level_idx
                            break

                    if best_trailing_sl is not None:
                        ts['trailing_active'] = True
                        # Ratchet SL: never move SL in unfavorable direction
                        if direction == 'BUY':
                            best_trailing_sl = max(best_trailing_sl, ts.get('locked_sl', 0))
                            ts['locked_sl'] = best_trailing_sl
                            if best_trailing_sl > stop_loss and candle_low <= best_trailing_sl:
                                close_reason = 'TRAILING_STOP'
                                close_price = best_trailing_sl
                        else:
                            # SELL: ratchet SL upward (tighter) — higher SL = more protective for shorts
                            best_trailing_sl = max(best_trailing_sl, ts.get('locked_sl', 0))
                            ts['locked_sl'] = best_trailing_sl
                            if best_trailing_sl < stop_loss and candle_high >= best_trailing_sl:
                                close_reason = 'TRAILING_STOP'
                                close_price = best_trailing_sl

                # Time-based exit (only if not already closing)
                if close_reason is None:
                    entry_time = trade.get('entry_time')
                    if entry_time is not None:
                        if isinstance(entry_time, str):
                            try:
                                entry_time = datetime.fromisoformat(entry_time)
                            except (ValueError, TypeError):
                                entry_time = None

                    if entry_time is not None:
                        hold_hours = (datetime.now() - entry_time).total_seconds() / 3600

                        if hold_hours >= ADAPTIVE_HYBRID_MAX_HOLD_HOURS:
                            close_reason = 'TIME_EXIT_48H'
                            close_price = current_price
                        elif hold_hours >= ADAPTIVE_HYBRID_HOLD_TP_CHECK_HOURS:
                            # Close if less than 50% of TP reached
                            if direction == 'BUY':
                                tp_dist = take_profit - entry_price
                                tp_progress = (current_price - entry_price) / tp_dist if tp_dist > 0 else 0
                            else:
                                tp_dist = entry_price - take_profit
                                tp_progress = (entry_price - current_price) / tp_dist if tp_dist > 0 else 0
                            if tp_progress < 0.5:
                                close_reason = 'TIME_EXIT_24H'
                                close_price = current_price

                if close_reason:
                    positions_to_close.append((position_id, close_price, close_reason))

        for position_id, close_price, reason in positions_to_close:
            closed_trade = self._close_paper_position(position_id, close_price, reason)
            if closed_trade:
                closed.append(closed_trade)
                # Clean up trailing stop state
                self.trailing_stops.pop(position_id, None)

        return closed

    def _close_paper_position(self, position_id: str, close_price: float, reason: str) -> dict:
        """Close a paper position with slippage and fees simulation. Fully thread-safe."""
        with self._position_lock:
            if position_id not in self.paper_positions:
                return None

            trade = self.paper_positions[position_id].copy()
            entry_price = trade['entry_price']
            direction = trade['direction']
            position_size = trade['position_size']
            symbol = trade['symbol']
            entry_fee = trade.get('entry_fee', 0)

            # Determine token class for slippage
            token_class = 'mid'
            for cls, profile in ADAPTIVE_HYBRID_ATR_PROFILES.items():
                if symbol in profile['tokens']:
                    token_class = cls
                    break

            slippage = PAPER_SLIPPAGE_V2.get(token_class, 0.001)
            exit_fee = position_size * PAPER_TAKER_FEE_V2

            # Apply slippage to close price (worse for trader)
            if direction == 'BUY':
                effective_close_price = close_price * (1 - slippage)
            else:
                effective_close_price = close_price * (1 + slippage)

            if direction == 'BUY':
                price_change_pct = (effective_close_price - entry_price) / entry_price
            else:
                price_change_pct = (entry_price - effective_close_price) / entry_price

            pnl = position_size * price_change_pct

            # Sanity check: PnL cannot exceed position size (prevent rounding bugs on micro-prices)
            if abs(pnl) > position_size * 1.5:
                cprint(f"  [WARNING] Absurd PnL detected: ${pnl:+,.2f} on ${position_size:.2f} position. "
                       f"Entry={entry_price}, Close={close_price}, Eff={effective_close_price}. Clamping.", "red")
                pnl = max(-position_size, min(position_size, pnl))

            # Deduct only exit fee (entry fee already deducted at open time)
            total_fees = exit_fee
            pnl = pnl - total_fees

            trade['close_price'] = close_price
            trade['effective_close_price'] = round(effective_close_price, 6)
            trade['exit_price'] = close_price
            trade['exit_time'] = datetime.now().isoformat()
            trade['close_reason'] = reason
            trade['exit_fee'] = round(exit_fee, 4)
            trade['total_fees'] = round(entry_fee + exit_fee, 4)
            trade['slippage'] = slippage
            trade['pnl'] = round(pnl, 2)
            trade['total_pnl'] = round(pnl + trade.get('partial_pnl_realized', 0), 2)
            trade['pnl_pct'] = round(price_change_pct * 100, 2)
            trade['status'] = 'CLOSED'

            self.daily_pnl += pnl
            self.paper_balance += pnl
            del self.paper_positions[position_id]

            # Consecutive loss circuit breaker with escalating cooldowns
            if pnl < 0:
                self.consecutive_losses += 1
                try:
                    from src.config import ADAPTIVE_HYBRID_ESCALATING_COOLDOWNS
                except ImportError:
                    ADAPTIVE_HYBRID_ESCALATING_COOLDOWNS = {3: 2, 4: 6, 5: 24}

                # Find the matching cooldown (highest applicable)
                cooldown_hours = 0
                for loss_count, hours in sorted((int(k), v) for k, v in ADAPTIVE_HYBRID_ESCALATING_COOLDOWNS.items()):
                    if self.consecutive_losses >= loss_count:
                        cooldown_hours = hours
                if cooldown_hours > 0:
                    self._loss_breaker_until = datetime.utcnow() + timedelta(hours=cooldown_hours)
                    # Set post-breaker size reduction counter
                    if not hasattr(self, '_post_breaker_trades'):
                        self._post_breaker_trades = 0
                    self._post_breaker_trades = 3
                    cprint(f"  [CIRCUIT BREAKER] {self.consecutive_losses} consecutive losses — pausing trading for {cooldown_hours}h", "red", attrs=['bold'])
            else:
                self.consecutive_losses = 0
                self._loss_breaker_until = None

            self.closed_positions.append(trade)
            balance_snapshot = self.paper_balance

        # Add benchmark alpha to trade record (OUTSIDE lock — _get_benchmark_alpha does HTTP calls)
        alpha = self._get_benchmark_alpha()
        trade['btc_alpha'] = alpha.get('BTC', {}).get('alpha', 0)
        trade['eth_alpha'] = alpha.get('ETH', {}).get('alpha', 0)
        trade['strategy_return_pct'] = alpha.get('strategy_return_pct', 0)

        color = 'green' if pnl > 0 else 'red'
        cprint(f"\n[ADAPTIVE HYBRID] Closed {trade['symbol']} ({reason})", color, attrs=['bold'])
        _pd = self._price_decimals(entry_price)
        cprint(f"  Entry: ${entry_price:,.{_pd}f} -> Exit: ${close_price:,.{_pd}f} (eff: ${effective_close_price:,.{_pd}f})", "white")
        cprint(f"  PnL: ${pnl:+,.2f} ({price_change_pct*100:+.2f}%) | Fees: ${total_fees:.4f} | Slip: {slippage:.3%}", color)
        cprint(f"  Balance: ${balance_snapshot:,.2f}", "white")

        # Discord alert
        try:
            from src.utils.alerting import get_alert_manager
            trade['balance_after'] = balance_snapshot
            get_alert_manager().trade_closed(trade)
        except Exception:
            pass

        # Update trade memory with outcome
        if 'memory_decision_id' in trade:
            try:
                entry_time = trade.get('entry_time')
                if isinstance(entry_time, str):
                    try:
                        entry_time = datetime.fromisoformat(entry_time)
                    except (ValueError, TypeError):
                        entry_time = None
                hold_hours = (datetime.now() - entry_time).total_seconds() / 3600 if entry_time else None

                self._trade_memory.update_outcome(
                    decision_id=trade['memory_decision_id'],
                    pnl=pnl,
                    hold_duration_hours=hold_hours,
                    max_adverse=trade.get('max_adverse_excursion', 0),
                    max_favorable=trade.get('max_favorable_excursion', 0),
                    close_reason=reason,
                )
            except Exception as e:
                cprint(f"  [Memory] Warning: could not update outcome: {e}", "yellow")

        # I/O operations outside the lock
        self._log_closed_trade(trade)
        self._update_position_status_in_csv(position_id, trade)

        # Feed ML components
        try:
            module_scores = {}
            modules_str = trade.get('modules', '{}')
            if modules_str and modules_str != '{}':
                module_scores = json.loads(modules_str) if isinstance(modules_str, str) else modules_str

            trade_result = {
                'pnl': pnl,
                'module_scores': module_scores,
                'market_regime': self._current_regime,
                'symbol': symbol,
                'direction': direction,
            }
            self._weight_optimizer.update(trade_result)
            self._feedback.record_trade(trade_result)
        except Exception as e:
            cprint(f"  [ML] Warning: feedback error: {e}", "yellow")

        # Post-trade learning (LLM analyzes what happened)
        if ADAPTIVE_HYBRID_LLM_LEARNER:
            try:
                trade_for_learning = trade.copy()
                trade_for_learning['market_regime'] = self._current_regime
                analyze_closed_trade(
                    trade=trade_for_learning,
                    trade_memory=self._trade_memory,
                    model=self._llm_model,
                    bypass=not ADAPTIVE_HYBRID_LLM_LEARNER,
                )
            except Exception as e:
                cprint(f"  [Trade Learner] Error: {e}", "yellow")

        return trade

    def _update_position_status_in_csv(self, position_id: str, trade: dict):
        """Update a position's status and exit data in paper_trades.csv when closed."""
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
                df.loc[mask, 'exit_price'] = trade.get('exit_price', trade.get('close_price', 0))
                df.loc[mask, 'exit_time'] = trade.get('exit_time', '')
                df.loc[mask, 'pnl'] = trade.get('pnl', 0)
                df.loc[mask, 'close_reason'] = trade.get('close_reason', '')
                df.to_csv(paper_trades_file, index=False)
        except Exception as e:
            cprint(f"[AdaptiveHybrid] Warning: Could not update CSV: {e}", "yellow")

    def _update_open_position_in_csv(self, position_id: str, trade: dict):
        """Update an open position's mutable fields (position_size, scale_out_level) in paper_trades.csv."""
        try:
            paper_trades_file = os.path.join(self.data_dir, 'paper_trades.csv')
            if not os.path.exists(paper_trades_file):
                return
            df = pd.read_csv(paper_trades_file)
            if df.empty:
                return
            mask = (df['position_id'] == position_id) & (df['status'] == 'OPEN')
            if mask.any():
                df.loc[mask, 'position_size'] = trade.get('position_size', 0)
                df.loc[mask, 'scale_out_level'] = trade.get('scale_out_level', 0)
                df.loc[mask, 'partial_pnl_realized'] = trade.get('partial_pnl_realized', 0.0)
                df.to_csv(paper_trades_file, index=False)
        except Exception as e:
            cprint(f"[AdaptiveHybrid] Warning: Could not update open position CSV: {e}", "yellow")

    def _log_closed_trade(self, trade: dict):
        """Log closed trade to separate CSV file."""
        try:
            log_file = os.path.join(self.data_dir, 'closed_trades.csv')
            df = pd.DataFrame([trade])
            if os.path.exists(log_file):
                df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                df.to_csv(log_file, index=False)
        except Exception as e:
            cprint(f"[AdaptiveHybrid] Error logging closed trade: {e}", "yellow")

    def get_paper_status(self) -> dict:
        """Get current paper trading status (for dashboard). Thread-safe."""
        with self._position_lock:
            return {
                'paper_balance': round(self.paper_balance, 2),
                'initial_balance': PAPER_TRADING_BALANCE,
                'total_pnl': round(self.paper_balance - PAPER_TRADING_BALANCE, 2),
                'daily_pnl': round(self.daily_pnl, 2),
                'daily_trades': self.daily_trades,
                'open_positions': len(self.paper_positions),
                'total_closed': len(self.closed_positions),
                'positions': [pos.copy() for pos in self.paper_positions.values()],
            }

    def get_daily_stats(self, date=None) -> dict:
        """Compute daily stats from closed_trades.csv for a given date.

        Args:
            date: datetime.date or None (defaults to yesterday)

        Returns dict with: date, total_pnl, trades_count, wins, losses, win_rate,
            best_trade, worst_trade, balance, open_positions, total_pnl_alltime,
            streak, alpha_btc
        """
        from datetime import date as date_type
        if date is None:
            date = (datetime.now() - timedelta(days=1)).date()
        elif isinstance(date, datetime):
            date = date.date()

        result = {
            'date': str(date),
            'total_pnl': 0,
            'trades_count': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'best_trade': {},
            'worst_trade': {},
            'balance': round(self.paper_balance, 2),
            'open_positions': len(self.paper_positions),
            'total_pnl_alltime': round(self.paper_balance - PAPER_TRADING_BALANCE, 2),
            'streak': 0,
            'alpha_btc': None,
        }

        closed_file = os.path.join(self.data_dir, 'closed_trades.csv')
        if not os.path.exists(closed_file):
            return result

        try:
            df = pd.read_csv(closed_file)
            if df.empty or 'exit_time' not in df.columns or 'pnl' not in df.columns:
                return result

            # Parse exit_time and filter by date
            df['exit_date'] = pd.to_datetime(df['exit_time'], errors='coerce').dt.date
            day_df = df[df['exit_date'] == date]

            if day_df.empty:
                return result

            pnls = day_df['pnl'].astype(float)
            result['total_pnl'] = round(float(pnls.sum()), 2)
            result['trades_count'] = len(day_df)
            result['wins'] = int((pnls > 0).sum())
            result['losses'] = int((pnls < 0).sum())
            result['win_rate'] = round(result['wins'] / result['trades_count'] * 100, 1) if result['trades_count'] > 0 else 0

            # Best / worst trade
            best_idx = pnls.idxmax()
            worst_idx = pnls.idxmin()
            result['best_trade'] = {
                'symbol': day_df.loc[best_idx, 'symbol'] if 'symbol' in day_df.columns else '?',
                'pnl': round(float(pnls.loc[best_idx]), 2),
            }
            result['worst_trade'] = {
                'symbol': day_df.loc[worst_idx, 'symbol'] if 'symbol' in day_df.columns else '?',
                'pnl': round(float(pnls.loc[worst_idx]), 2),
            }

            # Streak (consecutive wins/losses from most recent)
            all_pnls = df.sort_values('exit_time')['pnl'].astype(float).tolist()
            if all_pnls:
                streak = 0
                last_sign = 1 if all_pnls[-1] > 0 else -1
                for p in reversed(all_pnls):
                    current_sign = 1 if p > 0 else -1
                    if current_sign == last_sign:
                        streak += current_sign
                    else:
                        break
                result['streak'] = streak

            # Alpha vs BTC
            alpha = self._get_benchmark_alpha()
            if alpha and 'BTC' in alpha:
                result['alpha_btc'] = alpha['BTC'].get('alpha', 0)

        except Exception as e:
            from termcolor import cprint
            cprint(f"[DailyStats] Error reading closed_trades.csv: {e}", "yellow")

        return result

    def close_all_paper_positions(self) -> list:
        """Force close all open paper positions at current market price."""
        with self._position_lock:
            if not self.paper_positions:
                return []
            position_ids = list(self.paper_positions.keys())
            positions_copy = {pid: self.paper_positions[pid].copy() for pid in position_ids}

        closed = []
        for position_id in position_ids:
            trade = positions_copy[position_id]
            try:
                df = self._fetch_candles(trade['symbol'], interval='15m', candles=5)
                if df is not None and len(df) > 0:
                    current_price = float(df['close'].iloc[-1])
                    closed_trade = self._close_paper_position(position_id, current_price, 'MANUAL')
                    if closed_trade:
                        closed.append(closed_trade)
            except Exception as e:
                cprint(f"[AdaptiveHybrid] Error closing {position_id}: {e}", "red")

        return closed

    def _load_state_from_csv(self):
        """Load existing positions and balance from CSV files."""
        try:
            paper_trades_file = os.path.join(self.data_dir, 'paper_trades.csv')
            closed_trades_file = os.path.join(self.data_dir, 'closed_trades.csv')

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
                                'leverage': float(row.get('leverage', ADAPTIVE_HYBRID_LEVERAGE)),
                                'stop_loss': float(row.get('stop_loss', 0)),
                                'take_profit': float(row.get('take_profit', 0)),
                                'sl_pct': float(row.get('sl_pct', 1.5)),
                                'tp_pct': float(row.get('tp_pct', 2.5)),
                                'confidence': float(row.get('confidence', 0)),
                                'atr': float(row.get('atr', 0) or 0),
                                'entry_fee': float(row.get('entry_fee', 0) or 0),
                                'scale_out_level': int(float(row.get('scale_out_level', 0) or 0)),
                                'partial_pnl_realized': float(row.get('partial_pnl_realized', 0) or 0),
                                'status': 'OPEN',
                                'entry_time': row.get('timestamp', datetime.now().isoformat()),
                            }

                    if self.paper_positions:
                        max_counter = 0
                        for pos_id in self.paper_positions.keys():
                            parts = pos_id.split('_')
                            if len(parts) >= 4:
                                try:
                                    max_counter = max(max_counter, int(parts[-1]))
                                except ValueError:
                                    pass
                        self._position_counter = max_counter

            realized_pnl = 0.0
            closed_entry_fees = 0.0
            closed_partial_pnl = 0.0
            if os.path.exists(closed_trades_file):
                closed_df = pd.read_csv(closed_trades_file)
                if not closed_df.empty and 'pnl' in closed_df.columns:
                    realized_pnl = closed_df['pnl'].sum()
                    if 'entry_fee' in closed_df.columns:
                        closed_entry_fees = closed_df['entry_fee'].fillna(0).sum()
                    if 'partial_pnl_realized' in closed_df.columns:
                        closed_partial_pnl = closed_df['partial_pnl_realized'].fillna(0).sum()
                    self.closed_positions = closed_df.to_dict('records')

                    # Update trade memory outcomes for trades that closed while system was down
                    if hasattr(self, '_trade_memory') and self._trade_memory:
                        for _, row in closed_df.iterrows():
                            if 'memory_decision_id' in row and pd.notna(row.get('memory_decision_id')):
                                try:
                                    self._trade_memory.update_outcome(
                                        int(row['memory_decision_id']),
                                        pnl=row.get('pnl_usd', 0),
                                        correct=1 if row.get('pnl_usd', 0) > 0 else 0,
                                        hold_hours=row.get('hold_hours', 0),
                                        close_reason=row.get('close_reason', 'unknown')
                                    )
                                except Exception:
                                    pass  # Already updated or missing

            # Subtract entry fees of currently open positions (already deducted at open time)
            open_entry_fees = sum(t.get('entry_fee', 0) for t in self.paper_positions.values())
            # Add back partial PnL already credited to balance (not in realized_pnl from closed_trades.csv)
            open_partial_pnl = sum(t.get('partial_pnl_realized', 0) for t in self.paper_positions.values())
            # Balance = initial + closed PnL - ALL entry fees + open partial PnL
            # NOTE: closed_partial_pnl is NOT added here — it was already credited to
            # paper_balance during the session (line ~2400) and is included in realized_pnl
            # from closed_trades.csv. Only open_partial_pnl needs re-adding for positions
            # still open whose partials were credited in a previous session.
            self.paper_balance = (PAPER_TRADING_BALANCE + realized_pnl
                                  - closed_entry_fees - open_entry_fees
                                  + open_partial_pnl)

            # Restore daily_trades count from CSV so limit survives restarts
            today_str = datetime.utcnow().date().isoformat()
            today_trades = 0

            # Count open positions opened today
            for pos in self.paper_positions.values():
                ts = pos.get('timestamp', '')
                if ts and str(ts)[:10] == today_str:
                    today_trades += 1

            # Count closed trades from today
            if os.path.exists(closed_trades_file):
                try:
                    closed_df_today = pd.read_csv(closed_trades_file)
                    if not closed_df_today.empty and 'timestamp' in closed_df_today.columns:
                        closed_df_today['_date'] = closed_df_today['timestamp'].astype(str).str[:10]
                        today_trades += int((closed_df_today['_date'] == today_str).sum())
                except Exception:
                    pass  # Already counted what we could

            self.daily_trades = today_trades
            self.last_trade_date = datetime.utcnow().date()
            if today_trades > 0:
                cprint(f"[AdaptiveHybrid] Restored daily_trades={today_trades} from CSV (date={today_str})", "cyan")

        except Exception as e:
            cprint(f"[AdaptiveHybrid] Warning: Could not load state: {e}", "yellow")


# For standalone testing
if __name__ == "__main__":
    cprint("=" * 60, "cyan")
    cprint("  Testing Adaptive Hybrid Strategy", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")

    strategy = AdaptiveHybridStrategy()

    for symbol in ['BTC', 'ETH', 'SOL', 'LINK', 'ARB', 'DOGE']:
        try:
            cprint(f"\nAnalyzing {symbol}...", "white", attrs=['bold'])
            signal = strategy.generate_signals(symbol=symbol)

            if signal:
                if signal['direction'] != 'NEUTRAL':
                    cprint(f"  SIGNAL: {signal['direction']} (strength={signal['signal']:.0%})", "green")
                else:
                    score = signal['metadata'].get('score', 0)
                    threshold = signal['metadata'].get('threshold', 45)
                    cprint(f"  NEUTRAL (score={score:.1f}/{threshold:.0f})", "yellow")

        except Exception as e:
            cprint(f"Error testing {symbol}: {e}", "red")
            import traceback
            traceback.print_exc()

    cprint("\nTest completed!", "cyan")
