"""
🌙 Moon Dev's Configuration File
Built with love by Moon Dev 🚀
"""

import os

# 🔄 Exchange Selection
EXCHANGE = 'hyperliquid'  # Options: 'solana', 'hyperliquid'

# 💰 Trading Configuration
USDC_ADDRESS = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # Never trade or close
SOL_ADDRESS = "So11111111111111111111111111111111111111111"   # Never trade or close

# Create a list of addresses to exclude from trading/closing
EXCLUDED_TOKENS = [USDC_ADDRESS, SOL_ADDRESS]

# Token List for Trading 📋
# NOTE: Trading Agent now has its own token list - see src/agents/trading_agent.py lines 101-104
MONITORED_TOKENS = [
    # '9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump',    # 🌬️ FART
    # 'DitHyRMQiSDhn5cnKMJV2CDDt6sVct96YrECiM49pump'     # housecoin
]

# Moon Dev's Token Trading List 🚀
# Each token is carefully selected by Moon Dev for maximum moon potential! 🌙
tokens_to_trade = MONITORED_TOKENS  # Using the same list for trading

# ⚡ HyperLiquid Configuration
# HYPERLIQUID_SYMBOLS is derived from SNIPER_ASSETS (single source of truth)
# Defined after SNIPER_ASSETS below
HYPERLIQUID_LEVERAGE = 5  # Default leverage for HyperLiquid trades (1-50)

# 🔄 Exchange-Specific Token Lists
# Use this to determine which tokens/symbols to trade based on active exchange
def get_active_tokens():
    """Returns the appropriate token/symbol list based on active exchange"""
    if EXCHANGE == 'hyperliquid':
        return HYPERLIQUID_SYMBOLS
    else:
        return MONITORED_TOKENS

# Token to Exchange Mapping (for future hybrid trading)
TOKEN_EXCHANGE_MAP = {
    'BTC': 'hyperliquid',
    'ETH': 'hyperliquid',
    'SOL': 'hyperliquid',
    # All other tokens default to Solana
}

# Token and wallet settings
symbol = '9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump'
WALLET_ADDRESS = os.getenv('SOLANA_WALLET_ADDRESS', '')
address = WALLET_ADDRESS  # Backward compatibility alias

# Position sizing 🎯
usd_size = 25  # Size of position to hold
max_usd_order_size = 3  # Max order size
tx_sleep = 30  # Sleep between transactions
slippage = 199  # Slippage settings

# Risk Management Settings 🛡️
CASH_PERCENTAGE = 20  # Minimum % to keep in USDC as safety buffer (0-100)
MAX_POSITION_PERCENTAGE = 30  # Maximum % allocation per position (0-100)
SLEEP_AFTER_CLOSE = 600  # Prevent overtrading

MAX_LOSS_GAIN_CHECK_HOURS = 12  # How far back to check for max loss/gain limits (in hours)
# Note: SLEEP_BETWEEN_RUNS_MINUTES is defined further below (line ~115)


# Max Loss/Gain Settings FOR RISK AGENT 1/5/25
USE_PERCENTAGE = False  # If True, use percentage-based limits. If False, use USD-based limits

# USD-based limits (used if USE_PERCENTAGE is False)
MAX_LOSS_USD = 25  # Maximum loss in USD before stopping trading
MAX_GAIN_USD = 25 # Maximum gain in USD before stopping trading

# USD MINIMUM BALANCE RISK CONTROL
MINIMUM_BALANCE_USD = 50  # If balance falls below this, risk agent will consider closing all positions
USE_AI_CONFIRMATION = True  # If True, consult AI before closing positions. If False, close immediately on breach

# Percentage-based limits (used if USE_PERCENTAGE is True)
MAX_LOSS_PERCENT = 5  # Maximum loss as percentage (e.g., 20 = 20% loss)
MAX_GAIN_PERCENT = 5  # Maximum gain as percentage (e.g., 50 = 50% gain)

# Transaction settings ⚡
PRIORITY_FEE = 100000  # ~0.02 USD at current SOL prices
orders_per_open = 3  # Multiple orders for better fill rates

# Market maker settings 📊
buy_under = .0946
sell_over = 1

# Data collection settings 📈
DAYSBACK_4_DATA = 3
DATA_TIMEFRAME = '1H'  # 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 8H, 12H, 1D, 3D, 1W, 1M
SAVE_OHLCV_DATA = False  # 🌙 Set to True to save data permanently, False will only use temp data during run

# AI Model Settings 🤖
AI_MODEL = "claude-sonnet-4-5"            # Model Options:
                                           # - claude-haiku-4-5-20251001 (Fast, efficient)
                                           # - claude-sonnet-4-5 (Balanced, strong reasoning)
                                           # - claude-sonnet-4-6 (Latest balanced model)
                                           # - claude-opus-4-6 (Most powerful, expensive)
AI_MAX_TOKENS = 1024  # Max tokens for response
AI_TEMPERATURE = 0.7  # Creativity vs precision (0-1)

# Trading Strategy Agent Settings - MAY NOT BE USED YET 1/5/25
ENABLE_STRATEGIES = True  # Set this to True to use strategies
STRATEGY_MIN_CONFIDENCE = 0.7  # Minimum confidence to act on strategy signals

# Sleep time between main agent runs
SLEEP_BETWEEN_RUNS_MINUTES = 5  # How long to sleep between agent runs 🕒

# in our nice_funcs in token over view we look for minimum trades last hour
MIN_TRADES_LAST_HOUR = 2


# Real-Time Clips Agent Settings 🎬
REALTIME_CLIPS_ENABLED = True
REALTIME_CLIPS_OBS_FOLDER = '/Volumes/Moon 26/OBS'  # Your OBS recording folder
REALTIME_CLIPS_AUTO_INTERVAL = 120  # Check every N seconds (120 = 2 minutes)
REALTIME_CLIPS_LENGTH = 2  # Minutes to analyze per check
REALTIME_CLIPS_AI_MODEL = 'groq'  # Model type: groq, openai, claude, deepseek, xai, ollama
REALTIME_CLIPS_AI_MODEL_NAME = None  # None = use default for model type
REALTIME_CLIPS_TWITTER = True  # Auto-open Twitter compose after clip

# Multifactor Strategy Settings 📊
MULTIFACTOR_ASSETS = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT', 'MATIC']
MULTIFACTOR_SMALLCAPS = []  # Add small cap tokens here (e.g., ['WIF', 'BONK', 'PEPE'])
MULTIFACTOR_TIMEFRAME = '15m'  # Intraday timeframe
MULTIFACTOR_RISK_PER_TRADE = 0.04  # 4% risk per trade

# Multifactor weights (must sum to 1.0)
MULTIFACTOR_WEIGHTS = {
    'trend': 0.25,      # EMA alignment
    'momentum': 0.20,   # MACD
    'rsi': 0.20,        # RSI zones
    'volume': 0.15,     # Volume confirmation
    'sentiment': 0.20   # Twitter sentiment
}

# Multifactor thresholds
MULTIFACTOR_BUY_THRESHOLD = 0.6   # Score > 0.6 = BUY
MULTIFACTOR_SELL_THRESHOLD = -0.6  # Score < -0.6 = SELL

# ============================================================================
# RAMF Strategy Settings (Regime Adaptive Momentum Fade)
# ============================================================================
# A contrarian strategy that fades exhausted momentum in high volatility regimes
# Designed for small accounts with conservative risk management

RAMF_ASSETS = [
    # Blue chips
    'BTC', 'ETH', 'SOL', 'XRP',
    # AI tokens
    'FET', 'RENDER', 'TAO', 'NEAR', 'WLD',
    # Popular alts
    'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT',
    # L2 & DeFi
    'ARB', 'OP', 'SUI', 'SEI',
]  # Assets to trade with RAMF strategy (HyperLiquid perpetuals)
RAMF_LEVERAGE = 3                    # Conservative leverage for small accounts (1-5 recommended)
RAMF_STOP_LOSS_PCT = 1.0             # Base stop-loss percentage (dynamically adjusted by ATR)
RAMF_TAKE_PROFIT_PCT = 2.0           # Base take-profit percentage (dynamically adjusted by ATR)
RAMF_MIN_CONFIDENCE = 60             # Minimum confidence score to trade (0-100) - lowered from 70 for more signals
RAMF_MAX_DAILY_TRADES = 15           # Maximum trades per day (increased for paper trading)
RAMF_MAX_DAILY_LOSS_USD = 50         # Daily loss limit in USD (~10% of $500)
RAMF_MAX_DAILY_GAIN_USD = 75         # Daily gain limit in USD (~15% of $500)

# Volatility Regime Settings
# The strategy only trades in HIGH or LOW volatility regimes
# ATR percentile > HIGH threshold = mean-reversion mode (fade exhaustion)
# ATR percentile < LOW threshold = trend-following mode
# Between these thresholds = NORMAL (no trade) - widen bands for more signals
RAMF_VOLATILITY_HIGH_PERCENTILE = 55  # Top 45% of ATR = "high" volatility (reduced from 60 to minimize dead zone)
RAMF_VOLATILITY_LOW_PERCENTILE = 45   # Bottom 45% of ATR = "low" volatility (increased from 40 to minimize dead zone)

# Momentum Exhaustion Settings (for HIGH volatility regime)
RAMF_ATR_EXTENSION_THRESHOLD = 1.5    # ATRs from VWAP for exhaustion detection (reduced from 2.0)
RAMF_CONSECUTIVE_BAR_THRESHOLD = 3    # Consecutive bars in same direction for exhaustion (reduced from 5)

# RAMF Advanced Settings (v2.0 improvements)
RAMF_USE_ADAPTIVE_SL_TP = True       # Dynamic SL/TP based on ATR
RAMF_ATR_SL_MULTIPLIER = 1.5         # SL = ATR * multiplier (e.g., 1.5 ATR)
RAMF_ATR_TP_MULTIPLIER = 3.0         # TP = ATR * multiplier (e.g., 3.0 ATR for 2:1 R:R)
RAMF_MIN_SL_PCT = 0.5                # Minimum SL percentage (floor)
RAMF_MAX_SL_PCT = 2.0                # Maximum SL percentage (ceiling)

# Time Windows Settings (UTC hours)
RAMF_USE_TIME_WINDOWS = True         # Enable time-based confidence modifier
RAMF_OPTIMAL_HOURS = [7, 8, 9, 13, 14, 15, 19, 20, 21]  # London open, NY open, Asia close
RAMF_AVOID_HOURS = [0, 1, 2, 3, 4, 5]  # Low liquidity hours
RAMF_OPTIMAL_HOUR_BONUS = 15         # Bonus confidence during optimal hours
RAMF_AVOID_HOUR_PENALTY = 20         # Penalty during low liquidity hours

# Multi-Timeframe Settings
RAMF_USE_MTF = True                  # Enable multi-timeframe confluence
RAMF_MTF_TIMEFRAMES = ['5m', '15m', '1h', '4h']  # Timeframes to check
RAMF_MTF_AGREEMENT_BONUS = 10        # Bonus per timeframe agreement
RAMF_MTF_MIN_AGREEMENT = 2           # Minimum timeframes that must agree

# Funding Rate Divergence Settings
RAMF_USE_FUNDING_DIVERGENCE = True   # Enable funding divergence detection
RAMF_FUNDING_DIV_LOOKBACK = 24       # Hours to look back for divergence
RAMF_FUNDING_DIV_THRESHOLD = 0.3     # Minimum divergence score (-1 to +1)
RAMF_FUNDING_DIV_BONUS = 20          # Bonus for strong divergence signal

# Liquidation Cluster Settings
RAMF_USE_LIQ_CLUSTERS = True         # Enable liquidation cluster prediction
RAMF_LIQ_CLUSTER_THRESHOLD = 2.0     # Ratio threshold for cluster detection
RAMF_LIQ_CLUSTER_BONUS = 15          # Bonus when price near liquidation cluster

# ============================================================================
# Data Provider Settings (Replaces Moon Dev API)
# ============================================================================
# Liquidations: Binance Futures WebSocket (free, real-time)
# Funding/OI: HyperLiquid API (free, real-time)

BINANCE_WS_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"
LIQUIDATION_BUFFER_SIZE = 10000      # Max liquidations to keep in memory
LIQUIDATION_LOOKBACK_MINUTES = 15    # Default lookback for ratio calculation

# ============================================================================
# Paper Trading Mode
# ============================================================================
# Set PAPER_TRADING = True to simulate trades without real execution
# Recommended: Start with paper trading to validate strategy performance

PAPER_TRADING = True                 # True = simulation mode, False = live trading
PAPER_TRADING_BALANCE = 500          # Simulated starting balance in USD

# ============================================================================
# Active Strategy Selection
# ============================================================================
# Choose which strategy to run (only one should be active at a time)

ACTIVE_STRATEGY = 'adaptive_hybrid'  # Options: 'multifactor', 'ramf', 'sniper', 'hybrid', 'adaptive_hybrid', 'example'

# ============================================================================
# SNIPER AI Strategy Settings
# ============================================================================
# A precision trading strategy requiring ALL 7 checklist conditions to align.
# Target: 1-2 trades/day with 80%+ win rate.
# Philosophy: "Fewer trades, but quasi-certain trades"

# Assets to trade - univers de 14 tokens sélectionnés par OI HyperLiquid
# Retirés : MATIC (OI=$0), RENDER (OI<$1M), DOT (OI faible), FIL ($2M), ARB ($3M), OP ($3M), CRV (remplacé par ENA)
# Ajoutés : SUI ($22M OI), TAO ($15M OI), NEAR ($16M OI), ENA ($15M OI), kPEPE ($10M OI)
SNIPER_ASSETS = [
    'BTC', 'ETH', 'SOL',           # Majors
    'XRP', 'AVAX', 'SUI',          # L1 établis + émergent
    'TAO', 'NEAR',                  # AI
    'AAVE', 'ENA',                  # DeFi
    'LINK',                         # Oracle
    'DOGE', 'kPEPE',               # Memes
    'ADA',                          # L1 value
]

# Single source of truth for HyperLiquid symbols
HYPERLIQUID_SYMBOLS = SNIPER_ASSETS

# Position sizing
SNIPER_LEVERAGE = 3                    # Conservative leverage
SNIPER_STOP_LOSS_PCT = 1.5             # Base SL (AI may adjust)
SNIPER_TAKE_PROFIT_PCT = 3.0           # Base TP (2:1 minimum R:R)

# Daily limits (adjusted for 13 assets)
SNIPER_MAX_DAILY_TRADES = 5            # Max 5 trades per day (more opportunities with 13 assets)
SNIPER_MAX_DAILY_LOSS_USD = 30         # ~6% of $500
SNIPER_MAX_DAILY_GAIN_USD = 60         # ~12% of $500

# Checklist thresholds (Option A - modérément ajusté pour plus d'opportunités)
SNIPER_SIGMA_THRESHOLD = 1.8           # Condition 1: Extreme move sigma (was 2.5 - too strict)
SNIPER_FUNDING_EXTREME_THRESHOLD = 2.0 # Condition 2: Funding Z-score
SNIPER_LIQ_RATIO_THRESHOLD = 1.5       # Condition 3: Liquidation ratio
SNIPER_VOLUME_SPIKE_THRESHOLD = 3.0    # Condition 5: Volume climax multiplier

# Time window settings (UTC hours)
SNIPER_OPTIMAL_HOURS = [7, 8, 9, 13, 14, 15, 16]  # London + NY open (still used for scoring bonus)
SNIPER_ALLOW_NORMAL_HOURS = True       # 24/7 trading enabled - all hours allowed

# AI Validation settings
SNIPER_AI_MIN_CONFIDENCE = 85          # Minimum AI confidence to execute
SNIPER_AI_MODEL = 'claude-sonnet-4-5'  # Use capable model for reasoning
SNIPER_AI_TEMPERATURE = 0.3            # Low temp for analytical responses
SNIPER_AI_MAX_TOKENS = 1024

# Setup-specific thresholds (Option A - relaxed RSI for more opportunities)
SNIPER_CAPITULATION_MIN_DROP_PCT = 5.0  # Min 5% drop for capitulation fade
SNIPER_EUPHORIA_MIN_RISE_PCT = 5.0      # Min 5% rise for euphoria fade
SNIPER_RSI_OVERSOLD = 32                # RSI < 32 for capitulation (was 25 - too strict)
SNIPER_RSI_OVERBOUGHT = 68              # RSI > 68 for euphoria (was 75 - too strict)

# Lookback window for price change detection (Option 1 - extended for slower moves)
SNIPER_LOOKBACK_HOURS = 8               # Hours to look back for price moves (was 4h - too short)

# === ADVANCED IMPROVEMENTS ===

# 1. ATR-based trailing stop (replaces fixed SL when position is profitable)
SNIPER_USE_TRAILING_STOP = True         # Enable ATR-based trailing stop
SNIPER_TRAILING_ATR_MULTIPLIER = 2.0    # Trailing stop = 2 × ATR(14)
SNIPER_TRAILING_ACTIVATION_PCT = 1.0    # Activate trailing after +1% profit

# 2. Market regime filter (ADX) - thresholds raised +30% to allow moderate trends
SNIPER_USE_REGIME_FILTER = True         # Enable market regime detection
SNIPER_ADX_TRENDING_THRESHOLD = 40      # ADX > 40 = trending market (was 25 - too strict)
SNIPER_ADX_PERIOD = 14                  # ADX calculation period

# 3. Correlation filter (for position sizing)
SNIPER_USE_CORRELATION_SIZING = True    # Reduce size if correlated positions open
SNIPER_CORRELATION_THRESHOLD = 0.7      # High correlation threshold
SNIPER_CORRELATION_LOOKBACK_DAYS = 30   # Days of data for correlation calc

# 4. Funding Arbitrage setup
SNIPER_ENABLE_FUNDING_ARBITRAGE = True  # Enable funding arbitrage setup
SNIPER_FUNDING_ARBITRAGE_THRESHOLD = 0.1  # Funding > ±0.1% = extreme
SNIPER_FUNDING_ARBITRAGE_STABILITY_PCT = 1.0  # Max price move for "stable"

# 5. Weighted scoring (instead of binary 7/7)
SNIPER_USE_WEIGHTED_SCORING = True      # Enable weighted confidence scoring
SNIPER_MIN_WEIGHTED_SCORE = 7.0         # Minimum score out of 10 (was 8.5 - too strict)
SNIPER_WEIGHTS = {
    'extreme_move': 2.0,
    'funding_divergence': 1.5,
    'liquidation_cascade': 1.5,
    'multi_tf': 1.0,
    'volume_climax': 1.0,
    'time_window': 0.5,
    'ai_validation': 2.5,
}

# 6. Dynamic position sizing based on AI confidence
SNIPER_USE_CONFIDENCE_SIZING = True     # Scale position with confidence
SNIPER_CONFIDENCE_SIZE_MAP = {
    85: 0.5,   # 85-89% confidence = 50% size
    90: 0.75,  # 90-94% confidence = 75% size
    95: 1.0,   # 95%+ confidence = 100% size
}

# 7. Dynamic threshold adaptation (volatility-based)
SNIPER_USE_DYNAMIC_THRESHOLDS = True    # Enable adaptive thresholds based on market conditions
SNIPER_VOL_RATIO_HIGH = 1.3             # Recent vol > 1.3x historical = high volatility regime
SNIPER_VOL_RATIO_LOW = 0.7              # Recent vol < 0.7x historical = low volatility regime
SNIPER_MAX_THRESHOLD_ADJUSTMENT = 0.25  # Max threshold adjustment ±25%
SNIPER_ADX_RANGING_THRESHOLD = 20       # ADX < 20 = ranging market (mean-reversion friendly)
SNIPER_RECALIBRATION_HOURS = 4          # Recalibrate thresholds every N hours (0 = nightly only)

# ============================================================================
# TREND RIDER Strategy Settings (Trend-Following Pullback)
# ============================================================================
# Complements Sniper AI by trading WITH the trend instead of fading it
# Activates when market is trending (ADX > 35) and EMAs are aligned

TREND_RIDER_ENABLED = True
TREND_RIDER_ASSETS = ['BTC', 'ETH', 'SOL', 'LINK', 'AVAX']  # Liquid assets only

# Trend Detection
TREND_RIDER_ADX_MIN = 35                   # Minimum ADX for valid trend
TREND_RIDER_EMA_PERIODS = [20, 50, 200]    # EMAs for alignment check

# Pullback Conditions
TREND_RIDER_RSI_PULLBACK_LONG = (30, 45)   # RSI zone for long pullback
TREND_RIDER_RSI_PULLBACK_SHORT = (55, 70)  # RSI zone for short pullback
TREND_RIDER_VOLUME_DECLINE_RATIO = 0.8     # Volume < 80% avg during pullback

# Confirmation
TREND_RIDER_VOLUME_SPIKE_RATIO = 1.2       # Volume > 120% avg on confirmation
TREND_RIDER_MIN_CANDLE_BODY_PCT = 50       # Candle body > 50% of range

# Risk Management
TREND_RIDER_LEVERAGE = 2                    # Conservative leverage
TREND_RIDER_ATR_SL_MULT = 1.5              # Stop loss = 1.5 ATR
TREND_RIDER_ATR_TP_MULT = 2.0              # Take profit = 2.0 ATR
TREND_RIDER_TRAILING_ATR_MULT = 2.0        # Trailing stop = 2 ATR

# Daily Limits
TREND_RIDER_MAX_DAILY_TRADES = 3
TREND_RIDER_MAX_DAILY_LOSS_USD = 25

# AI Validation
TREND_RIDER_AI_MODEL = 'claude-sonnet-4-5'
TREND_RIDER_AI_MIN_CONFIDENCE = 70
TREND_RIDER_AI_TEMPERATURE = 0.3

# Scoring System
TREND_RIDER_MIN_SCORE = 7.0                # Minimum score to trade (out of 10)
TREND_RIDER_WEIGHTS = {
    'trend_alignment': 2.5,                # EMA stack properly aligned
    'pullback_quality': 2.0,               # RSI in zone + price near EMA
    'momentum_confirmation': 2.0,          # Confirmation candle + volume
    'htf_agreement': 1.5,                  # Higher timeframe agrees
    'ai_validation': 2.0,                  # LLM confidence >= 70%
}

# ============================================================================
# HYBRID Mode Settings (Sniper + Trend Rider)
# ============================================================================
# Enables both strategies to work together based on market regime

HYBRID_MODE_ENABLED = True
HYBRID_PREFER_SNIPER = True                # Sniper takes priority over Trend Rider
HYBRID_MAX_CONCURRENT_POSITIONS = 2        # Max positions across both strategies
HYBRID_SNIPER_MIN_SCORE_PRIORITY = 7.0     # Sniper needs 7.0+ to take priority

# ============================================================================
# ADAPTIVE HYBRID Strategy Settings
# ============================================================================
# Multi-module scoring strategy that aggregates 8 independent signal generators.
# Target: 1-3 trades/day with 55%+ win rate.

ADAPTIVE_HYBRID_BASE_THRESHOLD = 45      # Base score threshold (0-100) to trigger a trade
ADAPTIVE_HYBRID_URGENCY_START_HOURS = 4  # Start relaxing threshold after N hours without trade
ADAPTIVE_HYBRID_URGENCY_FLOOR = 42       # Minimum threshold (never go below this)
ADAPTIVE_HYBRID_MAX_DAILY_TRADES = 5     # Max trades per day
ADAPTIVE_HYBRID_MAX_DAILY_LOSS_USD = 30  # Daily loss limit in USD
ADAPTIVE_HYBRID_LEVERAGE = 3             # Default leverage
ADAPTIVE_HYBRID_ATR_SL_MULT = 1.5       # Stop loss = 1.5x ATR
ADAPTIVE_HYBRID_ATR_TP_MULT = 3.0       # Take profit = 3.0x ATR (must be >= SL_MULT * MIN_RR_RATIO)
ADAPTIVE_HYBRID_SKIP_LLM = True          # Skip LLM re-evaluation (strategy has own filters)
ADAPTIVE_HYBRID_MAX_POSITION_PCT = 25    # Max position size as % of paper balance
ADAPTIVE_HYBRID_MIN_CONVERGENT_MODULES = 2  # Minimum modules agreeing for a trade signal
ADAPTIVE_HYBRID_MIN_RR_RATIO = 2.0       # Minimum reward:risk ratio (TP must be >= SL * this)

# ATR SL/TP profiles by token class
ADAPTIVE_HYBRID_ATR_PROFILES = {
    'major':  {'sl_mult': 1.8, 'tp_mult': 3.6, 'tokens': ['BTC', 'ETH']},
    'mid':    {'sl_mult': 1.5, 'tp_mult': 3.0, 'tokens': ['SOL', 'XRP', 'AVAX', 'LINK', 'ADA', 'AAVE', 'NEAR', 'SUI', 'TAO']},
    'alt':    {'sl_mult': 1.2, 'tp_mult': 2.4, 'tokens': ['DOGE', 'kPEPE', 'ENA']},
}
ADAPTIVE_HYBRID_RESET_PAPER = False       # One-shot flag: reset paper trading state on next startup

# Trailing Stop (conservative)
ADAPTIVE_HYBRID_TRAILING_ACTIVATE_ATR = 1.5  # Activate trailing after +1.5 ATR profit
ADAPTIVE_HYBRID_TRAILING_DISTANCE_ATR = 2.0  # Trail at 2 ATR from high/low

# Session filter (UTC hours)
ADAPTIVE_HYBRID_OPTIMAL_HOURS = [7, 8, 9, 13, 14, 15, 19, 20, 21]  # London + NY + Asia open
ADAPTIVE_HYBRID_AVOID_HOURS = [0, 1, 2, 3, 4, 5]  # Low liquidity

# Time-based exit
ADAPTIVE_HYBRID_MAX_HOLD_HOURS = 48  # Force close after 48h
ADAPTIVE_HYBRID_HOLD_TP_CHECK_HOURS = 24  # Close if <50% TP after 24h

# Paper trading fees simulation
PAPER_TAKER_FEE = 0.00035  # HyperLiquid taker fee 0.035%
PAPER_SLIPPAGE = {
    'major': 0.001,   # 0.1% for BTC, ETH
    'mid': 0.002,     # 0.2% for SOL, XRP, AVAX, etc.
    'alt': 0.005,     # 0.5% for DOGE, kPEPE, ENA
}

# Leverage profiles by token class
ADAPTIVE_HYBRID_LEVERAGE_PROFILES = {
    'major': 3,   # BTC, ETH
    'mid': 3,     # SOL, XRP, AVAX, etc.
    'alt': 2,     # DOGE, kPEPE, ENA — lower leverage for volatile alts
}

# Module weights (must sum to 1.0)
ADAPTIVE_HYBRID_WEIGHTS = {
    'mean_reversion': 0.12,      # Bollinger Bands + RSI in ranging markets
    'momentum_breakout': 0.10,   # Range breakout + volume confirmation
    'ema_trend': 0.10,           # EMA trend (merged ema_crossover + trend_rider)
    'funding_contrarian': 0.08,  # Extreme funding rate contrarian
    'rsi_divergence': 0.08,      # Price vs RSI divergence
    'sniper_lite': 0.14,         # Relaxed Sniper (extreme move + volume)
    'trend_rider_lite': 0.00,    # Merged into ema_trend
    'ramf_lite': 0.08,           # Volatility regime (no dead zone)
    'oi_delta': 0.10,            # NEW: Open Interest delta
    'sentiment': 0.08,           # NEW: Sentiment composite
    'squeeze_detector': 0.06,    # NEW: Squeeze detection
    'order_imbalance': 0.06,     # NEW: Order book imbalance
}

# Weight profiles by token behavior class
ADAPTIVE_HYBRID_WEIGHT_PROFILES = {
    'ranging': {  # BTC, ETH — often range-bound
        'mean_reversion': 0.16, 'momentum_breakout': 0.06, 'ema_trend': 0.08,
        'funding_contrarian': 0.10, 'rsi_divergence': 0.10, 'sniper_lite': 0.14,
        'trend_rider_lite': 0.00, 'ramf_lite': 0.08,
        'oi_delta': 0.08, 'sentiment': 0.08, 'squeeze_detector': 0.06, 'order_imbalance': 0.06,
    },
    'trending': {  # DOGE, kPEPE, SUI, TAO — strong momentum
        'mean_reversion': 0.06, 'momentum_breakout': 0.14, 'ema_trend': 0.10,
        'funding_contrarian': 0.08, 'rsi_divergence': 0.06, 'sniper_lite': 0.12,
        'trend_rider_lite': 0.00, 'ramf_lite': 0.08,
        'oi_delta': 0.12, 'sentiment': 0.08, 'squeeze_detector': 0.08, 'order_imbalance': 0.08,
    },
}
ADAPTIVE_HYBRID_RANGING_TOKENS = ['BTC', 'ETH']
ADAPTIVE_HYBRID_TRENDING_TOKENS = ['DOGE', 'kPEPE', 'SUI', 'TAO']

# ============================================================================
# Risk Agent Settings (for paper trading mode)
# ============================================================================
RISK_MAX_DRAWDOWN_PCT = 15       # Pause trading if PnL < -15% of initial capital
RISK_MAX_DAILY_LOSS_USD = 30     # Pause trading for the day if daily PnL < -$30
RISK_MAX_POSITIONS = 4           # Maximum simultaneous open positions

# Cooling-off after drawdown breach
RISK_COOLING_OFF_HOURS = 4        # Wait 4h after drawdown breach before resuming
RISK_RECOVERY_SIZE_PCT = 50       # Resume with 50% size for 24h after recovery
RISK_RECOVERY_DURATION_HOURS = 24 # Duration of reduced-size recovery period

