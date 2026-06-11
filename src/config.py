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

ADAPTIVE_HYBRID_BASE_THRESHOLD = 48      # Base score threshold (0-100) — lowered from 52: LLM filters now handle quality control
ADAPTIVE_HYBRID_URGENCY_START_HOURS = 3  # Start relaxing threshold after N hours without trade (was 4)
ADAPTIVE_HYBRID_URGENCY_FLOOR = 42       # Minimum threshold (never go below this) — lowered from 45
ADAPTIVE_HYBRID_MAX_DAILY_TRADES = 6     # Max trades per day
ADAPTIVE_HYBRID_MAX_DAILY_LOSS_USD = 30  # Daily loss limit in USD
ADAPTIVE_HYBRID_LEVERAGE = 3             # Default leverage
ADAPTIVE_HYBRID_ATR_SL_MULT = 3.5       # Stop loss = 3.5x ATR (widened for fewer premature exits)
ADAPTIVE_HYBRID_ATR_TP_MULT = 3.8       # Take profit = 3.8x ATR (reduced from 4.5 — 60% of trades hit TIME_EXIT before TP)
ADAPTIVE_HYBRID_SKIP_LLM = True          # Skip LLM re-evaluation (strategy has own filters)
ADAPTIVE_HYBRID_RISK_PCT = 0.012         # Risque FIXE par trade : balance × 1.2% = perte max au SL (sizing simplifié Jun 2026 —
                                         # le Kelly floor + 6 multiplicateurs empilés écrasaient le notionnel à $25 médian).
                                         # Le levier ne multiplie PLUS le risque (il ne réduit que la marge immobilisée) :
                                         # notional = balance × RISK_PCT / sl_fraction, plafonné par marge et MAX_POSITION_PCT.
                                         # Équivaut aux notionnels de l'ancien 0.4% × levier 3, mais le risque est désormais
                                         # identique pour toutes les classes de tokens (lev 2 ou 3).
ADAPTIVE_HYBRID_VOLUME_FILTER_MIN = 0.05 # Minimum volume ratio to accept signal (was 0.15)

# LLM-Enhanced Pipeline Settings
ADAPTIVE_HYBRID_LLM_CONFIRMATION = True   # Enabled: LLM filters low-quality trades before execution
ADAPTIVE_HYBRID_LLM_REGIME = True         # Enabled: regime detection to avoid counter-trend trades
ADAPTIVE_HYBRID_LLM_LEARNER = True       # Enable post-trade LLM learning
ADAPTIVE_HYBRID_MTF_CONFLUENCE = True    # Enable multi-timeframe confluence scoring
ADAPTIVE_HYBRID_LLM_PROVIDER = 'anthropic'  # LLM provider — switched from groq (qwen3-32b rate-limited, 100% failures)
ADAPTIVE_HYBRID_LLM_TIMEOUT_S = 8       # Max seconds to wait for LLM response (raised from 5 for Sonnet latency)
ADAPTIVE_HYBRID_MAX_POSITION_PCT = 25    # Max position size as % of paper balance
ADAPTIVE_HYBRID_MIN_CONVERGENT_MODULES = 2  # Minimum modules agreeing for a trade signal
ADAPTIVE_HYBRID_MIN_RR_RATIO = 2.0       # Minimum reward:risk ratio

# ATR SL/TP profiles by token class
ADAPTIVE_HYBRID_ATR_PROFILES = {
    'btc':    {'sl_mult': 2.8, 'tp_mult': 3.5, 'tokens': ['BTC']},        # TP tightened 4.2→3.5 (60% of closes were TIME_EXIT_24H)
    'eth':    {'sl_mult': 3.0, 'tp_mult': 4.2, 'tokens': ['ETH']},        # TP tightened 5.0→4.2
    'mid':    {'sl_mult': 2.5, 'tp_mult': 3.8, 'tokens': ['SOL', 'XRP', 'AVAX', 'LINK', 'ADA', 'AAVE', 'NEAR', 'SUI', 'TAO']},  # TP tightened 4.5→3.8
    'alt':    {'sl_mult': 2.2, 'tp_mult': 3.5, 'tokens': ['DOGE', 'kPEPE', 'ENA']},  # TP tightened 4.0→3.5
}
ADAPTIVE_HYBRID_RESET_PAPER = False       # One-shot flag: reset paper trading state on next startup

# Legacy trailing stop (replaced by ADAPTIVE_HYBRID_TRAILING_LEVELS progressive trailing)
ADAPTIVE_HYBRID_TRAILING_ACTIVATE_ATR = 1.5  # Deprecated: use TRAILING_LEVELS instead
ADAPTIVE_HYBRID_TRAILING_DISTANCE_ATR = 2.0  # Deprecated: use TRAILING_LEVELS instead

# Session filter (UTC hours)
ADAPTIVE_HYBRID_OPTIMAL_HOURS = [7, 8, 9, 13, 14, 20, 21]  # London + NY + Asia open (removed 15, 19: 18% WR, -$46 PnL)
ADAPTIVE_HYBRID_AVOID_HOURS = [1, 2, 3, 4]  # Low liquidity hours (reduced from 7 to 4)

# Time-based exit
ADAPTIVE_HYBRID_MAX_HOLD_HOURS = 48  # Force close after 48h
ADAPTIVE_HYBRID_HOLD_TP_CHECK_HOURS = 12  # Close if <50% TP after 12h (was 24 — TIME_EXIT squattait les slots)

# Stagnation exit: close if after N hours the price is still within X ATR of entry
# (les positions stagnantes squattaient 28% des slot-heures pour -$2.46)
ADAPTIVE_HYBRID_STAGNATION_HOURS = 8
ADAPTIVE_HYBRID_STAGNATION_ATR = 0.3

# Legacy paper trading fees (used by backtester; live paper uses V2 below)
PAPER_TAKER_FEE = 0.00035  # Deprecated: use PAPER_TAKER_FEE_V2
PAPER_SLIPPAGE = {          # Deprecated: use PAPER_SLIPPAGE_V2
    'btc': 0.001,     # 0.1% for BTC
    'eth': 0.001,     # 0.1% for ETH
    'mid': 0.002,     # 0.2% for SOL, XRP, AVAX, etc.
    'alt': 0.005,     # 0.5% for DOGE, kPEPE, ENA
}

# Leverage profiles by token class
ADAPTIVE_HYBRID_LEVERAGE_PROFILES = {
    'btc': 3,     # BTC
    'eth': 3,     # ETH
    'mid': 3,     # SOL, XRP, AVAX, etc.
    'alt': 2,     # DOGE, kPEPE, ENA — lower leverage for volatile alts
}

# Module weights (must sum to 1.0) — 15 modules, noise modules removed
ADAPTIVE_HYBRID_WEIGHTS = {
    'mean_reversion': 0.09,
    'momentum_breakout': 0.07,
    'ema_trend': 0.07,
    'funding_composite': 0.08,
    'rsi_divergence': 0.06,
    'sniper_lite': 0.08,
    'ramf_lite': 0.06,
    'oi_delta': 0.07,
    'sentiment': 0.05,
    'squeeze_detector': 0.05,
    'order_imbalance': 0.07,
    'crowd_positioning': 0.07,
    'cvd': 0.09,
    'vwap_deviation': 0.06,
    'liquidation_cascade': 0.03,
}

# Weight profiles by token behavior class
ADAPTIVE_HYBRID_WEIGHT_PROFILES = {
    'ranging': {  # BTC, ETH — often range-bound
        'mean_reversion': 0.12, 'momentum_breakout': 0.04, 'ema_trend': 0.05,
        'funding_composite': 0.08, 'rsi_divergence': 0.07, 'sniper_lite': 0.09,
        'ramf_lite': 0.05,
        'oi_delta': 0.06, 'sentiment': 0.04, 'squeeze_detector': 0.05, 'order_imbalance': 0.05,
        'crowd_positioning': 0.07,
        'cvd': 0.08, 'vwap_deviation': 0.07,
        'liquidation_cascade': 0.08,
    },
    'trending': {  # DOGE, kPEPE, SUI, TAO — strong momentum
        'mean_reversion': 0.04, 'momentum_breakout': 0.09, 'ema_trend': 0.07,
        'funding_composite': 0.06, 'rsi_divergence': 0.05, 'sniper_lite': 0.08,
        'ramf_lite': 0.05,
        'oi_delta': 0.09, 'sentiment': 0.05, 'squeeze_detector': 0.06, 'order_imbalance': 0.06,
        'crowd_positioning': 0.06,
        'cvd': 0.09, 'vwap_deviation': 0.06,
        'liquidation_cascade': 0.09,
    },
}
ADAPTIVE_HYBRID_RANGING_TOKENS = ['BTC', 'ETH']
ADAPTIVE_HYBRID_TRENDING_TOKENS = ['DOGE', 'kPEPE', 'SUI', 'TAO']

# --- Risk: Notional exposure cap ---
ADAPTIVE_HYBRID_MAX_NOTIONAL_PCT = 500  # Max total notional exposure as % of balance

# --- Weekend size reduction ---
ADAPTIVE_HYBRID_WEEKEND_SIZE_REDUCTION = 0.30  # Reduce position size by 30% on weekends

# --- Escalating cooldowns after consecutive losses ---
ADAPTIVE_HYBRID_ESCALATING_COOLDOWNS = {3: 2, 4: 6, 5: 24}  # {consecutive_losses: cooldown_hours}

# --- Choppiness Index threshold ---
ADAPTIVE_HYBRID_CHOPPINESS_THRESHOLD = 61.8  # CI above this = choppy market, penalize signals

# --- Regime hysteresis ---
ADAPTIVE_HYBRID_REGIME_HYSTERESIS = 3  # Require N consecutive same classifications before switching

# --- Funding cluster cap ---
ADAPTIVE_HYBRID_FUNDING_CLUSTER_CAP = 0.10  # Max combined effective weight for funding-based modules

# --- Kelly-adaptive sizing (DEPRECATED Jun 2026: retiré du sizing, voir ADAPTIVE_HYBRID_RISK_PCT) ---
ADAPTIVE_HYBRID_KELLY_LOOKBACK = 30  # Deprecated — kept for backward compat
ADAPTIVE_HYBRID_KELLY_FRACTION = 0.5  # Deprecated — kept for backward compat

# --- Event calendar ---
ADAPTIVE_HYBRID_EVENT_CALENDAR_FILE = os.path.join(os.path.dirname(__file__), 'data', 'event_calendar.json')
ADAPTIVE_HYBRID_EVENT_SIZE_REDUCTION = 0.50  # Reduce position size by 50% near high-impact events
ADAPTIVE_HYBRID_EVENT_WINDOW_HOURS = 2  # Hours before/after event to apply reduction

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

# Portfolio Correlation
CORRELATION_HIGH_THRESHOLD = 0.75  # Above this, reduce sizing
CORRELATION_SIZING_FACTOR = 0.5   # Reduce to 50% when correlated

# Regime Detection
REGIME_ADX_TRENDING = 30
REGIME_ADX_RANGING = 20
REGIME_VOL_HIGH = 1.2
REGIME_VOL_LOW = 0.8

# ============================================================================
# PHASE 2-4: Enhanced Trading Improvements
# ============================================================================

# --- Trailing Stop Progressive (Phase 1) ---
# Production data (Mar 12, 2026): 88.6% of exits were trailing stops at +0.21%
# because breakeven locked at 1.0 ATR. Average win was $0.30 vs average loss $4.62
# (1:15 risk/reward). Raised breakeven to 2.0 ATR to let winners run.
# BUGFIX (May 2026): activate_atr levels 2/3 (4.0/5.5) were unreachable before TP
# (BTC TP=3.5, mid TP=3.8, alt TP=3.5). Lowered to 1.5/3.0/4.5 so progressive
# trailing actually activates and locks profit before TP fires.
# R:R FIX (Jun 2026): breakeven 1.5 → 2.5 ATR — le breakeven précoce écrêtait les gains
# (R:R réalisé 0.56), les winners étaient stoppés à breakeven avant de courir.
ADAPTIVE_HYBRID_TRAILING_LEVELS = [
    {'activate_atr': 2.5, 'distance_atr': None, 'breakeven': True},
    {'activate_atr': 3.0, 'distance_atr': 1.5},
    {'activate_atr': 4.5, 'distance_atr': 0.8},
]

# --- Scale-Out Partial Take Profit (Phase 3) ---
# R:R FIX (Jun 2026): 1er niveau (40% TP → close 33%) supprimé — le scale-out précoce
# amputait les gains et contribuait au R:R réalisé de 0.56.
ADAPTIVE_HYBRID_SCALE_OUT_LEVELS = [
    {'tp_pct': 0.70, 'close_pct': 0.50},  # 70% of TP path -> close 50%
]

# --- Volatility Targeting (Phase 3) ---
ADAPTIVE_HYBRID_VOL_TARGET_DAILY_PCT = 1.0  # DEPRECATED Jun 2026: retiré du sizing (empilement de multiplicateurs)
ADAPTIVE_HYBRID_VOL_MIN_POSITION_USD = 10   # Min position size in USD

# --- Monitoring Enhancement (Phase 1) ---
ADAPTIVE_HYBRID_USE_REALTIME_PRICE = True  # Use ask_bid() instead of 15m candles for monitoring

# --- Slippage Correction (Phase 1) ---
# Updated to be more realistic + applied symmetrically (entry AND exit)
PAPER_SLIPPAGE_V2 = {
    'btc': 0.0003,    # 0.03% (was 0.1%)
    'eth': 0.0005,    # 0.05% (was 0.1%)
    'mid': 0.0012,    # 0.12% (was 0.2%)
    'alt': 0.003,     # 0.30% (was 0.5%)
}
PAPER_TAKER_FEE_V2 = 0.00045  # HyperLiquid taker fee 0.045% (was 0.035%)

# --- Adaptive Weights (Phase 3) ---
ADAPTIVE_HYBRID_USE_BAYESIAN_WEIGHTS = False  # Disabled: use fixed weights
ADAPTIVE_HYBRID_BAYESIAN_MIN_TRADES = 15     # Min trades before adaptive weights activate
ADAPTIVE_HYBRID_BAYESIAN_DECAY = 0.95        # Decay factor for recency bias

# --- Anomaly Filter (Phase 4) ---
ADAPTIVE_HYBRID_USE_ANOMALY_FILTER = False    # Disabled: remove anomaly filter
ADAPTIVE_HYBRID_ANOMALY_SCORE_DIVISOR = 2    # Divide signal score by this if anomaly detected

# ============================================================================
# Smart Scheduling - Light Check
# ============================================================================
# Daemon thread that polls HyperLiquid allMids every 2 minutes (1 HTTP call)
# to detect sudden price spikes and trigger priority analysis.
LIGHT_CHECK_ENABLED = True
LIGHT_CHECK_INTERVAL_S = 120                   # 2 minutes between checks
LIGHT_CHECK_PRICE_THRESHOLDS = {               # Min move in 1 check (2min) to trigger
    'large': 0.008,   # BTC, ETH  -- 0.8%
    'mid':   0.015,   # SOL, XRP, ADA, etc. -- 1.5%
    'small': 0.025,   # DOGE, kPEPE, ENA -- 2.5%
}
LIGHT_CHECK_ROLLING_THRESHOLDS = {             # Min move over rolling window (10min)
    'large': 0.015,   # 1.5%
    'mid':   0.030,   # 3.0%
    'small': 0.050,   # 5.0%
}
LIGHT_CHECK_ROLLING_WINDOW = 5                 # 5 checks = 10 minutes

# Smart Scheduling - Priority Queue Scheduler (Phase 2)
SCHEDULER_ENABLED = True                           # Use smart scheduler instead of fixed cycle
FULL_CHECK_COOLDOWN_S = 120                        # Min 2min between two full checks of the same token
FULL_CHECK_BASE_INTERVAL_MIN = 10                  # Base recheck interval (no position, routine)
FULL_CHECK_POSITION_INTERVAL_MIN = 3               # Recheck interval when token has open position

# ============================================================================
# Signal Confirmation & Trend Filter
# ============================================================================

# Signal confirmation
ADAPTIVE_HYBRID_CONFIRMATION_BARS = 0  # Wait N bars to confirm signal
ADAPTIVE_HYBRID_SCORE_PERSISTENCE_CYCLES = 2  # Score must be above threshold for N cycles

# 4H trend filter
ADAPTIVE_HYBRID_4H_TREND_FILTER = True
ADAPTIVE_HYBRID_4H_TREND_PENALTY = 0.05  # 5% score reduction if 4h neutral (loosened from 0.10 to allow more BUY signals in bear bias)
ADAPTIVE_HYBRID_4H_TREND_REJECT = False  # Reject if 4h opposes 1h

# ═══════════════════════════════════════════════════════════════
# INDEPENDENT STRATEGY CONFIGS
# ═══════════════════════════════════════════════════════════════

# Strategy 1: Funding Rate Mean Reversion
# DISABLED (2026-06): signal cassé — le "z-score 7j" est en réalité calculé sur ~14h
# (deque de 168 échantillons remplie toutes les 5 min) → z-scores absurdes (jusqu'à 12.9)
# sans pouvoir prédictif. 11 SL = -$61 sur 32j. Refonte du signal requise avant réactivation.
FUNDING_MR_ENABLED = False
FUNDING_MR_TOKENS = ['BTC', 'ETH', 'SOL', 'XRP', 'AVAX', 'SUI', 'TAO', 'NEAR', 'AAVE', 'ENA', 'LINK', 'DOGE', 'kPEPE', 'ADA']
FUNDING_MR_ZSCORE_ENTRY = 1.3  # LEGACY/UNUSED: actual thresholds live in funding_mean_reversion.py ZSCORE_THRESHOLDS (1.5/1.3/1.0 per tier). Kept for backward compat, value is the weighted avg.
FUNDING_MR_ZSCORE_EXIT = 0.5   # Exit when funding normalizes
FUNDING_MR_RSI_CONFIRM = True  # Require RSI divergence confirmation
FUNDING_MR_MAX_HOLD_HOURS = 12  # Funding resets every 8h on HL
FUNDING_MR_RISK_PCT = 0.015    # 1.5% risk per trade
FUNDING_MR_SL_ATR_MULT = 3.0   # Stop loss in ATR multiples
FUNDING_MR_TP_ATR_MULT = 6.0   # Take profit in ATR multiples
FUNDING_MR_MAX_DAILY_TRADES = 4
FUNDING_MR_MAX_DAILY_LOSS_USD = 15.0
FUNDING_MR_LEVERAGE = {'btc': 3, 'eth': 3, 'mid': 3, 'alt': 2}

# Strategy 2: Volatility Compression Breakout
# Seule stratégie indépendante rentable sur 32j (PF 1.84). Extension PRUDENTE :
# univers élargi de 6 mids liquides HyperLiquid + cap journalier 3→4 + borne ADX max.
# NE PAS toucher RISK_PCT / VOLUME_SPIKE / squeeze / trailing (tunés, edge fragile n=29).
VOL_BREAKOUT_ENABLED = True
VOL_BREAKOUT_TOKENS = ['BTC', 'ETH', 'SOL', 'XRP', 'AVAX', 'SUI', 'TAO', 'NEAR', 'AAVE', 'LINK',
                       'DOGE', 'ADA', 'LTC', 'ARB', 'OP', 'INJ']  # +6 mids (existence HL vérifiée)
VOL_BREAKOUT_BB_PERCENTILE = 20  # BB width below Nth percentile = squeeze
VOL_BREAKOUT_VOLUME_SPIKE = 2.0  # Volume must be Nx average for breakout confirmation
VOL_BREAKOUT_ADX_THRESHOLD = 25  # ADX must rise above this
VOL_BREAKOUT_ADX_MAX = 32        # Reject entries when ADX already > 32 (late entries: 7 trades ADX>30 = -$10.42)
VOL_BREAKOUT_RISK_PCT = 0.015
VOL_BREAKOUT_TRAILING_ATR = 2.5  # Trailing stop distance
VOL_BREAKOUT_MAX_HOLD_HOURS = 24
VOL_BREAKOUT_MAX_DAILY_TRADES = 4  # 3→4 avec l'univers élargi
VOL_BREAKOUT_MAX_DAILY_LOSS_USD = 15.0
VOL_BREAKOUT_LEVERAGE = {'btc': 3, 'eth': 3, 'mid': 3, 'alt': 2}

# Strategy 3: Liquidation Cascade Fade
# Le feed liquidations est désormais Bybit WS (le WS Binance est bloqué depuis l'IP du
# serveur et le REST fallback allForceOrders a été supprimé par Binance).
LIQ_CASCADE_ENABLED = True
LIQ_CASCADE_TOKENS = ['BTC', 'ETH', 'SOL', 'XRP', 'AVAX', 'SUI']  # Only liquid enough for cascade fading
LIQ_CASCADE_SIGMA_THRESHOLD = 2.0  # Liquidation volume > Nx std dev (loosened from 3.0 to detect smaller cascades)
LIQ_CASCADE_RISK_PCT = 0.01  # 1% risk (smaller, high WR target)
LIQ_CASCADE_MAX_HOLD_HOURS = 4  # Quick fade, short hold
LIQ_CASCADE_SL_ATR_MULT = 2.0
LIQ_CASCADE_TP_ATR_MULT = 3.0
LIQ_CASCADE_MAX_DAILY_TRADES = 2
LIQ_CASCADE_MAX_DAILY_LOSS_USD = 10.0
LIQ_CASCADE_LEVERAGE = {'btc': 3, 'eth': 3, 'mid': 2, 'alt': 2}
LIQ_CASCADE_COOLDOWN_MINUTES = 30  # Min time between cascade trades

# Strategy 4: Fibonacci OTE Scalping (new — inspired by Casper's scalping strategy)
# DISABLED (2026-06): aucun edge brut (+$0.87 avant friction sur 157 trades, PF réel 0.26,
# -$119). La friction frais+slippage représente 101% de la perte — aucun retuning possible.
OTE_SCALP_ENABLED = False
OTE_SCALP_TOKENS = ['BTC', 'ETH', 'SOL']  # Start with most liquid only (tight spreads critical for scalping)
OTE_SCALP_TREND_TIMEFRAME = '1h'       # H1 for trend direction
OTE_SCALP_ENTRY_TIMEFRAME = '5m'       # M5 for entry execution
OTE_SCALP_TREND_EMA_FAST = 20
OTE_SCALP_TREND_EMA_SLOW = 50
OTE_SCALP_TREND_EMA_FILTER = 200
OTE_SCALP_ADX_MIN = 17                  # Minimum ADX on H1 for trend confirmation (re-tightened from 15 after WR 11% on 18 trades; was 20 pre-loosening)
OTE_SCALP_SWING_LOOKBACK = 24           # M5 candles (~2h of price action)
OTE_SCALP_SWING_MIN_RANGE_PCT = 0.0035  # 0.35% minimum impulse range (re-tightened from 0.003 after WR 11% on 18 trades; was 0.004 pre-loosening)
OTE_SCALP_OTE_LOW = 0.618
OTE_SCALP_OTE_HIGH = 0.786
OTE_SCALP_RR_RATIO = 2.0                # 1:2 risk/reward
OTE_SCALP_SL_BUFFER_ATR = 0.2           # Extra ATR beyond swing for SL
OTE_SCALP_RISK_PCT = 0.005              # 0.5% risk per trade (scalping = high freq)
OTE_SCALP_MAX_HOLD_MINUTES = 60         # Time stop — scalping shouldn't hold longer
OTE_SCALP_MAX_DAILY_TRADES = 6          # Tightened from 10 — over-trading fix (8.2 trades/day live, PnL -$0.37/trade avg)
OTE_SCALP_MAX_DAILY_LOSS_USD = 20.0     # Kill-switch
OTE_SCALP_MAX_POSITIONS = 2
OTE_SCALP_COOLDOWN_MINUTES = 15         # Between trades on same token (any close)
OTE_SCALP_LOSS_COOLDOWN_MINUTES = 60    # Extra cooldown after a STOP_LOSS on same token (prevents revenge re-entry)
OTE_SCALP_LEVERAGE = {'btc': 3, 'eth': 3, 'mid': 3, 'alt': 2}
OTE_SCALP_VOLUME_MIN = 0.15             # Dead market filter
OTE_SCALP_AVOID_HOURS_UTC = [1, 2, 3]   # Asia night low liquidity
OTE_SCALP_FUNDING_AVOID_WINDOW_MIN = 5  # Minutes before/after funding settlement to avoid
OTE_SCALP_BOS_BUFFER_PCT = 0.001        # 0.1% buffer past entry for BE after BOS
OTE_SCALP_CONFIRMATION_BARS = 0         # No confirmation delay (scalping needs speed)
OTE_SCALP_USE_LIMIT_ORDERS = False      # Market orders for paper; limit orders for live
OTE_SCALP_MAX_POSITION_PCT = 25         # Max 25% of balance per position

# Global independent strategy settings
INDEPENDENT_STRATEGIES_ENABLED = True
INDEPENDENT_STRATEGIES_MAX_TOTAL_DAILY_LOSS_USD = 50.0  # Was 30, accommodates 4th strategy (-$20 ote_scalp)
INDEPENDENT_STRATEGIES_MAX_POSITIONS = 6  # Across all 4 strategies combined (was 4)

