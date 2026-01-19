"""
🌙 Moon Dev's Configuration File
Built with love by Moon Dev 🚀
"""

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
# Uses MULTIFACTOR_ASSETS defined below for consistency
HYPERLIQUID_SYMBOLS = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'DOT', 'MATIC']
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
address = '4wgfCBf2WwLSRKLef9iW7JXZ2AfkxUxGM4XcKpHm3Sin' # YOUR WALLET ADDRESS HERE

# Position sizing 🎯
usd_size = 25  # Size of position to hold
max_usd_order_size = 3  # Max order size
tx_sleep = 30  # Sleep between transactions
slippage = 199  # Slippage settings

# Risk Management Settings 🛡️
CASH_PERCENTAGE = 20  # Minimum % to keep in USDC as safety buffer (0-100)
MAX_POSITION_PERCENTAGE = 30  # Maximum % allocation per position (0-100)
STOPLOSS_PRICE = 1 # NOT USED YET 1/5/25    
BREAKOUT_PRICE = .0001 # NOT USED YET 1/5/25
SLEEP_AFTER_CLOSE = 600  # Prevent overtrading

MAX_LOSS_GAIN_CHECK_HOURS = 12  # How far back to check for max loss/gain limits (in hours)
SLEEP_BETWEEN_RUNS_MINUTES = 15  # How long to sleep between agent runs 🕒


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
slippage = 199  # 500 = 5% and 50 = .5% slippage
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
AI_MODEL = "claude-3-haiku-20240307"  # Model Options:
                                     # - claude-3-haiku-20240307 (Fast, efficient Claude model)
                                     # - claude-3-sonnet-20240229 (Balanced Claude model)
                                     # - claude-3-opus-20240229 (Most powerful Claude model)
AI_MAX_TOKENS = 1024  # Max tokens for response
AI_TEMPERATURE = 0.7  # Creativity vs precision (0-1)

# Trading Strategy Agent Settings - MAY NOT BE USED YET 1/5/25
ENABLE_STRATEGIES = True  # Set this to True to use strategies
STRATEGY_MIN_CONFIDENCE = 0.7  # Minimum confidence to act on strategy signals

# Sleep time between main agent runs
SLEEP_BETWEEN_RUNS_MINUTES = 15  # How long to sleep between agent runs 🕒

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
RAMF_MIN_CONFIDENCE = 70             # Minimum confidence score to trade (0-100)
RAMF_MAX_DAILY_TRADES = 15           # Maximum trades per day (increased for paper trading)
RAMF_MAX_DAILY_LOSS_USD = 50         # Daily loss limit in USD (~10% of $500)
RAMF_MAX_DAILY_GAIN_USD = 75         # Daily gain limit in USD (~15% of $500)

# Volatility Regime Settings
# The strategy only trades in HIGH or LOW volatility regimes
# ATR percentile > HIGH threshold = mean-reversion mode (fade exhaustion)
# ATR percentile < LOW threshold = trend-following mode
# Between these thresholds = NORMAL (no trade) - widen bands for more signals
RAMF_VOLATILITY_HIGH_PERCENTILE = 60  # Top 40% of ATR = "high" volatility (default was 75)
RAMF_VOLATILITY_LOW_PERCENTILE = 40   # Bottom 40% of ATR = "low" volatility (default was 25)

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

ACTIVE_STRATEGY = 'ramf'             # Options: 'multifactor', 'ramf', 'example'

# Future variables (not active yet) 🔮
sell_at_multiple = 3
USDC_SIZE = 1
limit = 49
timeframe = '15m'
stop_loss_perctentage = -.24
EXIT_ALL_POSITIONS = False
DO_NOT_TRADE_LIST = ['777']
CLOSED_POSITIONS_TXT = '777'
minimum_trades_in_last_hour = 777
