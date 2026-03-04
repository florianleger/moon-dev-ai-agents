# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an experimental AI trading system that orchestrates 48+ specialized AI agents to analyze markets, execute strategies, and manage risk across cryptocurrency markets (HyperLiquid perpetuals + Solana). The project uses a modular agent architecture with unified LLM provider abstraction supporting Claude, GPT-4, DeepSeek, Groq, Gemini, and local Ollama models.

**Primary strategy:** Adaptive Hybrid Strategy with 14 scoring modules, LLM-enhanced pipeline (trade confirmation, Wyckoff regime classification, post-trade learning), and multi-timeframe confluence. See `STRATEGY_OPTIMIZATION.md` for full details.

## Key Development Commands

### Conductor Workspace Setup
```bash
# For Conductor workspaces, run the setup script first:
./setup-conductor.sh

# This will:
# 1. Copy .env from ~/dev/moon-dev-ai-agents/
# 2. Activate conda environment 'tflow'
# 3. Install all Python dependencies
```

### Manual Environment Setup
```bash
# Use existing conda environment (DO NOT create new virtual environments)
conda activate tflow

# Install/update dependencies
pip install -r requirements.txt

# IMPORTANT: Update requirements.txt every time you add a new package
pip freeze > requirements.txt
```

### Running the System
```bash
# Run main orchestrator (controls multiple agents)
python src/main.py

# Run individual agents standalone
python src/agents/trading_agent.py
python src/agents/risk_agent.py
python src/agents/rbi_agent.py
python src/agents/chat_agent.py
# ... any agent in src/agents/ can run independently
```

### Backtesting
```bash
# Run Adaptive Hybrid backtest (fetches live data from HyperLiquid)
python src/backtesting/backtest_adaptive_hybrid.py --symbol BTC --timeframe 1h --days 180

# Walk-forward validation
python src/backtesting/backtest_adaptive_hybrid.py --symbol BTC --timeframe 1h --days 180 --walk-forward

# Grid search optimization
python src/backtesting/optimize_params.py
python src/backtesting/optimize_params_v2.py

# Legacy: Use backtesting.py library with pandas_ta or talib for indicators
# Sample OHLCV data available at:
# /Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv
```

### Running Tests
```bash
conda activate tflow
python -m pytest tests/ -q                           # All tests
python -m pytest tests/test_adaptive_hybrid_scoring.py  # Strategy scoring
python -m pytest tests/test_signal_pipeline.py          # Signal pipeline
python -m pytest tests/test_risk_agent.py               # Risk agent
python -m pytest tests/test_data_providers.py           # Data providers
```

### Coolify Deployment
The application is deployed on Coolify at:
- **Coolify Console**: https://console.gallion.tech
- **Application UUID**: `v4cgs8wk8gsgw04ww8c8sk0o`
- **Dashboard URL**: `http://v4cgs8wk8gsgw04ww8c8sk0o.46.224.59.218.sslip.io/dashboard`

Deployments are triggered automatically on push to `main` branch.

## Architecture Overview

### Core Structure
```
src/
├── agents/              # 48+ specialized AI agents (each <800 lines)
├── models/              # LLM provider abstraction (ModelFactory pattern)
├── strategies/
│   ├── custom/          # Strategy implementations
│   │   └── adaptive_hybrid_strategy.py  # Primary strategy (14 modules)
│   └── modules/         # Scoring modules (independent, testable)
│       ├── mean_reversion.py, momentum.py, ema_trend.py, ...  # Technical (7)
│       ├── crowd_positioning.py, social_hype.py, funding_divergence.py  # Sentiment (3)
│       ├── llm_confirmation.py, llm_regime.py, trade_learner.py  # LLM pipeline (3)
│       └── mtf_confluence.py  # Multi-timeframe (1)
├── backtesting/         # Backtest engine + strategy-specific runners + optimizers
├── execution/           # LiveOrderManager (HyperLiquid SDK native orders)
├── data_providers/      # Market data (HyperLiquid, Binance, CoinGecko, Fear&Greed, etc.)
├── data/                # Agent outputs, memory, analysis results
│   └── trade_memory.py  # SQLite-based trade journal + lesson storage
├── utils/               # Logging (structured) + alerting (Discord/Telegram)
├── config.py            # Global configuration (positions, risk limits, API settings)
├── main.py              # Main orchestrator for multi-agent loop
├── nice_funcs.py        # ~1,200 lines of shared trading utilities
├── nice_funcs_hl.py     # Hyperliquid-specific utilities
└── scripts/             # Standalone utility scripts
tests/                   # 80+ pytest tests (scoring, pipeline, risk, data providers)
```

### Agent Ecosystem

**Trading Agents**: `trading_agent`, `strategy_agent`, `risk_agent`, `copybot_agent`
**Market Analysis**: `sentiment_agent`, `whale_agent`, `funding_agent`, `liquidation_agent`, `chartanalysis_agent`
**Content Creation**: `chat_agent`, `clips_agent`, `tweet_agent`, `video_agent`, `phone_agent`
**Strategy Development**: `rbi_agent` (Research-Based Inference - codes backtests from videos/PDFs), `research_agent`
**Specialized**: `sniper_agent`, `solana_agent`, `tx_agent`, `million_agent`, `tiktok_agent`, `compliance_agent`

Each agent can run independently or as part of the main orchestrator loop.

### LLM Integration (Model Factory)

Located at `src/models/model_factory.py` and `src/models/README.md`

**Unified Interface**: All agents use `ModelFactory.create_model()` for consistent LLM access
**Supported Providers**: Anthropic Claude (default), OpenAI, DeepSeek, Groq, Google Gemini, Ollama (local)
**Key Pattern**:
```python
from src.models.model_factory import ModelFactory

model = ModelFactory.create_model('anthropic')  # or 'openai', 'deepseek', 'groq', etc.
response = model.generate_response(system_prompt, user_content, temperature, max_tokens)
```

### Configuration Management

**Primary Config**: `src/config.py`
- Trading settings: `MONITORED_TOKENS`, `EXCLUDED_TOKENS`, position sizing (`usd_size`, `max_usd_order_size`)
- Risk management: `CASH_PERCENTAGE`, `MAX_POSITION_PERCENTAGE`, `MAX_LOSS_USD`, `MAX_GAIN_USD`, `MINIMUM_BALANCE_USD`
- Agent behavior: `SLEEP_BETWEEN_RUNS_MINUTES`, `ACTIVE_AGENTS` dict in `main.py`
- AI settings: `AI_MODEL`, `AI_MAX_TOKENS`, `AI_TEMPERATURE`
- **Adaptive Hybrid Strategy**: `ADAPTIVE_HYBRID_*` parameters (threshold, ATR profiles, module weights, LLM pipeline flags)
- **LLM Pipeline Flags**: `ADAPTIVE_HYBRID_LLM_CONFIRMATION`, `_LLM_REGIME`, `_LLM_LEARNER`, `_MTF_CONFLUENCE` (all True/False)

**Environment Variables**: `.env` (see `.env_example`)
- Trading APIs: `BIRDEYE_API_KEY`, `COINGECKO_API_KEY` (MOONDEV_API_KEY no longer needed)
- AI Services: `ANTHROPIC_KEY`, `OPENAI_KEY`, `DEEPSEEK_KEY`, `GROQ_API_KEY`, `GEMINI_KEY`
- Blockchain: `SOLANA_PRIVATE_KEY`, `HYPER_LIQUID_ETH_PRIVATE_KEY`, `RPC_ENDPOINT`

### Shared Utilities

**`src/nice_funcs.py`** (~1,200 lines): Core trading functions
- Data: `token_overview()`, `token_price()`, `get_position()`, `get_ohlcv_data()`
- Trading: `market_buy()`, `market_sell()`, `chunk_kill()`, `open_position()`
- Analysis: Technical indicators, PnL calculations, rug pull detection

**`src/data_providers/`**: Market data providers (replaces Moon Dev API)
- `market_data.py`: Unified interface for funding rates, OI (HyperLiquid), liquidations (Binance)
- `binance_futures.py`: Real-time liquidations via WebSocket + CSV persistence
- `binance_sentiment.py`: L/S ratio, Top Trader ratio, Taker Buy/Sell volume (free API)
- `coingecko_social.py`: Trending coins, global market cap, price momentum (free API)
- `cross_exchange_funding.py`: HyperLiquid vs Binance funding rate divergence (free API)
- `fear_greed.py`: Alternative.me Fear & Greed Index (contrarian signal)
- `defi_llama.py`: TVL and DEX volume data

**`src/strategies/modules/`**: 14 independent scoring modules (see `STRATEGY_OPTIMIZATION.md` for full inventory)

**`src/utils/`**: Infrastructure
- `logger.py`: Structured logging (colored console + JSON file)
- `alerting.py`: Discord/Telegram webhooks (circuit breaker alerts, daily summary)

### Data Flow Pattern

```
Config/Input → Agent Init → API Data Fetch → Data Parsing →
LLM Analysis (via ModelFactory) → Decision Output →
Result Storage (CSV/JSON in src/data/) → Optional Trade Execution
```

## Development Rules

### File Management
- **Keep files under 800 lines** - if longer, split into new files and update README
- **DO NOT move files without asking** - you can create new files but no moving
- **NEVER create new virtual environments** - use existing `conda activate tflow`
- **Update requirements.txt** after adding any new package

### Backtesting
- Use `backtesting.py` library (NOT their built-in indicators)
- Use `pandas_ta` or `talib` for technical indicators instead
- Sample data available at `/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv`

### Code Style
- **No fake/synthetic data** - always use real data or fail the script
- **Minimal error handling** - user wants to see errors, not over-engineered try/except blocks
- **No API key exposure** - never show keys from `.env` in output

### Agent Development Pattern

When creating new agents:
1. Inherit from base patterns in existing agents
2. Use `ModelFactory` for LLM access
3. Store outputs in `src/data/[agent_name]/`
4. Make agent independently executable (standalone script)
5. Add configuration to `config.py` if needed
6. Follow naming: `[purpose]_agent.py`

### Testing Strategies

Place strategy definitions in `src/strategies/` folder:
```python
class YourStrategy(BaseStrategy):
    name = "strategy_name"
    description = "what it does"

    def generate_signals(self, token_address, market_data):
        return {
            "action": "BUY"|"SELL"|"NOTHING",
            "confidence": 0-100,
            "reasoning": "explanation"
        }
```

## Important Context

### Risk-First Philosophy
- Risk Agent runs first in main loop before any trading decisions
- Configurable circuit breakers (`MAX_LOSS_USD`, `MINIMUM_BALANCE_USD`)
- AI confirmation for position-closing decisions (configurable via `USE_AI_CONFIRMATION`)

### Data Sources
1. **HyperLiquid API** - OHLCV, funding rates, open interest, order book, positions (free, real-time)
2. **Binance Futures API** - Real-time liquidations (WebSocket), L/S ratios, top trader positioning, taker volume (free)
3. **CoinGecko API** - Trending coins, global market cap, price momentum (free, 10K calls/month)
4. **Alternative.me** - Fear & Greed Index (free)
5. **DefiLlama** - TVL and DEX volumes (free)
6. **BirdEye API** - Solana token data (price, volume, liquidity, OHLCV)
7. **Helius RPC** - Solana blockchain interaction

### Autonomous Execution
- Main loop runs every 15 minutes by default (`SLEEP_BETWEEN_RUNS_MINUTES`)
- Agents handle errors gracefully and continue execution
- Keyboard interrupt for graceful shutdown
- All agents log to console with color-coded output (termcolor)

### AI-Driven Strategy Generation (RBI Agent)
1. User provides: YouTube video URL / PDF / trading idea text
2. DeepSeek-R1 analyzes and extracts strategy logic
3. Generates backtesting.py compatible code
4. Executes backtest and returns performance metrics
5. Cost: ~$0.027 per backtest execution (~6 minutes)

## Common Patterns

### Adding New Agent
1. Create `src/agents/your_agent.py`
2. Implement standalone execution logic
3. Add to `ACTIVE_AGENTS` in `main.py` if needed for orchestration
4. Use `ModelFactory` for LLM calls
5. Store results in `src/data/your_agent/`

### Switching AI Models
Edit `config.py`:
```python
AI_MODEL = "claude-3-haiku-20240307"  # Fast, cheap
# AI_MODEL = "claude-3-sonnet-20240229"  # Balanced
# AI_MODEL = "claude-3-opus-20240229"  # Most powerful
```

Or use different models per agent via ModelFactory:
```python
model = ModelFactory.create_model('deepseek')  # Reasoning tasks
model = ModelFactory.create_model('groq')      # Fast inference
```

### Reading Market Data
```python
from src.nice_funcs import token_overview, get_ohlcv_data, token_price

# Get comprehensive token data
overview = token_overview(token_address)

# Get price history
ohlcv = get_ohlcv_data(token_address, timeframe='1H', days_back=3)

# Get current price
price = token_price(token_address)
```

## Project Philosophy

This is an **experimental, educational project** demonstrating AI agent patterns through algorithmic trading:
- No guarantees of profitability (substantial risk of loss)
- Open source and free for learning
- YouTube-driven development with weekly updates
- Community-supported via Discord
- No token associated with project (avoid scams)

The goal is to democratize AI agent development and show practical multi-agent orchestration patterns that can be applied beyond trading.

## Adaptive Hybrid Strategy (Primary)

The primary trading strategy uses 14 independent scoring modules, LLM-enhanced pipeline, and optimized ATR-based risk management. Full documentation in `STRATEGY_OPTIMIZATION.md`.

### Key Performance (Backtest BTC 1h 180d)
- Return: +11.66% (benchmark: -34.57%), Alpha: +46.4%
- Win Rate: 55.4%, Profit Factor: 1.46, Sharpe: 2.75, Max DD: 3.25%
- Walk-forward: 3/3 folds profitable

### Quick Parameter Reference

| Parameter | Value | File |
|-----------|-------|------|
| Score threshold | 55 | `config.py` → `ADAPTIVE_HYBRID_BASE_THRESHOLD` |
| BTC SL/TP | 2.8 / 4.2 ATR | `config.py` → `ADAPTIVE_HYBRID_ATR_PROFILES` |
| ETH SL/TP | 4.0 / 8.0 ATR | Same |
| Min convergent modules | 2 | `config.py` → `ADAPTIVE_HYBRID_MIN_CONVERGENT_MODULES` |
| LLM provider (fast) | groq | `config.py` → `ADAPTIVE_HYBRID_LLM_PROVIDER` |

### Overfitting Awareness

Parameters were optimized on 180 days of bear market data. Three layers of mitigation:
1. **Walk-forward validation** (3/3 OOS folds profitable, cross-asset on ETH)
2. **LLM adaptation** (regime classifier, trade confirmation, post-trade learning - none existed during backtest)
3. **Mechanical safeguards** (max daily loss, drawdown circuit breakers, position limits)

Monitor PF > 1.0 and DD < 10% during first 2-4 weeks of paper trading. See `STRATEGY_OPTIMIZATION.md` for full robustness analysis.

## Legacy: RAMF Strategy Tuning Learnings

The RAMF strategy is now integrated as the `ramf_lite` module within the Adaptive Hybrid Strategy. Its standalone settings remain in `config.py` under `RAMF_*` for backward compatibility.
