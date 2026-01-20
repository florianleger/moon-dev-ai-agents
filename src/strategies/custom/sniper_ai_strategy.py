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
from termcolor import cprint
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

from ..base_strategy import BaseStrategy

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

        # Load existing state
        self._load_state_from_csv()

        cprint(f"[Sniper] Strategy initialized", "cyan")
        cprint(f"  - Assets: {self.assets}", "white")
        cprint(f"  - AI Confidence Required: {SNIPER_AI_MIN_CONFIDENCE}%", "white")
        cprint(f"  - Paper Trading: {PAPER_TRADING}", "white")
        cprint(f"  - Loaded positions: {len(self.paper_positions)}", "white")
        cprint(f"  - Current balance: ${self.paper_balance:,.2f}", "white")

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

        Returns:
            dict: {passed, sigma, direction, price_change_pct, threshold}
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

            passed = abs(z_score) >= SNIPER_SIGMA_THRESHOLD
            direction = 'oversold' if z_score < 0 else 'overbought'

            return {
                'passed': passed,
                'sigma': round(z_score, 2),
                'direction': direction,
                'price_change_pct': round(price_change_pct, 2),
                'threshold': SNIPER_SIGMA_THRESHOLD
            }

        except Exception as e:
            cprint(f"[Sniper] Error checking extreme move: {e}", "yellow")
            return {'passed': False, 'sigma': 0, 'direction': 'neutral', 'price_change_pct': 0, 'threshold': SNIPER_SIGMA_THRESHOLD}

    def check_funding_divergence(self, symbol: str, direction: str) -> dict:
        """
        Check condition 2: Contrarian funding signal.

        For LONG: Funding should be very negative (shorts paying longs)
        For SHORT: Funding should be very positive (longs paying shorts)

        Returns:
            dict: {passed, funding_rate, funding_zscore, is_contrarian}
        """
        try:
            if self._market_data is None:
                return {'passed': False, 'funding_rate': 0, 'funding_zscore': 0, 'is_contrarian': False}

            funding_zscore = self._market_data.get_funding_zscore(symbol)
            funding_data = self._market_data.get_funding_rate(symbol)

            funding_rate = funding_data['funding_rate'] if funding_data else 0
            annual_rate = funding_rate * 24 * 365 * 100 if funding_data else 0

            is_extreme_negative = funding_zscore < -SNIPER_FUNDING_EXTREME_THRESHOLD
            is_extreme_positive = funding_zscore > SNIPER_FUNDING_EXTREME_THRESHOLD

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
                'is_contrarian': is_contrarian
            }

        except Exception as e:
            cprint(f"[Sniper] Error checking funding divergence: {e}", "yellow")
            return {'passed': False, 'funding_rate': 0, 'funding_zscore': 0, 'is_contrarian': False}

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

    def check_volume_climax(self, df: pd.DataFrame) -> dict:
        """
        Check condition 5: Volume exhaustion spike detected.

        Looking for: Recent spike > 3x average, now declining.

        Returns:
            dict: {passed, current_ratio, peak_ratio, is_climax, is_declining}
        """
        try:
            volume = df['volume'].tail(50)
            avg_volume = volume.mean()

            current_volume = volume.iloc[-1]
            current_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            recent_volume = volume.tail(10)
            peak_volume = recent_volume.max()
            peak_ratio = peak_volume / avg_volume if avg_volume > 0 else 1.0

            is_declining = current_volume < peak_volume * 0.7
            is_climax = peak_ratio >= SNIPER_VOLUME_SPIKE_THRESHOLD

            passed = is_climax and is_declining

            return {
                'passed': passed,
                'current_ratio': round(current_ratio, 2),
                'peak_ratio': round(peak_ratio, 2),
                'is_climax': is_climax,
                'is_declining': is_declining,
                'threshold': SNIPER_VOLUME_SPIKE_THRESHOLD
            }

        except Exception as e:
            cprint(f"[Sniper] Error checking volume climax: {e}", "yellow")
            return {'passed': False, 'current_ratio': 0, 'peak_ratio': 0, 'is_climax': False, 'is_declining': False}

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
        - Price drops >5% in <4h
        - Funding very negative
        - Liquidation spike (longs liquidated)
        - RSI < 25
        """
        try:
            # Check 4h price drop
            lookback = min(16, len(df))
            price_4h_ago = df['close'].iloc[-lookback]
            current_price = df['close'].iloc[-1]
            price_change = (current_price - price_4h_ago) / price_4h_ago * 100

            # Check RSI
            rsi = RSIIndicator(df['close'], window=14).rsi().iloc[-1]

            is_capitulation = (
                price_change <= -SNIPER_CAPITULATION_MIN_DROP_PCT and
                rsi <= SNIPER_RSI_OVERSOLD
            )

            return {
                'detected': is_capitulation,
                'type': 'capitulation_fade',
                'direction': 'BUY',
                'price_change_4h': round(price_change, 2),
                'rsi': round(rsi, 1),
                'current_price': float(current_price)
            }

        except Exception as e:
            cprint(f"[Sniper] Error detecting capitulation: {e}", "yellow")
            return {'detected': False, 'type': 'capitulation_fade', 'direction': 'BUY'}

    def detect_euphoria_fade(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Detect euphoria fade setup (SHORT entry).

        Conditions:
        - Price rises >5% in <4h
        - Funding very positive
        - RSI > 75
        """
        try:
            lookback = min(16, len(df))
            price_4h_ago = df['close'].iloc[-lookback]
            current_price = df['close'].iloc[-1]
            price_change = (current_price - price_4h_ago) / price_4h_ago * 100

            rsi = RSIIndicator(df['close'], window=14).rsi().iloc[-1]

            is_euphoria = (
                price_change >= SNIPER_EUPHORIA_MIN_RISE_PCT and
                rsi >= SNIPER_RSI_OVERBOUGHT
            )

            return {
                'detected': is_euphoria,
                'type': 'euphoria_fade',
                'direction': 'SELL',
                'price_change_4h': round(price_change, 2),
                'rsi': round(rsi, 1),
                'current_price': float(current_price)
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
        volume_climax = self.check_volume_climax(df)
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

        # Only run AI validation if first 6 pass
        ai_validation = {'passed': False, 'confidence': 0, 'reasoning': 'First 6 conditions not met'}

        if passed_count == 6:
            cprint(f"\n  All 6 conditions passed! Running AI validation...", "cyan")
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
            cprint(f"\n  Only {passed_count}/6 conditions passed - skipping AI validation", "yellow")

        # Final result
        all_passed = passed_count == 6 and ai_validation['passed']

        return {
            'all_passed': all_passed,
            'passed_count': passed_count + (1 if ai_validation['passed'] else 0),
            'total': 7,
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

            # Check for capitulation fade (LONG)
            if capitulation['detected']:
                cprint(f"[Sniper] Capitulation fade detected! Running full checklist...", "green")
                result = self.run_checklist(symbol, df, 'capitulation_fade', 'BUY')

                if result['all_passed']:
                    return self._build_signal(symbol, result)

            # Check for euphoria fade (SHORT)
            if euphoria['detected']:
                cprint(f"[Sniper] Euphoria fade detected! Running full checklist...", "red")
                result = self.run_checklist(symbol, df, 'euphoria_fade', 'SELL')

                if result['all_passed']:
                    return self._build_signal(symbol, result)

            # No valid setup
            if not capitulation['detected'] and not euphoria['detected']:
                cprint(f"[Sniper] No setup detected for {symbol}", "white")

            return {
                'token': symbol,
                'signal': 0.0,
                'direction': 'NEUTRAL',
                'metadata': {
                    'strategy_type': 'sniper_ai',
                    'reason': 'No valid setup or checklist not passed'
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

        signal = {
            'token': symbol,
            'signal': round(ai_val.get('confidence', 85) / 100, 3),
            'direction': result['direction'],
            'metadata': {
                'strategy_type': 'sniper_ai',
                'setup_type': result['type'],
                'current_price': result['current_price'],
                'checklist_score': f"{result['passed_count']}/{result['total']}",
                'ai_confidence': ai_val.get('confidence', 0),
                'ai_reasoning': ai_val.get('reasoning', ''),
                'stop_loss_pct': ai_val.get('suggested_sl_pct', SNIPER_STOP_LOSS_PCT),
                'take_profit_pct': ai_val.get('suggested_tp_pct', SNIPER_TAKE_PROFIT_PCT),
                'leverage': SNIPER_LEVERAGE,
                'extreme_move': result['extreme_move'],
                'funding_divergence': result['funding_divergence'],
                'time_window': result['time_window']
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
        """Monitor all open paper positions and close those that hit SL/TP."""
        if not PAPER_TRADING or not self.paper_positions:
            return []

        closed = []

        symbols_to_check = set(pos['symbol'] for pos in self.paper_positions.values())
        current_prices = {}

        for symbol in symbols_to_check:
            try:
                df = self._fetch_candles(symbol, interval='15m', candles=5)
                if df is not None and len(df) > 0:
                    current_prices[symbol] = float(df['close'].iloc[-1])
            except Exception as e:
                cprint(f"[Sniper] Could not fetch price for {symbol}: {e}", "yellow")

        positions_to_close = []

        for position_id, trade in self.paper_positions.items():
            symbol = trade['symbol']
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            direction = trade['direction']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']

            close_reason = None

            if direction == 'BUY':
                if current_price <= stop_loss:
                    close_reason = 'STOP_LOSS'
                elif current_price >= take_profit:
                    close_reason = 'TAKE_PROFIT'
            else:
                if current_price >= stop_loss:
                    close_reason = 'STOP_LOSS'
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
