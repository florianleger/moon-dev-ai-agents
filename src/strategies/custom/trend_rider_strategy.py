"""
Trend Rider Strategy - Trend-Following Pullback Strategy with LLM Validation

This strategy identifies pullback opportunities within established trends:
- Detects trend via EMA alignment (20 > 50 > 200 for bullish)
- Waits for pullback to key EMA levels
- Confirms with candle pattern and volume
- Uses Claude LLM for final validation

Designed to complement Sniper AI (mean-reversion) in hybrid mode.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from termcolor import cprint
from typing import Dict, Optional, Tuple

from ..base_strategy import BaseStrategy

# Import config with fallbacks
try:
    from src.config import (
        TREND_RIDER_ENABLED,
        TREND_RIDER_ASSETS,
        TREND_RIDER_ADX_MIN,
        TREND_RIDER_RSI_PULLBACK_LONG,
        TREND_RIDER_RSI_PULLBACK_SHORT,
        TREND_RIDER_VOLUME_DECLINE_RATIO,
        TREND_RIDER_VOLUME_SPIKE_RATIO,
        TREND_RIDER_MIN_CANDLE_BODY_PCT,
        TREND_RIDER_LEVERAGE,
        TREND_RIDER_ATR_SL_MULT,
        TREND_RIDER_ATR_TP_MULT,
        TREND_RIDER_TRAILING_ATR_MULT,
        TREND_RIDER_MAX_DAILY_TRADES,
        TREND_RIDER_MAX_DAILY_LOSS_USD,
        TREND_RIDER_AI_MODEL,
        TREND_RIDER_AI_MIN_CONFIDENCE,
        TREND_RIDER_AI_TEMPERATURE,
        TREND_RIDER_MIN_SCORE,
        TREND_RIDER_WEIGHTS,
    )
except ImportError:
    # Defaults if config not yet updated
    TREND_RIDER_ENABLED = True
    TREND_RIDER_ASSETS = ['BTC', 'ETH', 'SOL', 'LINK', 'AVAX']
    TREND_RIDER_ADX_MIN = 35
    TREND_RIDER_RSI_PULLBACK_LONG = (30, 45)
    TREND_RIDER_RSI_PULLBACK_SHORT = (55, 70)
    TREND_RIDER_VOLUME_DECLINE_RATIO = 0.8
    TREND_RIDER_VOLUME_SPIKE_RATIO = 1.2
    TREND_RIDER_MIN_CANDLE_BODY_PCT = 50
    TREND_RIDER_LEVERAGE = 2
    TREND_RIDER_ATR_SL_MULT = 1.5
    TREND_RIDER_ATR_TP_MULT = 2.0
    TREND_RIDER_TRAILING_ATR_MULT = 2.0
    TREND_RIDER_MAX_DAILY_TRADES = 3
    TREND_RIDER_MAX_DAILY_LOSS_USD = 25
    TREND_RIDER_AI_MODEL = 'claude-sonnet-4-5'
    TREND_RIDER_AI_MIN_CONFIDENCE = 70
    TREND_RIDER_AI_TEMPERATURE = 0.3
    TREND_RIDER_MIN_SCORE = 7.0
    TREND_RIDER_WEIGHTS = {
        'trend_alignment': 2.5,
        'pullback_quality': 2.0,
        'momentum_confirmation': 2.0,
        'htf_agreement': 1.5,
        'ai_validation': 2.0,
    }

# LLM System Prompt
TREND_RIDER_SYSTEM_PROMPT = """You are a professional trend-following trading analyst specializing in crypto perpetuals on HyperLiquid.

Your role is to evaluate pullback opportunities within established trends. You must:
1. Assess trend quality and sustainability
2. Validate pullback as healthy retracement (not trend reversal)
3. Evaluate entry timing and risk/reward
4. Consider macro context (BTC correlation, market sentiment)

CRITICAL PRINCIPLES:
- Only trade WITH the trend, never against it
- Pullbacks should be 20-50% retracement of the last swing (not too deep)
- Volume should DECREASE during pullback (profit-taking, not distribution)
- Volume should INCREASE on resumption candle (buyers stepping in)
- Avoid catching falling knives - wait for confirmation
- Be cautious near major support/resistance levels
- Consider BTC's trend for altcoin trades

REJECTION CRITERIA (any of these = REJECT):
- Trend showing signs of exhaustion (momentum divergence)
- Pullback too deep (>61.8% Fibonacci retracement)
- Volume increasing during pullback (distribution)
- Higher timeframe in opposite direction
- BTC showing weakness while trading altcoin long

You must respond with a valid JSON object (no markdown, no explanation outside JSON):
{
    "decision": "EXECUTE" or "REJECT",
    "confidence": 0-100,
    "trend_quality": "STRONG" or "MODERATE" or "WEAK",
    "pullback_type": "HEALTHY" or "DEEP" or "REVERSAL_RISK",
    "entry_timing": "OPTIMAL" or "EARLY" or "LATE",
    "risk_factors": ["list", "of", "concerns"],
    "reasoning": "detailed explanation of your analysis"
}"""


class TrendRiderStrategy(BaseStrategy):
    """Trend-following pullback strategy with LLM validation"""

    def __init__(self):
        super().__init__("Trend Rider")
        self.name = "Trend Rider"

        # Daily counters
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()

        # LLM model
        self._claude_model = None
        self._init_llm()

        cprint(f"[TrendRider] Strategy initialized", "cyan")
        cprint(f"  Assets: {TREND_RIDER_ASSETS}", "cyan")
        cprint(f"  ADX min: {TREND_RIDER_ADX_MIN}", "cyan")
        cprint(f"  Min score: {TREND_RIDER_MIN_SCORE}/10", "cyan")

    def _init_llm(self):
        """Initialize LLM model for validation"""
        try:
            from src.models.model_factory import ModelFactory
            model_factory = ModelFactory()
            self._claude_model = model_factory.get_model('claude', TREND_RIDER_AI_MODEL)
            cprint(f"[TrendRider] AI model initialized: {TREND_RIDER_AI_MODEL}", "green")
        except Exception as e:
            cprint(f"[TrendRider] Warning: Could not initialize AI model: {e}", "yellow")
            self._claude_model = None

    def _reset_daily_counters(self):
        """Reset daily counters if new day"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset_date = today
            cprint("[TrendRider] New trading day - counters reset", "cyan")

    def _check_daily_limits(self) -> Tuple[bool, str]:
        """Check if daily limits are reached"""
        if self.daily_trades >= TREND_RIDER_MAX_DAILY_TRADES:
            return False, f"Daily trade limit reached ({self.daily_trades}/{TREND_RIDER_MAX_DAILY_TRADES})"
        if self.daily_pnl <= -TREND_RIDER_MAX_DAILY_LOSS_USD:
            return False, f"Daily loss limit reached (${self.daily_pnl:.2f})"
        return True, ""

    def generate_signals(self, symbol: str = None, df: pd.DataFrame = None) -> dict:
        """
        Generate trading signal for a symbol.

        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            df: Optional pre-fetched DataFrame with OHLCV data

        Returns:
            Signal dict with token, signal strength, direction, metadata
        """
        self._reset_daily_counters()

        # Check daily limits
        can_trade, limit_reason = self._check_daily_limits()
        if not can_trade:
            return self._neutral_signal(symbol, limit_reason)

        # Skip if symbol not in our asset list
        if symbol and symbol not in TREND_RIDER_ASSETS:
            return self._neutral_signal(symbol, f"Not in TREND_RIDER_ASSETS")

        # Fetch data if not provided
        if df is None:
            df = self._fetch_data(symbol)
            if df is None or len(df) < 50:
                return self._neutral_signal(symbol, "Insufficient data")

        # Add technical indicators
        df = self._add_indicators(df)

        cprint(f"\n{'='*60}", "cyan")
        cprint(f"[TrendRider] Analyzing {symbol}...", "cyan")
        cprint(f"{'='*60}", "cyan")

        # 1. Detect trend
        trend = self._detect_trend(df, symbol)
        if not trend['valid']:
            return self._neutral_signal(
                symbol,
                trend.get('reason', 'No valid trend'),
                market_state=self._get_market_state(df)
            )

        cprint(f"  [TrendRider] {trend['direction']} trend detected (ADX={trend['adx']:.1f})", "green")

        # 2. Detect pullback
        pullback = self._detect_pullback(df, trend['direction'])
        if not pullback['valid']:
            return self._neutral_signal(
                symbol,
                pullback.get('reason', 'No valid pullback'),
                market_state=self._get_market_state(df),
                diagnostic={'trend': trend}
            )

        cprint(f"  [TrendRider] Pullback detected: RSI={pullback['rsi']:.1f}, dist_ema20={pullback['dist_ema20']:.2f}%", "green")

        # 3. Check confirmation
        confirmation = self._check_confirmation(df, trend['direction'])

        # 4. Run checklist
        checklist = self._run_checklist(df, trend, pullback, confirmation)
        pre_ai_score = sum(c['score'] for c in checklist.values() if c.get('score'))

        cprint(f"  [TrendRider] Pre-AI score: {pre_ai_score:.1f}/8.0", "cyan")

        # Skip AI if pre-score too low
        if pre_ai_score < 5.0:
            return self._neutral_signal(
                symbol,
                f"Pre-AI score too low: {pre_ai_score:.1f}/8.0",
                checklist=checklist,
                market_state=self._get_market_state(df)
            )

        # 5. LLM Validation
        ai_result = self._get_llm_evaluation(symbol, df, trend, pullback, confirmation)
        checklist['ai_validation'] = {
            'passed': ai_result.get('confidence', 0) >= TREND_RIDER_AI_MIN_CONFIDENCE,
            'score': TREND_RIDER_WEIGHTS['ai_validation'] if ai_result.get('confidence', 0) >= TREND_RIDER_AI_MIN_CONFIDENCE else 0,
            'confidence': ai_result.get('confidence', 0),
            'trend_quality': ai_result.get('trend_quality', 'N/A'),
            'pullback_type': ai_result.get('pullback_type', 'N/A'),
        }

        final_score = sum(c.get('score', 0) for c in checklist.values())

        cprint(f"  [TrendRider] Final score: {final_score:.1f}/10.0 (AI: {ai_result.get('confidence', 0)}%)", "cyan")

        # 6. Generate signal if score meets threshold
        if final_score >= TREND_RIDER_MIN_SCORE and ai_result.get('decision') == 'EXECUTE':
            direction = 'BUY' if trend['direction'] == 'BULLISH' else 'SELL'
            stops = self._calculate_stops(df, direction)

            self.daily_trades += 1

            cprint(f"  [TrendRider] SIGNAL: {direction} {symbol} @ {stops['entry']:.4f}", "green", attrs=['bold'])
            cprint(f"    SL: {stops['stop_loss']:.4f} ({stops['sl_pct']:.2f}%)", "yellow")
            cprint(f"    TP: {stops['take_profit']:.4f} ({stops['tp_pct']:.2f}%)", "green")

            return {
                'token': symbol,
                'signal': ai_result.get('confidence', 70) / 100.0,
                'direction': direction,
                'metadata': {
                    'strategy_type': 'trend_rider',
                    'strategy_source': 'TrendRider',
                    'setup_type': f"pullback_{trend['direction'].lower()}",
                    'current_price': float(df['close'].iloc[-1]),
                    'weighted_score': final_score,
                    'checklist_score': f"{sum(1 for c in checklist.values() if c.get('passed'))}/5",
                    'checklist_details': checklist,
                    'ai_confidence': ai_result.get('confidence', 0),
                    'ai_reasoning': ai_result.get('reasoning', ''),
                    'risk_factors': ai_result.get('risk_factors', []),
                    'trend_quality': ai_result.get('trend_quality', 'N/A'),
                    'pullback_type': ai_result.get('pullback_type', 'N/A'),
                    'entry_timing': ai_result.get('entry_timing', 'N/A'),
                    'stop_loss': stops['stop_loss'],
                    'stop_loss_pct': stops['sl_pct'],
                    'take_profit': stops['take_profit'],
                    'take_profit_pct': stops['tp_pct'],
                    'leverage': TREND_RIDER_LEVERAGE,
                    'adx': trend['adx'],
                    'rsi': pullback['rsi'],
                    'ema_alignment': trend['ema_status'],
                }
            }
        else:
            reason = f"Score {final_score:.1f}/10 < {TREND_RIDER_MIN_SCORE}" if final_score < TREND_RIDER_MIN_SCORE else "AI rejected"
            return self._neutral_signal(
                symbol,
                reason,
                checklist=checklist,
                market_state=self._get_market_state(df),
                ai_reasoning=ai_result.get('reasoning', '')
            )

    def _fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for symbol"""
        try:
            from src.nice_funcs_hyperliquid import get_data
            df = get_data(symbol=symbol, timeframe='1h', bars=100, add_indicators=False)
            return df
        except Exception as e:
            cprint(f"[TrendRider] Error fetching data for {symbol}: {e}", "red")
            return None

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""
        try:
            from ta.trend import EMAIndicator, ADXIndicator, MACD
            from ta.momentum import RSIIndicator
            from ta.volatility import AverageTrueRange

            # EMAs
            df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()
            df['ema_200'] = EMAIndicator(df['close'], window=200).ema_indicator() if len(df) >= 200 else df['ema_50']

            # ADX
            adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=14)
            df['adx'] = adx_indicator.adx()
            df['adx_pos'] = adx_indicator.adx_pos()
            df['adx_neg'] = adx_indicator.adx_neg()

            # RSI
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi()

            # ATR
            df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

            # MACD
            macd = MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

            # Volume MA
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']

            # Candle body percentage
            df['candle_range'] = df['high'] - df['low']
            df['candle_body'] = abs(df['close'] - df['open'])
            df['body_pct'] = (df['candle_body'] / df['candle_range'] * 100).fillna(0)

            # Price distance from EMAs
            df['dist_ema20'] = ((df['close'] - df['ema_20']) / df['ema_20'] * 100)
            df['dist_ema50'] = ((df['close'] - df['ema_50']) / df['ema_50'] * 100)

        except Exception as e:
            cprint(f"[TrendRider] Error adding indicators: {e}", "red")

        return df

    def _detect_trend(self, df: pd.DataFrame, symbol: str) -> dict:
        """Detect trend via EMA alignment and ADX"""
        try:
            current = df.iloc[-1]
            adx = current['adx']
            ema_20 = current['ema_20']
            ema_50 = current['ema_50']
            ema_200 = current['ema_200']
            price = current['close']

            # Check ADX strength
            if adx < TREND_RIDER_ADX_MIN:
                return {
                    'valid': False,
                    'reason': f"ADX too low ({adx:.1f} < {TREND_RIDER_ADX_MIN})",
                    'adx': adx
                }

            # Check EMA alignment for bullish trend
            bullish_alignment = ema_20 > ema_50 > ema_200 and price > ema_20
            bearish_alignment = ema_20 < ema_50 < ema_200 and price < ema_20

            if bullish_alignment:
                # Count how long trend has been aligned
                trend_bars = self._count_trend_duration(df, 'BULLISH')
                return {
                    'valid': True,
                    'direction': 'BULLISH',
                    'adx': adx,
                    'ema_20': ema_20,
                    'ema_50': ema_50,
                    'ema_200': ema_200,
                    'ema_status': '20 > 50 > 200',
                    'trend_duration': trend_bars,
                    'adx_pos': current['adx_pos'],
                    'adx_neg': current['adx_neg'],
                }
            elif bearish_alignment:
                trend_bars = self._count_trend_duration(df, 'BEARISH')
                return {
                    'valid': True,
                    'direction': 'BEARISH',
                    'adx': adx,
                    'ema_20': ema_20,
                    'ema_50': ema_50,
                    'ema_200': ema_200,
                    'ema_status': '20 < 50 < 200',
                    'trend_duration': trend_bars,
                    'adx_pos': current['adx_pos'],
                    'adx_neg': current['adx_neg'],
                }
            else:
                return {
                    'valid': False,
                    'reason': 'EMAs not aligned (mixed trend)',
                    'adx': adx
                }

        except Exception as e:
            return {'valid': False, 'reason': f'Error: {e}'}

    def _count_trend_duration(self, df: pd.DataFrame, direction: str) -> int:
        """Count how many bars the trend has been aligned"""
        count = 0
        for i in range(len(df) - 1, -1, -1):
            row = df.iloc[i]
            if direction == 'BULLISH':
                if row['ema_20'] > row['ema_50']:
                    count += 1
                else:
                    break
            else:
                if row['ema_20'] < row['ema_50']:
                    count += 1
                else:
                    break
        return count

    def _detect_pullback(self, df: pd.DataFrame, trend_direction: str) -> dict:
        """Detect pullback to EMA with RSI in target zone"""
        try:
            current = df.iloc[-1]
            rsi = current['rsi']
            dist_ema20 = current['dist_ema20']
            dist_ema50 = current['dist_ema50']

            # Define target zones based on trend direction
            if trend_direction == 'BULLISH':
                rsi_min, rsi_max = TREND_RIDER_RSI_PULLBACK_LONG
                # Price should be near EMA20 (within 1%) or touched it recently
                near_ema = dist_ema20 <= 1.0 and dist_ema20 >= -2.0
                touched_ema50 = dist_ema50 <= 1.0 and dist_ema50 >= -2.0
            else:
                rsi_min, rsi_max = TREND_RIDER_RSI_PULLBACK_SHORT
                near_ema = dist_ema20 >= -1.0 and dist_ema20 <= 2.0
                touched_ema50 = dist_ema50 >= -1.0 and dist_ema50 <= 2.0

            # Check RSI in pullback zone
            rsi_in_zone = rsi_min <= rsi <= rsi_max

            # Check volume declining during pullback (last 3-5 bars)
            recent_volume = df['volume_ratio'].iloc[-5:-1].mean()
            volume_declining = recent_volume < TREND_RIDER_VOLUME_DECLINE_RATIO

            if rsi_in_zone and (near_ema or touched_ema50):
                # Calculate pullback depth
                if trend_direction == 'BULLISH':
                    recent_high = df['high'].iloc[-20:].max()
                    pullback_depth = (recent_high - current['close']) / recent_high * 100
                else:
                    recent_low = df['low'].iloc[-20:].min()
                    pullback_depth = (current['close'] - recent_low) / recent_low * 100

                return {
                    'valid': True,
                    'rsi': rsi,
                    'rsi_zone': f"{rsi_min}-{rsi_max}",
                    'dist_ema20': dist_ema20,
                    'dist_ema50': dist_ema50,
                    'near_ema20': near_ema,
                    'touched_ema50': touched_ema50,
                    'volume_declining': volume_declining,
                    'recent_volume_ratio': recent_volume,
                    'pullback_depth': pullback_depth,
                }
            else:
                reasons = []
                if not rsi_in_zone:
                    reasons.append(f"RSI {rsi:.1f} not in zone [{rsi_min}-{rsi_max}]")
                if not near_ema and not touched_ema50:
                    reasons.append(f"Price not near EMA (dist_ema20={dist_ema20:.2f}%)")
                return {
                    'valid': False,
                    'reason': ', '.join(reasons),
                    'rsi': rsi,
                    'dist_ema20': dist_ema20,
                }

        except Exception as e:
            return {'valid': False, 'reason': f'Error: {e}'}

    def _check_confirmation(self, df: pd.DataFrame, trend_direction: str) -> dict:
        """Check for confirmation candle with volume"""
        try:
            current = df.iloc[-1]
            prev = df.iloc[-2]

            # Determine if current candle is confirmation
            if trend_direction == 'BULLISH':
                is_bullish_candle = current['close'] > current['open']
                closes_above_prev = current['close'] > prev['high']
            else:
                is_bullish_candle = current['close'] < current['open']
                closes_above_prev = current['close'] < prev['low']

            body_pct = current['body_pct']
            strong_body = body_pct >= TREND_RIDER_MIN_CANDLE_BODY_PCT

            volume_spike = current['volume_ratio'] >= TREND_RIDER_VOLUME_SPIKE_RATIO

            confirmed = (is_bullish_candle or closes_above_prev) and (strong_body or volume_spike)

            return {
                'confirmed': confirmed,
                'is_trend_candle': is_bullish_candle,
                'closes_beyond_prev': closes_above_prev,
                'body_pct': body_pct,
                'strong_body': strong_body,
                'volume_ratio': current['volume_ratio'],
                'volume_spike': volume_spike,
            }

        except Exception as e:
            return {'confirmed': False, 'error': str(e)}

    def _run_checklist(self, df: pd.DataFrame, trend: dict, pullback: dict, confirmation: dict) -> dict:
        """Run 5-point checklist"""
        checklist = {}

        # 1. Trend Alignment (weight: 2.5)
        checklist['trend_alignment'] = {
            'passed': trend['valid'],
            'score': TREND_RIDER_WEIGHTS['trend_alignment'] if trend['valid'] else 0,
            'adx': trend.get('adx', 0),
            'ema_status': trend.get('ema_status', 'N/A'),
            'duration': trend.get('trend_duration', 0),
        }

        # 2. Pullback Quality (weight: 2.0)
        pullback_quality = pullback['valid'] and pullback.get('volume_declining', False)
        checklist['pullback_quality'] = {
            'passed': pullback_quality,
            'score': TREND_RIDER_WEIGHTS['pullback_quality'] if pullback_quality else 0,
            'rsi': pullback.get('rsi', 0),
            'dist_ema20': pullback.get('dist_ema20', 0),
            'volume_declining': pullback.get('volume_declining', False),
        }

        # 3. Momentum Confirmation (weight: 2.0)
        checklist['momentum_confirmation'] = {
            'passed': confirmation.get('confirmed', False),
            'score': TREND_RIDER_WEIGHTS['momentum_confirmation'] if confirmation.get('confirmed', False) else 0,
            'candle_type': 'trend' if confirmation.get('is_trend_candle') else 'counter',
            'body_pct': confirmation.get('body_pct', 0),
            'volume_ratio': confirmation.get('volume_ratio', 0),
        }

        # 4. Higher TF Agreement (weight: 1.5)
        # Check if MACD is aligned with trend
        current = df.iloc[-1]
        macd_aligned = (trend['direction'] == 'BULLISH' and current['macd'] > current['macd_signal']) or \
                       (trend['direction'] == 'BEARISH' and current['macd'] < current['macd_signal'])

        checklist['htf_agreement'] = {
            'passed': macd_aligned,
            'score': TREND_RIDER_WEIGHTS['htf_agreement'] if macd_aligned else 0,
            'macd': current['macd'],
            'macd_signal': current['macd_signal'],
            'aligned': macd_aligned,
        }

        return checklist

    def _get_llm_evaluation(self, symbol: str, df: pd.DataFrame, trend: dict, pullback: dict, confirmation: dict) -> dict:
        """Get LLM evaluation of the setup"""
        if not self._claude_model:
            return {'decision': 'REJECT', 'confidence': 0, 'reasoning': 'LLM not available'}

        try:
            current = df.iloc[-1]

            # Build user prompt
            user_prompt = f"""Analyze this trend-following pullback setup for {symbol}:

## TREND ANALYSIS
- Direction: {trend['direction']}
- ADX: {trend['adx']:.1f} (strength of trend, >35 = strong)
- EMA Alignment: {trend.get('ema_status', 'N/A')}
  - EMA20: {trend.get('ema_20', 0):.4f}
  - EMA50: {trend.get('ema_50', 0):.4f}
  - EMA200: {trend.get('ema_200', 0):.4f}
  - Current Price: {current['close']:.4f}
- Trend Duration: {trend.get('trend_duration', 0)} bars since alignment
- DI+: {trend.get('adx_pos', 0):.1f}, DI-: {trend.get('adx_neg', 0):.1f}

## PULLBACK ANALYSIS
- RSI: {pullback.get('rsi', 0):.1f} (target zone: {pullback.get('rsi_zone', 'N/A')})
- Distance from EMA20: {pullback.get('dist_ema20', 0):.2f}%
- Distance from EMA50: {pullback.get('dist_ema50', 0):.2f}%
- Pullback Depth: {pullback.get('pullback_depth', 0):.1f}% from recent swing
- Volume During Pullback: {pullback.get('recent_volume_ratio', 0):.2f}x average (want <0.8)

## CONFIRMATION SIGNALS
- Last Candle Type: {'Bullish' if confirmation.get('is_trend_candle') else 'Bearish'}
- Body Size: {confirmation.get('body_pct', 0):.0f}% of range
- Volume: {confirmation.get('volume_ratio', 0):.2f}x average
- Closes Beyond Previous: {'Yes' if confirmation.get('closes_beyond_prev') else 'No'}

## MARKET CONTEXT
- MACD: {current['macd']:.4f} vs Signal: {current['macd_signal']:.4f}
- MACD Histogram: {current['macd_diff']:.4f}
- ATR(14): {current['atr']:.4f}

## PROPOSED TRADE
- Direction: {'LONG' if trend['direction'] == 'BULLISH' else 'SHORT'}
- Entry: {current['close']:.4f}
- Stop Loss: {TREND_RIDER_ATR_SL_MULT} ATR = {current['close'] - (current['atr'] * TREND_RIDER_ATR_SL_MULT):.4f} for long
- Take Profit: {TREND_RIDER_ATR_TP_MULT} ATR = {current['close'] + (current['atr'] * TREND_RIDER_ATR_TP_MULT):.4f} for long
- Risk/Reward: {TREND_RIDER_ATR_TP_MULT / TREND_RIDER_ATR_SL_MULT:.1f}:1

Should we execute this trade?"""

            # Get LLM response
            response = self._claude_model.generate_response(
                system_prompt=TREND_RIDER_SYSTEM_PROMPT,
                user_content=user_prompt,
                temperature=TREND_RIDER_AI_TEMPERATURE,
                max_tokens=1024
            )

            # Parse JSON response
            return self._parse_llm_response(response)

        except Exception as e:
            cprint(f"[TrendRider] LLM error: {e}", "red")
            return {'decision': 'REJECT', 'confidence': 0, 'reasoning': f'LLM error: {e}'}

    def _parse_llm_response(self, response: str) -> dict:
        """Parse JSON response from LLM"""
        try:
            # Try to extract JSON from response
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0].strip()
            else:
                json_str = response.strip()

            result = json.loads(json_str)
            return result

        except json.JSONDecodeError:
            # Try to find JSON object in response
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass

            cprint(f"[TrendRider] Could not parse LLM response", "yellow")
            return {
                'decision': 'REJECT',
                'confidence': 0,
                'reasoning': 'Could not parse LLM response'
            }

    def _calculate_stops(self, df: pd.DataFrame, direction: str) -> dict:
        """Calculate entry, stop loss, and take profit levels"""
        current = df.iloc[-1]
        entry = current['close']
        atr = current['atr']

        if direction == 'BUY':
            stop_loss = entry - (atr * TREND_RIDER_ATR_SL_MULT)
            take_profit = entry + (atr * TREND_RIDER_ATR_TP_MULT)
        else:
            stop_loss = entry + (atr * TREND_RIDER_ATR_SL_MULT)
            take_profit = entry - (atr * TREND_RIDER_ATR_TP_MULT)

        sl_pct = abs(stop_loss - entry) / entry * 100
        tp_pct = abs(take_profit - entry) / entry * 100

        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'sl_pct': sl_pct,
            'tp_pct': tp_pct,
            'atr': atr,
        }

    def _get_market_state(self, df: pd.DataFrame) -> dict:
        """Get current market state for diagnostics"""
        try:
            current = df.iloc[-1]
            return {
                'adx': round(current['adx'], 1),
                'rsi': round(current['rsi'], 1),
                'ema_20': round(current['ema_20'], 4),
                'ema_50': round(current['ema_50'], 4),
                'dist_ema20': round(current['dist_ema20'], 2),
                'volume_ratio': round(current['volume_ratio'], 2),
            }
        except:
            return {}

    def _neutral_signal(self, symbol: str, reason: str, checklist: dict = None,
                        market_state: dict = None, diagnostic: dict = None,
                        ai_reasoning: str = None) -> dict:
        """Return neutral signal with metadata"""
        return {
            'token': symbol,
            'signal': 0.0,
            'direction': 'NEUTRAL',
            'metadata': {
                'strategy_type': 'trend_rider',
                'strategy_source': 'TrendRider',
                'reason': reason,
                'checklist_details': checklist,
                'market_state': market_state,
                'diagnostic': diagnostic,
                'ai_reasoning': ai_reasoning,
            }
        }


# Standalone test
if __name__ == "__main__":
    cprint("\n" + "="*60, "cyan")
    cprint("Testing Trend Rider Strategy", "cyan", attrs=['bold'])
    cprint("="*60, "cyan")

    strategy = TrendRiderStrategy()

    for symbol in ['BTC', 'ETH', 'SOL']:
        cprint(f"\nTesting {symbol}...", "yellow")
        signal = strategy.generate_signals(symbol=symbol)
        cprint(f"Result: {signal['direction']} (confidence: {signal['signal']*100:.0f}%)",
               "green" if signal['direction'] != 'NEUTRAL' else "gray")
        if signal.get('metadata', {}).get('reason'):
            cprint(f"Reason: {signal['metadata']['reason']}", "gray")
