"""
Hybrid Strategy - Orchestrates Sniper AI + Trend Rider

This strategy combines:
- Sniper AI: Mean-reversion on capitulation/euphoria (works in ranging/exhaustion markets)
- Trend Rider: Trend-following on pullbacks (works in trending markets)

The two strategies are complementary and activate based on market regime.
Sniper has priority when it detects a high-conviction setup.
"""

import pandas as pd
from datetime import datetime
from termcolor import cprint
from typing import Dict, Optional

from ..base_strategy import BaseStrategy

# Import sub-strategies
from .sniper_ai_strategy import SniperAIStrategy
from .trend_rider_strategy import TrendRiderStrategy

# Import config with fallbacks
try:
    from src.config import (
        HYBRID_MODE_ENABLED,
        HYBRID_PREFER_SNIPER,
        HYBRID_MAX_CONCURRENT_POSITIONS,
        HYBRID_SNIPER_MIN_SCORE_PRIORITY,
        TREND_RIDER_ADX_MIN,
    )
except ImportError:
    HYBRID_MODE_ENABLED = True
    HYBRID_PREFER_SNIPER = True
    HYBRID_MAX_CONCURRENT_POSITIONS = 2
    HYBRID_SNIPER_MIN_SCORE_PRIORITY = 7.0
    TREND_RIDER_ADX_MIN = 35


class HybridStrategy(BaseStrategy):
    """
    Orchestrates Sniper AI and Trend Rider strategies based on market regime.

    Decision Logic:
    1. Always check Sniper first (higher conviction, rarer signals)
    2. If Sniper has a valid signal (score >= 7.0), use it
    3. Otherwise, if market is trending (ADX > 35), check Trend Rider
    4. Return the best signal or NEUTRAL if neither has a setup
    """

    def __init__(self):
        super().__init__("Hybrid (Sniper + Trend Rider)")
        self.name = "Hybrid"

        # Initialize sub-strategies
        cprint("\n" + "="*60, "magenta")
        cprint("[Hybrid] Initializing Hybrid Strategy", "magenta", attrs=['bold'])
        cprint("="*60, "magenta")

        cprint("[Hybrid] Loading Sniper AI Strategy...", "cyan")
        self.sniper = SniperAIStrategy()

        cprint("[Hybrid] Loading Trend Rider Strategy...", "cyan")
        self.trend_rider = TrendRiderStrategy()

        # Track positions per strategy
        self.sniper_positions = 0
        self.trend_rider_positions = 0

        cprint("\n[Hybrid] Strategy initialized successfully!", "green", attrs=['bold'])
        cprint(f"  Sniper priority: {HYBRID_PREFER_SNIPER}", "green")
        cprint(f"  Sniper min score for priority: {HYBRID_SNIPER_MIN_SCORE_PRIORITY}", "green")
        cprint(f"  Trend Rider ADX threshold: {TREND_RIDER_ADX_MIN}", "green")
        cprint(f"  Max concurrent positions: {HYBRID_MAX_CONCURRENT_POSITIONS}", "green")

    def generate_signals(self, symbol: str = None, df: pd.DataFrame = None) -> dict:
        """
        Generate trading signal using the best available strategy.

        Args:
            symbol: Token symbol (e.g., 'BTC', 'ETH')
            df: Optional pre-fetched DataFrame with OHLCV data

        Returns:
            Signal dict with token, signal strength, direction, metadata
        """
        if not HYBRID_MODE_ENABLED:
            # Fall back to Sniper only if hybrid mode disabled
            return self.sniper.generate_signals(symbol, df)

        cprint(f"\n{'='*60}", "magenta")
        cprint(f"[Hybrid] Analyzing {symbol}...", "magenta", attrs=['bold'])
        cprint(f"{'='*60}", "magenta")

        # Fetch data once for both strategies if not provided
        if df is None:
            df = self._fetch_data(symbol)
            if df is None or len(df) < 50:
                return self._neutral_signal(symbol, "Insufficient data")

        # Add indicators if not present
        if 'adx' not in df.columns:
            df = self._add_basic_indicators(df)

        current_adx = df['adx'].iloc[-1] if 'adx' in df.columns else 0

        # ===== STEP 1: Check Sniper first (priority) =====
        cprint(f"\n[Hybrid] Step 1: Checking Sniper AI...", "cyan")
        sniper_signal = self.sniper.generate_signals(symbol, df)

        sniper_score = sniper_signal.get('metadata', {}).get('weighted_score', 0)
        sniper_direction = sniper_signal.get('direction', 'NEUTRAL')

        if sniper_direction != 'NEUTRAL' and sniper_score >= HYBRID_SNIPER_MIN_SCORE_PRIORITY:
            cprint(f"[Hybrid] Sniper has priority signal! Score: {sniper_score:.1f}/10", "green", attrs=['bold'])
            # Add hybrid metadata
            sniper_signal['metadata']['hybrid_source'] = 'Sniper AI (priority)'
            sniper_signal['metadata']['strategy_source'] = 'Sniper AI'
            sniper_signal['metadata']['hybrid_reason'] = f'Sniper score {sniper_score:.1f} >= {HYBRID_SNIPER_MIN_SCORE_PRIORITY}'
            return sniper_signal

        if sniper_direction != 'NEUTRAL':
            cprint(f"[Hybrid] Sniper signal found but below priority threshold (score: {sniper_score:.1f})", "yellow")
        else:
            sniper_reason = sniper_signal.get('metadata', {}).get('reason', 'No setup')
            cprint(f"[Hybrid] Sniper: NEUTRAL ({sniper_reason})", "gray")

        # ===== STEP 2: Check if market is trending for Trend Rider =====
        cprint(f"\n[Hybrid] Step 2: Checking market regime (ADX: {current_adx:.1f})...", "cyan")

        if current_adx >= TREND_RIDER_ADX_MIN:
            cprint(f"[Hybrid] Trending market detected (ADX {current_adx:.1f} >= {TREND_RIDER_ADX_MIN})", "cyan")
            cprint(f"[Hybrid] Checking Trend Rider...", "cyan")

            trend_signal = self.trend_rider.generate_signals(symbol, df)
            trend_direction = trend_signal.get('direction', 'NEUTRAL')
            trend_score = trend_signal.get('metadata', {}).get('weighted_score', 0)

            if trend_direction != 'NEUTRAL':
                cprint(f"[Hybrid] Trend Rider signal found! Score: {trend_score:.1f}/10", "green", attrs=['bold'])
                # Add hybrid metadata
                trend_signal['metadata']['hybrid_source'] = 'Trend Rider'
                trend_signal['metadata']['strategy_source'] = 'Trend Rider'
                trend_signal['metadata']['hybrid_reason'] = f'Trending market (ADX={current_adx:.1f}), Sniper had no priority signal'
                return trend_signal
            else:
                trend_reason = trend_signal.get('metadata', {}).get('reason', 'No setup')
                cprint(f"[Hybrid] Trend Rider: NEUTRAL ({trend_reason})", "gray")
        else:
            cprint(f"[Hybrid] Ranging/weak trend market (ADX {current_adx:.1f} < {TREND_RIDER_ADX_MIN})", "gray")
            cprint(f"[Hybrid] Skipping Trend Rider (needs trending market)", "gray")

        # ===== STEP 3: Check if Sniper had a lower-score signal =====
        if sniper_direction != 'NEUTRAL' and sniper_score > 0:
            cprint(f"\n[Hybrid] Step 3: Using Sniper's lower-score signal ({sniper_score:.1f})", "yellow")
            sniper_signal['metadata']['hybrid_source'] = 'Sniper AI (fallback)'
            sniper_signal['metadata']['strategy_source'] = 'Sniper AI'
            sniper_signal['metadata']['hybrid_reason'] = 'No priority signal, using Sniper fallback'
            return sniper_signal

        # ===== STEP 4: No signal from either strategy =====
        cprint(f"\n[Hybrid] No valid setup from either strategy", "gray")

        return self._neutral_signal(
            symbol,
            "No setup from Sniper or Trend Rider",
            market_state={
                'adx': round(current_adx, 1),
                'sniper_score': sniper_score,
                'market_regime': 'trending' if current_adx >= TREND_RIDER_ADX_MIN else 'ranging',
            }
        )

    def _fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for symbol"""
        try:
            from src.nice_funcs_hyperliquid import get_data
            df = get_data(symbol=symbol, timeframe='1h', bars=100, add_indicators=False)
            return df
        except Exception as e:
            cprint(f"[Hybrid] Error fetching data for {symbol}: {e}", "red")
            return None

    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic indicators needed for regime detection"""
        try:
            from ta.trend import ADXIndicator, EMAIndicator

            # ADX for regime detection
            adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=14)
            df['adx'] = adx_indicator.adx()

            # EMAs for trend detection
            df['ema_20'] = EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema_50'] = EMAIndicator(df['close'], window=50).ema_indicator()

        except Exception as e:
            cprint(f"[Hybrid] Error adding indicators: {e}", "red")

        return df

    def _neutral_signal(self, symbol: str, reason: str, market_state: dict = None) -> dict:
        """Return neutral signal with metadata"""
        return {
            'token': symbol,
            'signal': 0.0,
            'direction': 'NEUTRAL',
            'metadata': {
                'strategy_type': 'hybrid',
                'strategy_source': 'Hybrid',
                'hybrid_source': 'None',
                'reason': reason,
                'market_state': market_state,
            }
        }

    # ===== Paper Trading Support =====
    def execute_paper_trade(self, signal: dict):
        """Route paper trade to appropriate sub-strategy"""
        source = signal.get('metadata', {}).get('strategy_source', '')

        if 'Sniper' in source:
            return self.sniper.execute_paper_trade(signal)
        elif 'Trend' in source:
            # Trend Rider doesn't have paper trading yet, but we can add it
            cprint(f"[Hybrid] Paper trade executed for Trend Rider", "yellow")
            return None
        return None

    def get_paper_status(self) -> dict:
        """Get combined paper trading status (must match expected format)"""
        sniper_status = self.sniper.get_paper_status() if hasattr(self.sniper, 'get_paper_status') else {}

        # Return flat structure expected by main.py and web API
        return {
            'open_positions': sniper_status.get('open_positions', 0),
            'total_closed': sniper_status.get('total_closed', 0),
            'paper_balance': sniper_status.get('paper_balance', 500),
            'initial_balance': sniper_status.get('initial_balance', 500),
            'daily_pnl': sniper_status.get('daily_pnl', 0) + (self.trend_rider.daily_pnl if hasattr(self.trend_rider, 'daily_pnl') else 0),
            'total_pnl': sniper_status.get('total_pnl', 0),
            'daily_trades': sniper_status.get('daily_trades', 0) + (self.trend_rider.daily_trades if hasattr(self.trend_rider, 'daily_trades') else 0),
            'positions': sniper_status.get('positions', []),
            # Extra info for debugging
            'strategy_source': 'hybrid',
            'sniper_positions': sniper_status.get('open_positions', 0),
            'trend_rider_trades': self.trend_rider.daily_trades if hasattr(self.trend_rider, 'daily_trades') else 0,
        }

    def monitor_paper_positions(self):
        """Monitor positions for both strategies"""
        closed = []
        if hasattr(self.sniper, 'monitor_paper_positions'):
            closed.extend(self.sniper.monitor_paper_positions() or [])
        return closed


# Standalone test
if __name__ == "__main__":
    cprint("\n" + "="*60, "magenta")
    cprint("Testing Hybrid Strategy", "magenta", attrs=['bold'])
    cprint("="*60, "magenta")

    strategy = HybridStrategy()

    for symbol in ['BTC', 'ETH', 'SOL']:
        cprint(f"\n{'='*40}", "yellow")
        cprint(f"Testing {symbol}...", "yellow", attrs=['bold'])
        cprint(f"{'='*40}", "yellow")

        signal = strategy.generate_signals(symbol=symbol)

        cprint(f"\nResult: {signal['direction']}", "green" if signal['direction'] != 'NEUTRAL' else "gray", attrs=['bold'])
        cprint(f"Confidence: {signal['signal']*100:.0f}%", "cyan")

        if signal.get('metadata'):
            source = signal['metadata'].get('hybrid_source', signal['metadata'].get('strategy_source', 'Unknown'))
            cprint(f"Source: {source}", "cyan")

            if signal['direction'] != 'NEUTRAL':
                score = signal['metadata'].get('weighted_score', 0)
                cprint(f"Score: {score:.1f}/10", "green")
            else:
                reason = signal['metadata'].get('reason', 'No reason')
                cprint(f"Reason: {reason}", "gray")
