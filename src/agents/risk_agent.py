"""
Moon Dev's Risk Management Agent
Refactored for HyperLiquid paper trading with circuit breakers.
"""

import os
import time
from datetime import datetime, timedelta
from termcolor import cprint
from dotenv import load_dotenv
from src.config import (
    PAPER_TRADING,
    PAPER_TRADING_BALANCE,
    RISK_MAX_DRAWDOWN_PCT,
    RISK_MAX_DAILY_LOSS_USD,
    RISK_MAX_POSITIONS,
    ACTIVE_STRATEGY,
    RISK_COOLING_OFF_HOURS,
    RISK_RECOVERY_SIZE_PCT,
    RISK_RECOVERY_DURATION_HOURS,
)
from src.agents.base_agent import BaseAgent

load_dotenv()


class RiskAgent(BaseAgent):
    def __init__(self):
        """Initialize Risk Agent for paper trading mode."""
        super().__init__('risk')

        self.initial_balance = PAPER_TRADING_BALANCE
        self.peak_balance = PAPER_TRADING_BALANCE  # High Water Mark tracking
        self.trading_paused = False
        self.pause_reason = None
        self.daily_pause = False
        self.daily_pause_date = None

        # Cooling-off and recovery state
        self.pause_timestamp = None
        self.recovery_mode = False
        self.recovery_start = None

        # Strategy reference (set externally by main.py)
        self._strategy = None

        cprint("[RiskAgent] Initialized", "cyan")
        cprint(f"  Max drawdown: {RISK_MAX_DRAWDOWN_PCT}%", "white")
        cprint(f"  Max daily loss: ${RISK_MAX_DAILY_LOSS_USD}", "white")
        cprint(f"  Max positions: {RISK_MAX_POSITIONS}", "white")

    def set_strategy(self, strategy):
        """Set reference to the active strategy (called from main.py)."""
        self._strategy = strategy
        cprint(f"[RiskAgent] Strategy linked: {type(strategy).__name__}", "cyan")

    def _get_strategy(self):
        """Get the strategy instance, trying singleton fallback."""
        if self._strategy:
            return self._strategy

        # Fallback: try to get the singleton instance
        try:
            from src.strategies.custom.adaptive_hybrid_strategy import AdaptiveHybridStrategy
            if AdaptiveHybridStrategy._instance:
                self._strategy = AdaptiveHybridStrategy._instance
                return self._strategy
        except Exception:
            pass

        return None

    def get_portfolio_value(self):
        """Get portfolio value from paper trading state."""
        strategy = self._get_strategy()
        if strategy and hasattr(strategy, 'get_paper_status'):
            status = strategy.get_paper_status()
            return status.get('paper_balance', self.initial_balance)
        return self.initial_balance

    def get_current_pnl(self):
        """Calculate current PnL from paper balance."""
        return self.get_portfolio_value() - self.initial_balance

    def get_daily_pnl(self):
        """Get today's PnL from strategy."""
        strategy = self._get_strategy()
        if strategy and hasattr(strategy, 'get_paper_status'):
            status = strategy.get_paper_status()
            return status.get('daily_pnl', 0.0)
        return 0.0

    def get_open_position_count(self):
        """Get number of open positions."""
        strategy = self._get_strategy()
        if strategy and hasattr(strategy, 'get_paper_status'):
            status = strategy.get_paper_status()
            return status.get('open_positions', 0)
        return 0

    def close_all_positions(self):
        """Close all open paper positions."""
        strategy = self._get_strategy()
        if strategy and hasattr(strategy, 'close_all_paper_positions'):
            closed = strategy.close_all_paper_positions()
            if closed:
                cprint(f"[RiskAgent] Force-closed {len(closed)} positions", "red", attrs=['bold'])
            return closed
        return []

    def get_recovery_size_factor(self) -> float:
        """Returns position size multiplier. 1.0 = full size, <1.0 = reduced (recovery mode)."""
        if self.recovery_mode and self.recovery_start:
            elapsed = (datetime.now() - self.recovery_start).total_seconds() / 3600
            if elapsed < RISK_RECOVERY_DURATION_HOURS:
                factor = RISK_RECOVERY_SIZE_PCT / 100.0
                cprint(f"[RiskAgent] Recovery mode: {RISK_RECOVERY_SIZE_PCT}% size "
                       f"({elapsed:.1f}h / {RISK_RECOVERY_DURATION_HOURS}h)", "yellow")
                return factor
            else:
                self.recovery_mode = False
                self.recovery_start = None
                cprint("[RiskAgent] Recovery period ended, full size restored", "green")
        return 1.0

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed (used by strategy agent before opening new positions)."""
        # Reset daily pause if new day
        today = datetime.now().date()
        if self.daily_pause and self.daily_pause_date != today:
            self.daily_pause = False
            self.daily_pause_date = None
            cprint("[RiskAgent] Daily pause reset (new day)", "green")

        if self.trading_paused:
            # Check cooling-off period before allowing resume
            if self.pause_timestamp:
                elapsed = (datetime.now() - self.pause_timestamp).total_seconds() / 3600
                if elapsed < RISK_COOLING_OFF_HOURS:
                    cprint(f"[RiskAgent] Cooling-off: {elapsed:.1f}h / {RISK_COOLING_OFF_HOURS}h", "yellow")
            return False
        if self.daily_pause:
            return False

        # Check max positions
        open_count = self.get_open_position_count()
        if open_count >= RISK_MAX_POSITIONS:
            cprint(f"[RiskAgent] Max positions reached ({open_count}/{RISK_MAX_POSITIONS})", "yellow")
            return False

        return True

    def run(self):
        """Run risk checks. Returns True if a limit was breached."""
        strategy = self._get_strategy()
        if not strategy:
            cprint("[RiskAgent] No strategy linked, skipping checks", "yellow")
            return False

        status = strategy.get_paper_status() if hasattr(strategy, 'get_paper_status') else {}

        balance = status.get('paper_balance', self.initial_balance)
        total_pnl = balance - self.initial_balance
        daily_pnl = status.get('daily_pnl', 0.0)
        open_positions = status.get('open_positions', 0)

        # High Water Mark tracking
        current_balance = balance
        self.peak_balance = max(self.peak_balance, current_balance)
        hwm_drawdown_pct = (self.peak_balance - current_balance) / self.peak_balance * 100 if self.peak_balance > 0 else 0
        pnl_pct = (total_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0

        cprint(f"\n[RiskAgent] Balance: ${balance:,.2f} | PnL: ${total_pnl:+,.2f} ({pnl_pct:+.1f}%) | "
               f"Daily: ${daily_pnl:+,.2f} | Positions: {open_positions}/{RISK_MAX_POSITIONS}", "cyan")
        cprint(f"[RiskAgent] HWM: ${self.peak_balance:,.2f} | HWM Drawdown: {hwm_drawdown_pct:.1f}%"
               f"{' | RECOVERY MODE' if self.recovery_mode else ''}", "cyan")

        # Reset daily pause if new day
        today = datetime.now().date()
        if self.daily_pause and self.daily_pause_date != today:
            self.daily_pause = False
            self.daily_pause_date = None

        breach = False

        # Recovery check: resume trading if HWM drawdown has recovered + cooling-off elapsed
        if self.trading_paused and self.pause_timestamp:
            cooling_elapsed = (datetime.now() - self.pause_timestamp).total_seconds() / 3600
            recovery_threshold_pct = RISK_MAX_DRAWDOWN_PCT * 0.66  # Resume when drawdown < 2/3 of limit
            if hwm_drawdown_pct < recovery_threshold_pct and cooling_elapsed >= RISK_COOLING_OFF_HOURS:
                cprint(f"[RiskAgent] Drawdown recovered (HWM DD: {hwm_drawdown_pct:.1f}%) "
                       f"and cooling-off elapsed ({cooling_elapsed:.1f}h), resuming trading", "green", attrs=['bold'])
                self.trading_paused = False
                self.pause_reason = None
                self.pause_timestamp = None
                # Enter recovery mode with reduced size
                self.recovery_mode = True
                self.recovery_start = datetime.now()
                cprint(f"[RiskAgent] Entering recovery mode: {RISK_RECOVERY_SIZE_PCT}% size "
                       f"for {RISK_RECOVERY_DURATION_HOURS}h", "yellow")

        # Circuit breaker 1: Max drawdown (using HWM-based drawdown)
        if hwm_drawdown_pct >= RISK_MAX_DRAWDOWN_PCT:
            cprint(f"[RiskAgent] MAX HWM DRAWDOWN BREACHED: {hwm_drawdown_pct:.1f}% "
                   f"(limit: {RISK_MAX_DRAWDOWN_PCT}% from peak ${self.peak_balance:,.2f})", "red", attrs=['bold'])
            if not self.trading_paused:
                self.trading_paused = True
                self.pause_timestamp = datetime.now()
            self.pause_reason = f"HWM drawdown {RISK_MAX_DRAWDOWN_PCT}% reached"
            self.close_all_positions()
            breach = True

        # Circuit breaker 2: Daily loss limit
        if daily_pnl <= -RISK_MAX_DAILY_LOSS_USD:
            cprint(f"[RiskAgent] DAILY LOSS LIMIT BREACHED: ${daily_pnl:+,.2f} "
                   f"(limit: -${RISK_MAX_DAILY_LOSS_USD:,.2f})", "red", attrs=['bold'])
            self.daily_pause = True
            self.daily_pause_date = today
            self.pause_reason = f"Daily loss limit ${RISK_MAX_DAILY_LOSS_USD} reached"
            self.close_all_positions()
            breach = True

        # Circuit breaker 3: Max positions (informational, enforcement is in is_trading_allowed)
        if open_positions >= RISK_MAX_POSITIONS:
            cprint(f"[RiskAgent] Max positions ({RISK_MAX_POSITIONS}) reached — no new trades allowed", "yellow")

        if not breach:
            cprint("[RiskAgent] All risk checks OK", "green")

        return breach


def main():
    """Main function to run the risk agent standalone."""
    cprint("[RiskAgent] Starting standalone mode...", "cyan")

    agent = RiskAgent()

    while True:
        try:
            agent.run()
            time.sleep(300)
        except KeyboardInterrupt:
            print("\n[RiskAgent] Shutting down...")
            break
        except Exception as e:
            print(f"[RiskAgent] Error: {e}")
            time.sleep(300)


if __name__ == "__main__":
    main()
