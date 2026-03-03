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
)
from src.agents.base_agent import BaseAgent

load_dotenv()


class RiskAgent(BaseAgent):
    def __init__(self):
        """Initialize Risk Agent for paper trading mode."""
        super().__init__('risk')

        self.initial_balance = PAPER_TRADING_BALANCE
        self.trading_paused = False
        self.pause_reason = None
        self.daily_pause = False
        self.daily_pause_date = None

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

    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed (used by strategy agent before opening new positions)."""
        # Reset daily pause if new day
        today = datetime.now().date()
        if self.daily_pause and self.daily_pause_date != today:
            self.daily_pause = False
            self.daily_pause_date = None
            cprint("[RiskAgent] Daily pause reset (new day)", "green")

        if self.trading_paused:
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
        pnl_pct = (total_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0

        cprint(f"\n[RiskAgent] Balance: ${balance:,.2f} | PnL: ${total_pnl:+,.2f} ({pnl_pct:+.1f}%) | "
               f"Daily: ${daily_pnl:+,.2f} | Positions: {open_positions}/{RISK_MAX_POSITIONS}", "cyan")

        # Reset daily pause if new day
        today = datetime.now().date()
        if self.daily_pause and self.daily_pause_date != today:
            self.daily_pause = False
            self.daily_pause_date = None

        breach = False

        # Recovery check: resume trading if drawdown has recovered
        max_drawdown_usd = self.initial_balance * (RISK_MAX_DRAWDOWN_PCT / 100)
        recovery_threshold = max_drawdown_usd * 0.66  # Resume when drawdown < 2/3 of limit
        if self.trading_paused and total_pnl > -recovery_threshold:
            cprint(f"[RiskAgent] Drawdown recovered (PnL: ${total_pnl:+,.2f}), resuming trading", "green", attrs=['bold'])
            self.trading_paused = False
            self.pause_reason = None

        # Circuit breaker 1: Max drawdown
        if total_pnl <= -max_drawdown_usd:
            cprint(f"[RiskAgent] MAX DRAWDOWN BREACHED: ${total_pnl:+,.2f} "
                   f"(limit: -${max_drawdown_usd:,.2f} / -{RISK_MAX_DRAWDOWN_PCT}%)", "red", attrs=['bold'])
            self.trading_paused = True
            self.pause_reason = f"Max drawdown {RISK_MAX_DRAWDOWN_PCT}% reached"
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
