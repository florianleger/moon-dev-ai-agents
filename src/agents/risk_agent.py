"""
Moon Dev's Risk Management Agent
Refactored for HyperLiquid paper trading with circuit breakers.
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
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
    MINIMUM_BALANCE_USD,
    CORRELATION_HIGH_THRESHOLD,
    CORRELATION_SIZING_FACTOR,
)
from src.agents.base_agent import BaseAgent

load_dotenv()

RISK_STATE_FILE = Path("src/data/risk_agent_state.json")


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

        # Independent daily PnL tracking
        self._daily_start_balance = None
        self._daily_date = None

        # Strategy reference (set externally by main.py)
        self._strategy = None

        # Why new entries are currently blocked (set by allows_new_entries)
        self.entries_blocked_reason = None

        # Load persisted state
        self._load_state()

        cprint("[RiskAgent] Initialized", "cyan")
        cprint(f"  Max drawdown: {RISK_MAX_DRAWDOWN_PCT}%", "white")
        cprint(f"  Max daily loss: ${RISK_MAX_DAILY_LOSS_USD}", "white")
        cprint(f"  Max positions: {RISK_MAX_POSITIONS}", "white")
        cprint(f"  Minimum balance: ${MINIMUM_BALANCE_USD}", "white")

    def _save_state(self):
        """Persist risk agent state to disk."""
        state = {
            'peak_balance': self.peak_balance,
            'trading_paused': self.trading_paused,
            'pause_reason': self.pause_reason,
            'pause_timestamp': self.pause_timestamp.isoformat() if self.pause_timestamp else None,
            'recovery_mode': self.recovery_mode,
            'recovery_start': self.recovery_start.isoformat() if self.recovery_start else None,
            'daily_pause': self.daily_pause,
            'daily_pause_date': str(self.daily_pause_date) if self.daily_pause_date else None,
            'daily_start_balance': self._daily_start_balance,
            'daily_date': str(self._daily_date) if self._daily_date else None,
        }
        RISK_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        RISK_STATE_FILE.write_text(json.dumps(state, indent=2))

    def _load_state(self):
        """Load persisted risk agent state from disk."""
        if not RISK_STATE_FILE.exists():
            return
        try:
            state = json.loads(RISK_STATE_FILE.read_text())
            self.peak_balance = state.get('peak_balance', self.peak_balance)
            self.trading_paused = state.get('trading_paused', False)
            self.pause_reason = state.get('pause_reason', None)
            ts = state.get('pause_timestamp')
            self.pause_timestamp = datetime.fromisoformat(ts) if ts else None
            self.recovery_mode = state.get('recovery_mode', False)
            rs = state.get('recovery_start')
            self.recovery_start = datetime.fromisoformat(rs) if rs else None
            self.daily_pause = state.get('daily_pause', False)
            dp = state.get('daily_pause_date')
            self.daily_pause_date = datetime.strptime(dp, '%Y-%m-%d').date() if dp else None
            self._daily_start_balance = state.get('daily_start_balance')
            dd = state.get('daily_date')
            self._daily_date = datetime.strptime(dd, '%Y-%m-%d').date() if dd else None
            cprint(f"[RiskAgent] Restored state (peak=${self.peak_balance:,.2f}, paused={self.trading_paused})", "cyan")
        except Exception as e:
            cprint(f"[RiskAgent] Could not load state: {e}", "yellow")

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

    def _get_current_balance(self):
        """Get current balance from strategy or fallback."""
        strategy = self._get_strategy()
        if strategy and hasattr(strategy, 'get_paper_status'):
            status = strategy.get_paper_status()
            return status.get('paper_balance', self.initial_balance)
        return self.initial_balance

    def get_portfolio_value(self):
        """Get portfolio value from paper trading state."""
        return self._get_current_balance()

    def get_current_pnl(self):
        """Calculate current PnL from paper balance."""
        return self.get_portfolio_value() - self.initial_balance

    def get_daily_pnl(self):
        """Get today's PnL independently tracked by the risk agent."""
        today = datetime.now().date()
        if self._daily_date != today:
            self._daily_date = today
            self._daily_start_balance = self._get_current_balance()
            self._save_state()
        current = self._get_current_balance()
        if self._daily_start_balance is None:
            self._daily_start_balance = current
            self._save_state()
        return current - self._daily_start_balance

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
        if not strategy or not hasattr(strategy, 'close_all_paper_positions'):
            cprint("[RiskAgent] CRITICAL: Cannot close positions - strategy not linked!", "red")
            return []
        closed = strategy.close_all_paper_positions()
        if closed:
            cprint(f"[RiskAgent] Force-closed {len(closed)} positions", "red", attrs=['bold'])
        return closed

    def get_drawdown_scaling_factor(self) -> float:
        """Returns position sizing factor based on current drawdown from HWM.

        Linearly scales from 1.0 (no drawdown) to 0.25 (at max drawdown threshold).
        This is an anti-martingale approach: reduce exposure as losses mount.
        """
        current_balance = self._get_current_balance()

        # Update HWM
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
            self._save_state()

        if self.peak_balance <= 0:
            return 1.0

        drawdown_pct = (self.peak_balance - current_balance) / self.peak_balance * 100

        if drawdown_pct <= 0:
            return 1.0

        # Linear scaling: 0% DD -> 1.0, RISK_MAX_DRAWDOWN_PCT -> 0.25
        scaling = max(0.25, 1.0 - (drawdown_pct / RISK_MAX_DRAWDOWN_PCT) * 0.75)

        if scaling < 1.0:
            cprint(f"[RiskAgent] Drawdown-adjusted sizing: {scaling:.0%} (DD={drawdown_pct:.1f}%)", "yellow")

        return scaling

    def get_recovery_size_factor(self) -> float:
        """Returns position size multiplier combining recovery mode AND drawdown scaling."""
        recovery_factor = 1.0
        if self.recovery_mode and self.recovery_start:
            elapsed = (datetime.now() - self.recovery_start).total_seconds() / 3600
            if elapsed < RISK_RECOVERY_DURATION_HOURS:
                recovery_factor = RISK_RECOVERY_SIZE_PCT / 100.0
                cprint(f"[RiskAgent] Recovery mode: {RISK_RECOVERY_SIZE_PCT}% size "
                       f"({elapsed:.1f}h / {RISK_RECOVERY_DURATION_HOURS}h)", "yellow")
            else:
                self.recovery_mode = False
                self.recovery_start = None
                self._save_state()
                cprint("[RiskAgent] Recovery period ended, full size restored", "green")

        drawdown_factor = self.get_drawdown_scaling_factor()
        return min(recovery_factor, drawdown_factor)

    def is_trading_allowed(self) -> bool:
        """Hard circuit breakers only: drawdown/balance pause and daily loss pause.

        Max positions does NOT block this anymore — a full book is a normal
        state that must only prevent NEW entries (see allows_new_entries()),
        never the scan loop or position monitoring/exits.
        """
        # Reset daily pause if new day
        today = datetime.now().date()
        if self.daily_pause and self.daily_pause_date != today:
            self.daily_pause = False
            self.daily_pause_date = None
            self._save_state()
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

        return True

    def allows_new_entries(self) -> bool:
        """Check if NEW positions may be opened (circuit breakers + max positions).

        Sets self.entries_blocked_reason for logging by the caller.
        """
        if not self.is_trading_allowed():
            self.entries_blocked_reason = self.pause_reason or "risk circuit breaker active"
            return False

        open_count = self.get_open_position_count()
        if open_count >= RISK_MAX_POSITIONS:
            self.entries_blocked_reason = f"max positions reached ({open_count}/{RISK_MAX_POSITIONS})"
            return False

        self.entries_blocked_reason = None
        return True

    # Static correlation table for major crypto pairs
    _CORRELATION_TABLE = {
        ('BTC', 'ETH'): 0.85, ('ETH', 'BTC'): 0.85,
        ('BTC', 'SOL'): 0.75, ('SOL', 'BTC'): 0.75,
        ('ETH', 'SOL'): 0.80, ('SOL', 'ETH'): 0.80,
        ('BTC', 'XRP'): 0.70, ('XRP', 'BTC'): 0.70,
        ('ETH', 'XRP'): 0.65, ('XRP', 'ETH'): 0.65,
        ('BTC', 'ADA'): 0.70, ('ADA', 'BTC'): 0.70,
        ('BTC', 'AVAX'): 0.70, ('AVAX', 'BTC'): 0.70,
        ('BTC', 'LINK'): 0.70, ('LINK', 'BTC'): 0.70,
        ('ETH', 'AVAX'): 0.75, ('AVAX', 'ETH'): 0.75,
        ('ETH', 'LINK'): 0.75, ('LINK', 'ETH'): 0.75,
    }
    _DEFAULT_MAJOR_ALT_CORR = 0.60

    def _get_pair_correlation(self, sym_a: str, sym_b: str) -> float:
        """Get static correlation between two symbols."""
        if sym_a == sym_b:
            return 1.0
        return self._CORRELATION_TABLE.get((sym_a, sym_b), self._DEFAULT_MAJOR_ALT_CORR)

    def check_portfolio_correlation(self) -> float:
        """Check correlation across open positions.
        Returns a sizing factor: 1.0 = OK, <1.0 = high correlation detected.
        """
        strategy = self._get_strategy()
        if not strategy or not hasattr(strategy, 'paper_positions'):
            return 1.0

        positions = list(strategy.paper_positions.values())
        if len(positions) <= 1:
            return 1.0

        symbols = [p['symbol'] for p in positions]
        max_corr = 0.0
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = self._get_pair_correlation(symbols[i], symbols[j])
                max_corr = max(max_corr, corr)

        if max_corr >= CORRELATION_HIGH_THRESHOLD:
            cprint(f"[RiskAgent] High portfolio correlation detected: {max_corr:.2f} "
                   f"(threshold: {CORRELATION_HIGH_THRESHOLD})", "yellow")
            return CORRELATION_SIZING_FACTOR

        return 1.0

    def get_correlation_sizing_factor(self, new_symbol: str, positions: list) -> float:
        """Get sizing factor for a new position given existing positions.
        Args:
            new_symbol: Symbol of the new position to open.
            positions: List of dicts with at least 'symbol' key.
        Returns:
            float: 1.0 = OK, CORRELATION_SIZING_FACTOR if too correlated.
        """
        if not positions:
            return 1.0

        for pos in positions:
            corr = self._get_pair_correlation(new_symbol, pos['symbol'])
            if corr >= CORRELATION_HIGH_THRESHOLD:
                cprint(f"[RiskAgent] New {new_symbol} correlated with {pos['symbol']}: {corr:.2f}", "yellow")
                return CORRELATION_SIZING_FACTOR

        return 1.0

    def run(self):
        """Run risk checks. Returns True if a limit was breached."""
        strategy = self._get_strategy()
        if not strategy:
            cprint("[RiskAgent] No strategy linked, skipping checks", "yellow")
            return False

        status = strategy.get_paper_status() if hasattr(strategy, 'get_paper_status') else {}

        balance = status.get('paper_balance', self.initial_balance)
        total_pnl = balance - self.initial_balance
        daily_pnl = self.get_daily_pnl()
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

        # Circuit breaker 0: Minimum balance check
        if current_balance <= MINIMUM_BALANCE_USD:
            cprint(f"[RiskAgent] MINIMUM BALANCE BREACHED: ${current_balance:,.2f} "
                   f"(minimum: ${MINIMUM_BALANCE_USD:,.2f})", "red", attrs=['bold'])
            self.trading_paused = True
            self.pause_reason = f"Balance ${current_balance:.2f} below minimum ${MINIMUM_BALANCE_USD}"
            self.pause_timestamp = datetime.now()
            self.close_all_positions()
            self._save_state()
            return True

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
                self._save_state()
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
            self._save_state()
            breach = True

        # Circuit breaker 2: Daily loss limit
        if daily_pnl <= -RISK_MAX_DAILY_LOSS_USD:
            cprint(f"[RiskAgent] DAILY LOSS LIMIT BREACHED: ${daily_pnl:+,.2f} "
                   f"(limit: -${RISK_MAX_DAILY_LOSS_USD:,.2f})", "red", attrs=['bold'])
            self.daily_pause = True
            self.daily_pause_date = today
            self.pause_reason = f"Daily loss limit ${RISK_MAX_DAILY_LOSS_USD} reached"
            self.close_all_positions()
            self._save_state()
            breach = True

        # Circuit breaker 3: Max positions (informational, enforcement is in allows_new_entries)
        if open_positions >= RISK_MAX_POSITIONS:
            cprint(f"[RiskAgent] Max positions ({RISK_MAX_POSITIONS}) reached — no new trades allowed", "yellow")

        if not breach:
            cprint("[RiskAgent] All risk checks OK", "green")

        # Save state after every run
        self._save_state()

        return breach


# ---------------------------------------------------------------------------
# Per-strategy kill switch + global daily-loss breaker (independent strategies)
# ---------------------------------------------------------------------------

KILL_SWITCH_STATE_FILE = Path("src/data/strategy_kill_switch.json")
KILL_SWITCH_ROLLING_WINDOW = 40   # trades used for rolling PF
KILL_SWITCH_MIN_TRADES = 25       # min trades before PF can trigger
KILL_SWITCH_PF_MIN = 0.7          # pause if rolling PF below this
KILL_SWITCH_MAX_DD_PCT = 12.0     # pause if account drawdown above this


def corrected_trade_pnls(df):
    """Per-trade corrected PnL series from a closed_trades DataFrame.

    Uses total_pnl when available (includes partial scale-out PnL, e.g.
    adaptive_hybrid), otherwise pnl. Entry fee is then deducted in both cases:
    every strategy deducts it from balance at open and never includes it in
    pnl/total_pnl — same convention as dashboard and calibration_agent.
    """
    import pandas as pd
    if 'total_pnl' in df.columns:
        pnl = pd.to_numeric(df['total_pnl'], errors='coerce')
        if 'pnl' in df.columns:
            pnl = pnl.fillna(pd.to_numeric(df['pnl'], errors='coerce'))
    else:
        pnl = pd.to_numeric(df['pnl'], errors='coerce')
    if 'entry_fee' in df.columns:
        pnl = pnl - pd.to_numeric(df['entry_fee'], errors='coerce').fillna(0)
    return pnl.fillna(0)


def independent_daily_loss_breached(instances, limit_usd: float):
    """Global breaker: sum of daily PnL across independent strategy instances.

    Returns (breached: bool, total_daily_pnl: float). Implements
    INDEPENDENT_STRATEGIES_MAX_TOTAL_DAILY_LOSS_USD (previously dead config).
    """
    total = sum(float(getattr(inst, 'daily_pnl', 0.0) or 0.0) for inst in instances)
    return total <= -abs(limit_usd), total


class StrategyKillSwitch:
    """Auto-pause a strategy when its rolling PF or account drawdown degrades.

    Pause criteria (evaluated on the strategy's closed_trades.csv):
      - rolling PF over the last KILL_SWITCH_ROLLING_WINDOW trades < KILL_SWITCH_PF_MIN
        (only with at least KILL_SWITCH_MIN_TRADES trades in the window), or
      - current account drawdown from peak > KILL_SWITCH_MAX_DD_PCT.

    A paused strategy stays paused (state persisted to JSON) until manually
    resumed (resume() or deleting its entry in the state file). The caller
    must keep managing the paused strategy's open positions for a clean close.
    """

    def __init__(self, state_file=KILL_SWITCH_STATE_FILE, initial_balance=PAPER_TRADING_BALANCE):
        self.state_file = Path(state_file)
        self.initial_balance = initial_balance
        self.state = {}
        if self.state_file.exists():
            try:
                self.state = json.loads(self.state_file.read_text())
            except Exception as e:
                cprint(f"[KillSwitch] Could not load state: {e}", "yellow")

    def _save(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def is_paused(self, name: str) -> bool:
        return bool(self.state.get(name, {}).get('paused', False))

    def resume(self, name: str):
        """Manually un-pause a strategy."""
        if name in self.state:
            del self.state[name]
            self._save()
            cprint(f"[KillSwitch] {name} resumed", "green")

    def evaluate(self, name: str, closed_trades_csv: str) -> dict:
        """Evaluate kill-switch criteria for a strategy. Returns its state dict."""
        if self.is_paused(name):
            return self.state[name]

        import pandas as pd
        if not os.path.exists(closed_trades_csv):
            return {'paused': False, 'n_trades': 0}
        df = pd.read_csv(closed_trades_csv)
        if df.empty or ('pnl' not in df.columns and 'total_pnl' not in df.columns):
            return {'paused': False, 'n_trades': 0}

        pnls = corrected_trade_pnls(df)

        # Current account drawdown from all-time peak (full history)
        equity = self.initial_balance + pnls.cumsum()
        peak = max(self.initial_balance, float(equity.max()))
        dd_pct = (peak - float(equity.iloc[-1])) / peak * 100 if peak > 0 else 0.0

        # Rolling profit factor on the last N trades
        window = pnls.tail(KILL_SWITCH_ROLLING_WINDOW)
        n = len(window)
        gains = float(window[window > 0].sum())
        losses = float(-window[window < 0].sum())
        pf = gains / losses if losses > 0 else float('inf')

        reasons = []
        if n >= KILL_SWITCH_MIN_TRADES and pf < KILL_SWITCH_PF_MIN:
            reasons.append(f"rolling PF {pf:.2f} < {KILL_SWITCH_PF_MIN} (last {n} trades)")
        if dd_pct > KILL_SWITCH_MAX_DD_PCT:
            reasons.append(f"account drawdown {dd_pct:.1f}% > {KILL_SWITCH_MAX_DD_PCT}%")

        if reasons:
            reason = " + ".join(reasons)
            entry = {
                'paused': True,
                'reason': reason,
                'paused_at': datetime.now().isoformat(),
                'pf': round(pf, 3) if pf != float('inf') else None,
                'n_trades': int(n),
                'dd_pct': round(dd_pct, 2),
            }
            self.state[name] = entry
            self._save()
            cprint(f"[KillSwitch] STRATEGY PAUSED: {name} — {reason}", "red", attrs=['bold'])
            try:
                from src.utils.alerting import get_alert_manager
                get_alert_manager().circuit_breaker_triggered(
                    f"Kill switch: {name}", reason)
            except Exception as e:
                cprint(f"[KillSwitch] Alert failed: {e}", "yellow")
            return entry

        return {'paused': False, 'pf': round(pf, 3) if pf != float('inf') else None,
                'n_trades': int(n), 'dd_pct': round(dd_pct, 2)}


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
