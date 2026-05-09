"""
🌙 Moon Dev's AI Trading System
Main entry point for running trading agents
"""

import faulthandler
import os
import sys
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from termcolor import cprint
from dotenv import load_dotenv
import time
from datetime import datetime, timedelta
from config import *

# Enable faulthandler for deadlock diagnostics (dumps all thread stacks on SIGSEGV/SIGABRT)
faulthandler.enable()
# Also dump stacks on SIGUSR1 (kill -USR1 <pid>)
import signal as _signal
faulthandler.register(_signal.SIGUSR1)

# Global socket timeout to prevent infinite HTTP hangs (HyperLiquid SDK, etc.)
import socket
socket.setdefaulttimeout(30)

# Add project root to Python path BEFORE any 'src.*' imports so that when this
# file is executed as `python src/main.py` (script dir = src/), the top-level
# `src` package is still resolvable. Previously SmartScheduler and LightCheck
# imports silently failed with "No module named 'src'", forcing the legacy
# fixed-cycle loop instead of the adaptive scheduler.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Light Check for spike detection
try:
    from src.scheduling.light_check import LightCheck
    LIGHT_CHECK_AVAILABLE = True
except Exception as e:
    cprint(f"[Main] LightCheck import failed: {e}", "yellow")
    LIGHT_CHECK_AVAILABLE = False

# Import Smart Scheduler (Phase 2)
try:
    from src.scheduling.scheduler import SmartScheduler
    SMART_SCHEDULER_AVAILABLE = True
except Exception as e:
    cprint(f"[Main] SmartScheduler import failed: {e}", "yellow")
    SMART_SCHEDULER_AVAILABLE = False

# Import web state for dashboard control
try:
    from src.web.state import is_strategy_running, update_paper_status, ensure_state_initialized
    WEB_STATE_AVAILABLE = True
except ImportError:
    WEB_STATE_AVAILABLE = False
    def is_strategy_running():
        return True  # Default to running if web state not available
    def ensure_state_initialized():
        return {"running": True}

# Load environment variables
load_dotenv()

# Agent Configuration
# Note: Only RAMF strategy is active. Other agents disabled to simplify system.
# Data sources: HyperLiquid (funding, OI) + Binance WebSocket (liquidations)
# No Moon Dev API dependency.
ACTIVE_AGENTS = {
    'risk': True,       # Risk management agent (paper trading circuit breakers)
    'trading': False,   # LLM trading agent (disabled)
    'strategy': True,   # RAMF Strategy only
    'copybot': False,   # CopyBot agent (disabled)
    'sentiment': False, # Sentiment agent (disabled)
    'calibration': True, # Auto-calibration agent (24h cycle + trigger file)
}

# Safety: Force risk agent ON when live trading
if not ACTIVE_AGENTS['risk'] and not PAPER_TRADING:
    cprint("⚠️ CRITICAL: Risk Agent forced ON for live trading safety", "red", attrs=['bold'])
    ACTIVE_AGENTS['risk'] = True

# Import agents conditionally to avoid loading unnecessary dependencies
TradingAgent = None
RiskAgent = None
StrategyAgent = None
CopyBotAgent = None
SentimentAgent = None

if ACTIVE_AGENTS['trading']:
    from src.agents.trading_agent import TradingAgent
if ACTIVE_AGENTS['risk']:
    from src.agents.risk_agent import RiskAgent
if ACTIVE_AGENTS['strategy']:
    from src.agents.strategy_agent import StrategyAgent
if ACTIVE_AGENTS['copybot']:
    from src.agents.copybot_agent import CopyBotAgent
if ACTIVE_AGENTS['sentiment']:
    from src.agents.sentiment_agent import SentimentAgent

# Import alerting for circuit breaker notifications
try:
    from src.utils.alerting import AlertManager
    alert_manager = AlertManager()
except ImportError:
    alert_manager = None

# Import CalibrationAgent (auto-calibration daemon)
try:
    from src.agents.calibration_agent import CalibrationAgent
    CALIBRATION_AVAILABLE = True
except Exception as e:
    cprint(f"[Main] CalibrationAgent import failed: {e}", "yellow")
    CALIBRATION_AVAILABLE = False

# Import independent strategies (Phase 2)
# Previous version had an `if ... or True:` outer guard which was always-true
# dead code. The actual gating happens via INDEPENDENT_STRATEGIES_ENABLED below.
IndependentStrategies = {}
try:
    from config import INDEPENDENT_STRATEGIES_ENABLED
except ImportError:
    INDEPENDENT_STRATEGIES_ENABLED = False

if INDEPENDENT_STRATEGIES_ENABLED:
    if True:  # preserves indentation of the import block below
        try:
            from src.strategies.custom.funding_mean_reversion import FundingMeanReversionStrategy
            IndependentStrategies['funding_mr'] = FundingMeanReversionStrategy
            cprint("[Main] Funding Mean Reversion strategy loaded", "cyan")
        except Exception as e:
            cprint(f"[Main] Funding MR import failed: {e}", "yellow")

        try:
            from src.strategies.custom.volatility_breakout import VolatilityBreakoutStrategy
            IndependentStrategies['vol_breakout'] = VolatilityBreakoutStrategy
            cprint("[Main] Volatility Breakout strategy loaded", "cyan")
        except Exception as e:
            cprint(f"[Main] Vol Breakout import failed: {e}", "yellow")

        try:
            from src.strategies.custom.liquidation_cascade_fade import LiquidationCascadeFadeStrategy
            IndependentStrategies['liq_cascade'] = LiquidationCascadeFadeStrategy
            cprint("[Main] Liquidation Cascade Fade strategy loaded", "cyan")
        except Exception as e:
            cprint(f"[Main] Liq Cascade import failed: {e}", "yellow")

        try:
            from src.strategies.custom.ote_scalper_strategy import OteScalperStrategy
            IndependentStrategies['ote_scalp'] = OteScalperStrategy
            cprint("[Main] OTE Scalper strategy loaded", "cyan")
        except Exception as e:
            cprint(f"[Main] OTE Scalper import failed: {e}", "yellow")

def position_monitor_loop(strategy, interval=30):
    """Dedicated thread for monitoring SL/TP every 30 seconds."""
    while True:
        try:
            strategy.monitor_paper_positions()
        except Exception as e:
            cprint(f"[Monitor] Error: {e}", "red")
        time.sleep(interval)


def independent_strategy_loop(strategy_instance, name, interval_seconds=300):
    """Dedicated thread for running an independent strategy on its own cycle."""
    cprint(f"[{name}] Strategy thread started (cycle: {interval_seconds}s)", "cyan")
    while True:
        try:
            if not is_strategy_running():
                time.sleep(30)
                continue
            strategy_instance.run_cycle(strategy_instance.tokens)
        except Exception as e:
            cprint(f"[{name}] Error in cycle: {e}", "red")
        time.sleep(interval_seconds)


def calibration_loop(calibration_agent, interval_hours=24):
    """Daemon thread: runs calibration every 24h and on trigger file."""
    trigger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'calibration_trigger.json')
    while True:
        try:
            # Check trigger file (written after N closed trades)
            triggered = False
            if os.path.exists(trigger_path):
                try:
                    with open(trigger_path, 'r') as f:
                        trigger = json.load(f)
                    if trigger.get('run', False):
                        cprint("[Calibration] Trigger file detected, running calibration...", "cyan")
                        triggered = True
                        # Clear trigger
                        with open(trigger_path, 'w') as f:
                            json.dump({'run': False, 'cleared_at': datetime.now().isoformat()}, f)
                except Exception as e:
                    cprint(f"[Calibration] Error reading trigger file: {e}", "yellow")

            if triggered:
                calibration_agent.run()
        except Exception as e:
            cprint(f"[Calibration] Error in trigger check: {e}", "red")

        # Sleep for the full interval, checking trigger every 5 minutes
        sleep_total = interval_hours * 3600
        elapsed = 0
        check_interval = 300  # Check trigger file every 5 minutes
        while elapsed < sleep_total:
            time.sleep(min(check_interval, sleep_total - elapsed))
            elapsed += check_interval
            # Check trigger during wait
            try:
                if os.path.exists(trigger_path):
                    with open(trigger_path, 'r') as f:
                        trigger = json.load(f)
                    if trigger.get('run', False):
                        cprint("[Calibration] Trigger file detected mid-cycle, running calibration...", "cyan")
                        with open(trigger_path, 'w') as f:
                            json.dump({'run': False, 'cleared_at': datetime.now().isoformat()}, f)
                        calibration_agent.run()
            except Exception:
                pass

        # Scheduled 24h run
        try:
            cprint("[Calibration] Scheduled 24h calibration run...", "cyan")
            calibration_agent.run()
        except Exception as e:
            cprint(f"[Calibration] Error in scheduled run: {e}", "red")


def daily_report_loop(strategy):
    """Daemon thread: sends a daily report at 9h00 local time."""
    while True:
        now = datetime.now()
        target = now.replace(hour=9, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        sleep_seconds = (target - now).total_seconds()
        time.sleep(sleep_seconds)

        try:
            yesterday = (datetime.now() - timedelta(days=1)).date()
            stats = strategy.get_daily_stats(yesterday)
            from src.utils.alerting import get_alert_manager
            get_alert_manager().daily_summary(stats)
            cprint(f"[DailyReport] Sent report for {yesterday}", "cyan")
        except Exception as e:
            cprint(f"[DailyReport] Error: {e}", "red")


def run_agents():
    """Run all active agents in sequence"""
    try:
        # Initialize active agents
        trading_agent = TradingAgent() if ACTIVE_AGENTS['trading'] else None
        risk_agent = RiskAgent() if ACTIVE_AGENTS['risk'] else None
        strategy_agent = StrategyAgent() if ACTIVE_AGENTS['strategy'] else None
        copybot_agent = CopyBotAgent() if ACTIVE_AGENTS['copybot'] else None
        sentiment_agent = SentimentAgent() if ACTIVE_AGENTS['sentiment'] else None

        # Link risk agent to strategy for paper trading monitoring
        # and start dedicated SL/TP monitoring thread
        monitor_strategy = None
        if risk_agent and strategy_agent:
            for strategy in strategy_agent.enabled_strategies:
                if hasattr(strategy, 'get_paper_status'):
                    risk_agent.set_strategy(strategy)
                    # Connect risk agent to strategy for recovery mode sizing
                    if hasattr(strategy, 'set_risk_agent'):
                        strategy.set_risk_agent(risk_agent)
                    monitor_strategy = strategy
                    break

        if monitor_strategy and hasattr(monitor_strategy, 'monitor_paper_positions'):
            monitor_thread = threading.Thread(
                target=position_monitor_loop,
                args=(monitor_strategy, 30),
                daemon=True,
                name="SL_TP_Monitor"
            )
            monitor_thread.start()
            cprint("[Main] SL/TP monitor thread started (30s interval)", "cyan")

        # Start daily report thread (sends Discord summary at 9h00)
        if monitor_strategy and hasattr(monitor_strategy, 'get_daily_stats'):
            report_thread = threading.Thread(
                target=daily_report_loop,
                args=(monitor_strategy,),
                daemon=True,
                name="Daily_Report"
            )
            report_thread.start()
            cprint("[Main] Daily report thread started (9h00 daily)", "cyan")

        # Start CalibrationAgent daemon thread (24h cycle + trigger file)
        if CALIBRATION_AVAILABLE and ACTIVE_AGENTS.get('calibration', False):
            try:
                cal_agent = CalibrationAgent()
                cal_thread = threading.Thread(
                    target=calibration_loop,
                    args=(cal_agent, 24),
                    daemon=True,
                    name="Calibration_Agent"
                )
                cal_thread.start()
                cprint("[Main] Calibration agent thread started (24h cycle + trigger file)", "cyan")
            except Exception as e:
                cprint(f"[Main] Failed to start CalibrationAgent: {e}", "red")

        # Start independent strategy threads (Phase 2)
        # Each strategy is wrapped in its own try/except so a single failure
        # (e.g. constructor exception) does not silently kill the launch loop
        # and prevent siblings from starting. We log the full traceback to make
        # diagnosis trivial in production logs.
        independent_instances = {}
        for strat_key, strat_class in IndependentStrategies.items():
            try:
                instance = strat_class()
                independent_instances[strat_key] = instance
                # Determine cycle interval based on strategy type
                intervals = {'funding_mr': 300, 'vol_breakout': 300, 'liq_cascade': 60, 'ote_scalp': 60}
                interval = intervals.get(strat_key, 300)
                t = threading.Thread(
                    target=independent_strategy_loop,
                    args=(instance, strat_key, interval),
                    daemon=True,
                    name=f"Strategy_{strat_key}"
                )
                t.start()
                cprint(
                    f"[Main] Independent strategy '{strat_key}' thread started "
                    f"(interval={interval}s, thread={t.name})",
                    "green",
                )
            except Exception as e:
                cprint(f"[Main][ERROR] Failed to start {strat_key}: {e}", "red")
                import traceback
                traceback.print_exc()

        # Start Light Check daemon thread (spike detection every 2 minutes)
        light_check = None
        if LIGHT_CHECK_AVAILABLE and LIGHT_CHECK_ENABLED:
            light_check = LightCheck()
            light_check.start()

        # Start Scheduler Healthcheck daemon thread (detects scheduler freezes)
        # Polls scheduler_state.json every 5 minutes; alerts to Discord if
        # min(last_check_ago_s) > 30 minutes (1h cooldown to avoid spam).
        try:
            from src.utils.scheduler_healthcheck import start_healthcheck_thread
            start_healthcheck_thread()
            cprint("[Main] Scheduler healthcheck thread started (5min interval, "
                   "30min freeze threshold, 1h alert cooldown)", "cyan")
        except Exception as e:
            cprint(f"[Main] Failed to start scheduler healthcheck: {e}", "yellow")

        # Initialize Smart Scheduler (Phase 2)
        use_scheduler = (SMART_SCHEDULER_AVAILABLE and SCHEDULER_ENABLED
                         and strategy_agent is not None)
        scheduler = None
        if use_scheduler:
            scheduler = SmartScheduler()
            scheduler.load_state()
            active_tokens = [t for t in get_active_tokens() if t not in EXCLUDED_TOKENS]
            scheduler.enqueue_all_routine(active_tokens)
            cprint(f"[Scheduler] Smart scheduler initialized with {len(active_tokens)} tokens", "cyan")
        elif SCHEDULER_ENABLED and not SMART_SCHEDULER_AVAILABLE:
            cprint("[Scheduler] SCHEDULER_ENABLED=True but module not available, using fixed cycle", "yellow")

        # Shared helpers
        def _show_paper_status():
            """Display paper trading status and sync to dashboard."""
            if not strategy_agent:
                return
            for strategy in strategy_agent.enabled_strategies:
                if hasattr(strategy, 'get_paper_status'):
                    status = strategy.get_paper_status()
                    if status['open_positions'] > 0 or status['total_closed'] > 0:
                        cprint(f"\n  Paper Trading Status:", "magenta")
                        cprint(f"   Balance: ${status['paper_balance']:,.2f} (started: ${status['initial_balance']:,.2f})", "white")
                        cprint(f"   Total PnL: ${status['total_pnl']:+,.2f}", "green" if status['total_pnl'] >= 0 else "red")
                        cprint(f"   Daily PnL: ${status['daily_pnl']:+,.2f}", "white")
                        cprint(f"   Open: {status['open_positions']} | Closed: {status['total_closed']}", "white")
                    if WEB_STATE_AVAILABLE:
                        try:
                            update_paper_status(
                                balance=status['paper_balance'],
                                positions=status.get('positions', []),
                                daily_pnl=status['daily_pnl'],
                                total_pnl=status['total_pnl'],
                                trades_today=status.get('daily_trades', 0)
                            )
                        except Exception:
                            pass

        def _analyze_token(token):
            """Analyze a single token - sequential for thread-safety."""
            try:
                return strategy_agent.get_signals(token)
            except Exception as e:
                cprint(f"Error analyzing {token}: {e}", "red")
                return None

        # Watchdog executor for scheduler loop: enforces a hard timeout on
        # _analyze_token calls so a hung LLM SDK request (TCP/TLS zombie) cannot
        # freeze the main scheduler thread for hours. max_workers=1 preserves
        # sequential execution order (trade execution path is not thread-safe).
        # Created once at module-level scope (not per-iteration) to avoid the
        # cost of spawning a new thread on every scheduler tick.
        _watchdog_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scheduler-watchdog")
        _ANALYZE_TOKEN_TIMEOUT_S = 120  # 30s LLM timeout * 2 retries + overhead

        def _analyze_token_with_watchdog(token):
            """Run _analyze_token in a worker thread with a hard timeout.

            Returns (signal, timed_out: bool). On timeout, logs a warning and
            returns (None, True) so the caller can record a fail result and
            keep last_check fresh for the healthcheck.
            """
            from concurrent.futures import TimeoutError as FuturesTimeoutError
            future = _watchdog_executor.submit(_analyze_token, token)
            try:
                return future.result(timeout=_ANALYZE_TOKEN_TIMEOUT_S), False
            except FuturesTimeoutError:
                cprint(
                    f"  [Watchdog] scheduler watchdog timeout for token {token} "
                    f"(exceeded {_ANALYZE_TOKEN_TIMEOUT_S}s) - skipping, will retry next cycle",
                    "yellow",
                )
                # Note: we cannot cancel the running future (Python limitation),
                # but the LLM SDK timeouts (30s + retries) will eventually free it.
                return None, True

        def _extract_result(token, signal):
            """Extract scheduler-relevant fields from a generate_signals result."""
            if signal and signal.get('metadata'):
                meta = signal['metadata']
                return {
                    'score': meta.get('score', 0),
                    'threshold': meta.get('threshold', 40),
                    'regime': (meta.get('llm_regime') or {}).get('regime', ''),
                    'atr_pct': meta.get('atr', 0) / max(meta.get('current_price', 1), 1e-9) if meta.get('atr') else 0,
                }
            return {'score': 0, 'threshold': 40, 'regime': '', 'atr_pct': 0}

        def _has_position(token):
            """Check if there is an open paper position for this token."""
            for strategy in strategy_agent.enabled_strategies:
                if hasattr(strategy, 'paper_positions'):
                    with strategy._position_lock:
                        for pos in strategy.paper_positions.values():
                            if pos.get('symbol') == token:
                                return True
            return False

        def _write_heartbeat(cycle_start):
            """Write heartbeat file for monitoring."""
            cycle_duration = time.time() - cycle_start
            heartbeat_path = os.path.join(os.path.dirname(__file__), 'data', 'bot_heartbeat.json')
            try:
                os.makedirs(os.path.dirname(heartbeat_path), exist_ok=True)
                with open(heartbeat_path, 'w') as f:
                    json.dump({
                        'last_cycle': datetime.now().isoformat(),
                        'timestamp': time.time(),
                        'status': 'running',
                        'cycle_duration_s': round(cycle_duration, 1)
                    }, f)
            except Exception:
                pass

        # =====================================================================
        # SCHEDULER-BASED LOOP (Phase 2) — fast 30s iterations
        # =====================================================================
        if use_scheduler:
            last_risk_run = 0
            RISK_INTERVAL_S = 5 * 60  # Risk agent every 5 minutes

            # ---- Scheduler self-heal state (root-cause fix for prod freeze) -
            # If get_due_symbols() returns empty AND the queue is empty for too
            # many consecutive ticks, OR no record_result has succeeded in too
            # long, we re-seed the queue with all monitored symbols. Defense-
            # in-depth against the over-represented full-skip drain that left
            # 6 tokens orphaned (priority=null, next_recheck=null) in prod.
            empty_due_streak = 0
            EMPTY_DUE_STREAK_HEAL_THRESHOLD = 3   # 3 ticks * 30s = 90s starved
            last_record_result_ts = time.time()
            RECORD_RESULT_STALE_S = 600           # 10 min without any record_result

            cprint(
                f"[Scheduler] Loop started, watchdog active "
                f"(timeout={_ANALYZE_TOKEN_TIMEOUT_S}s), "
                f"self-heal armed (empty_streak={EMPTY_DUE_STREAK_HEAL_THRESHOLD} ticks, "
                f"record_stale={RECORD_RESULT_STALE_S}s), "
                f"queue size={scheduler.queue_size()}",
                "green",
            )

            def _self_heal_reseed(reason: str):
                """Re-seed the scheduler queue with all monitored symbols."""
                try:
                    symbols = scheduler.get_all_symbols()
                    if not symbols:
                        symbols = [t for t in get_active_tokens() if t not in EXCLUDED_TOKENS]
                    cprint(
                        f"[Scheduler] SELF-HEAL: {reason} — re-enqueueing "
                        f"{len(symbols)} symbols (was queue size {scheduler.queue_size()})",
                        "yellow",
                        attrs=['bold'],
                    )
                    scheduler.enqueue_all_routine(symbols, reason_label="self_heal")
                    cprint(
                        f"[Scheduler] SELF-HEAL complete — new queue size {scheduler.queue_size()}",
                        "green",
                    )
                except Exception as _e:
                    cprint(f"[Scheduler] SELF-HEAL failed: {_e}", "red")

            while True:
                try:
                    iter_start = time.time()

                    # --- Risk Agent (every 5 min) ---
                    if risk_agent and (time.time() - last_risk_run >= RISK_INTERVAL_S):
                        cprint("\n  Running Risk Management...", "cyan")
                        risk_agent.run()
                        last_risk_run = time.time()
                        _show_paper_status()

                    # --- Strategy paused check ---
                    if not is_strategy_running():
                        cprint("  Strategy paused (Start from web dashboard to enable)", "yellow")
                        time.sleep(30)
                        continue

                    # --- Risk gate ---
                    if risk_agent and not risk_agent.is_trading_allowed():
                        cprint(f"  Trading paused by Risk Agent: {risk_agent.pause_reason}", "red")
                        time.sleep(30)
                        continue

                    # --- Inject spike triggers from LightCheck ---
                    if light_check:
                        spike_tokens = light_check.get_and_clear_triggered()
                        for spike_token in spike_tokens:
                            scheduler.enqueue_spike(spike_token, f"spike detected by LightCheck")

                    # --- Get due tokens from scheduler ---
                    due = scheduler.get_due_symbols()

                    # --- Self-heal: empty-queue streak watchdog ---------------
                    # Tokens disappear from the queue when schedule_recheck()
                    # returns without re-enqueueing (e.g. silent exception, or
                    # the over-represented full-skip path that orphaned 6
                    # tokens in prod). We re-seed if we observe an empty due-list
                    # AND an empty queue for too many consecutive ticks.
                    if not due:
                        empty_due_streak += 1
                        if empty_due_streak >= EMPTY_DUE_STREAK_HEAL_THRESHOLD:
                            qsize = scheduler.queue_size()
                            # Only re-seed if queue is genuinely starved.
                            # Non-empty queue with no due means tokens are
                            # scheduled in the future — that's the normal idle
                            # state.
                            if qsize == 0:
                                _self_heal_reseed(
                                    f"queue empty for {empty_due_streak} consecutive ticks (~{empty_due_streak*30}s)"
                                )
                            empty_due_streak = 0
                    else:
                        empty_due_streak = 0

                    if due:
                        cprint(f"\n  [Scheduler] {len(due)} token(s) due (queue: {scheduler.queue_size()} remaining)", "cyan")

                        for i, req in enumerate(due):
                            if risk_agent and not risk_agent.is_trading_allowed():
                                cprint(f"  Trading paused mid-cycle by Risk Agent: {risk_agent.pause_reason}", "red")
                                # Re-enqueue current + all remaining unprocessed tokens
                                for remaining in due[i:]:
                                    scheduler.schedule_recheck(remaining.symbol, scheduler.get_last_result(remaining.symbol), _has_position(remaining.symbol))
                                break

                            cprint(f"  [Scheduler] Analyzing {req.symbol} ({req.reason})", "cyan")
                            signal, timed_out = _analyze_token_with_watchdog(req.symbol)

                            # Always record_result (even on timeout) so last_check stays
                            # fresh and the healthcheck reflects scheduler liveness.
                            result = _extract_result(req.symbol, signal)
                            if timed_out:
                                # Mark fail so calling code can distinguish a timeout
                                # from a normal "no signal" result if needed.
                                result['status'] = 'fail'
                                result['fail_reason'] = 'watchdog_timeout'
                            scheduler.record_result(req.symbol, result)
                            last_record_result_ts = time.time()
                            scheduler.schedule_recheck(req.symbol, result, _has_position(req.symbol))

                        _write_heartbeat(iter_start)

                    # --- Self-heal: record_result-stale watchdog --------------
                    # Catches the case where the queue holds items but dispatch
                    # is stuck somewhere upstream of record_result (unlikely
                    # given the per-token watchdog; defense-in-depth).
                    if time.time() - last_record_result_ts > RECORD_RESULT_STALE_S:
                        _self_heal_reseed(
                            f"no record_result in {int(time.time() - last_record_result_ts)}s "
                            f"(threshold {RECORD_RESULT_STALE_S}s)"
                        )
                        last_record_result_ts = time.time()  # avoid re-trigger every tick

                    # --- CopyBot / Sentiment (run after risk agent, same cadence) ---
                    risk_just_ran = (time.time() - last_risk_run < RISK_INTERVAL_S + 30)
                    if copybot_agent and risk_just_ran:
                        copybot_agent.run_analysis_cycle()
                    if sentiment_agent and risk_just_ran:
                        sentiment_agent.run()

                    # Wait 30s before next iteration
                    time.sleep(30)

                except Exception as e:
                    cprint(f"\n  Error in scheduler loop: {str(e)}", "red")
                    try:
                        import traceback as _tb
                        _tb.print_exc()
                    except Exception:
                        pass
                    cprint("  Continuing to next iteration...", "yellow")
                    time.sleep(30)

        # =====================================================================
        # FIXED-CYCLE LOOP (legacy fallback when SCHEDULER_ENABLED=False)
        # =====================================================================
        while True:
            try:
                cycle_start = time.time()

                # Run Risk Management
                if risk_agent:
                    cprint("\n  Running Risk Management...", "cyan")
                    risk_agent.run()

                # Run Trading Analysis
                if trading_agent:
                    cprint("\n  Running Trading Analysis...", "cyan")
                    trading_agent.run()

                # Run Strategy Analysis (only if enabled via web dashboard)
                if strategy_agent:
                    # Check if strategy should run (controlled by web dashboard)
                    if not is_strategy_running():
                        cprint("\n  Strategy paused (Start from web dashboard to enable)", "yellow")
                    else:
                        cprint("\n  Running Strategy Analysis...", "cyan")

                    for strategy in strategy_agent.enabled_strategies:
                        # Show paper trading status and sync to web dashboard
                        if hasattr(strategy, 'get_paper_status'):
                            status = strategy.get_paper_status()
                            if status['open_positions'] > 0 or status['total_closed'] > 0:
                                cprint(f"\n  Paper Trading Status:", "magenta")
                                cprint(f"   Balance: ${status['paper_balance']:,.2f} (started: ${status['initial_balance']:,.2f})", "white")
                                cprint(f"   Total PnL: ${status['total_pnl']:+,.2f}", "green" if status['total_pnl'] >= 0 else "red")
                                cprint(f"   Daily PnL: ${status['daily_pnl']:+,.2f}", "white")
                                cprint(f"   Open: {status['open_positions']} | Closed: {status['total_closed']}", "white")

                            # Sync paper trading status to web dashboard
                            if WEB_STATE_AVAILABLE:
                                try:
                                    update_paper_status(
                                        balance=status['paper_balance'],
                                        positions=status.get('positions', []),
                                        daily_pnl=status['daily_pnl'],
                                        total_pnl=status['total_pnl'],
                                        trades_today=status.get('daily_trades', 0)
                                    )
                                except Exception as e:
                                    pass  # Don't fail if web state update fails

                    # Only analyze new signals if strategy is running (controlled by dashboard)
                    if is_strategy_running():
                        # Check risk agent before opening new positions
                        if risk_agent and not risk_agent.is_trading_allowed():
                            cprint(f"\n  Trading paused by Risk Agent: {risk_agent.pause_reason}", "red")
                        else:
                            # Check for spike-triggered tokens from LightCheck (priority analysis)
                            spike_tokens = set()
                            if light_check:
                                spike_tokens = light_check.get_and_clear_triggered()
                                if spike_tokens:
                                    cprint(f"\n[LightCheck] Priority analysis for {len(spike_tokens)} spiked token(s): {', '.join(sorted(spike_tokens))}", "yellow", attrs=['bold'])
                                    for spike_token in sorted(spike_tokens):
                                        if risk_agent and not risk_agent.is_trading_allowed():
                                            cprint(f"\n  Trading paused by Risk Agent: {risk_agent.pause_reason}", "red")
                                            break
                                        _analyze_token(spike_token)

                            active_tokens = get_active_tokens()  # Uses HYPERLIQUID_SYMBOLS when exchange is hyperliquid
                            # Exclude tokens already analyzed via spike priority
                            tokens_to_analyze = [t for t in active_tokens if t not in EXCLUDED_TOKENS and t not in spike_tokens]
                            cprint(f"\n  Analyzing {len(tokens_to_analyze)} tokens...", "cyan")

                            # Use batch signal generation for pre-computation if available
                            for strat in strategy_agent.enabled_strategies:
                                if hasattr(strat, 'generate_signals_batch'):
                                    try:
                                        results = strat.generate_signals_batch(tokens_to_analyze)
                                        if results:
                                            cprint(f"[Batch] Pre-computed {len(results)} signal results", "cyan")
                                    except Exception as e:
                                        cprint(f"[Batch] Error: {e}", "yellow")
                                    break

                            # Sequential execution: get_signals does trade execution which is not thread-safe
                            for token in tokens_to_analyze:
                                if risk_agent and not risk_agent.is_trading_allowed():
                                    cprint(f"\n  Trading paused mid-cycle by Risk Agent: {risk_agent.pause_reason}", "red")
                                    break
                                _analyze_token(token)

                # Run CopyBot Analysis
                if copybot_agent:
                    cprint("\n  Running CopyBot Portfolio Analysis...", "cyan")
                    copybot_agent.run_analysis_cycle()

                # Run Sentiment Analysis
                if sentiment_agent:
                    cprint("\n  Running Sentiment Analysis...", "cyan")
                    sentiment_agent.run()

                # Write heartbeat
                _write_heartbeat(cycle_start)

                # Sleep until next cycle
                next_run = datetime.now() + timedelta(minutes=SLEEP_BETWEEN_RUNS_MINUTES)
                cprint(f"\n  Sleeping until {next_run.strftime('%H:%M:%S')}", "cyan")
                time.sleep(60 * SLEEP_BETWEEN_RUNS_MINUTES)

            except Exception as e:
                cprint(f"\n  Error running agents: {str(e)}", "red")
                cprint("  Continuing to next cycle...", "yellow")
                time.sleep(60)  # Sleep for 1 minute on error before retrying

    except KeyboardInterrupt:
        cprint("\n👋 Gracefully shutting down...", "yellow")
        if scheduler:
            scheduler.save_state()
            cprint("[Scheduler] State saved to disk", "cyan")
        if light_check:
            light_check.stop()
            cprint("[LightCheck] State saved to disk", "cyan")
    except Exception as e:
        cprint(f"\n❌ Fatal error in main loop: {str(e)}", "red")
        raise

if __name__ == "__main__":
    cprint("\n🌙 Moon Dev AI Agent Trading System Starting...", "white", "on_blue")

    # Initialize and validate state (handles AUTO_START logic)
    if WEB_STATE_AVAILABLE:
        cprint("\n📋 Initializing state...", "cyan")
        ensure_state_initialized()

    # Show paper trading status
    try:
        from src.config import PAPER_TRADING, PAPER_TRADING_BALANCE, ACTIVE_STRATEGY
        if PAPER_TRADING:
            cprint("\n⚠️  PAPER TRADING MODE ENABLED", "yellow", "on_red", attrs=['bold'])
            cprint(f"   Simulated Balance: ${PAPER_TRADING_BALANCE}", "yellow")
            cprint("   No real trades will be executed\n", "yellow")
        else:
            cprint("\n🔴 LIVE TRADING MODE", "white", "on_red", attrs=['bold'])
            cprint("   Real trades will be executed!\n", "red")

        cprint(f"📈 Active Strategy: {ACTIVE_STRATEGY.upper()}", "cyan")
    except ImportError:
        pass

    cprint("\n📊 Active Agents:", "white", "on_blue")
    for agent, active in ACTIVE_AGENTS.items():
        status = "✅ ON" if active else "❌ OFF"
        cprint(f"  • {agent.title()}: {status}", "white", "on_blue")
    print("\n")

    run_agents()