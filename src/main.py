"""
🌙 Moon Dev's AI Trading System
Main entry point for running trading agents
"""

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

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

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

def position_monitor_loop(strategy, interval=30):
    """Dedicated thread for monitoring SL/TP every 30 seconds."""
    while True:
        try:
            strategy.monitor_paper_positions()
        except Exception as e:
            cprint(f"[Monitor] Error: {e}", "red")
        time.sleep(interval)


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

        while True:
            try:
                cycle_start = time.time()

                # Run Risk Management
                if risk_agent:
                    cprint("\n🛡️ Running Risk Management...", "cyan")
                    risk_agent.run()

                # Run Trading Analysis
                if trading_agent:
                    cprint("\n🤖 Running Trading Analysis...", "cyan")
                    trading_agent.run()

                # Run Strategy Analysis (only if enabled via web dashboard)
                if strategy_agent:
                    # Check if strategy should run (controlled by web dashboard)
                    if not is_strategy_running():
                        cprint("\n⏸️  Strategy paused (Start from web dashboard to enable)", "yellow")
                    else:
                        cprint("\n📊 Running Strategy Analysis...", "cyan")

                    for strategy in strategy_agent.enabled_strategies:
                        # Show paper trading status and sync to web dashboard
                        if hasattr(strategy, 'get_paper_status'):
                            status = strategy.get_paper_status()
                            if status['open_positions'] > 0 or status['total_closed'] > 0:
                                cprint(f"\n💰 Paper Trading Status:", "magenta")
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
                            cprint(f"\n⛔ Trading paused by Risk Agent: {risk_agent.pause_reason}", "red")
                        else:
                            active_tokens = get_active_tokens()  # Uses HYPERLIQUID_SYMBOLS when exchange is hyperliquid
                            tokens_to_analyze = [t for t in active_tokens if t not in EXCLUDED_TOKENS]
                            cprint(f"\n🔍 Analyzing {len(tokens_to_analyze)} tokens...", "cyan")

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

                            # Token analysis: get_signals() includes trade execution,
                            # which mutates shared state (paper_positions, balance).
                            # Data fetching is already parallelized by generate_signals_batch above.
                            # Trade execution stays sequential for thread-safety.
                            def analyze_token(token):
                                """Analyze a single token - wrapped for parallel execution."""
                                try:
                                    return strategy_agent.get_signals(token)
                                except Exception as e:
                                    cprint(f"Error analyzing {token}: {e}", "red")
                                    return None

                            # Sequential execution: get_signals does trade execution which is not thread-safe
                            # TODO: Separate signal generation (parallelizable) from trade execution (sequential)
                            # to enable full parallelization. For now, batch pre-computation above handles
                            # the IO-bound candle fetching in parallel.
                            for token in tokens_to_analyze:
                                if risk_agent and not risk_agent.is_trading_allowed():
                                    cprint(f"\n⛔ Trading paused mid-cycle by Risk Agent: {risk_agent.pause_reason}", "red")
                                    break
                                analyze_token(token)

                # Run CopyBot Analysis
                if copybot_agent:
                    cprint("\n🤖 Running CopyBot Portfolio Analysis...", "cyan")
                    copybot_agent.run_analysis_cycle()

                # Run Sentiment Analysis
                if sentiment_agent:
                    cprint("\n🎭 Running Sentiment Analysis...", "cyan")
                    sentiment_agent.run()

                # Write heartbeat
                cycle_end = time.time()
                cycle_duration = cycle_end - cycle_start
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

                # Sleep until next cycle
                next_run = datetime.now() + timedelta(minutes=SLEEP_BETWEEN_RUNS_MINUTES)
                cprint(f"\n😴 Sleeping until {next_run.strftime('%H:%M:%S')}", "cyan")
                time.sleep(60 * SLEEP_BETWEEN_RUNS_MINUTES)

            except Exception as e:
                cprint(f"\n❌ Error running agents: {str(e)}", "red")
                cprint("🔄 Continuing to next cycle...", "yellow")
                time.sleep(60)  # Sleep for 1 minute on error before retrying

    except KeyboardInterrupt:
        cprint("\n👋 Gracefully shutting down...", "yellow")
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