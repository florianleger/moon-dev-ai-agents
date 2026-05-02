"""
Calibration Agent -- Auto-tunes strategy parameters based on recent trade performance.

Reads closed trades from closed_trades.csv and TradeMemory, diagnoses performance
problems (SL too tight, TP never reached, declining win rate, etc.), computes
mechanical adjustments, optionally validates via LLM, and writes overrides to
data/calibration_overrides.json consumed by the strategy at runtime.

Follows the same agent pattern as risk_agent.py (init, run, persistence JSON).
"""

import json
import os
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from termcolor import cprint

from src.agents.base_agent import BaseAgent
from src.utils.calibration import apply_guardrail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_BASE = Path(__file__).parent.parent / 'data'
OVERRIDES_PATH = DATA_BASE / 'calibration_overrides.json'
CALIBRATION_STATE_PATH = DATA_BASE / 'calibration_agent_state.json'


def _get_closed_trades_csv() -> str:
    """Return path to closed_trades.csv for the active strategy."""
    from src.config import ACTIVE_STRATEGY
    folder_map = {
        'sniper': 'sniper',
        'adaptive_hybrid': 'adaptive_hybrid',
        'hybrid': 'sniper',
    }
    folder = folder_map.get(ACTIVE_STRATEGY, 'ramf')
    return str(DATA_BASE / folder / 'closed_trades.csv')


# ---------------------------------------------------------------------------
# CalibrationAgent
# ---------------------------------------------------------------------------
class CalibrationAgent(BaseAgent):
    """Self-calibrating agent that adjusts strategy parameters based on recent performance."""

    # Minimum trades required before any calibration
    MIN_TRADES = 15  # Raised from 5: reduce noise (3 SL streak no longer triggers adjustment)
    # Minimum history span (days) required before any calibration -- guards against early bursts of trades
    MIN_HISTORY_DAYS = 3  # Lowered from 7 to allow earlier convergence given low trade frequency in prod

    def __init__(self):
        super().__init__('calibration')
        self._state = {}
        self._load_state()
        cprint("[CalibrationAgent] Initialized", "cyan")

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _save_state(self):
        CALIBRATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._state['last_updated'] = datetime.now().isoformat()
        tmp = str(CALIBRATION_STATE_PATH) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self._state, f, indent=2, default=str)
        os.replace(tmp, str(CALIBRATION_STATE_PATH))

    def _load_state(self):
        if CALIBRATION_STATE_PATH.exists():
            try:
                self._state = json.loads(CALIBRATION_STATE_PATH.read_text())
                cprint(f"[CalibrationAgent] Restored state (last run: {self._state.get('last_run', 'never')})", "cyan")
            except Exception as e:
                cprint(f"[CalibrationAgent] Could not load state: {e}", "yellow")
                self._state = {}
        else:
            self._state = {}

    # ------------------------------------------------------------------
    # 1. Collect performance data
    # ------------------------------------------------------------------
    def _collect_performance_data(self, days: int = 14) -> dict:
        """Read closed trades from CSV and TradeMemory. Returns metrics dict."""
        trades = []

        # --- Source 1: closed_trades.csv (primary) ---
        csv_path = _get_closed_trades_csv()
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty and 'exit_time' in df.columns and 'pnl' in df.columns:
                    df['exit_dt'] = pd.to_datetime(df['exit_time'], errors='coerce')
                    cutoff = datetime.now() - timedelta(days=days)
                    df = df[df['exit_dt'] >= cutoff]
                    for _, row in df.iterrows():
                        trades.append({
                            'symbol': row.get('symbol', ''),
                            'direction': row.get('direction', ''),
                            'pnl': float(row.get('pnl', 0) or 0),
                            'pnl_pct': float(row.get('pnl_pct', 0) or 0),
                            'close_reason': str(row.get('close_reason', '')),
                            'entry_time': str(row.get('entry_time', row.get('timestamp', ''))),
                            'exit_time': str(row.get('exit_time', '')),
                            'score': float(row.get('score', 0) or 0),
                            'entry_price': float(row.get('entry_price', 0) or 0),
                            'close_price': float(row.get('close_price', 0) or 0),
                        })
            except Exception as e:
                cprint(f"[CalibrationAgent] Error reading closed_trades.csv: {e}", "yellow")

        # --- Source 2: TradeMemory (supplement for win rate / lessons) ---
        memory_summary = {}
        try:
            from src.data.trade_memory import TradeMemory
            tm = TradeMemory.get_instance()
            memory_summary = tm.get_performance_summary(days=days)
        except Exception as e:
            cprint(f"[CalibrationAgent] TradeMemory unavailable: {e}", "yellow")

        # --- Compute metrics ---
        if not trades:
            return {'trades': 0, 'memory_summary': memory_summary}

        total = len(trades)
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        profit_factor = gross_profit / max(gross_loss, 0.01)
        win_rate = len(wins) / total if total > 0 else 0

        # Close reason distribution
        reason_counts = Counter(t['close_reason'] for t in trades)
        sl_pct = reason_counts.get('STOP_LOSS', 0) / total if total > 0 else 0
        tp_count = reason_counts.get('TAKE_PROFIT', 0)

        # Daily trade frequency
        try:
            exit_dates = [pd.to_datetime(t['exit_time']).date() for t in trades if t['exit_time']]
            unique_days = len(set(exit_dates)) or 1
            trades_per_day = total / unique_days
        except Exception:
            trades_per_day = total / max(days, 1)

        # Hour-based analysis
        hour_stats = {}
        for t in trades:
            try:
                hour = pd.to_datetime(t['entry_time']).hour
                if hour not in hour_stats:
                    hour_stats[hour] = {'wins': 0, 'losses': 0}
                if t['pnl'] > 0:
                    hour_stats[hour]['wins'] += 1
                else:
                    hour_stats[hour]['losses'] += 1
            except Exception:
                pass

        # Current balance from WebState
        current_balance = None
        try:
            from src.web.state import get_dashboard_stats
            stats = get_dashboard_stats()
            current_balance = stats.get('balance')
        except Exception:
            pass

        return {
            'trades': total,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(win_rate, 4),
            'profit_factor': round(profit_factor, 3),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'total_pnl': round(gross_profit - gross_loss, 2),
            'avg_win': round(gross_profit / max(len(wins), 1), 2),
            'avg_loss': round(gross_loss / max(len(losses), 1), 2),
            'sl_pct': round(sl_pct, 3),
            'tp_count': tp_count,
            'reason_counts': dict(reason_counts),
            'trades_per_day': round(trades_per_day, 2),
            'hour_stats': hour_stats,
            'current_balance': current_balance,
            'memory_summary': memory_summary,
            'raw_trades': trades,
        }

    # ------------------------------------------------------------------
    # 2. Diagnose problems
    # ------------------------------------------------------------------
    def _diagnose_problems(self, metrics: dict) -> list:
        """Identify performance patterns that suggest parameter adjustments."""
        problems = []
        total = metrics.get('trades', 0)
        if total < self.MIN_TRADES:
            return problems

        # SL_TOO_TIGHT: >60% of exits are stop losses
        if metrics.get('sl_pct', 0) > 0.60:
            problems.append({
                'type': 'SL_TOO_TIGHT',
                'detail': f"Stop loss exits: {metrics['sl_pct']:.0%} of {total} trades",
                'severity': 'HIGH',
            })

        # TP_NEVER_REACHED: 0 take-profit exits in the period
        if metrics.get('tp_count', 0) == 0 and total >= 5:
            problems.append({
                'type': 'TP_NEVER_REACHED',
                'detail': f"0 take-profit exits across {total} trades",
                'severity': 'MEDIUM',
            })

        # WR_DECLINING: win rate below 40%
        if metrics.get('win_rate', 0) < 0.40:
            problems.append({
                'type': 'WR_DECLINING',
                'detail': f"Win rate: {metrics['win_rate']:.1%}",
                'severity': 'HIGH',
            })

        # BAD_HOURS: certain hours with WR < 25% and >= 3 trades
        bad_hours = []
        for hour, stats in metrics.get('hour_stats', {}).items():
            total_h = stats['wins'] + stats['losses']
            if total_h >= 3:
                wr = stats['wins'] / total_h
                if wr < 0.25:
                    bad_hours.append(hour)
        if bad_hours:
            problems.append({
                'type': 'BAD_HOURS',
                'detail': f"Hours with WR < 25%: {bad_hours}",
                'severity': 'LOW',
                'hours': bad_hours,
            })

        # THRESHOLD_TOO_LOW: PF < 0.8 with > 3 trades/day
        if metrics.get('profit_factor', 999) < 0.8 and metrics.get('trades_per_day', 0) > 3:
            problems.append({
                'type': 'THRESHOLD_TOO_LOW',
                'detail': f"PF={metrics['profit_factor']:.2f} with {metrics['trades_per_day']:.1f} trades/day",
                'severity': 'HIGH',
            })

        # THRESHOLD_TOO_HIGH: < 0.3 trades/day with PF > 1.5
        if metrics.get('trades_per_day', 999) < 0.3 and metrics.get('profit_factor', 0) > 1.5:
            problems.append({
                'type': 'THRESHOLD_TOO_HIGH',
                'detail': f"PF={metrics['profit_factor']:.2f} but only {metrics['trades_per_day']:.2f} trades/day",
                'severity': 'LOW',
            })

        return problems

    # ------------------------------------------------------------------
    # 3. Compute adjustments (mechanical rules)
    # ------------------------------------------------------------------
    def _compute_adjustments(self, problems: list, metrics: dict) -> dict:
        """Produce parameter adjustment proposals from diagnosed problems."""
        adjustments = {}
        problem_types = {p['type'] for p in problems}

        # Load current overrides as baseline
        current = self._load_current_overrides()

        # Get config defaults
        try:
            from src.config import (
                ADAPTIVE_HYBRID_BASE_THRESHOLD,
                ADAPTIVE_HYBRID_ATR_PROFILES,
            )
        except ImportError:
            ADAPTIVE_HYBRID_BASE_THRESHOLD = 55
            ADAPTIVE_HYBRID_ATR_PROFILES = {}

        current_threshold = current.get(
            'ADAPTIVE_HYBRID_BASE_THRESHOLD', {}).get('value', ADAPTIVE_HYBRID_BASE_THRESHOLD)

        # --- SL_TOO_TIGHT: widen stop loss by 10% ---
        if 'SL_TOO_TIGHT' in problem_types:
            atr_profiles = current.get('ADAPTIVE_HYBRID_ATR_PROFILES', {}).get('value')
            if not atr_profiles:
                atr_profiles = dict(ADAPTIVE_HYBRID_ATR_PROFILES) if ADAPTIVE_HYBRID_ATR_PROFILES else {}
            new_profiles = {}
            for symbol, profile in atr_profiles.items():
                new_profiles[symbol] = dict(profile)
                if 'sl' in new_profiles[symbol]:
                    new_profiles[symbol]['sl'] = round(new_profiles[symbol]['sl'] * 1.10, 2)
            if new_profiles:
                adjustments['ADAPTIVE_HYBRID_ATR_PROFILES'] = {
                    'value': new_profiles,
                    'reason': 'SL_TOO_TIGHT: widening SL by 10%',
                }

        # --- TP_NEVER_REACHED: reduce TP multiplier by 10% ---
        if 'TP_NEVER_REACHED' in problem_types:
            atr_profiles = adjustments.get('ADAPTIVE_HYBRID_ATR_PROFILES', {}).get('value')
            if not atr_profiles:
                atr_profiles = current.get('ADAPTIVE_HYBRID_ATR_PROFILES', {}).get('value')
            if not atr_profiles:
                try:
                    from src.config import ADAPTIVE_HYBRID_ATR_PROFILES
                    atr_profiles = dict(ADAPTIVE_HYBRID_ATR_PROFILES)
                except ImportError:
                    atr_profiles = {}
            new_profiles = {}
            for symbol, profile in atr_profiles.items():
                new_profiles[symbol] = dict(profile)
                if 'tp' in new_profiles[symbol]:
                    new_profiles[symbol]['tp'] = round(new_profiles[symbol]['tp'] * 0.90, 2)
            if new_profiles:
                adjustments['ADAPTIVE_HYBRID_ATR_PROFILES'] = {
                    'value': new_profiles,
                    'reason': adjustments.get('ADAPTIVE_HYBRID_ATR_PROFILES', {}).get(
                        'reason', '') + ' | TP_NEVER_REACHED: reducing TP by 10%',
                }

        # --- WR_DECLINING: raise threshold by 5 ---
        if 'WR_DECLINING' in problem_types:
            new_val = current_threshold + 5
            new_val, _ = apply_guardrail(
                'ADAPTIVE_HYBRID_BASE_THRESHOLD', new_val, current_threshold,
                ADAPTIVE_HYBRID_BASE_THRESHOLD)
            adjustments['ADAPTIVE_HYBRID_BASE_THRESHOLD'] = {
                'value': round(new_val, 1),
                'reason': f'WR_DECLINING: raising threshold from {current_threshold} to {new_val}',
            }

        # --- THRESHOLD_TOO_LOW: raise threshold by 5 ---
        if 'THRESHOLD_TOO_LOW' in problem_types and 'ADAPTIVE_HYBRID_BASE_THRESHOLD' not in adjustments:
            new_val = current_threshold + 5
            new_val, _ = apply_guardrail(
                'ADAPTIVE_HYBRID_BASE_THRESHOLD', new_val, current_threshold,
                ADAPTIVE_HYBRID_BASE_THRESHOLD)
            adjustments['ADAPTIVE_HYBRID_BASE_THRESHOLD'] = {
                'value': round(new_val, 1),
                'reason': f'THRESHOLD_TOO_LOW: raising threshold from {current_threshold} to {new_val}',
            }

        # --- THRESHOLD_TOO_HIGH: lower threshold by 3 ---
        if 'THRESHOLD_TOO_HIGH' in problem_types and 'ADAPTIVE_HYBRID_BASE_THRESHOLD' not in adjustments:
            new_val = current_threshold - 3
            new_val, _ = apply_guardrail(
                'ADAPTIVE_HYBRID_BASE_THRESHOLD', new_val, current_threshold,
                ADAPTIVE_HYBRID_BASE_THRESHOLD)
            adjustments['ADAPTIVE_HYBRID_BASE_THRESHOLD'] = {
                'value': round(new_val, 1),
                'reason': f'THRESHOLD_TOO_HIGH: lowering threshold from {current_threshold} to {new_val}',
            }

        # --- BAD_HOURS: add to avoid list ---
        for p in problems:
            if p['type'] == 'BAD_HOURS':
                existing = current.get('ADAPTIVE_HYBRID_AVOID_HOURS', {}).get('value', [])
                merged = sorted(set(existing + p['hours']))
                adjustments['ADAPTIVE_HYBRID_AVOID_HOURS'] = {
                    'value': merged,
                    'reason': f'BAD_HOURS: avoiding hours {p["hours"]}',
                }

        return adjustments

    # ------------------------------------------------------------------
    # 4. LLM review
    # ------------------------------------------------------------------
    def _llm_review(self, metrics: dict, problems: list, proposed: dict) -> dict:
        """Ask LLM to review proposed adjustments. Returns approval dict."""
        try:
            from src.models.model_factory import ModelFactory
            model = ModelFactory.create_model('anthropic')
        except Exception as e:
            cprint(f"[CalibrationAgent] LLM unavailable ({e}), auto-approving", "yellow")
            return {'approved': True, 'modifications': {}, 'reasoning': 'LLM unavailable, auto-approved'}

        if model is None:
            cprint("[CalibrationAgent] No LLM model available, auto-approving", "yellow")
            return {'approved': True, 'modifications': {}, 'reasoning': 'No LLM model, auto-approved'}

        # Build concise metrics summary (exclude raw trades)
        metrics_summary = {k: v for k, v in metrics.items() if k not in ('raw_trades', 'hour_stats')}

        system_prompt = (
            "You are a quantitative trading strategy calibration reviewer. "
            "Given performance metrics, diagnosed problems, and proposed parameter adjustments, "
            "decide whether to approve or reject. Respond ONLY with valid JSON, no markdown."
        )

        user_content = json.dumps({
            'task': 'Review proposed calibration adjustments for a crypto trading bot.',
            'metrics': metrics_summary,
            'problems': problems,
            'proposed_adjustments': {k: {'value': v['value'], 'reason': v['reason']} for k, v in proposed.items()},
            'response_format': {
                'approved': 'bool -- true to apply adjustments, false to reject',
                'modifications': 'dict -- any changes to proposed values (param_name -> new_value), or empty',
                'reasoning': 'str -- brief explanation (1-2 sentences)',
            },
        }, indent=2)

        try:
            response = model.generate_response(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=0.3,
                max_tokens=1000,
            )
            # Parse JSON from response
            text = response.strip()
            # Handle potential markdown code fences
            if text.startswith('```'):
                text = text.split('\n', 1)[1] if '\n' in text else text[3:]
                if text.endswith('```'):
                    text = text[:-3]
                text = text.strip()
            result = json.loads(text)
            cprint(f"[CalibrationAgent] LLM review: approved={result.get('approved')}", "cyan")
            cprint(f"  Reasoning: {result.get('reasoning', 'N/A')}", "white")
            return result
        except Exception as e:
            cprint(f"[CalibrationAgent] LLM parse error ({e}), auto-approving", "yellow")
            return {'approved': True, 'modifications': {}, 'reasoning': f'LLM parse error: {e}'}

    # ------------------------------------------------------------------
    # 5. Apply overrides (atomic write)
    # ------------------------------------------------------------------
    def _apply_overrides(self, adjustments: dict):
        """Write adjustments to calibration_overrides.json with atomic write."""
        # Save rollback snapshot first
        self._save_rollback_snapshot()

        overrides_data = {
            'last_calibration': datetime.now().isoformat(),
            'overrides': {},
        }

        # Preserve existing overrides not being changed
        existing = self._load_current_overrides()
        for k, v in existing.items():
            overrides_data['overrides'][k] = v

        # Apply new adjustments
        for param, adj in adjustments.items():
            overrides_data['overrides'][param] = {
                'value': adj['value'],
                'reason': adj['reason'],
                'applied_at': datetime.now().isoformat(),
            }

        # Atomic write: tmp + rename
        OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(OVERRIDES_PATH) + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(overrides_data, f, indent=2, default=str)
        os.replace(tmp, str(OVERRIDES_PATH))

        cprint(f"[CalibrationAgent] Applied {len(adjustments)} override(s) to {OVERRIDES_PATH}", "green")
        for param, adj in adjustments.items():
            cprint(f"  {param} = {adj['value']} ({adj['reason']})", "white")

    def _save_rollback_snapshot(self):
        """Save current overrides as rollback target."""
        if OVERRIDES_PATH.exists():
            try:
                data = json.loads(OVERRIDES_PATH.read_text())
                self._state['rollback_snapshot'] = data
                self._state['rollback_timestamp'] = datetime.now().isoformat()
                # Also save balance at time of snapshot for DD comparison
                try:
                    from src.web.state import get_dashboard_stats
                    self._state['rollback_balance'] = get_dashboard_stats().get('balance')
                except Exception:
                    pass
                self._save_state()
            except Exception:
                pass

    def _load_current_overrides(self) -> dict:
        """Load current overrides from file."""
        if OVERRIDES_PATH.exists():
            try:
                data = json.loads(OVERRIDES_PATH.read_text())
                return data.get('overrides', {})
            except Exception:
                pass
        return {}

    # ------------------------------------------------------------------
    # 6. Rollback check
    # ------------------------------------------------------------------
    def _check_rollback(self) -> bool:
        """If drawdown increased >5% since last adjustment, rollback."""
        rollback_balance = self._state.get('rollback_balance')
        if rollback_balance is None:
            return False

        try:
            from src.web.state import get_dashboard_stats
            current_balance = get_dashboard_stats().get('balance')
        except Exception:
            return False

        if current_balance is None or rollback_balance is None:
            return False

        dd_pct = (rollback_balance - current_balance) / rollback_balance * 100
        if dd_pct > 5.0:
            cprint(f"[CalibrationAgent] ROLLBACK triggered: DD={dd_pct:.1f}% since last calibration "
                   f"(${rollback_balance:.2f} -> ${current_balance:.2f})", "red", attrs=['bold'])

            snapshot = self._state.get('rollback_snapshot')
            if snapshot:
                OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
                tmp = str(OVERRIDES_PATH) + '.tmp'
                with open(tmp, 'w') as f:
                    json.dump(snapshot, f, indent=2, default=str)
                os.replace(tmp, str(OVERRIDES_PATH))
                cprint("[CalibrationAgent] Rolled back to previous overrides", "yellow")
            else:
                # No snapshot -- remove overrides entirely
                if OVERRIDES_PATH.exists():
                    os.remove(str(OVERRIDES_PATH))
                    cprint("[CalibrationAgent] Removed all overrides (no snapshot available)", "yellow")

            # Clear rollback state
            self._state.pop('rollback_balance', None)
            self._state.pop('rollback_snapshot', None)
            self._state.pop('rollback_timestamp', None)
            self._save_state()
            return True

        return False

    # ------------------------------------------------------------------
    # 7. Main run
    # ------------------------------------------------------------------
    def run(self):
        """Orchestrate the full calibration cycle."""
        cprint("\n[CalibrationAgent] Starting calibration run...", "cyan", attrs=['bold'])

        # Step 0: Check if rollback is needed
        if self._check_rollback():
            self._state['last_run'] = datetime.now().isoformat()
            self._state['last_action'] = 'ROLLBACK'
            self._save_state()
            return

        # Step 1: Collect performance data
        metrics = self._collect_performance_data(days=14)
        trade_count = metrics.get('trades', 0)
        cprint(f"[CalibrationAgent] Collected {trade_count} trades (last 14 days)", "white")

        if trade_count < self.MIN_TRADES:
            cprint(f"[CalibrationAgent] Not enough trades ({trade_count}/{self.MIN_TRADES}), skipping", "yellow")
            self._state['last_run'] = datetime.now().isoformat()
            self._state['last_action'] = 'SKIP_INSUFFICIENT_DATA'
            self._save_state()
            return

        # Guard: require at least MIN_HISTORY_DAYS of trade history span
        try:
            raw_trades = metrics.get('raw_trades', [])
            exit_times = [pd.to_datetime(t['exit_time']) for t in raw_trades if t.get('exit_time')]
            if exit_times:
                history_span_days = (max(exit_times) - min(exit_times)).total_seconds() / 86400
                if history_span_days < self.MIN_HISTORY_DAYS:
                    cprint(f"[CalibrationAgent] History span too short ({history_span_days:.1f}d/{self.MIN_HISTORY_DAYS}d), skipping", "yellow")
                    self._state['last_run'] = datetime.now().isoformat()
                    self._state['last_action'] = 'SKIP_INSUFFICIENT_HISTORY_SPAN'
                    self._save_state()
                    return
        except Exception as e:
            cprint(f"[CalibrationAgent] Could not compute history span: {e}", "yellow")

        # Log key metrics
        cprint(f"  Win rate: {metrics.get('win_rate', 0):.1%} | PF: {metrics.get('profit_factor', 0):.2f} | "
               f"PnL: ${metrics.get('total_pnl', 0):+,.2f} | Trades/day: {metrics.get('trades_per_day', 0):.1f}",
               "white")
        cprint(f"  SL exits: {metrics.get('sl_pct', 0):.0%} | TP exits: {metrics.get('tp_count', 0)} | "
               f"Close reasons: {metrics.get('reason_counts', {})}", "white")

        # Step 2: Diagnose problems
        problems = self._diagnose_problems(metrics)
        if not problems:
            cprint("[CalibrationAgent] No problems detected, no adjustments needed", "green")
            self._state['last_run'] = datetime.now().isoformat()
            self._state['last_action'] = 'NO_PROBLEMS'
            self._state['last_metrics'] = {k: v for k, v in metrics.items() if k != 'raw_trades'}
            self._save_state()
            return

        for p in problems:
            cprint(f"  [{p['severity']}] {p['type']}: {p['detail']}", "yellow")

        # Step 3: Compute adjustments
        adjustments = self._compute_adjustments(problems, metrics)
        if not adjustments:
            cprint("[CalibrationAgent] Problems found but no actionable adjustments", "yellow")
            self._state['last_run'] = datetime.now().isoformat()
            self._state['last_action'] = 'NO_ADJUSTMENTS'
            self._save_state()
            return

        cprint(f"[CalibrationAgent] Proposed {len(adjustments)} adjustment(s):", "cyan")
        for param, adj in adjustments.items():
            cprint(f"  {param}: {adj['value']} -- {adj['reason']}", "white")

        # Step 4: LLM review
        review = self._llm_review(metrics, problems, adjustments)

        if not review.get('approved', False):
            cprint(f"[CalibrationAgent] LLM REJECTED adjustments: {review.get('reasoning', 'N/A')}", "red")
            self._state['last_run'] = datetime.now().isoformat()
            self._state['last_action'] = 'LLM_REJECTED'
            self._state['last_review'] = review
            self._save_state()
            return

        # Apply LLM modifications if any
        modifications = review.get('modifications', {})
        if modifications:
            for param, new_val in modifications.items():
                if param in adjustments:
                    old_val = adjustments[param]['value']
                    adjustments[param]['value'] = new_val
                    adjustments[param]['reason'] += f' (LLM modified: {old_val} -> {new_val})'
            cprint(f"[CalibrationAgent] LLM modified {len(modifications)} value(s)", "cyan")

        # Step 5: Apply overrides
        self._apply_overrides(adjustments)

        # Update state
        self._state['last_run'] = datetime.now().isoformat()
        self._state['last_action'] = 'APPLIED'
        self._state['last_problems'] = [p['type'] for p in problems]
        self._state['last_adjustments'] = {k: v['value'] for k, v in adjustments.items()}
        self._state['last_metrics'] = {k: v for k, v in metrics.items() if k != 'raw_trades'}
        self._state['last_review'] = review
        self._save_state()

        cprint("[CalibrationAgent] Calibration complete", "green", attrs=['bold'])


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def main():
    """Run the calibration agent standalone."""
    cprint("[CalibrationAgent] Starting standalone mode...", "cyan")
    agent = CalibrationAgent()

    while True:
        try:
            agent.run()
            # Run every 4 hours
            cprint("[CalibrationAgent] Next run in 4 hours", "white")
            time.sleep(4 * 3600)
        except KeyboardInterrupt:
            print("\n[CalibrationAgent] Shutting down...")
            break
        except Exception as e:
            cprint(f"[CalibrationAgent] Error: {e}", "red")
            import traceback
            traceback.print_exc()
            time.sleep(300)


if __name__ == '__main__':
    main()
