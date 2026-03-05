"""Quantitative feedback loop for trade learning."""
import json
import os
import threading
from collections import deque
from datetime import datetime
from termcolor import cprint

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'adaptive_weights')


class QuantitativeFeedback:
    """Computes actionable metrics from closed trades to feed back into strategy."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.recent_trades = deque(maxlen=window_size)
        self._lock = threading.RLock()
        self._load_state()

    def record_trade(self, trade: dict):
        """Record a closed trade for analysis.

        Args:
            trade: dict with pnl, module_scores, market_regime, symbol, direction, etc.
        """
        with self._lock:
            self.recent_trades.append({
                'pnl': trade.get('pnl', 0),
                'module_scores': trade.get('module_scores', {}),
                'market_regime': trade.get('market_regime', 'unknown'),
                'symbol': trade.get('symbol', ''),
                'direction': trade.get('direction', ''),
                'timestamp': datetime.now().isoformat(),
            })
        self._save_state()

    def compute_module_attribution(self) -> dict:
        """Compute each module's contribution score across recent trades.

        Returns dict of module_name -> attribution_score (positive = good, negative = bad)
        """
        with self._lock:
            trades = list(self.recent_trades)

        attribution = {}
        for trade in trades:
            pnl = trade.get('pnl', 0)
            sign = 1 if pnl > 0 else -1
            modules = trade.get('module_scores', {})
            for module, score in modules.items():
                if isinstance(score, (int, float)) and score > 0:
                    if module not in attribution:
                        attribution[module] = 0
                    attribution[module] += score * sign

        return attribution

    def compute_regime_performance(self) -> dict:
        """Win rate and PF by market regime."""
        with self._lock:
            trades = list(self.recent_trades)

        regime_stats = {}
        for t in trades:
            regime = t.get('market_regime', 'unknown')
            if regime not in regime_stats:
                regime_stats[regime] = {'wins': 0, 'losses': 0, 'pnl': 0.0}
            if t.get('pnl', 0) > 0:
                regime_stats[regime]['wins'] += 1
            else:
                regime_stats[regime]['losses'] += 1
            regime_stats[regime]['pnl'] += t.get('pnl', 0)

        # Add win rate and profit factor
        for regime, stats in regime_stats.items():
            total = stats['wins'] + stats['losses']
            stats['win_rate'] = stats['wins'] / total if total > 0 else 0
            gross_profit = sum(t['pnl'] for t in trades if t.get('market_regime') == regime and t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in trades if t.get('market_regime') == regime and t['pnl'] < 0))
            stats['profit_factor'] = gross_profit / max(gross_loss, 0.01)

        return regime_stats

    def suggest_threshold_adjustment(self, current_threshold: float) -> float:
        """Suggest threshold adjustment based on recent performance.

        If PF < 1.0 on last 20 trades -> raise threshold 5%
        If PF > 1.8 -> lower threshold 3% (take more trades)
        """
        with self._lock:
            trades = list(self.recent_trades)

        if len(trades) < 10:
            return current_threshold

        losses = sum(1 for t in trades if t.get('pnl', 0) < 0)
        if losses < 3:
            return current_threshold

        gross_profit = sum(t['pnl'] for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t.get('pnl', 0) < 0))
        pf = gross_profit / max(gross_loss, 0.01)

        if pf < 1.0:
            new_threshold = min(current_threshold * 1.05, 65)
            cprint(f"[Feedback] PF={pf:.2f} < 1.0, raising threshold {current_threshold:.0f} -> {new_threshold:.0f}", "yellow")
            return new_threshold
        elif pf > 1.8:
            new_threshold = max(current_threshold * 0.97, 40)
            cprint(f"[Feedback] PF={pf:.2f} > 1.8, lowering threshold {current_threshold:.0f} -> {new_threshold:.0f}", "green")
            return new_threshold

        return current_threshold

    def get_summary(self) -> dict:
        """Get summary of recent performance."""
        with self._lock:
            trades = list(self.recent_trades)

        if not trades:
            return {'trades': 0}

        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        losses = len(trades) - wins
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        avg_win = sum(t['pnl'] for t in trades if t['pnl'] > 0) / max(wins, 1)
        avg_loss = sum(t['pnl'] for t in trades if t['pnl'] < 0) / max(losses, 1)

        return {
            'trades': len(trades),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(trades),
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': sum(t['pnl'] for t in trades if t['pnl'] > 0) / max(abs(sum(t['pnl'] for t in trades if t['pnl'] < 0)), 0.01),
        }

    def _save_state(self):
        """Persist to disk."""
        os.makedirs(DATA_DIR, exist_ok=True)
        filepath = os.path.join(DATA_DIR, 'feedback_state.json')
        with self._lock:
            data = list(self.recent_trades)
        tmp = filepath + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, filepath)

    def _load_state(self):
        """Load persisted state."""
        filepath = os.path.join(DATA_DIR, 'feedback_state.json')
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            with self._lock:
                for t in data:
                    self.recent_trades.append(t)
            cprint(f"[Feedback] Restored {len(data)} trade records", "cyan")
        except Exception as e:
            cprint(f"[Feedback] Could not load state: {e}", "yellow")
