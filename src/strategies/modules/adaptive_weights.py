"""Bayesian adaptive weight optimizer using Thompson Sampling."""
import numpy as np
import json
import os
import threading
from datetime import datetime
from termcolor import cprint

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'adaptive_weights')


class BayesianWeightOptimizer:
    """Thompson Sampling for module weight optimization.

    Each module has a Beta(alpha, beta) distribution.
    - alpha increments on winning trades where the module fired
    - beta increments on losing trades where the module fired
    - Decay factor ensures recency bias
    """

    def __init__(self, module_names: list, prior_alpha: float = 2.0, prior_beta: float = 2.0,
                 decay: float = 0.95, min_trades: int = 15):
        self.module_names = list(module_names)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.decay = decay
        self.min_trades = min_trades  # Min trades before activating adaptive weights
        self.trade_count = 0
        self._lock = threading.Lock()

        self.alphas = {m: prior_alpha for m in module_names}
        self.betas = {m: prior_beta for m in module_names}

        # Load persisted state
        self._load_state()

    def get_weights(self, static_weights: dict) -> dict:
        """Get adaptive weights. Falls back to static_weights if not enough trades.

        Args:
            static_weights: The original static weight dict from config

        Returns:
            dict of module_name -> weight (sums to 1.0)
        """
        with self._lock:
            if self.trade_count < self.min_trades:
                return static_weights

            # Use Beta distribution mean (deterministic) for stable weight production
            samples = {}
            for m in self.module_names:
                if m in self.alphas:
                    samples[m] = self.alphas[m] / (self.alphas[m] + self.betas[m])
                else:
                    samples[m] = 0.5  # neutral prior for unknown modules

            # Normalize to sum to 1.0
            total = sum(samples.values())
            if total <= 0:
                return static_weights

            weights = {m: s / total for m, s in samples.items()}

            # Blend with static weights (50/50) for stability
            blended = {}
            for m in static_weights:
                adaptive_w = weights.get(m, 0)
                static_w = static_weights.get(m, 0)
                blended[m] = 0.5 * adaptive_w + 0.5 * static_w

            # Re-normalize
            total_blended = sum(blended.values())
            if total_blended > 0:
                blended = {m: w / total_blended for m, w in blended.items()}

            return blended

    def update(self, trade_result: dict):
        """Update module priors after a trade closes.

        Args:
            trade_result: dict with keys:
                - 'pnl': float (positive = win, negative = loss)
                - 'module_scores': dict of {module_name: score} for modules that fired
        """
        pnl = trade_result.get('pnl', 0)
        module_scores = trade_result.get('module_scores', {})
        is_win = pnl > 0

        # Identify modules that actively contributed (score > 0)
        active_modules = [m for m, s in module_scores.items() if isinstance(s, (int, float)) and s > 0]

        if not active_modules:
            return

        with self._lock:
            # Decay only active modules (preserve learned signal for inactive ones)
            for m in active_modules:
                if m in self.alphas:
                    self.alphas[m] = max(self.alphas[m] * self.decay, 0.5)
                    self.betas[m] = max(self.betas[m] * self.decay, 0.5)

            # Update modules that fired
            for m in active_modules:
                if m not in self.alphas:
                    self.alphas[m] = self.prior_alpha
                    self.betas[m] = self.prior_beta

                if is_win:
                    self.alphas[m] += 1.0
                else:
                    self.betas[m] += 1.0

            self.trade_count += 1

        self._save_state()

        # Log top/bottom modules
        if self.trade_count % 5 == 0:
            self._log_module_rankings()

    def _log_module_rankings(self):
        """Log current module rankings for diagnostics."""
        with self._lock:
            rankings = {}
            for m in self.module_names:
                a = self.alphas.get(m, self.prior_alpha)
                b = self.betas.get(m, self.prior_beta)
                rankings[m] = a / (a + b)  # Expected value of Beta distribution

        sorted_modules = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_modules[:3]
        bottom3 = sorted_modules[-3:]

        cprint(f"[AdaptiveWeights] Top: {', '.join(f'{m}={v:.2f}' for m, v in top3)} | "
               f"Bottom: {', '.join(f'{m}={v:.2f}' for m, v in bottom3)} | "
               f"Trades: {self.trade_count}", "cyan")

    def get_module_stats(self) -> dict:
        """Get current module performance estimates."""
        with self._lock:
            stats = {}
            for m in self.module_names:
                a = self.alphas.get(m, self.prior_alpha)
                b = self.betas.get(m, self.prior_beta)
                stats[m] = {
                    'expected_value': a / (a + b),
                    'alpha': a,
                    'beta': b,
                    'confidence': (a + b - 2 * self.prior_alpha) / max(self.trade_count, 1),
                }
            return stats

    def _save_state(self):
        """Persist optimizer state to disk."""
        os.makedirs(DATA_DIR, exist_ok=True)
        state = {
            'alphas': self.alphas,
            'betas': self.betas,
            'trade_count': self.trade_count,
            'timestamp': datetime.now().isoformat(),
        }
        filepath = os.path.join(DATA_DIR, 'bayesian_state.json')
        tmp = filepath + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, filepath)

    def _load_state(self):
        """Load persisted state."""
        filepath = os.path.join(DATA_DIR, 'bayesian_state.json')
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            self.alphas = {m: state['alphas'].get(m, self.prior_alpha) for m in self.module_names}
            self.betas = {m: state['betas'].get(m, self.prior_beta) for m in self.module_names}
            self.trade_count = state.get('trade_count', 0)
            cprint(f"[AdaptiveWeights] Restored state: {self.trade_count} trades", "cyan")
        except Exception as e:
            cprint(f"[AdaptiveWeights] Could not load state: {e}", "yellow")
