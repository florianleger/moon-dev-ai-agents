"""Trade Memory System - SQLite-based persistent memory for trading decisions.

Allows agents to:
1. Log every trading decision with context
2. Record outcomes when trades close
3. Retrieve similar historical situations
4. Generate context prompts enriched with past experience
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta


class TradeMemory:
    """Persistent memory for trading decisions and outcomes."""

    DB_PATH = os.path.join(os.path.dirname(__file__), 'trade_memory.db')
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, db_path=None):
        self.db_path = db_path or self.DB_PATH
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL,
                    source TEXT,
                    reasoning TEXT,
                    market_regime TEXT,
                    key_indicators TEXT,
                    modules_firing TEXT,
                    outcome_pnl REAL,
                    outcome_correct INTEGER,
                    hold_duration_hours REAL,
                    max_adverse_excursion REAL,
                    max_favorable_excursion REAL,
                    close_reason TEXT
                );

                CREATE TABLE IF NOT EXISTS market_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL,
                    rsi REAL,
                    adx REAL,
                    atr REAL,
                    volume_ratio REAL,
                    funding_rate REAL,
                    fear_greed INTEGER,
                    btc_correlation REAL
                );

                CREATE TABLE IF NOT EXISTS lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT NOT NULL UNIQUE,
                    success_rate REAL,
                    sample_size INTEGER DEFAULT 0,
                    avg_pnl REAL,
                    last_updated TEXT,
                    notes TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol);
                CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_decisions_outcome ON decisions(outcome_correct);
            ''')

    def log_decision(self, symbol, direction, confidence, source,
                     reasoning=None, market_regime=None, key_indicators=None,
                     modules_firing=None):
        """Log a trading decision. Returns the decision ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO decisions
                (timestamp, symbol, direction, confidence, source, reasoning,
                 market_regime, key_indicators, modules_firing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                symbol, direction, confidence, source, reasoning,
                market_regime,
                json.dumps(key_indicators) if key_indicators else None,
                json.dumps(modules_firing) if modules_firing else None
            ))
            return conn.execute('SELECT last_insert_rowid()').fetchone()[0]

    def update_outcome(self, decision_id, pnl, hold_duration_hours=None,
                       max_adverse=None, max_favorable=None, close_reason=None):
        """Update a decision with its outcome."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE decisions SET
                    outcome_pnl = ?, outcome_correct = ?,
                    hold_duration_hours = ?,
                    max_adverse_excursion = ?, max_favorable_excursion = ?,
                    close_reason = ?
                WHERE id = ?
            ''', (pnl, 1 if pnl > 0 else 0, hold_duration_hours,
                  max_adverse, max_favorable, close_reason, decision_id))

        self._update_lessons(decision_id)

    def get_win_rate(self, symbol=None, source=None, days=30):
        """Get win rate for symbol/source over period."""
        query = '''
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN outcome_correct = 1 THEN 1 ELSE 0 END) as wins
            FROM decisions
            WHERE outcome_correct IS NOT NULL
              AND timestamp > ?
        '''
        params = [(datetime.utcnow() - timedelta(days=days)).isoformat()]

        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        if source:
            query += ' AND source = ?'
            params.append(source)

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(query, params).fetchone()
            total, wins = row[0], row[1] or 0
            return (wins / total * 100) if total > 0 else None

    def get_recent_decisions(self, symbol=None, limit=10):
        """Get recent decisions with outcomes."""
        query = '''
            SELECT symbol, direction, confidence, source,
                   outcome_pnl, outcome_correct, reasoning, market_regime,
                   timestamp
            FROM decisions
            WHERE outcome_correct IS NOT NULL
        '''
        params = []
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def get_common_mistakes(self, days=30, min_occurrences=3):
        """Identify recurring losing patterns."""
        query = '''
            SELECT market_regime, direction,
                   COUNT(*) as count,
                   AVG(outcome_pnl) as avg_loss,
                   GROUP_CONCAT(symbol) as symbols
            FROM decisions
            WHERE outcome_correct = 0
              AND timestamp > ?
              AND market_regime IS NOT NULL
            GROUP BY market_regime, direction
            HAVING COUNT(*) >= ?
            ORDER BY avg_loss ASC
        '''
        params = [(datetime.utcnow() - timedelta(days=days)).isoformat(), min_occurrences]

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def build_context_prompt(self, symbol, direction=None):
        """Generate a context-enriched prompt section from historical data."""
        parts = []

        # Win rate
        win_rate = self.get_win_rate(symbol, days=30)
        if win_rate is not None:
            parts.append(f"Historical win rate on {symbol} (30d): {win_rate:.1f}%")

        # Recent decisions on this symbol
        recent = self.get_recent_decisions(symbol, limit=5)
        if recent:
            parts.append(f"Last {len(recent)} trades on {symbol}:")
            for d in recent:
                outcome = "WIN" if d['outcome_correct'] else "LOSS"
                parts.append(
                    f"  - {d['direction']} {outcome} (${d['outcome_pnl']:.2f}) "
                    f"in {d['market_regime'] or 'unknown'} regime"
                )

        # Common mistakes
        mistakes = self.get_common_mistakes()
        if mistakes:
            parts.append("Common losing patterns to AVOID:")
            for m in mistakes[:3]:
                parts.append(
                    f"  - {m['direction']} in {m['market_regime']}: "
                    f"{m['count']} losses, avg ${m['avg_loss']:.2f}"
                )

        return "\n".join(parts) if parts else ""

    def log_market_snapshot(self, symbol, price, rsi=None, adx=None, atr=None,
                           volume_ratio=None, funding_rate=None, fear_greed=None,
                           btc_correlation=None):
        """Log a market state snapshot for future analysis."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO market_snapshots
                (timestamp, symbol, price, rsi, adx, atr, volume_ratio,
                 funding_rate, fear_greed, btc_correlation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.utcnow().isoformat(), symbol, price, rsi, adx, atr,
                  volume_ratio, funding_rate, fear_greed, btc_correlation))

    def _update_lessons(self, decision_id):
        """Update lessons table based on a completed trade."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            decision = conn.execute(
                'SELECT * FROM decisions WHERE id = ?', (decision_id,)
            ).fetchone()

            if not decision or decision['market_regime'] is None:
                return

            pattern = f"{decision['direction']}_{decision['market_regime']}"

            stats = conn.execute('''
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN outcome_correct = 1 THEN 1 ELSE 0 END) as wins,
                       AVG(outcome_pnl) as avg_pnl
                FROM decisions
                WHERE market_regime = ? AND direction = ? AND outcome_correct IS NOT NULL
            ''', (decision['market_regime'], decision['direction'])).fetchone()

            success_rate = (stats['wins'] / stats['total'] * 100) if stats['total'] > 0 else 0

            conn.execute('''
                INSERT INTO lessons (pattern, success_rate, sample_size, avg_pnl, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(pattern) DO UPDATE SET
                    success_rate = excluded.success_rate,
                    sample_size = excluded.sample_size,
                    avg_pnl = excluded.avg_pnl,
                    last_updated = excluded.last_updated
            ''', (pattern, success_rate, stats['total'], stats['avg_pnl'],
                  datetime.utcnow().isoformat()))

    def get_performance_summary(self, days=7):
        """Get overall performance summary."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            summary = conn.execute('''
                SELECT COUNT(*) as total_trades,
                       SUM(CASE WHEN outcome_correct = 1 THEN 1 ELSE 0 END) as wins,
                       SUM(outcome_pnl) as total_pnl,
                       AVG(outcome_pnl) as avg_pnl,
                       MAX(outcome_pnl) as best_trade,
                       MIN(outcome_pnl) as worst_trade,
                       AVG(hold_duration_hours) as avg_hold_hours
                FROM decisions
                WHERE outcome_correct IS NOT NULL AND timestamp > ?
            ''', (cutoff,)).fetchone()

            return dict(summary) if summary else {}

    def cleanup(self, days=90):
        """Remove old data beyond retention period."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM decisions WHERE timestamp < ?', (cutoff,))
            conn.execute('DELETE FROM market_snapshots WHERE timestamp < ?', (cutoff,))
