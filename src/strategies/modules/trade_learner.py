"""Post-Trade Learning module.

After each trade closes, analyzes what happened using LLM,
generates actionable lessons, and stores them in TradeMemory.
Builds context prompts for future trades using recent lessons.
"""

import json
import time
from datetime import datetime
from termcolor import cprint


SYSTEM_PROMPT = (
    "You are a professional trading coach analyzing completed trades. "
    "Extract actionable lessons from each trade outcome. "
    "You always respond with valid JSON."
)

ANALYSIS_PROMPT_TEMPLATE = """Analyze this completed trade:

## Trade Details
- Symbol: {symbol}
- Direction: {direction}
- Entry: ${entry_price:,.2f} -> Exit: ${exit_price:,.2f}
- PnL: ${pnl:+,.2f} ({pnl_pct:+.2f}%)
- Hold Duration: {hold_hours:.1f} hours
- Close Reason: {close_reason}
- Entry Score: {score}
- Confidence: {confidence}%

## Market Context at Entry
- Regime: {regime}
- Modules Firing: {modules}

## Recent Performance
- Win Rate (30d): {win_rate}
- Total Trades (30d): {total_trades}

What worked and what didn't? Focus on ACTIONABLE improvements.

Respond ONLY with valid JSON:
{{
  "outcome": "WIN" | "LOSS",
  "lesson": "One-sentence actionable lesson (max 100 chars)",
  "pattern": "category_key (e.g. 'BUY_markup_high_score')",
  "what_worked": "brief positive takeaway or 'nothing' if loss",
  "what_failed": "brief failure analysis or 'nothing' if win",
  "suggested_adjustment": "one specific parameter/rule change or 'none'"
}}
"""


def _parse_lesson_response(response_text):
    """Extract JSON lesson from LLM response."""
    text = response_text.strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    result = json.loads(text)

    return {
        'outcome': result.get('outcome', 'LOSS').upper(),
        'lesson': result.get('lesson', '')[:200],
        'pattern': result.get('pattern', 'unknown'),
        'what_worked': result.get('what_worked', ''),
        'what_failed': result.get('what_failed', ''),
        'suggested_adjustment': result.get('suggested_adjustment', 'none'),
    }


def analyze_closed_trade(
    trade: dict,
    trade_memory=None,
    model=None,
    bypass: bool = False,
) -> dict:
    """Analyze a closed trade and generate lessons.

    Args:
        trade: Closed trade dict with entry/exit data, PnL, etc.
        trade_memory: TradeMemory instance for storing lessons and querying history.
        model: LLM model instance. If None, uses rule-based analysis.
        bypass: If True, skip LLM call (for backtesting).

    Returns:
        dict with lesson analysis or None if analysis skipped.
    """
    symbol = trade.get('symbol', '')
    direction = trade.get('direction', '')
    pnl = trade.get('pnl', 0)
    entry_price = trade.get('entry_price', 0)
    exit_price = trade.get('close_price', trade.get('exit_price', 0))
    close_reason = trade.get('close_reason', 'UNKNOWN')

    if not symbol or not direction:
        return None

    # Rule-based analysis (bypass mode or no model)
    if bypass or model is None:
        result = _rule_based_analysis(trade)
        if trade_memory:
            _store_lesson(trade_memory, trade, result)
        return result

    # Calculate derived fields
    pnl_pct = trade.get('pnl_pct', 0)
    if pnl_pct == 0 and entry_price > 0:
        pnl_pct = ((exit_price - entry_price) / entry_price * 100) if direction == 'BUY' else (
            (entry_price - exit_price) / entry_price * 100
        )

    hold_hours = 0
    entry_time = trade.get('entry_time')
    exit_time = trade.get('exit_time')
    if entry_time and exit_time:
        try:
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            if isinstance(exit_time, str):
                exit_time = datetime.fromisoformat(exit_time)
            hold_hours = (exit_time - entry_time).total_seconds() / 3600
        except (ValueError, TypeError):
            pass

    # Get trade memory context
    win_rate = ""
    total_trades = ""
    if trade_memory:
        wr = trade_memory.get_win_rate(symbol, days=30)
        if wr is not None:
            win_rate = f"{wr:.1f}%"
        perf = trade_memory.get_performance_summary(days=30)
        total_trades = str(perf.get('total_trades', 0))

    if not win_rate:
        win_rate = "N/A"
    if not total_trades:
        total_trades = "N/A"

    user_content = ANALYSIS_PROMPT_TEMPLATE.format(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        pnl_pct=pnl_pct,
        hold_hours=hold_hours,
        close_reason=close_reason,
        score=trade.get('score', 'N/A'),
        confidence=trade.get('confidence', 'N/A'),
        regime=trade.get('market_regime', 'unknown'),
        modules=trade.get('modules', 'N/A'),
        win_rate=win_rate,
        total_trades=total_trades,
    )

    try:
        start_time = time.time()
        response = model.generate_response(
            system_prompt=SYSTEM_PROMPT,
            user_content=user_content,
            temperature=0.3,
            max_tokens=384,
        )
        latency_ms = int((time.time() - start_time) * 1000)

        try:
            from src.models.model_factory import ModelFactory
            ModelFactory.log_call(getattr(model, 'model_type', 'unknown'), True, latency_ms)
        except Exception:
            pass

        if response is None:
            result = _rule_based_analysis(trade)
            if trade_memory:
                _store_lesson(trade_memory, trade, result)
            return result

        response_text = response.content if hasattr(response, 'content') else str(response)
        result = _parse_lesson_response(response_text)

        # Store lesson in trade memory
        if trade_memory:
            _store_lesson(trade_memory, trade, result)

        color = 'green' if result['outcome'] == 'WIN' else 'red'
        cprint(f"  [Trade Learner] {symbol} {direction}: {result['outcome']} - {result['lesson']}", color)

        return result

    except Exception as e:
        cprint(f"  [Trade Learner] Error analyzing {symbol}: {e}", "yellow")
        result = _rule_based_analysis(trade)
        if trade_memory:
            _store_lesson(trade_memory, trade, result)
        return result


def _rule_based_analysis(trade: dict) -> dict:
    """Simple rule-based post-trade analysis (no LLM needed)."""
    pnl = trade.get('pnl', 0)
    close_reason = trade.get('close_reason', 'UNKNOWN')
    direction = trade.get('direction', '')
    score = trade.get('score', 0)

    outcome = 'WIN' if pnl > 0 else 'LOSS'

    if outcome == 'WIN':
        if close_reason == 'TAKE_PROFIT':
            lesson = f"TP hit on {direction} - scoring system correctly identified entry"
            what_worked = f"Score {score} signal was accurate"
            what_failed = "nothing"
        elif close_reason == 'TRAILING_STOP':
            lesson = f"Trailing stop captured profit - good trend continuation"
            what_worked = "Trailing stop let profits run"
            what_failed = "nothing"
        else:
            lesson = f"Profitable close ({close_reason})"
            what_worked = "Entry timing was good"
            what_failed = "nothing"
    else:
        if close_reason == 'STOP_LOSS':
            lesson = f"SL hit - entry timing or direction was wrong"
            what_worked = "nothing"
            what_failed = f"Entry signal (score {score}) was premature or wrong direction"
        elif close_reason in ('TIME_EXIT_24H', 'TIME_EXIT_48H'):
            lesson = f"Time exit without reaching TP - weak momentum after entry"
            what_worked = "nothing"
            what_failed = "Signal lacked follow-through"
        else:
            lesson = f"Loss on {close_reason}"
            what_worked = "nothing"
            what_failed = "Trade did not work as expected"

    pattern = f"{direction}_{trade.get('market_regime', 'unknown')}_{close_reason.lower()}"

    return {
        'outcome': outcome,
        'lesson': lesson,
        'pattern': pattern,
        'what_worked': what_worked,
        'what_failed': what_failed,
        'suggested_adjustment': 'none',
    }


def _store_lesson(trade_memory, trade: dict, analysis: dict):
    """Store a lesson in TradeMemory."""
    try:
        pattern = analysis.get('pattern', 'unknown')
        # Use log_lesson if available, else use _update_lessons indirectly
        # The lessons table uses pattern as unique key
        import sqlite3
        with sqlite3.connect(trade_memory.db_path) as conn:
            # Check existing lesson
            existing = conn.execute(
                'SELECT sample_size, notes FROM lessons WHERE pattern = ?',
                (pattern,)
            ).fetchone()

            pnl = trade.get('pnl', 0)
            lesson_text = analysis.get('lesson', '')
            suggested = analysis.get('suggested_adjustment', 'none')
            notes = f"{lesson_text} | Suggested: {suggested}"

            if existing:
                old_size = existing[0] or 0
                new_size = old_size + 1
                # Append new lesson note (keep last 3 notes)
                old_notes = existing[1] or ''
                note_parts = old_notes.split(' || ')[-2:]  # Keep last 2
                note_parts.append(notes)
                combined_notes = ' || '.join(note_parts)

                conn.execute('''
                    UPDATE lessons SET
                        sample_size = ?,
                        last_updated = ?,
                        notes = ?
                    WHERE pattern = ?
                ''', (new_size, datetime.now().isoformat(), combined_notes, pattern))
            else:
                conn.execute('''
                    INSERT INTO lessons (pattern, success_rate, sample_size, avg_pnl, last_updated, notes)
                    VALUES (?, ?, 1, ?, ?, ?)
                ''', (
                    pattern,
                    100.0 if pnl > 0 else 0.0,
                    pnl,
                    datetime.now().isoformat(),
                    notes,
                ))

    except Exception as e:
        cprint(f"  [Trade Learner] Warning: could not store lesson: {e}", "yellow")


def build_lessons_context(trade_memory, symbol: str, limit: int = 5) -> str:
    """Build a context string from recent lessons for use in LLM prompts.

    Args:
        trade_memory: TradeMemory instance
        symbol: Trading symbol
        limit: Max number of lessons to include

    Returns:
        Formatted string with recent lessons
    """
    if trade_memory is None:
        return ""

    try:
        import sqlite3
        with sqlite3.connect(trade_memory.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT pattern, success_rate, sample_size, avg_pnl, notes
                FROM lessons
                WHERE pattern LIKE ? OR pattern LIKE ?
                ORDER BY last_updated DESC
                LIMIT ?
            ''', (f'BUY_%', f'SELL_%', limit)).fetchall()

            if not rows:
                return ""

            lines = ["Recent trading lessons:"]
            for row in rows:
                sr = row['success_rate'] or 0
                n = row['sample_size'] or 0
                avg = row['avg_pnl'] or 0
                notes = row['notes'] or ''
                # Truncate notes
                if len(notes) > 100:
                    notes = notes[:100] + "..."
                lines.append(
                    f"  - {row['pattern']}: {sr:.0f}% WR ({n} trades, avg ${avg:.2f}) - {notes}"
                )

            return "\n".join(lines)

    except Exception:
        return ""
