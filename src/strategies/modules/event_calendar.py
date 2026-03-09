"""
Event Calendar Module
Checks for upcoming high-impact macro events and reduces trading activity near them.
"""

import json
import os
from datetime import datetime
from typing import Optional, Dict

# Cache to avoid re-reading file every cycle
_calendar_cache = None
_cache_timestamp = 0
_CACHE_TTL = 3600  # Re-read file every hour


def _load_calendar() -> list:
    """Load event calendar from JSON file."""
    global _calendar_cache, _cache_timestamp
    import time
    now = time.time()

    if _calendar_cache is not None and now - _cache_timestamp < _CACHE_TTL:
        return _calendar_cache

    try:
        from src.config import ADAPTIVE_HYBRID_EVENT_CALENDAR_FILE
        calendar_path = ADAPTIVE_HYBRID_EVENT_CALENDAR_FILE
    except ImportError:
        calendar_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'event_calendar.json')

    if not os.path.exists(calendar_path):
        _calendar_cache = []
        _cache_timestamp = now
        return []

    try:
        with open(calendar_path, 'r') as f:
            data = json.load(f)
        _calendar_cache = data.get('events', [])
        _cache_timestamp = now
        return _calendar_cache
    except Exception:
        _calendar_cache = []
        _cache_timestamp = now
        return []


def check_upcoming_events(window_hours: float = None) -> Optional[Dict]:
    """
    Check if there's a high-impact event within the specified window.

    Returns:
        Dict with event info if an event is near, None otherwise.
        {'event': description, 'type': event_type, 'hours_until': float, 'size_reduction': float}
    """
    events = _load_calendar()
    if not events:
        return None

    try:
        from src.config import ADAPTIVE_HYBRID_EVENT_WINDOW_HOURS, ADAPTIVE_HYBRID_EVENT_SIZE_REDUCTION
    except ImportError:
        ADAPTIVE_HYBRID_EVENT_WINDOW_HOURS = 2.0
        ADAPTIVE_HYBRID_EVENT_SIZE_REDUCTION = 0.50

    if window_hours is None:
        window_hours = ADAPTIVE_HYBRID_EVENT_WINDOW_HOURS
    now = datetime.utcnow()

    for event in events:
        try:
            event_dt = datetime.strptime(f"{event['date']} {event.get('time', '12:00')}", "%Y-%m-%d %H:%M")
            hours_until = (event_dt - now).total_seconds() / 3600

            # Check if within window (before or after the event)
            if -window_hours <= hours_until <= window_hours:
                return {
                    'event': event.get('description', event.get('type', 'Unknown')),
                    'type': event.get('type', 'UNKNOWN'),
                    'hours_until': round(hours_until, 1),
                    'size_reduction': ADAPTIVE_HYBRID_EVENT_SIZE_REDUCTION,
                    'impact': event.get('impact', 'high'),
                }
        except (ValueError, KeyError):
            continue

    return None


def get_next_event() -> Optional[Dict]:
    """Get the next upcoming event (for dashboard display)."""
    events = _load_calendar()
    now = datetime.utcnow()

    for event in sorted(events, key=lambda e: e.get('date', '')):
        try:
            event_dt = datetime.strptime(f"{event['date']} {event.get('time', '12:00')}", "%Y-%m-%d %H:%M")
            if event_dt > now:
                hours_until = (event_dt - now).total_seconds() / 3600
                return {
                    'event': event.get('description', ''),
                    'type': event.get('type', ''),
                    'date': event['date'],
                    'time': event.get('time', ''),
                    'hours_until': round(hours_until, 1),
                }
        except (ValueError, KeyError):
            continue

    return None
