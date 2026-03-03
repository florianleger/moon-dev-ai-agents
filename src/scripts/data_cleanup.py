"""
Data rotation script - clean old files from src/data/

Removes files older than MAX_AGE_DAYS from data directories to prevent
unbounded disk growth in Docker containers.

Called at startup from entrypoint.sh and can be run standalone.
"""

import os
import time
from pathlib import Path

MAX_AGE_DAYS = 30
DATA_DIRS = [
    'src/data/ramf',
    'src/data/execution_results',
    'src/data/signals',
]

# Files to never delete (state files, configs)
PROTECTED_FILES = {
    'web_state.json',
    'bot_heartbeat.json',
    '.gitkeep',
}


def cleanup_old_files(base_path: str = '/app'):
    """
    Remove files older than MAX_AGE_DAYS from data directories.

    Args:
        base_path: Base path for the application (default: /app for Docker)
    """
    cutoff = time.time() - (MAX_AGE_DAYS * 86400)
    removed = 0
    freed_bytes = 0

    for dir_path in DATA_DIRS:
        full_path = Path(base_path) / dir_path
        if not full_path.exists():
            continue

        for f in full_path.rglob('*'):
            if not f.is_file():
                continue
            if f.name in PROTECTED_FILES:
                continue
            try:
                stat = f.stat()
                if stat.st_mtime < cutoff:
                    freed_bytes += stat.st_size
                    f.unlink()
                    removed += 1
            except OSError:
                pass

    freed_mb = freed_bytes / (1024 * 1024)
    print(f"[DataCleanup] Removed {removed} files older than {MAX_AGE_DAYS} days ({freed_mb:.1f} MB freed)")


def rotate_signals_csv(base_path: str = '/app', max_lines: int = 10000, keep_lines: int = 5000):
    """
    Rotate signals.csv if it exceeds max_lines.
    Keeps the most recent keep_lines entries.

    Args:
        base_path: Base path for the application
        max_lines: Trigger rotation when file exceeds this many lines
        keep_lines: Number of recent lines to keep after rotation
    """
    csv_path = Path(base_path) / 'src/data/ramf/signals.csv'
    if not csv_path.exists():
        return

    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        if len(lines) <= max_lines:
            return

        # Keep header + most recent entries
        header = lines[0] if lines else ''
        recent = lines[-(keep_lines):]

        # Ensure header is preserved
        if recent and recent[0] != header:
            recent.insert(0, header)

        with open(csv_path, 'w') as f:
            f.writelines(recent)

        trimmed = len(lines) - len(recent)
        print(f"[DataCleanup] Rotated signals.csv: removed {trimmed} old entries, kept {len(recent)}")
    except Exception as e:
        print(f"[DataCleanup] Error rotating signals.csv: {e}")


if __name__ == '__main__':
    # Determine base path: /app in Docker, project root locally
    if Path('/app/src').exists():
        base = '/app'
    else:
        base = str(Path(__file__).parent.parent.parent)

    cleanup_old_files(base)
    rotate_signals_csv(base)
