"""Structured logging for the trading bot."""
import logging
import json
import os
from datetime import datetime


def setup_logger(name, level=None):
    """Create a structured logger with both console and file output."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        log_level = level or os.getenv('LOG_LEVEL', 'INFO')
        logger.setLevel(getattr(logging, log_level))

        # Console handler with colored output (preserving the cprint feel)
        console = logging.StreamHandler()
        console.setFormatter(ColoredFormatter())
        logger.addHandler(console)

        # File handler with JSON structured logs
        os.makedirs('src/data/logs', exist_ok=True)
        file_handler = logging.FileHandler(
            f'src/data/logs/trading_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger


class ColoredFormatter(logging.Formatter):
    """Console formatter that mimics cprint colors."""
    COLORS = {
        'DEBUG': '\033[36m',    # cyan
        'INFO': '\033[32m',     # green
        'WARNING': '\033[33m',  # yellow
        'ERROR': '\033[31m',    # red
        'CRITICAL': '\033[35m', # magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        return f"{color}[{record.name}] {record.getMessage()}{self.RESET}"


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured log analysis."""
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
        }
        if hasattr(record, 'signal'):
            log_entry['signal'] = record.signal
        if hasattr(record, 'trade'):
            log_entry['trade'] = record.trade
        if hasattr(record, 'metrics'):
            log_entry['metrics'] = record.metrics
        return json.dumps(log_entry)


def log_signal(logger, symbol, direction, confidence, source, **extra):
    """Log a trading signal with structured data."""
    logger.info(
        f"Signal: {direction} {symbol} (confidence={confidence}%, source={source})",
        extra={'signal': {'symbol': symbol, 'direction': direction,
                         'confidence': confidence, 'source': source, **extra}}
    )


def log_trade(logger, symbol, direction, size, price, reason, **extra):
    """Log a trade execution with structured data."""
    logger.info(
        f"Trade: {direction} {symbol} size=${size:.2f} @ {price}",
        extra={'trade': {'symbol': symbol, 'direction': direction,
                        'size': size, 'price': price, 'reason': reason, **extra}}
    )
