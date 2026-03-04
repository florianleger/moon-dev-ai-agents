"""Alert system for critical trading events."""
import os
import requests
import json
from datetime import datetime


class AlertManager:
    """Sends alerts via Discord/Telegram webhooks for critical events."""

    def __init__(self):
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self._enabled = bool(self.discord_webhook or self.telegram_token)

    @property
    def is_enabled(self):
        return self._enabled

    def alert(self, title, message, level='info'):
        """Send an alert via all configured channels."""
        if not self._enabled:
            return

        if self.discord_webhook:
            self._send_discord(title, message, level)
        if self.telegram_token and self.telegram_chat_id:
            self._send_telegram(title, message, level)

    def circuit_breaker_triggered(self, breaker_name, details):
        """Alert when a circuit breaker triggers."""
        self.alert(
            f"CIRCUIT BREAKER: {breaker_name}",
            f"Trading paused. {details}",
            level='critical'
        )

    def large_loss(self, symbol, pnl, pct):
        """Alert on significant loss."""
        self.alert(
            f"LARGE LOSS: {symbol}",
            f"PnL: ${pnl:.2f} ({pct:.1f}%)",
            level='warning'
        )

    def bot_error(self, error_msg):
        """Alert on bot errors."""
        self.alert("BOT ERROR", error_msg, level='error')

    def daily_summary(self, total_pnl, trades_count, win_rate):
        """Send daily performance summary."""
        self.alert(
            "Daily Summary",
            f"PnL: ${total_pnl:.2f} | Trades: {trades_count} | Win Rate: {win_rate:.1f}%",
            level='info'
        )

    def _send_discord(self, title, message, level):
        color_map = {
            'info': 3447003,
            'warning': 16776960,
            'error': 15158332,
            'critical': 10038562,
        }
        payload = {
            "embeds": [{
                "title": title,
                "description": message,
                "color": color_map.get(level, 3447003),
                "timestamp": datetime.utcnow().isoformat(),
                "footer": {"text": "Moon Dev Trading Bot"}
            }]
        }
        try:
            requests.post(self.discord_webhook, json=payload, timeout=5)
        except Exception:
            pass

    def _send_telegram(self, title, message, level):
        emoji = {
            'info': 'i',
            'warning': '!',
            'error': 'X',
            'critical': '!!',
        }.get(level, '')
        text = f"[{emoji}] *{title}*\n{message}"
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            requests.post(
                url,
                json={
                    "chat_id": self.telegram_chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=5,
            )
        except Exception:
            pass


_alert_manager = None


def get_alert_manager():
    """Return singleton AlertManager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
