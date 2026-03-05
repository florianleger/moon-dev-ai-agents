"""Alert system for trading events — beautiful Discord embeds with anti-spam."""
import os
import requests
import time as _time
from datetime import datetime


class AlertManager:
    """Sends rich, human-readable alerts via Discord/Telegram webhooks.

    Anti-spam: rate-limits alerts per category to avoid flooding Discord.
    """

    # Minimum seconds between alerts of the same type
    COOLDOWNS = {
        'trade_opened': 30,       # Max 1 trade alert per 30s
        'trade_closed': 30,
        'signal': 300,            # Max 1 signal alert per 5min
        'cycle_summary': 900,     # Max 1 cycle summary per 15min
        'circuit_breaker': 600,   # Max 1 circuit breaker per 10min
        'error': 300,             # Max 1 error per 5min
        'default': 60,
    }

    def __init__(self):
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self._enabled = bool(self.discord_webhook or self.telegram_token)
        self._last_sent = {}  # {category: timestamp}

    @property
    def is_enabled(self):
        return self._enabled

    def _is_rate_limited(self, category: str) -> bool:
        """Check if this alert category is on cooldown. Returns True if we should skip."""
        now = _time.time()
        cooldown = self.COOLDOWNS.get(category, self.COOLDOWNS['default'])
        last = self._last_sent.get(category, 0)
        if now - last < cooldown:
            return True
        self._last_sent[category] = now
        return False

    def alert(self, title, message, level='info'):
        """Send an alert via all configured channels."""
        if not self._enabled:
            return

        if self.discord_webhook:
            self._send_discord(title, message, level)
        if self.telegram_token and self.telegram_chat_id:
            self._send_telegram(title, message, level)

    # ------------------------------------------------------------------
    # Rich trade alerts
    # ------------------------------------------------------------------

    def trade_opened(self, trade: dict, metadata: dict = None):
        """Beautiful Discord embed when a trade is opened."""
        if not self._enabled or self._is_rate_limited('trade_opened'):
            return

        metadata = metadata or {}
        symbol = trade.get('symbol', '?')
        direction = trade.get('direction', '?')
        price = trade.get('entry_price', 0)
        size = trade.get('position_size', 0)
        leverage = trade.get('leverage', 1)
        sl = trade.get('stop_loss', 0)
        tp = trade.get('take_profit', 0)
        score = metadata.get('score', 0)
        modules = metadata.get('active_modules', 0)
        total = metadata.get('total_fired', 0)

        is_long = direction == 'BUY'
        arrow = "LONG" if is_long else "SHORT"
        color = 0x00D166 if is_long else 0xED4245  # Green / Red

        # Calculate R:R ratio
        if price > 0 and sl > 0 and tp > 0:
            risk = abs(price - sl)
            reward = abs(tp - price)
            rr = reward / risk if risk > 0 else 0
        else:
            rr = 0

        # Fun commentary based on score
        if score >= 70:
            vibe = "Signal en beton arme"
        elif score >= 55:
            vibe = "Bonne convergence des modules"
        elif score >= 42:
            vibe = "Signal correct, on tente le coup"
        else:
            vibe = "On y va doucement"

        embed = {
            "title": f"{'LONG' if is_long else 'SHORT'} {symbol}",
            "description": f"**{vibe}** — {modules}/{total} modules d'accord",
            "color": color,
            "fields": [
                {"name": "Prix d'entree", "value": f"`${price:,.2f}`", "inline": True},
                {"name": "Taille", "value": f"`${size:,.2f}` ({leverage}x)", "inline": True},
                {"name": "Score", "value": f"`{score:.1f}/100`", "inline": True},
                {"name": "Stop Loss", "value": f"`${sl:,.2f}`", "inline": True},
                {"name": "Take Profit", "value": f"`${tp:,.2f}`", "inline": True},
                {"name": "R:R", "value": f"`1:{rr:.1f}`", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot | Paper Trading"},
        }

        # Top module scores as a compact string
        module_scores = metadata.get('module_scores', {})
        if module_scores:
            top_modules = sorted(module_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            mod_str = ' | '.join(f"{n}: {s}" for n, s in top_modules)
            embed["fields"].append({"name": "Top modules", "value": f"```{mod_str}```", "inline": False})

        self._send_discord_embed(embed)

    def trade_closed(self, trade: dict):
        """Beautiful Discord embed when a trade is closed."""
        if not self._enabled or self._is_rate_limited('trade_closed'):
            return

        symbol = trade.get('symbol', '?')
        direction = trade.get('direction', '?')
        entry = trade.get('entry_price', 0)
        exit_price = trade.get('close_price', 0)
        pnl = trade.get('pnl', 0)
        pnl_pct = trade.get('pnl_pct', 0)
        reason = trade.get('close_reason', '?')
        balance = trade.get('balance_after', 0)
        hold_time = ''

        # Calculate hold duration
        entry_time = trade.get('entry_time', '')
        exit_time = trade.get('exit_time', '')
        if entry_time and exit_time:
            try:
                t1 = datetime.fromisoformat(str(entry_time))
                t2 = datetime.fromisoformat(str(exit_time))
                delta = t2 - t1
                hours = delta.total_seconds() / 3600
                if hours < 1:
                    hold_time = f"{delta.total_seconds() / 60:.0f}min"
                elif hours < 24:
                    hold_time = f"{hours:.1f}h"
                else:
                    hold_time = f"{hours / 24:.1f}j"
            except Exception:
                pass

        is_win = pnl > 0
        is_long = direction == 'BUY'

        # Fun commentary
        if pnl_pct >= 3:
            vibe = "Joli coup, on prend les gains !"
        elif pnl_pct >= 1:
            vibe = "Petit profit, c'est toujours ca de pris"
        elif pnl_pct >= 0:
            vibe = "Quasi flat, on s'en sort bien"
        elif pnl_pct >= -1:
            vibe = "Petite perte maitrisee, le risk management fait son job"
        elif pnl_pct >= -3:
            vibe = "Ca pique un peu, mais rien de grave"
        else:
            vibe = "Aie, grosse perte. On analyse et on rebondit"

        # Color: green for win, red for loss
        color = 0x00D166 if is_win else 0xED4245

        # Reason in French
        reason_fr = {
            'stop_loss': 'Stop Loss touche',
            'take_profit': 'Take Profit atteint !',
            'trailing_stop': 'Trailing Stop active',
            'time_exit': 'Duree max atteinte',
            'partial_tp': 'Take Profit partiel',
            'manual': 'Fermeture manuelle',
        }.get(reason, reason)

        embed = {
            "title": f"{'LONG' if is_long else 'SHORT'} {symbol} ferme — {reason_fr}",
            "description": f"**{vibe}**",
            "color": color,
            "fields": [
                {"name": "PnL", "value": f"```{'+ ' if is_win else ''}{pnl:+.2f}$ ({pnl_pct:+.2f}%)```", "inline": False},
                {"name": "Entree", "value": f"`${entry:,.2f}`", "inline": True},
                {"name": "Sortie", "value": f"`${exit_price:,.2f}`", "inline": True},
                {"name": "Duree", "value": f"`{hold_time}`" if hold_time else "`?`", "inline": True},
                {"name": "Balance", "value": f"`${balance:,.2f}`", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot | Paper Trading"},
        }

        self._send_discord_embed(embed)

    def signal_detected(self, symbol: str, direction: str, score: float, threshold: float,
                        modules_fired: int, total_modules: int, module_details: dict = None):
        """Notify when a notable signal is detected (even if not traded)."""
        if not self._enabled or self._is_rate_limited('signal'):
            return

        is_long = direction == 'BUY'
        passed = score >= threshold
        color = 0x5865F2 if passed else 0x99AAB5  # Blurple if passed, grey if not

        status = "Signal valide !" if passed else "Signal trop faible"

        embed = {
            "title": f"{'LONG' if is_long else 'SHORT'} {symbol} — {status}",
            "description": f"Score `{score:.1f}` / seuil `{threshold:.0f}` — {modules_fired}/{total_modules} modules actifs",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot | Analyse"},
        }

        if module_details:
            top = sorted(module_details.items(), key=lambda x: x[1], reverse=True)[:6]
            mod_str = '\n'.join(f"  {n}: {s}" for n, s in top)
            embed["fields"] = [{"name": "Modules", "value": f"```{mod_str}```", "inline": False}]

        self._send_discord_embed(embed)

    def cycle_summary(self, tokens_analyzed: int, signals_found: int, trades_opened: int,
                      balance: float, open_positions: int):
        """End-of-cycle summary — compact one-liner."""
        if not self._enabled or not self.discord_webhook or self._is_rate_limited('cycle_summary'):
            return

        if trades_opened > 0:
            status = f"{trades_opened} nouveau(x) trade(s) ouvert(s)"
            color = 0x00D166
        elif signals_found > 0:
            status = f"{signals_found} signal(aux) detecte(s) mais pas trade(s)"
            color = 0xFEE75C
        else:
            status = "Aucun signal, le marche dort"
            color = 0x99AAB5

        embed = {
            "title": f"Cycle termine — {tokens_analyzed} tokens analyses",
            "description": f"**{status}**\nBalance: `${balance:,.2f}` | Positions ouvertes: `{open_positions}`",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot"},
        }

        self._send_discord_embed(embed)

    # ------------------------------------------------------------------
    # Legacy methods (kept for backward compatibility)
    # ------------------------------------------------------------------

    def circuit_breaker_triggered(self, breaker_name, details):
        """Alert when a circuit breaker triggers."""
        embed = {
            "title": f"CIRCUIT BREAKER : {breaker_name}",
            "description": f"**Trading en pause.** {details}\n\nLe bot se protege automatiquement. Il reprendra quand les conditions seront meilleures.",
            "color": 0xED4245,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot | Protection"},
        }
        self._send_discord_embed(embed)

    def large_loss(self, symbol, pnl, pct):
        """Alert on significant loss."""
        embed = {
            "title": f"Grosse perte sur {symbol}",
            "description": f"PnL: `{pnl:+.2f}$` (`{pct:+.1f}%`)\n\nPas de panique, ca fait partie du jeu. Le risk management limite les degats.",
            "color": 0xED4245,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot"},
        }
        self._send_discord_embed(embed)

    def bot_error(self, error_msg):
        """Alert on bot errors."""
        embed = {
            "title": "Erreur du bot",
            "description": f"```{error_msg[:1800]}```\nLe bot va tenter de continuer malgre l'erreur.",
            "color": 0xED4245,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot | Debug"},
        }
        self._send_discord_embed(embed)

    def daily_summary(self, total_pnl, trades_count, win_rate):
        """Send daily performance summary."""
        if total_pnl > 0:
            vibe = "Bonne journee pour le bot !"
        elif total_pnl == 0:
            vibe = "Journee calme, pas de mouvement"
        else:
            vibe = "Journee rouge, mais demain est un autre jour"

        color = 0x00D166 if total_pnl >= 0 else 0xED4245

        embed = {
            "title": f"Resume du jour — {vibe}",
            "description": "",
            "color": color,
            "fields": [
                {"name": "PnL", "value": f"`{total_pnl:+.2f}$`", "inline": True},
                {"name": "Trades", "value": f"`{trades_count}`", "inline": True},
                {"name": "Win Rate", "value": f"`{win_rate:.0f}%`", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot | Daily Report"},
        }
        self._send_discord_embed(embed)

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _send_discord_embed(self, embed):
        """Send a rich embed to Discord."""
        if not self.discord_webhook:
            return
        try:
            requests.post(self.discord_webhook, json={"embeds": [embed]}, timeout=5)
        except Exception:
            pass

    def _send_discord(self, title, message, level):
        """Legacy simple embed — routes to rich embed."""
        color_map = {
            'info': 0x3498DB,
            'warning': 0xFEE75C,
            'error': 0xED4245,
            'critical': 0xED4245,
        }
        self._send_discord_embed({
            "title": title,
            "description": message,
            "color": color_map.get(level, 0x3498DB),
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot"},
        })

    def _send_telegram(self, title, message, level):
        emoji = {'info': 'i', 'warning': '!', 'error': 'X', 'critical': '!!'}.get(level, '')
        text = f"[{emoji}] *{title}*\n{message}"
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            requests.post(url, json={"chat_id": self.telegram_chat_id, "text": text, "parse_mode": "Markdown"}, timeout=5)
        except Exception:
            pass


_alert_manager = None


def get_alert_manager():
    """Return singleton AlertManager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
