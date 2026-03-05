"""Alert system for trading events — beautiful Discord embeds with anti-spam.

Only 3 types of notifications:
1. Trade closed — performance du trade (seule alerte individuelle)
2. Alerte critique — circuit breaker, grosse perte
3. Rapport journalier — résumé de la veille (envoyé par le scheduler dans main.py)
"""
import os
import random
import requests
import time as _time
from datetime import datetime


class AlertManager:
    """Sends rich, human-readable alerts via Discord/Telegram webhooks.

    Anti-spam: rate-limits alerts per category to avoid flooding Discord.
    """

    # Minimum seconds between alerts of the same type
    COOLDOWNS = {
        'trade_opened': 30,       # Kept but no longer called from strategy
        'trade_closed': 30,
        'signal': 300,
        'cycle_summary': 900,
        'circuit_breaker': 3600,  # 1h cooldown (was 600s)
        'large_loss': 1800,       # 30min cooldown
        'service_down': 1800,     # 30min cooldown per service
        'error': 300,
        'default': 60,
    }

    # Seuil de grosse perte (% de la balance)
    LARGE_LOSS_THRESHOLD_PCT = float(os.getenv('LARGE_LOSS_THRESHOLD_PCT', '3.0'))

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
        """Discord embed when a trade is opened. DORMANT — kept for potential future use."""
        # No longer called from the strategy. Kept as a no-op.
        pass

    def trade_closed(self, trade: dict):
        """Beautiful Discord embed when a trade is closed — the main alert."""
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
        score = trade.get('score', 0)
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
        is_flat = abs(pnl_pct) < 0.1
        is_long = direction == 'BUY'

        # Realized R:R
        sl = trade.get('stop_loss', 0)
        if entry > 0 and sl > 0:
            planned_risk = abs(entry - sl)
            if planned_risk > 0:
                realized_rr = abs(exit_price - entry) / planned_risk
                if not is_win:
                    realized_rr = -realized_rr
            else:
                realized_rr = 0
        else:
            realized_rr = 0

        # Emoji contextuel
        if is_flat:
            emoji = random.choice(["", "", ""])
        elif pnl_pct >= 5:
            emoji = random.choice(["", "", ""])
        elif pnl_pct >= 2:
            emoji = random.choice(["", "", ""])
        elif pnl_pct > 0:
            emoji = random.choice(["", "", ""])
        elif pnl_pct >= -2:
            emoji = random.choice(["", "", ""])
        else:
            emoji = random.choice(["", "", ""])

        # Fun commentary — plus de variété
        win_comments = [
            "Joli coup, on prend les gains !",
            "Le plan a fonctionné, beau trade",
            "Les modules avaient raison",
            "Cash is king, on encaisse",
            "Propre et net, on continue",
        ]
        small_win_comments = [
            "Petit profit, c'est toujours ça de pris",
            "Un vert de plus, ça s'accumule",
            "Pas spectaculaire mais positif",
        ]
        flat_comments = [
            "Quasi flat, on s'en sort bien",
            "Ni gain ni perte, le marché hésite",
            "Break-even, on passe au suivant",
        ]
        small_loss_comments = [
            "Petite perte maîtrisée, le RM fait son job",
            "SL respecté, c'est la discipline qui paie",
            "Perte contrôlée, on reste serein",
        ]
        big_loss_comments = [
            "Ça pique, mais on analyse et on rebondit",
            "Mauvais timing, ça arrive. On apprend",
            "Le marché a décidé autrement, next trade",
        ]

        if is_flat:
            vibe = random.choice(flat_comments)
        elif pnl_pct >= 2:
            vibe = random.choice(win_comments)
        elif pnl_pct > 0:
            vibe = random.choice(small_win_comments)
        elif pnl_pct >= -2:
            vibe = random.choice(small_loss_comments)
        else:
            vibe = random.choice(big_loss_comments)

        # Color: green for win, red for loss, grey for flat
        if is_flat:
            color = 0x99AAB5
        elif is_win:
            color = 0x00D166
        else:
            color = 0xED4245

        # Reason in French
        reason_fr = {
            'stop_loss': 'Stop Loss',
            'STOP_LOSS': 'Stop Loss',
            'take_profit': 'Take Profit',
            'TAKE_PROFIT': 'Take Profit',
            'trailing_stop': 'Trailing Stop',
            'TRAILING_STOP': 'Trailing Stop',
            'time_exit': 'Durée max',
            'TIME_EXIT_24H': 'Durée max (24h)',
            'partial_tp': 'TP partiel',
            'manual': 'Manuel',
            'MANUAL': 'Manuel',
        }.get(reason, reason)

        title = f"{emoji} {'LONG' if is_long else 'SHORT'} {symbol} — {reason_fr}"

        embed = {
            "title": title,
            "description": f"**{vibe}**",
            "color": color,
            "fields": [
                {"name": "PnL", "value": f"```{'+ ' if is_win else ''}{pnl:+.2f}$ ({pnl_pct:+.2f}%)```", "inline": False},
                {"name": "Entrée", "value": f"`${entry:,.2f}`", "inline": True},
                {"name": "Sortie", "value": f"`${exit_price:,.2f}`", "inline": True},
                {"name": "R:R réalisé", "value": f"`{realized_rr:+.1f}R`", "inline": True},
                {"name": "Durée", "value": f"`{hold_time}`" if hold_time else "`?`", "inline": True},
                {"name": "Balance", "value": f"`${balance:,.2f}`", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot | Paper Trading"},
        }

        # Score d'entrée + top modules
        if score:
            embed["fields"].append({"name": "Score entrée", "value": f"`{score:.1f}/100`", "inline": True})

        modules_str = trade.get('modules', '')
        if modules_str and modules_str != '{}':
            try:
                if isinstance(modules_str, str):
                    # Parse the stringified dict
                    module_scores = eval(modules_str) if modules_str.startswith('{') else {}
                else:
                    module_scores = modules_str
                if module_scores:
                    top_3 = sorted(module_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    mod_display = ' | '.join(f"{n}: {s:+.0f}" for n, s in top_3)
                    embed["fields"].append({"name": "Top modules", "value": f"`{mod_display}`", "inline": False})
            except Exception:
                pass

        self._send_discord_embed(embed)

        # Auto-detect large loss and fire critical alert
        if balance > 0 and pnl < 0:
            loss_pct_of_balance = abs(pnl) / balance * 100
            if loss_pct_of_balance >= self.LARGE_LOSS_THRESHOLD_PCT:
                self.large_loss(symbol, pnl, pnl_pct)

    def signal_detected(self, symbol: str, direction: str, score: float, threshold: float,
                        modules_fired: int, total_modules: int, module_details: dict = None):
        """Notify when a notable signal is detected (even if not traded)."""
        if not self._enabled or self._is_rate_limited('signal'):
            return

        is_long = direction == 'BUY'
        passed = score >= threshold
        color = 0x5865F2 if passed else 0x99AAB5

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
            status = f"{signals_found} signal(aux) détecté(s) mais pas tradé(s)"
            color = 0xFEE75C
        else:
            status = "Aucun signal, le marché dort"
            color = 0x99AAB5

        embed = {
            "title": f"Cycle terminé — {tokens_analyzed} tokens analysés",
            "description": f"**{status}**\nBalance: `${balance:,.2f}` | Positions ouvertes: `{open_positions}`",
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot"},
        }

        self._send_discord_embed(embed)

    # ------------------------------------------------------------------
    # Critical alerts
    # ------------------------------------------------------------------

    def circuit_breaker_triggered(self, breaker_name, details):
        """Alert when a circuit breaker triggers. 1h cooldown."""
        if not self._enabled or self._is_rate_limited('circuit_breaker'):
            return

        embed = {
            "title": f"CIRCUIT BREAKER : {breaker_name}",
            "description": f"**Trading en pause.** {details}\n\nLe bot se protège automatiquement. Il reprendra quand les conditions seront meilleures.",
            "color": 0xED4245,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot | Protection"},
        }
        self._send_discord_embed(embed)

    def large_loss(self, symbol, pnl, pct):
        """Alert on significant loss (> LARGE_LOSS_THRESHOLD_PCT of balance)."""
        if not self._enabled or self._is_rate_limited('large_loss'):
            return

        embed = {
            "title": f"Grosse perte sur {symbol}",
            "description": f"PnL: `{pnl:+.2f}$` (`{pct:+.1f}%`)\n\nPas de panique, ça fait partie du jeu. Le risk management limite les dégâts.",
            "color": 0xED4245,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot | Alerte critique"},
        }
        self._send_discord_embed(embed)

    def bot_error(self, error_msg):
        """Alert on bot errors."""
        embed = {
            "title": "Erreur du bot",
            "description": f"```{error_msg[:1800]}```\nLe bot va tenter de continuer malgré l'erreur.",
            "color": 0xED4245,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot | Debug"},
        }
        self._send_discord_embed(embed)

    def service_down(self, service_name: str, error_msg: str):
        """Alert when a critical third-party service is unreachable.

        Rate-limited per service (30min cooldown each).
        """
        if not self._enabled:
            return
        # Per-service rate limiting
        key = f"service_down:{service_name}"
        if self._is_rate_limited(key):
            return

        embed = {
            "title": f"Service indisponible : {service_name}",
            "description": (
                f"```{str(error_msg)[:1500]}```\n"
                "Le bot continue avec les données en cache si disponibles."
            ),
            "color": 0xF0B232,  # Orange/amber
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Moon Dev Bot | Infra"},
        }
        self._send_discord_embed(embed)

    # ------------------------------------------------------------------
    # Daily summary — rich embed
    # ------------------------------------------------------------------

    def daily_summary(self, stats: dict):
        """Send daily performance summary with rich embed.

        Accepts a dict with keys:
            date, total_pnl, trades_count, wins, losses, win_rate,
            best_trade (dict: symbol, pnl), worst_trade (dict: symbol, pnl),
            balance, open_positions, total_pnl_alltime, streak,
            alpha_btc
        """
        if not self._enabled:
            return

        total_pnl = stats.get('total_pnl', 0)
        trades_count = stats.get('trades_count', 0)
        win_rate = stats.get('win_rate', 0)
        balance = stats.get('balance', 0)
        open_positions = stats.get('open_positions', 0)
        best = stats.get('best_trade', {})
        worst = stats.get('worst_trade', {})
        streak = stats.get('streak', 0)
        alpha_btc = stats.get('alpha_btc', None)
        total_pnl_alltime = stats.get('total_pnl_alltime', None)
        report_date = stats.get('date', 'Hier')

        # Fun commentary
        good_vibes = [
            "Belle journée, le bot performe !",
            "Les algos sont en forme aujourd'hui",
            "Journée verte, on continue comme ça",
            "Le plan fonctionne, on encaisse",
        ]
        flat_vibes = [
            "Journée calme, pas de mouvement",
            "Le marché hésite, patience...",
            "Rien de spécial, on reste en veille",
        ]
        bad_vibes = [
            "Journée rouge, demain est un autre jour",
            "Ça pique mais le RM fait son job",
            "Pas notre jour, on analyse et on ajuste",
        ]
        no_trade_vibes = [
            "Aucun trade hier, le bot était en veille",
            "Zéro signal, marché trop calme",
            "Pas de trade = pas de perte, c'est déjà ça",
        ]

        if trades_count == 0:
            vibe = random.choice(no_trade_vibes)
            color = 0x99AAB5  # Grey
            emoji = ""
        elif total_pnl > 0:
            vibe = random.choice(good_vibes)
            color = 0x00D166  # Green
            emoji = "" if total_pnl > 50 else ""
        elif abs(total_pnl) < 1:
            vibe = random.choice(flat_vibes)
            color = 0xFEE75C  # Yellow
            emoji = ""
        else:
            vibe = random.choice(bad_vibes)
            color = 0xED4245  # Red
            emoji = ""

        title = f"{emoji} Rapport du {report_date} — {vibe}"

        fields = []

        # PnL section
        if trades_count > 0:
            pnl_bar = _pnl_bar(total_pnl)
            fields.append({"name": "PnL du jour", "value": f"```{total_pnl:+.2f}$\n{pnl_bar}```", "inline": False})

        # Stats
        if trades_count > 0:
            fields.append({"name": "Trades", "value": f"`{trades_count}`", "inline": True})
            fields.append({"name": "Win Rate", "value": f"`{win_rate:.0f}%`", "inline": True})
            if streak:
                streak_str = f"{streak}W" if streak > 0 else f"{abs(streak)}L"
                fields.append({"name": "Streak", "value": f"`{streak_str}`", "inline": True})

        # Best / Worst trade
        if best:
            fields.append({"name": "Meilleur trade", "value": f"`{best.get('symbol', '?')}` `{best.get('pnl', 0):+.2f}$`", "inline": True})
        if worst:
            fields.append({"name": "Pire trade", "value": f"`{worst.get('symbol', '?')}` `{worst.get('pnl', 0):+.2f}$`", "inline": True})

        # Balance & positions
        if balance:
            fields.append({"name": "Balance", "value": f"`${balance:,.2f}`", "inline": True})
        if open_positions is not None:
            fields.append({"name": "Positions ouvertes", "value": f"`{open_positions}`", "inline": True})

        # Alpha vs BTC
        if alpha_btc is not None:
            alpha_emoji = "" if alpha_btc > 0 else ""
            fields.append({"name": f"{alpha_emoji} Alpha vs BTC", "value": f"`{alpha_btc:+.2f}%`", "inline": True})

        # All-time PnL
        if total_pnl_alltime is not None:
            fields.append({"name": "PnL total (all-time)", "value": f"`{total_pnl_alltime:+.2f}$`", "inline": True})

        embed = {
            "title": title,
            "color": color,
            "fields": fields,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": f"Moon Dev Bot | Daily Report | {report_date}"},
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


def _pnl_bar(pnl, max_width=20):
    """Visual PnL bar for the daily summary."""
    if pnl == 0:
        return "|" + " " * max_width + "|"
    # Scale: each block = $5
    blocks = min(int(abs(pnl) / 5) + 1, max_width)
    if pnl > 0:
        return " " * max_width + "|" + "=" * blocks + " +"
    else:
        pad = max_width - blocks
        return " " * pad + "- " + "=" * blocks + "|"


_alert_manager = None


def get_alert_manager():
    """Return singleton AlertManager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def alert_service_down(service_name: str, error):
    """Convenience: fire a service-down Discord alert (rate-limited per service)."""
    get_alert_manager().service_down(service_name, str(error))
