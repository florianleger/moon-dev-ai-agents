"""
Settings API endpoints
"""

import os
from typing import Dict, Any
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.web.auth import verify_credentials

router = APIRouter()


class SettingsUpdate(BaseModel):
    """Settings update model."""
    paper_trading: bool = None
    leverage: int = None
    min_confidence: int = None
    max_daily_trades: int = None
    assets: str = None  # Comma-separated list


@router.get("")
async def get_settings(username: str = Depends(verify_credentials)) -> Dict:
    """Get current settings."""
    try:
        from src.config import (
            PAPER_TRADING,
            RAMF_LEVERAGE,
            RAMF_MIN_CONFIDENCE,
            RAMF_MAX_DAILY_TRADES,
            RAMF_ASSETS,
            RAMF_STOP_LOSS_PCT,
            RAMF_TAKE_PROFIT_PCT,
            RAMF_USE_ADAPTIVE_SL_TP,
            RAMF_USE_TIME_WINDOWS,
            RAMF_USE_MTF,
            RAMF_USE_FUNDING_DIVERGENCE,
            RAMF_USE_LIQ_CLUSTERS,
        )

        return {
            "paper_trading": PAPER_TRADING,
            "leverage": RAMF_LEVERAGE,
            "min_confidence": RAMF_MIN_CONFIDENCE,
            "max_daily_trades": RAMF_MAX_DAILY_TRADES,
            "assets": RAMF_ASSETS,
            "stop_loss_pct": RAMF_STOP_LOSS_PCT,
            "take_profit_pct": RAMF_TAKE_PROFIT_PCT,
            "features": {
                "adaptive_sl_tp": RAMF_USE_ADAPTIVE_SL_TP,
                "time_windows": RAMF_USE_TIME_WINDOWS,
                "mtf_confluence": RAMF_USE_MTF,
                "funding_divergence": RAMF_USE_FUNDING_DIVERGENCE,
                "liquidation_clusters": RAMF_USE_LIQ_CLUSTERS,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def update_settings(
    settings: SettingsUpdate,
    username: str = Depends(verify_credentials)
) -> Dict:
    """Update settings (modifies config.py)."""
    try:
        config_path = Path(__file__).parent.parent.parent / "config.py"

        if not config_path.exists():
            raise HTTPException(status_code=404, detail="Config file not found")

        # Read current config
        with open(config_path, "r") as f:
            content = f.read()

        # Update values
        updates = []

        if settings.paper_trading is not None:
            content = _replace_config_value(content, "PAPER_TRADING", str(settings.paper_trading))
            updates.append(f"PAPER_TRADING={settings.paper_trading}")

        if settings.leverage is not None:
            if not 1 <= settings.leverage <= 10:
                raise HTTPException(status_code=400, detail="Leverage must be 1-10")
            content = _replace_config_value(content, "RAMF_LEVERAGE", str(settings.leverage))
            updates.append(f"RAMF_LEVERAGE={settings.leverage}")

        if settings.min_confidence is not None:
            if not 0 <= settings.min_confidence <= 100:
                raise HTTPException(status_code=400, detail="Confidence must be 0-100")
            content = _replace_config_value(content, "RAMF_MIN_CONFIDENCE", str(settings.min_confidence))
            updates.append(f"RAMF_MIN_CONFIDENCE={settings.min_confidence}")

        if settings.max_daily_trades is not None:
            if not 1 <= settings.max_daily_trades <= 50:
                raise HTTPException(status_code=400, detail="Max trades must be 1-50")
            content = _replace_config_value(content, "RAMF_MAX_DAILY_TRADES", str(settings.max_daily_trades))
            updates.append(f"RAMF_MAX_DAILY_TRADES={settings.max_daily_trades}")

        if settings.assets is not None:
            # Parse comma-separated list
            assets_list = [a.strip().upper() for a in settings.assets.split(",") if a.strip()]
            if not assets_list:
                raise HTTPException(status_code=400, detail="At least one asset required")
            assets_str = str(assets_list)
            content = _replace_config_value(content, "RAMF_ASSETS", assets_str)
            updates.append(f"RAMF_ASSETS={assets_list}")

        # Write updated config
        with open(config_path, "w") as f:
            f.write(content)

        return {
            "status": "updated",
            "changes": updates,
            "message": "Restart strategy for changes to take effect",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _replace_config_value(content: str, key: str, value: str) -> str:
    """Replace a config value in the content."""
    import re

    # Pattern to match: KEY = value  or  KEY=value
    pattern = rf"^({key}\s*=\s*).*$"
    replacement = rf"\g<1>{value}"

    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)

    if count == 0:
        # Key not found, append at end
        new_content = content.rstrip() + f"\n{key} = {value}\n"

    return new_content
