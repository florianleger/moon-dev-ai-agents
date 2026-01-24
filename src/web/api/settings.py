"""
Settings API endpoints - supports both RAMF and Sniper AI strategies
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
    stop_loss_pct: float = None
    take_profit_pct: float = None
    # Sniper advanced features
    use_trailing_stop: bool = None
    use_regime_filter: bool = None
    use_correlation_sizing: bool = None
    enable_funding_arbitrage: bool = None
    use_weighted_scoring: bool = None
    use_confidence_sizing: bool = None
    min_weighted_score: float = None


@router.get("")
async def get_settings(username: str = Depends(verify_credentials)) -> Dict:
    """Get current settings based on active strategy."""
    try:
        from src.config import ACTIVE_STRATEGY, PAPER_TRADING

        if ACTIVE_STRATEGY == 'sniper':
            return _get_sniper_settings()
        else:
            return _get_ramf_settings()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_sniper_settings() -> Dict:
    """Get Sniper AI strategy settings."""
    from src.config import (
        ACTIVE_STRATEGY,
        PAPER_TRADING,
        PAPER_TRADING_BALANCE,
        SNIPER_ASSETS,
        SNIPER_LEVERAGE,
        SNIPER_STOP_LOSS_PCT,
        SNIPER_TAKE_PROFIT_PCT,
        SNIPER_MAX_DAILY_TRADES,
        SNIPER_MAX_DAILY_LOSS_USD,
        SNIPER_MAX_DAILY_GAIN_USD,
        SNIPER_AI_MIN_CONFIDENCE,
        SNIPER_SIGMA_THRESHOLD,
        SNIPER_FUNDING_EXTREME_THRESHOLD,
        SNIPER_LIQ_RATIO_THRESHOLD,
        SNIPER_VOLUME_SPIKE_THRESHOLD,
        SNIPER_OPTIMAL_HOURS,
        SNIPER_ALLOW_NORMAL_HOURS,
        SNIPER_RSI_OVERSOLD,
        SNIPER_RSI_OVERBOUGHT,
        # Advanced improvements
        SNIPER_USE_TRAILING_STOP,
        SNIPER_TRAILING_ATR_MULTIPLIER,
        SNIPER_TRAILING_ACTIVATION_PCT,
        SNIPER_USE_REGIME_FILTER,
        SNIPER_ADX_TRENDING_THRESHOLD,
        SNIPER_USE_CORRELATION_SIZING,
        SNIPER_CORRELATION_THRESHOLD,
        SNIPER_ENABLE_FUNDING_ARBITRAGE,
        SNIPER_FUNDING_ARBITRAGE_THRESHOLD,
        SNIPER_USE_WEIGHTED_SCORING,
        SNIPER_MIN_WEIGHTED_SCORE,
        SNIPER_WEIGHTS,
        SNIPER_USE_CONFIDENCE_SIZING,
        SNIPER_CONFIDENCE_SIZE_MAP,
    )

    return {
        "strategy": "sniper",
        "paper_trading": PAPER_TRADING,
        "paper_balance": PAPER_TRADING_BALANCE,
        "leverage": SNIPER_LEVERAGE,
        "min_confidence": SNIPER_AI_MIN_CONFIDENCE,
        "max_daily_trades": SNIPER_MAX_DAILY_TRADES,
        "assets": SNIPER_ASSETS,
        "stop_loss_pct": SNIPER_STOP_LOSS_PCT,
        "take_profit_pct": SNIPER_TAKE_PROFIT_PCT,
        "max_daily_loss_usd": SNIPER_MAX_DAILY_LOSS_USD,
        "max_daily_gain_usd": SNIPER_MAX_DAILY_GAIN_USD,
        # Checklist thresholds
        "checklist": {
            "sigma_threshold": SNIPER_SIGMA_THRESHOLD,
            "funding_zscore_threshold": SNIPER_FUNDING_EXTREME_THRESHOLD,
            "liq_ratio_threshold": SNIPER_LIQ_RATIO_THRESHOLD,
            "volume_spike_threshold": SNIPER_VOLUME_SPIKE_THRESHOLD,
            "optimal_hours": SNIPER_OPTIMAL_HOURS,
            "allow_normal_hours": SNIPER_ALLOW_NORMAL_HOURS,
            "rsi_oversold": SNIPER_RSI_OVERSOLD,
            "rsi_overbought": SNIPER_RSI_OVERBOUGHT,
        },
        # Advanced features (toggleable)
        "features": {
            "trailing_stop": SNIPER_USE_TRAILING_STOP,
            "regime_filter": SNIPER_USE_REGIME_FILTER,
            "correlation_sizing": SNIPER_USE_CORRELATION_SIZING,
            "funding_arbitrage": SNIPER_ENABLE_FUNDING_ARBITRAGE,
            "weighted_scoring": SNIPER_USE_WEIGHTED_SCORING,
            "confidence_sizing": SNIPER_USE_CONFIDENCE_SIZING,
        },
        # Feature parameters
        "feature_params": {
            "trailing_atr_multiplier": SNIPER_TRAILING_ATR_MULTIPLIER,
            "trailing_activation_pct": SNIPER_TRAILING_ACTIVATION_PCT,
            "adx_trending_threshold": SNIPER_ADX_TRENDING_THRESHOLD,
            "correlation_threshold": SNIPER_CORRELATION_THRESHOLD,
            "funding_arb_threshold": SNIPER_FUNDING_ARBITRAGE_THRESHOLD,
            "min_weighted_score": SNIPER_MIN_WEIGHTED_SCORE,
            "weights": SNIPER_WEIGHTS,
            "confidence_size_map": SNIPER_CONFIDENCE_SIZE_MAP,
        }
    }


def _get_ramf_settings() -> Dict:
    """Get RAMF strategy settings."""
    from src.config import (
        ACTIVE_STRATEGY,
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
        "strategy": "ramf",
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


@router.post("")
async def update_settings(
    settings: SettingsUpdate,
    username: str = Depends(verify_credentials)
) -> Dict:
    """Update settings (modifies config.py)."""
    try:
        from src.config import ACTIVE_STRATEGY

        config_path = Path(__file__).parent.parent.parent / "config.py"

        if not config_path.exists():
            raise HTTPException(status_code=404, detail="Config file not found")

        # Read current config
        with open(config_path, "r") as f:
            content = f.read()

        updates = []
        prefix = "SNIPER_" if ACTIVE_STRATEGY == 'sniper' else "RAMF_"

        # Common settings
        if settings.paper_trading is not None:
            content = _replace_config_value(content, "PAPER_TRADING", str(settings.paper_trading))
            updates.append(f"PAPER_TRADING={settings.paper_trading}")

        if settings.leverage is not None:
            if not 1 <= settings.leverage <= 10:
                raise HTTPException(status_code=400, detail="Leverage must be 1-10")
            content = _replace_config_value(content, f"{prefix}LEVERAGE", str(settings.leverage))
            updates.append(f"{prefix}LEVERAGE={settings.leverage}")

        if settings.min_confidence is not None:
            if not 0 <= settings.min_confidence <= 100:
                raise HTTPException(status_code=400, detail="Confidence must be 0-100")
            key = "SNIPER_AI_MIN_CONFIDENCE" if ACTIVE_STRATEGY == 'sniper' else "RAMF_MIN_CONFIDENCE"
            content = _replace_config_value(content, key, str(settings.min_confidence))
            updates.append(f"{key}={settings.min_confidence}")

        if settings.max_daily_trades is not None:
            if not 1 <= settings.max_daily_trades <= 50:
                raise HTTPException(status_code=400, detail="Max trades must be 1-50")
            content = _replace_config_value(content, f"{prefix}MAX_DAILY_TRADES", str(settings.max_daily_trades))
            updates.append(f"{prefix}MAX_DAILY_TRADES={settings.max_daily_trades}")

        if settings.assets is not None:
            assets_list = [a.strip().upper() for a in settings.assets.split(",") if a.strip()]
            if not assets_list:
                raise HTTPException(status_code=400, detail="At least one asset required")
            assets_str = str(assets_list)
            content = _replace_config_value(content, f"{prefix}ASSETS", assets_str)
            updates.append(f"{prefix}ASSETS={assets_list}")

        if settings.stop_loss_pct is not None:
            if not 0.1 <= settings.stop_loss_pct <= 10:
                raise HTTPException(status_code=400, detail="Stop loss must be 0.1-10%")
            content = _replace_config_value(content, f"{prefix}STOP_LOSS_PCT", str(settings.stop_loss_pct))
            updates.append(f"{prefix}STOP_LOSS_PCT={settings.stop_loss_pct}")

        if settings.take_profit_pct is not None:
            if not 0.1 <= settings.take_profit_pct <= 20:
                raise HTTPException(status_code=400, detail="Take profit must be 0.1-20%")
            content = _replace_config_value(content, f"{prefix}TAKE_PROFIT_PCT", str(settings.take_profit_pct))
            updates.append(f"{prefix}TAKE_PROFIT_PCT={settings.take_profit_pct}")

        # Sniper-specific feature toggles
        if ACTIVE_STRATEGY == 'sniper':
            if settings.use_trailing_stop is not None:
                content = _replace_config_value(content, "SNIPER_USE_TRAILING_STOP", str(settings.use_trailing_stop))
                updates.append(f"SNIPER_USE_TRAILING_STOP={settings.use_trailing_stop}")

            if settings.use_regime_filter is not None:
                content = _replace_config_value(content, "SNIPER_USE_REGIME_FILTER", str(settings.use_regime_filter))
                updates.append(f"SNIPER_USE_REGIME_FILTER={settings.use_regime_filter}")

            if settings.use_correlation_sizing is not None:
                content = _replace_config_value(content, "SNIPER_USE_CORRELATION_SIZING", str(settings.use_correlation_sizing))
                updates.append(f"SNIPER_USE_CORRELATION_SIZING={settings.use_correlation_sizing}")

            if settings.enable_funding_arbitrage is not None:
                content = _replace_config_value(content, "SNIPER_ENABLE_FUNDING_ARBITRAGE", str(settings.enable_funding_arbitrage))
                updates.append(f"SNIPER_ENABLE_FUNDING_ARBITRAGE={settings.enable_funding_arbitrage}")

            if settings.use_weighted_scoring is not None:
                content = _replace_config_value(content, "SNIPER_USE_WEIGHTED_SCORING", str(settings.use_weighted_scoring))
                updates.append(f"SNIPER_USE_WEIGHTED_SCORING={settings.use_weighted_scoring}")

            if settings.use_confidence_sizing is not None:
                content = _replace_config_value(content, "SNIPER_USE_CONFIDENCE_SIZING", str(settings.use_confidence_sizing))
                updates.append(f"SNIPER_USE_CONFIDENCE_SIZING={settings.use_confidence_sizing}")

            if settings.min_weighted_score is not None:
                if not 0 <= settings.min_weighted_score <= 10:
                    raise HTTPException(status_code=400, detail="Min weighted score must be 0-10")
                content = _replace_config_value(content, "SNIPER_MIN_WEIGHTED_SCORE", str(settings.min_weighted_score))
                updates.append(f"SNIPER_MIN_WEIGHTED_SCORE={settings.min_weighted_score}")

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
