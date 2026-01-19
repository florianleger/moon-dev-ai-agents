"""
API endpoints for Moon Dev Trading Dashboard
"""

from fastapi import APIRouter

# Create main API router
api_router = APIRouter(prefix="/api")

# Import and include sub-routers
from .dashboard import router as dashboard_router
from .positions import router as positions_router
from .settings import router as settings_router
from .strategy import router as strategy_router

api_router.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])
api_router.include_router(positions_router, prefix="/positions", tags=["positions"])
api_router.include_router(settings_router, prefix="/settings", tags=["settings"])
api_router.include_router(strategy_router, prefix="/strategy", tags=["strategy"])

__all__ = ['api_router']
