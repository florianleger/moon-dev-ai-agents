"""
Moon Dev Trading Dashboard - FastAPI Application
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.web.auth import verify_credentials
from src.web.api import api_router
from src.web.state import get_strategy_state


# Lifespan for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("[Web] Starting Moon Dev Trading Dashboard...")
    yield
    print("[Web] Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Moon Dev Trading Dashboard",
    description="Trading dashboard for RAMF strategy",
    version="1.0.0",
    lifespan=lifespan,
)

# Static files and templates
static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "templates"

if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

templates = Jinja2Templates(directory=str(templates_path))


# Include API routes
app.include_router(api_router)


# Health check endpoint (no auth required for Docker/Coolify healthchecks)
@app.get("/health")
async def health_check():
    """Health check endpoint - no authentication required."""
    return {"status": "healthy", "service": "moon-dev-trading-dashboard"}


# Page routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request, username: str = Depends(verify_credentials)):
    """Redirect to dashboard."""
    return RedirectResponse(url="/dashboard", status_code=302)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, username: str = Depends(verify_credentials)):
    """Main dashboard page."""
    state = get_strategy_state()
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "username": username,
            "strategy_running": state.get("running", False),
        }
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, username: str = Depends(verify_credentials)):
    """Settings page."""
    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "username": username,
        }
    )


@app.get("/signals", response_class=HTMLResponse)
async def signals_page(request: Request, username: str = Depends(verify_credentials)):
    """Signals history page."""
    return templates.TemplateResponse(
        "signals.html",
        {
            "request": request,
            "username": username,
        }
    )


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("WEB_PORT", "8080"))
    host = os.getenv("WEB_HOST", "0.0.0.0")

    print(f"\n[Web] Starting server at http://{host}:{port}")
    print(f"[Web] Login with WEB_USERNAME/WEB_PASSWORD from .env\n")

    uvicorn.run(
        "src.web.app:app",
        host=host,
        port=port,
        reload=False,
    )
