"""
Moon Dev Trading Dashboard - FastAPI Application
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.web.auth import verify_credentials, _sign_session, SESSION_COOKIE, SESSION_MAX_AGE
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

# Rate limiting state
_request_counts: dict = defaultdict(list)
_RATE_LIMIT = 60  # max requests per minute per IP


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiter: 60 requests per minute per IP."""
    # Skip rate limiting for health checks
    if request.url.path == "/health":
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Clean old entries (older than 60s)
    _request_counts[client_ip] = [t for t in _request_counts[client_ip] if now - t < 60]

    if len(_request_counts[client_ip]) >= _RATE_LIMIT:
        return JSONResponse(status_code=429, content={"detail": "Too many requests"})

    _request_counts[client_ip].append(now)
    return await call_next(request)


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


@app.get("/api/health")
async def api_health():
    """Rich health snapshot for monitoring + alerting integrations.

    No auth required so external uptime checks (UptimeRobot, etc.) can hit it.
    Returns scheduler freshness + active strategy count + uptime so we can
    detect frozen-bot scenarios externally.
    """
    try:
        from src.utils.scheduler_healthcheck import get_health_snapshot
        return get_health_snapshot()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": str(e)[:200]},
        )


# Helper to set session cookie on page responses
def _set_session_cookie(response, username: str):
    """Set a signed session cookie so fetch()/EventSource can authenticate."""
    response.set_cookie(
        key=SESSION_COOKIE,
        value=_sign_session(username),
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return response


# Page routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request, username: str = Depends(verify_credentials)):
    """Redirect to dashboard."""
    response = RedirectResponse(url="/dashboard", status_code=302)
    _set_session_cookie(response, username)
    return response


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request, username: str = Depends(verify_credentials)):
    """Main dashboard page."""
    state = get_strategy_state()
    response = templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "username": username,
            "strategy_running": state.get("running", False),
        }
    )
    _set_session_cookie(response, username)
    return response


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, username: str = Depends(verify_credentials)):
    """Settings page."""
    response = templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "username": username,
        }
    )
    _set_session_cookie(response, username)
    return response


@app.get("/signals", response_class=HTMLResponse)
async def signals_page(request: Request, username: str = Depends(verify_credentials)):
    """Signals history page."""
    response = templates.TemplateResponse(
        "signals.html",
        {
            "request": request,
            "username": username,
        }
    )
    _set_session_cookie(response, username)
    return response


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
        log_level="warning",
    )
