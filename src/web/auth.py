"""
Basic Authentication + Cookie Session for Moon Dev Trading Dashboard

Page routes use Basic Auth and set a session cookie on success.
API routes accept either Basic Auth or the session cookie.
This ensures fetch() and EventSource work reliably without
embedding credentials in JavaScript.
"""

import hashlib
import hmac
import os
import secrets
import time

from fastapi import Cookie, Depends, HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

# Session cookie name
SESSION_COOKIE = "moon_session"
# Session valid for 24 hours
SESSION_MAX_AGE = 86400


def _get_session_secret() -> str:
    """Derive a stable session signing secret from WEB_PASSWORD."""
    password = os.getenv("WEB_PASSWORD", "")
    return hashlib.sha256(f"moon-session-{password}".encode()).hexdigest()


def _sign_session(username: str) -> str:
    """Create a signed session token: username:expiry:signature."""
    expiry = int(time.time()) + SESSION_MAX_AGE
    payload = f"{username}:{expiry}"
    sig = hmac.new(
        _get_session_secret().encode(), payload.encode(), hashlib.sha256
    ).hexdigest()[:32]
    return f"{payload}:{sig}"


def _verify_session(token: str) -> str | None:
    """Verify a signed session token. Returns username or None."""
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return None
        username, expiry_str, sig = parts
        expiry = int(expiry_str)
        if time.time() > expiry:
            return None
        payload = f"{username}:{expiry_str}"
        expected_sig = hmac.new(
            _get_session_secret().encode(), payload.encode(), hashlib.sha256
        ).hexdigest()[:32]
        if not hmac.compare_digest(sig, expected_sig):
            return None
        return username
    except Exception:
        return None


def _check_basic_auth(credentials: HTTPBasicCredentials) -> str | None:
    """Check Basic Auth credentials. Returns username or None."""
    expected_username = os.getenv("WEB_USERNAME", "admin")
    expected_password = os.getenv("WEB_PASSWORD")
    if not expected_password:
        return None

    correct_username = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        expected_username.encode("utf-8"),
    )
    correct_password = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        expected_password.encode("utf-8"),
    )

    if correct_username and correct_password:
        return credentials.username
    return None


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Verify HTTP Basic Auth credentials (used for page routes).

    Returns:
        str: Username if credentials are valid

    Raises:
        HTTPException: 401 if credentials are invalid
    """
    expected_password = os.getenv("WEB_PASSWORD")
    if not expected_password:
        raise ValueError("WEB_PASSWORD environment variable must be set")

    username = _check_basic_auth(credentials)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return username


def verify_credentials_or_cookie(request: Request) -> str:
    """
    Verify auth via Basic Auth OR session cookie (used for API routes).

    Tries Basic Auth first (for direct API calls), then falls back to
    session cookie (for browser fetch/EventSource calls).
    """
    # Try Basic Auth from Authorization header
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("basic "):
        import base64
        try:
            decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            username_str, _, password_str = decoded.partition(":")
            creds = HTTPBasicCredentials(username=username_str, password=password_str)
            result = _check_basic_auth(creds)
            if result:
                return result
        except Exception:
            pass

    # Try session cookie
    cookie_value = request.cookies.get(SESSION_COOKIE)
    if cookie_value:
        username = _verify_session(cookie_value)
        if username:
            return username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Basic"},
    )
