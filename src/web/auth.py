"""
Basic Authentication for Moon Dev Trading Dashboard
"""

import os
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    Verify HTTP Basic Auth credentials.

    Returns:
        str: Username if credentials are valid

    Raises:
        HTTPException: 401 if credentials are invalid
    """
    expected_username = os.getenv("WEB_USERNAME", "admin")
    expected_password = os.getenv("WEB_PASSWORD", "changeme")

    correct_username = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        expected_username.encode("utf-8")
    )
    correct_password = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        expected_password.encode("utf-8")
    )

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username
