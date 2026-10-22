import hmac
import os

from fastapi import Request
from fastapi.responses import JSONResponse


def configured_token() -> str:
    return os.environ.get("AGENTDOCK_NODE_API_TOKEN", "").strip()


def is_authorized(request: Request) -> bool:
    token = configured_token()
    authorization = request.headers.get("authorization", "")
    supplied = (
        authorization[7:].strip()
        if authorization.lower().startswith("bearer ")
        else request.headers.get("x-api-key", "").strip()
    )
    if not supplied:
        supplied = request.headers.get("x-api-key", "").strip()
    return bool(
        token
        and supplied
        and hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8"))
    )


async def auth_middleware(request: Request, call_next):
    if request.method == "OPTIONS" or request.url.path in {"/", "/alive", "/health"}:
        return await call_next(request)
    if not configured_token():
        return JSONResponse(
            {"detail": "AGENTDOCK_NODE_API_TOKEN is not configured"},
            status_code=503,
        )
    if not is_authorized(request):
        return JSONResponse(
            {"detail": "Authentication required"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)
