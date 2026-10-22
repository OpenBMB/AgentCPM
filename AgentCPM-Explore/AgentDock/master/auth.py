import hmac
import os

from fastapi import Request
from fastapi.responses import JSONResponse


AUTH_COOKIE = "agentdock_token"


def configured_token() -> str:
    return os.environ.get("AGENTDOCK_API_TOKEN", "").strip()


def configured_node_token() -> str:
    return os.environ.get("AGENTDOCK_NODE_API_TOKEN", "").strip()


def tokens_are_distinct() -> bool:
    manager_token = configured_token().encode("utf-8")
    node_token = configured_node_token().encode("utf-8")
    return bool(manager_token and node_token and not hmac.compare_digest(manager_token, node_token))


def security_configured() -> bool:
    return tokens_are_distinct()


def request_token(request: Request) -> str:
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
        if supplied:
            return supplied
    return request.headers.get("x-api-key", "").strip() or request.cookies.get(AUTH_COOKIE, "")


def is_authorized(request: Request) -> bool:
    token = configured_token()
    supplied = request_token(request)
    return bool(
        token
        and supplied
        and hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8"))
    )


def node_proxy_headers(request: Request) -> dict[str, str]:
    if not security_configured():
        raise RuntimeError("AgentDock API tokens are not configured or are not distinct")

    return {
        key: value
        for key, value in request.headers.items()
        if key.lower()
        not in {"host", "authorization", "proxy-authorization", "x-api-key", "cookie"}
    } | {"X-API-Key": configured_node_token()}


async def auth_middleware(request: Request, call_next):
    if request.method == "OPTIONS" or request.url.path in {"/", "/alive", "/auth/login"}:
        return await call_next(request)
    if request.url.path.startswith("/static/"):
        return await call_next(request)
    if not security_configured():
        return JSONResponse(
            {"detail": "AgentDock API tokens are not configured or are not distinct"},
            status_code=503,
        )
    if not is_authorized(request):
        return JSONResponse(
            {"detail": "Authentication required"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await call_next(request)
