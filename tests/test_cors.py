from __future__ import annotations

import httpx
from fastapi import FastAPI

from app.config import Settings
from app.main import _configure_cors


def _test_app(app_settings: Settings) -> FastAPI:
    application = FastAPI()
    _configure_cors(application, app_settings)

    @application.post("/knowledge/files")
    async def upload() -> dict[str, bool]:
        return {"ok": True}

    return application


async def test_cors_is_disabled_by_default() -> None:
    application = _test_app(Settings(_env_file=None))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://backend.test",
    ) as client:
        response = await client.post(
            "/knowledge/files",
            headers={"Origin": "http://localhost:8080"},
        )

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


async def test_cors_origin_regex_allows_http_server_port_8080_only() -> None:
    application = _test_app(
        Settings(
            _env_file=None,
            cors_allow_origin_regex=r"^http://(?:localhost|127\.0\.0\.1):8080$",
        )
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://backend.test",
    ) as client:
        allowed = await client.options(
            "/knowledge/files",
            headers={
                "Origin": "http://localhost:8080",
                "Access-Control-Request-Method": "POST",
            },
        )
        rejected = await client.options(
            "/knowledge/files",
            headers={
                "Origin": "http://localhost:8081",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:8080"
    assert rejected.status_code == 400
    assert "access-control-allow-origin" not in rejected.headers


async def test_cors_exact_origins_are_comma_separated_and_trimmed() -> None:
    app_settings = Settings(
        _env_file=None,
        cors_allow_origins=" http://localhost:8080, http://rag.test:8080 ,,",
    )
    application = _test_app(app_settings)

    assert app_settings.cors_allow_origin_list == [
        "http://localhost:8080",
        "http://rag.test:8080",
    ]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application),
        base_url="http://backend.test",
    ) as client:
        response = await client.post(
            "/knowledge/files",
            headers={"Origin": "http://rag.test:8080"},
        )

    assert response.headers["access-control-allow-origin"] == "http://rag.test:8080"
