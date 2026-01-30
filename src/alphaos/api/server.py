
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from starlette.responses import Response

from alphaos.core.config import AlphaOSConfig

logger = logging.getLogger("alphaos.api")


class SPAStaticFiles(StaticFiles):
    """
    Serve a Vite/React SPA from a dist folder, with HTML fallback.

    - Serves real files when present (e.g. /assets/*).
    - For unknown paths, falls back to dist/index.html so React Router works.
    """

    def __init__(self, directory: str | Path, index: str = "index.html"):
        super().__init__(directory=str(directory), html=True, check_dir=False)
        self._index_path = Path(directory) / index

    async def get_response(self, path: str, scope) -> Response:  # type: ignore[override]
        response = await super().get_response(path, scope)
        if response.status_code != 404:
            return response

        # Do not swallow API routes
        req_path = scope.get("path", "")
        if req_path.startswith("/api"):
            return response

        if self._index_path.exists():
            return await super().get_response(self._index_path.name, scope)

        return PlainTextResponse(
            "UI dist not found. Build UI first: cd ui && npm ci && npm run build",
            status_code=503,
        )


def create_app(config: AlphaOSConfig, ui_dist_path: str | Path | None = None) -> FastAPI:
    """
    Create an AlphaOS FastAPI app.

    - Exposes /api/* endpoints for the UI.
    - Optionally serves the built UI from ui_dist_path (Vite build output).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO)
        logger.info("Starting AlphaOS API Server...")

        # DB engine (avoid module-level globals so this can be embedded)
        app.state.db_engine = create_async_engine(
            config.database.connection_string,
            echo=False,
        )
        try:
            yield
        finally:
            logger.info("Shutting down AlphaOS API Server...")
            engine: AsyncEngine | None = getattr(app.state, "db_engine", None)
            if engine is not None:
                await engine.dispose()

    app = FastAPI(title="AlphaOS API", lifespan=lifespan)

    # CORS - Allow UI to connect
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health_check():
        return {"status": "ok", "service": "alphaos-api"}

    @app.get("/api/history/runtime", response_model=list[dict[str, Any]])
    async def get_runtime_history(limit: int = Query(1000, le=5000), offset: int = 0):
        """
        Fetch historical runtime state (Temperature/Entropy).
        Returns raw dictionaries to avoid tight coupling with RuntimeSnapshot validation
        if DB schema slightly diverges (e.g. timestamp format).
        """
        engine: AsyncEngine | None = getattr(app.state, "db_engine", None)
        if engine is None:
            raise HTTPException(status_code=500, detail="DB engine not initialized")

        async with AsyncSession(engine) as session:
            try:
                query = text(
                    """
                    SELECT * FROM runtime_state
                    ORDER BY timestamp DESC
                    LIMIT :limit OFFSET :offset
                    """
                )
                result = await session.execute(query, {"limit": limit, "offset": offset})
                rows = result.mappings().all()
                payload: list[dict[str, Any]] = []
                for row in rows:
                    data = dict(row)
                    if "timestamp" not in data and "time" in data:
                        ts = data["time"]
                        try:
                            data["timestamp"] = ts.timestamp() if ts is not None else None
                        except Exception:
                            data["timestamp"] = None
                    payload.append(data)
                return payload
            except Exception as e:
                logger.error("Error fetching runtime history: %s", e)
                if "does not exist" in str(e):
                    return []
                raise HTTPException(status_code=500, detail=str(e))

    # Static UI hosting (prod)
    if ui_dist_path is not None:
        dist = Path(ui_dist_path)
        # Mount last so /api routes keep priority.
        app.mount("/", SPAStaticFiles(dist), name="ui")

        index_path = dist / "index.html"
        if index_path.exists():
            # Explicit SPA routes for common client-side paths.
            @app.get("/live")
            async def ui_live():
                return FileResponse(index_path)

            @app.get("/analytics")
            async def ui_analytics():
                return FileResponse(index_path)

            @app.get("/architecture")
            async def ui_architecture():
                return FileResponse(index_path)

    return app


# Expose 'app' for uvicorn workers (uv run python -m alphaos.api.server runs via uvicorn)
# When running as module, this variable must exist
app = create_app(AlphaOSConfig(), ui_dist_path=Path("ui") / "dist")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
