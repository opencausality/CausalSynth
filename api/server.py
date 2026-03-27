"""CausalSynth FastAPI application factory.

Usage:
    # Programmatic
    from causalsynth.api.server import create_app
    app = create_app()

    # Via uvicorn (used by CLI)
    uvicorn.run("causalsynth.api.server:create_app", factory=True, ...)
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from causalsynth import __version__

logger = logging.getLogger("causalsynth.api.server")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="CausalSynth API",
        description=(
            "Generate causally-faithful synthetic data that preserves the "
            "structural causal model of the original data."
        ),
        version=__version__,
        contact={
            "name": "CausalSynth",
            "url": "https://github.com/causalsynth/causalsynth",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    # CORS — allow all origins for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from causalsynth.api.routes import router

    app.include_router(router, prefix="")

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("CausalSynth API v%s started.", __version__)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("CausalSynth API shutting down.")

    return app
