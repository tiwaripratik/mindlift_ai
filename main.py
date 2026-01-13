# app/main.py
"""
MindLift.ai - Main FastAPI Application

Proactive Mental Health Assessment Platform
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time
import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.routes import conversation_router, websocket_router
from app.api.models import HealthCheckResponse, ErrorResponse
from app.config.settings import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for application startup and shutdown
    """
    # Startup
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"📊 Debug mode: {settings.debug}")
    logger.info(f"🔐 Crisis detection: {settings.enable_crisis_detection}")
    logger.info(f"💬 Proactive injection: {settings.enable_proactive_injection}")

    # Validate API keys
    if not settings.gemini_api_key and not settings.openai_api_key:
        logger.warning("⚠️  No AI API keys configured!")

    yield

    # Shutdown
    logger.info(f"👋 Shutting down {settings.app_name}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    ## MindLift.ai - Proactive Mental Health Assessment Platform

    A conversational AI system that conducts natural, empathetic mental health assessments
    using the PHQ-8 depression screening framework.

    ### Features
    - 🤖 **Multi-Model AI**: Combines Gemini, GPT, and Claude for optimal responses
    - 🧠 **Proactive Assessment**: Subtly explores all 8 PHQ-8 dimensions
    - 🚨 **Crisis Detection**: Immediate identification and resource provision
    - 📊 **Real-Time Tracking**: Live progress and symptom detection
    - 💬 **WebSocket Support**: Real-time bidirectional communication

    ### Authentication
    Currently in development mode. Production deployment will require authentication.

    ### Rate Limiting
    - REST API: 100 requests/minute per IP
    - WebSocket: 50 messages/minute per connection

    ### Support
    For issues or questions, contact support@mindlift.ai
    """,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "conversation",
            "description": "Mental health conversation management"
        },
        {
            "name": "websocket",
            "description": "Real-time chat via WebSocket"
        },
        {
            "name": "health",
            "description": "Health checks and monitoring"
        }
    ]
)

# ============================================================================
# Middleware
# ============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)

# GZip Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request ID and timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request timing and ID to response headers"""
    import uuid

    request_id = str(uuid.uuid4())
    start_time = time.time()

    # Add request ID to request state
    request.state.request_id = request_id

    response = await call_next(request)

    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id

    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    logger.info(f"📥 {request.method} {request.url.path}")

    response = await call_next(request)

    logger.info(f"📤 {request.method} {request.url.path} - {response.status_code}")

    return response


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="ValidationError",
            message="Invalid request data",
            detail=str(exc.errors())
        ).model_dump()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"❌ Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
            detail=str(exc) if settings.debug else None
        ).model_dump()
    )


# ============================================================================
# Routes
# ============================================================================

# Include routers
app.include_router(conversation_router)
app.include_router(websocket_router)


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["health"],
    summary="Health check",
    description="Check if the service is healthy and get status of dependencies"
)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint

    **Returns:**
    - Service status
    - Version information
    - Status of dependent services
    """
    services_status = {
        "api": "healthy",
        "orchestrator": "healthy",
        "gemini": "configured" if settings.gemini_api_key else "not_configured",
        "openai": "configured" if settings.openai_api_key else "not_configured",
        "anthropic": "configured" if settings.anthropic_api_key else "not_configured"
    }

    return HealthCheckResponse(
        status="healthy",
        version=settings.app_version,
        services=services_status
    )


# Root endpoint
@app.get(
    "/",
    tags=["health"],
    summary="API root",
    description="API information and links"
)
async def root():
    """
    API root endpoint

    **Returns basic API information and useful links**
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "documentation": "/docs" if settings.debug else "Contact admin for docs",
        "health_check": "/health",
        "websocket": "ws://your-domain/api/v1/ws/chat/{user_id}",
        "message": "Welcome to MindLift.ai - Proactive Mental Health Assessment Platform"
    }


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", 8000)),
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
