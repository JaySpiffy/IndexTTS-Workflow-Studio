"""
IndexTTS2 FastAPI Backend Only
Simplified version that starts the FastAPI server without Gradio dependencies.
"""

import os
import sys
import argparse
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import logging

# Configure request logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import API routers
from backend.api.routers import (
    speakers,
    speakers_tools,
    conversation,
    conversation_results,
    audio_process,
    files,
    frontend,
    timeline,
    emotion_timeline,
    emotion_detection
)

# Import configuration and utilities
from backend.api.config import settings
from backend.api.exceptions import IndexTTSException
from backend.api.models import HealthResponse


def resolve_model_dir() -> str:
    """Resolve the model directory relative to the project root."""
    model_dir = Path(settings.model_path) if settings.model_path else project_root / "shared" / "models" / "checkpoints"

    if not model_dir.is_absolute():
        model_dir = (project_root / model_dir).resolve()

    return str(model_dir)


def build_runtime_args(verbose: bool = False):
    """Build the runtime arguments used to initialize the TTS model."""

    class RuntimeArgs:
        def __init__(self):
            self.verbose = verbose
            self.model_dir = resolve_model_dir()
            self.device = settings.resolve_device()
            self.fp16 = settings.resolve_fp16(self.device)
            self.deepspeed = settings.use_deepspeed
            self.cuda_kernel = settings.use_cuda_kernel

    return RuntimeArgs()


def get_requested_device_label() -> str:
    """Return the configured device label shown in health/status responses."""
    return settings.resolve_device() or "auto"


def get_runtime_device(app: FastAPI) -> str | None:
    """Return the actual loaded runtime device when available."""
    tts = getattr(app.state, "tts", None)
    device = getattr(tts, "device", None)
    if device is None:
        return None
    return str(device)


def is_deepspeed_enabled(app: FastAPI) -> bool:
    """Return whether the active runtime is using DeepSpeed acceleration."""
    tts = getattr(app.state, "tts", None)
    return bool(getattr(tts, "use_deepspeed", False))


def is_gpu_device(device: str | None) -> bool:
    """Return whether the resolved runtime device is an accelerator."""
    if not device:
        return False

    normalized = device.strip().lower()
    return normalized.startswith("cuda") or normalized == "xpu" or normalized == "mps"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    print("🚀 Starting IndexTTS2 FastAPI Backend...")
    
    # Initialize TTS model and other resources
    try:
        # Import core components directly without Gradio dependencies
        from backend.indextts.infer_v2 import IndexTTS2
        
        cmd_args = build_runtime_args()
        logger.info(
            "TTS runtime request: model_dir=%s device_setting=%s resolved_device=%s fp16_mode=%s use_fp16=%s deepspeed=%s cuda_kernel=%s",
            cmd_args.model_dir,
            settings.device,
            cmd_args.device or "auto",
            settings.fp16_mode,
            cmd_args.fp16,
            cmd_args.deepspeed,
            cmd_args.cuda_kernel,
        )
        
        # Initialize TTS engine
        tts = IndexTTS2(
            model_dir=cmd_args.model_dir,
            cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
            use_fp16=cmd_args.fp16,
            device=cmd_args.device,
            use_deepspeed=cmd_args.deepspeed,
            use_cuda_kernel=cmd_args.cuda_kernel,
        )
        
        # Store components in app state
        app.state.tts = tts
        app.state.tts_core = tts  # Add for compatibility with conversation.py
        app.state.cmd_args = cmd_args
        
        # Mark model as loaded
        app.state.model_loaded = True
        print("✅ TTS model loaded successfully")
        print(f"DEBUG: Stored TTS model in app.state.tts and app.state.tts_core")
        
        logger.info(
            "TTS runtime ready: device=%s use_fp16=%s use_deepspeed=%s use_cuda_kernel=%s",
            getattr(tts, "device", "unknown"),
            getattr(tts, "use_fp16", False),
            getattr(tts, "use_deepspeed", False),
            getattr(tts, "use_cuda_kernel", False),
        )

        # Initialize services
        from backend.api.services import (
            TTSService, SpeakerService, ConversationService,
            AudioProcessingService, FileService, TimelineService, EmotionService
        )
        
        # Import and initialize ConversationManager from standalone implementation
        from backend.api.core.conversation_manager import ConversationManager
        
        # Create a wrapper class to match the expected interface
        class TTSCoreWrapper:
            def __init__(self, tts_instance):
                self.tts = tts_instance
            
            # Expose the TTS instance directly for compatibility
            @property
            def tts_core(self):
                return self.tts
        
        # Initialize ConversationManager
        tts_core_wrapper = TTSCoreWrapper(tts)
        conversation_manager = ConversationManager(
            tts_core=tts_core_wrapper,
            cmd_args=cmd_args
        )
        
        # Store conversation manager in app state
        app.state.conversation_manager = conversation_manager
        
        # Initialize TTSCore for API usage
        from backend.api.core.tts_core import TTSCore
        app.state.tts_core_api = TTSCore(tts, cmd_args)
        
        # Initialize services with dependencies
        app.state.tts_service = TTSService(
            tts_core=tts,
            cmd_args=cmd_args
        )
        
        app.state.speaker_service = SpeakerService()
        app.state.conversation_service = ConversationService(
            conversation_manager=conversation_manager
        )
        app.state.audio_processing_service = AudioProcessingService()
        app.state.file_service = FileService()
        app.state.emotion_service = EmotionService()
        app.state.timeline_service = TimelineService(
            tts_service=app.state.tts_service,
            conversation_service=app.state.conversation_service,
            emotion_service=app.state.emotion_service
        )
        
        print("✅ Services initialized successfully")
        print(f"DEBUG: Initialized ConversationManager and stored in app.state.conversation_manager")
        
    except Exception as e:
        print(f"❌ Failed to initialize TTS backend: {e}")
        # Continue without TTS - endpoints will return appropriate errors
        app.state.model_loaded = False
    
    yield
    
    # Shutdown
    print("🛑 Shutting down IndexTTS2 FastAPI Backend...")
    # Cleanup resources if needed


# Create FastAPI application
app = FastAPI(
    title="IndexTTS2 API",
    description="API for IndexTTS2 text-to-speech system with emotion control and conversation generation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging."""
    method = request.method
    url = str(request.url)
    
    # Log the request
    logger.info(f"🔍 Incoming request: {method} {url}")
    
    # Process the request
    response = await call_next(request)
    
    # Log the response status
    logger.info(f"📤 Response status: {response.status_code} for {method} {url}")
    
    return response


# Exception handlers
@app.exception_handler(IndexTTSException)
async def indextts_exception_handler(request, exc: IndexTTSException):
    """Handle custom IndexTTS exceptions."""
    logger.error(f"❌ IndexTTS Exception: {exc.error_code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"❌ General Exception: {type(exc).__name__} - {str(exc)}")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request Method: {request.method}")
    
    # Try to get request body for debugging
    try:
        if hasattr(request, '_body') and request._body:
            logger.error(f"Request Body: {request._body.decode('utf-8')}")
    except Exception as body_error:
        logger.error(f"Could not read request body: {body_error}")
    
    # Create detailed error response
    error_details = {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "request_url": str(request.url),
        "request_method": request.method
    }
    
    # Add debug info if enabled
    if settings.debug:
        import traceback
        error_details["traceback"] = traceback.format_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "INTERNAL_ERROR",
            "message": f"An unexpected error occurred: {type(exc).__name__}",
            "details": error_details
        }
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    runtime_device = get_runtime_device(app)
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        model_loaded=getattr(app.state, 'model_loaded', False),
        requested_device=get_requested_device_label(),
        runtime_device=runtime_device,
        using_gpu=is_gpu_device(runtime_device),
        using_deepspeed=is_deepspeed_enabled(app),
    )


# Include API routers
app.include_router(frontend.router, tags=["frontend"])
app.include_router(speakers.router, prefix="/api/speakers", tags=["speakers"])
app.include_router(speakers_tools.router, prefix="/api/speakers-tools", tags=["speakers-tools"])
app.include_router(conversation.router, prefix="/api/conversation", tags=["conversation"])
app.include_router(conversation_results.router, prefix="/api/conversation/results", tags=["conversation-results"])
app.include_router(audio_process.router, prefix="/api/audio-process", tags=["audio-processing"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(timeline.router, prefix="/api/timeline", tags=["timeline"])
app.include_router(emotion_timeline.router, prefix="/api/emotion-timeline", tags=["emotion-timeline"])
app.include_router(emotion_detection.router, prefix="/api/emotion-detection", tags=["emotion-detection"])


def main():
    """Main entry point for running the server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="IndexTTS FastAPI Backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the web UI on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
    parser.add_argument("--model_dir", type=str, default=resolve_model_dir(), help="Model checkpoints directory")
    parser.add_argument("--device", type=str, default=settings.device, help="Inference device: auto, cpu, cuda, cuda:0, xpu, or mps")
    parser.add_argument("--fp16_mode", type=str, default=settings.fp16_mode, help="FP16 mode: auto, true, or false")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
    parser.add_argument("--deepspeed", action="store_true", default=settings.use_deepspeed, help="Use DeepSpeed to accelerate if available")
    parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Update settings with command line arguments
    settings.host = args.host
    settings.port = args.port
    settings.model_path = args.model_dir
    settings.device = args.device
    settings.fp16_mode = "true" if args.fp16 else args.fp16_mode
    settings.use_deepspeed = args.deepspeed
    settings.use_cuda_kernel = args.cuda_kernel
    
    # Run the server
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level
    )


if __name__ == "__main__":
    main()
