"""
Emotion detection endpoints for IndexTTS2 API.
Handles text-based emotion detection using the QwenEmotion model.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field, root_validator
import logging

from ..models import BaseResponse
from ..exceptions import (
    IndexTTSException, ModelNotLoadedError, ValidationError
)
from ..config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


class EmotionDetectionRequest(BaseModel):
    """Request model for emotion detection from text."""
    text: Optional[str] = Field(None, description="Single text to analyze for emotions")
    batch_texts: Optional[List[str]] = Field(None, description="Optional batch of texts to analyze")
    
    # Allow camelCase alias for compatibility
    class Config:
        extra = "allow"  # Allow extra fields to prevent validation errors
        
    @root_validator(pre=True)
    def validate_request_payload(cls, values):
        values = dict(values or {})

        if values.get('batch_texts') is None and values.get('batchTexts') is not None:
            values['batch_texts'] = values.get('batchTexts')

        if values.get('text') is None and not values.get('batch_texts'):
            raise ValueError("Either 'text' or 'batch_texts' must be provided")

        return values


class EmotionDetectionResponse(BaseModel):
    """Response model for emotion detection results."""
    emotion_vectors: List[float] = Field(..., description="8-dimensional emotion vector")
    emotion_dict: Dict[str, float] = Field(..., description="Emotion dictionary with labels")
    text: str = Field(..., description="Original text that was analyzed")


class BatchEmotionDetectionResponse(BaseModel):
    """Response model for batch emotion detection results."""
    results: List[EmotionDetectionResponse] = Field(..., description="List of emotion detection results")


def get_tts_core(request: Request):
    """Get TTS core from app state."""
    app = request.app
    if not getattr(app.state, 'model_loaded', False):
        raise ModelNotLoadedError()
    return app.state.tts_core


def get_qwen_emo_model(tts_core):
    """Resolve the lazily-loaded Qwen emotion model from the runtime."""
    get_qwen_emo = getattr(tts_core, '_get_qwen_emo', None)
    if callable(get_qwen_emo):
        return get_qwen_emo()

    qwen_emo = getattr(tts_core, 'qwen_emo', None)
    if qwen_emo is None:
        raise ValidationError("Emotion detection model not available")

    return qwen_emo


@router.post("/detect", response_model=BaseResponse)
async def detect_emotion_from_text(
    request: EmotionDetectionRequest,
    http_request: Request
):
    """
    Detect emotions from text using the QwenEmotion model.
    
    Args:
        request: Emotion detection request
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Emotion detection results
    """
    try:
        # Add detailed logging for debugging
        logger.info(f"🔍 Emotion detection request received:")
        logger.info(f"  - text: '{request.text}'")
        logger.info(f"  - batch_texts: {request.batch_texts}")
        logger.info(f"  - Request type: {'batch' if request.batch_texts else 'single'}")
        
        tts_core = get_tts_core(http_request)
        qwen_emo = get_qwen_emo_model(tts_core)
        
        # Check if QwenEmotion model is available
        if not hasattr(tts_core, 'qwen_emo'):
            logger.error("❌ Emotion detection model not available")
            raise ValidationError("Emotion detection model not available")
        
        logger.info("✅ Emotion detection model is available")
        
        # Process single text or batch
        if request.batch_texts:
            logger.info(f"🔄 Processing batch of {len(request.batch_texts)} texts")
            # Pure batch processing - only process batch_texts, ignore text field
            results = []
            texts_to_process = request.batch_texts
            
            for text in texts_to_process:
                if not text or not text.strip():
                    # Skip empty texts, use default calm emotion
                    emotion_dict = {"calm": 1.0}
                    emotion_vector = [0, 0, 0, 0, 0, 0, 0, 1.0]
                else:
                    # Detect emotions using QwenEmotion model
                    emotion_dict = qwen_emo.inference(text.strip())
                    # Convert to vector format
                    emotion_vector = list(emotion_dict.values())
                
                results.append(EmotionDetectionResponse(
                    emotion_vectors=emotion_vector,
                    emotion_dict=emotion_dict,
                    text=text
                ))
            
            return BaseResponse(
                message=f"Emotion detection completed for {len(results)} texts",
                details={"results": [result.dict() for result in results]}
            )
        else:
            # Single text processing
            logger.info(f"🔄 Processing single text: '{request.text}'")
            
            if not request.text or not request.text.strip():
                # Empty text, use default calm emotion
                logger.warning(f"⚠️ Empty single text detected, using default calm emotion")
                emotion_dict = {"calm": 1.0}
                emotion_vector = [0, 0, 0, 0, 0, 0, 0, 1.0]
            else:
                # Detect emotions using QwenEmotion model
                try:
                    logger.info(f"🧠 Running emotion inference on single text: '{request.text.strip()}'")
                    emotion_dict = qwen_emo.inference(request.text.strip())
                    logger.info(f"✅ Emotion inference result: {emotion_dict}")
                    # Convert to vector format
                    emotion_vector = list(emotion_dict.values())
                    logger.info(f"✅ Emotion vector: {emotion_vector}")
                except Exception as emo_error:
                    logger.error(f"❌ Emotion inference failed for single text '{request.text}': {emo_error}")
                    # Use default calm emotion on failure
                    emotion_dict = {"calm": 1.0}
                    emotion_vector = [0, 0, 0, 0, 0, 0, 0, 1.0]
            
            result = EmotionDetectionResponse(
                emotion_vectors=emotion_vector,
                emotion_dict=emotion_dict,
                text=request.text
            )
            
            logger.info(f"✅ Single text processing completed")
            return BaseResponse(
                message="Emotion detection completed",
                details=result.dict()
            )
        
    except Exception as e:
        logger.error(f"❌ Emotion detection failed: {type(e).__name__} - {str(e)}")
        logger.error(f"❌ Request details: text='{request.text}', batch_texts={request.batch_texts}")
        
        if isinstance(e, IndexTTSException):
            logger.error(f"❌ IndexTTS Exception: {e.error_code} - {e.message}")
            raise
        logger.error(f"❌ Raising ValidationError with detailed error info")
        raise ValidationError(f"Failed to detect emotions: {str(e)}")


@router.get("/emotions", response_model=BaseResponse)
async def get_supported_emotions():
    """
    Get the list of supported emotions and their order.
    
    Returns:
        BaseResponse: Supported emotions information
    """
    try:
        # Return the standard emotion order used by IndexTTS2
        emotions = [
            {"name": "happy", "index": 0, "description": "Happy, joyful emotion"},
            {"name": "sad", "index": 1, "description": "Sad, melancholic emotion"},
            {"name": "angry", "index": 2, "description": "Angry, frustrated emotion"},
            {"name": "afraid", "index": 3, "description": "Fearful, scared emotion"},
            {"name": "disgusted", "index": 4, "description": "Disgusted, repulsed emotion"},
            {"name": "melancholic", "index": 5, "description": "Melancholic, pensive emotion"},
            {"name": "surprised", "index": 6, "description": "Surprised, amazed emotion"},
            {"name": "calm", "index": 7, "description": "Calm, neutral emotion"}
        ]
        
        return BaseResponse(
            message="Supported emotions retrieved",
            details={
                "emotions": emotions,
                "vector_dimensions": 8,
                "vector_range": {"min": 0.0, "max": 1.2}
            }
        )
        
    except Exception as e:
        raise ValidationError(f"Failed to get supported emotions: {str(e)}")
