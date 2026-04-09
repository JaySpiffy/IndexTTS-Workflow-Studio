"""
Conversation results management endpoints for IndexTTS2 API.
Handles results processing, concatenation, and export.
"""

import uuid
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse

from ..models import (
    ConversationLineResult, LineVersion, BaseResponse, LineRegenerationRequest, LineRegenerationMode
)
from ..exceptions import (
    IndexTTSException, FileNotFoundError, ValidationError,
    AudioProcessingError
)
from ..config import settings
from ..core.app_paths import TEMP_CONVERSATION_SEGMENTS_DIR
from ..services import ConversationService, AudioProcessingService

router = APIRouter()


def get_conversation_service(request: Request) -> ConversationService:
    """Get conversation service from app state."""
    if not hasattr(request.app.state, 'conversation_service'):
        request.app.state.conversation_service = ConversationService(
            conversation_manager=request.app.state.conversation_manager if hasattr(request.app.state, 'conversation_manager') else None
        )
    
    return request.app.state.conversation_service


def get_audio_processing_service(request: Request) -> AudioProcessingService:
    """Get audio processing service from app state."""
    if not hasattr(request.app.state, 'audio_processing_service'):
        request.app.state.audio_processing_service = AudioProcessingService()
    
    return request.app.state.audio_processing_service


def _normalize_version_payload(conversation_service: ConversationService, version: Optional[Dict[str, Any]], is_selected: bool = False) -> Optional[Dict[str, Any]]:
    """Ensure regenerated versions have the fields the frontend expects."""
    if not isinstance(version, dict):
        return None

    normalized = deepcopy(version)
    audio_path = normalized.get("audio_path")
    if audio_path:
        normalized.setdefault("audio_filename", Path(audio_path).name)
        normalized.setdefault("audio_url", conversation_service._build_audio_url(audio_path))
    normalized["is_selected"] = bool(is_selected)
    return normalized


def _version_score(version: Optional[Dict[str, Any]]) -> float:
    """Compare versions using the same quality-first ordering as the UI."""
    if not isinstance(version, dict):
        return float("-inf")
    return float(version.get("quality_score", version.get("similarity_score", float("-inf"))))


def _best_version_index(versions: List[Dict[str, Any]]) -> int:
    """Return the index of the highest scoring version."""
    best_index = 0
    best_score = float("-inf")
    for index, version in enumerate(versions):
        score = _version_score(version)
        if score > best_score:
            best_index = index
            best_score = score
    return best_index


def _ensure_single_selection(versions: List[Dict[str, Any]], preferred_index: Optional[int] = None) -> List[Dict[str, Any]]:
    """Keep version selection stable and guarantee at most one selected item."""
    if not versions:
        return versions

    selected_indices = [index for index, version in enumerate(versions) if version.get("is_selected")]
    if preferred_index is None:
        preferred_index = selected_indices[0] if selected_indices else _best_version_index(versions)

    for index, version in enumerate(versions):
        version["is_selected"] = index == preferred_index
    return versions


def _merge_threshold_versions(
    conversation_service: ConversationService,
    existing_versions: List[Dict[str, Any]],
    regenerated_versions: List[Dict[str, Any]],
    slots_to_regenerate: List[int],
) -> List[Dict[str, Any]]:
    """Keep the best version in each slot while preserving slot order."""
    updated_versions: List[Dict[str, Any]] = []
    regenerated_index = 0
    slot_lookup = set(slots_to_regenerate)

    for slot_index in range(max(len(existing_versions), len(slots_to_regenerate))):
        existing_version = deepcopy(existing_versions[slot_index]) if slot_index < len(existing_versions) else None
        selected_before = bool(existing_version and existing_version.get("is_selected"))

        if slot_index in slot_lookup:
            candidate = (
                deepcopy(regenerated_versions[regenerated_index])
                if regenerated_index < len(regenerated_versions)
                else None
            )
            regenerated_index += 1
            chosen_version = candidate if _version_score(candidate) > _version_score(existing_version) else existing_version
        else:
            chosen_version = existing_version

        normalized = _normalize_version_payload(
            conversation_service,
            chosen_version,
            is_selected=selected_before,
        )
        if normalized:
            updated_versions.append(normalized)

    return _ensure_single_selection(updated_versions)


@router.get("/{conversation_id}/results")
async def get_conversation_results(conversation_id: str, request: Request):
    """
    Get the results of a completed conversation generation.
    
    Args:
        conversation_id: ID of the conversation
        request: HTTP request object
        
    Returns:
        BaseResponse: Conversation results
    """
    try:
        conversation_service = get_conversation_service(request)
        
        # Get conversation status from service
        status = conversation_service.get_conversation_status(conversation_id)
        
        if status["status"] != "completed":
            raise ValidationError(f"Conversation not completed: {conversation_id}")
        
        return BaseResponse(
            message=f"Retrieved results for conversation: {conversation_id}",
            details={
                "conversation_id": conversation_id,
                "lines": status.get("lines", []),
                "total_versions": status.get("total_versions", 0),
                "generation_time": status.get("end_time", 0) - status.get("start_time", 0),
                "generation_params": status.get("generation_params", {}),
                "seed_runtime_metadata": status.get("seed_runtime_metadata"),
                "concatenation_completed": status.get("concatenation_completed", False),
                "concatenation_output_path": status.get("concatenation_output_path"),
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to get conversation results: {str(e)}")


@router.get("/{conversation_id}/line/{line_number}/versions")
async def get_line_versions(conversation_id: str, line_number: int, request: Request):
    """
    Get all versions for a specific line in a conversation.
    
    Args:
        conversation_id: ID of the conversation
        line_number: Line number (0-based)
        request: HTTP request object
        
    Returns:
        BaseResponse: Line versions
    """
    try:
        conversation_service = get_conversation_service(request)
        
        # Get conversation status from service
        status = conversation_service.get_conversation_status(conversation_id)
        
        if status["status"] != "completed":
            raise ValidationError(f"Conversation not completed: {conversation_id}")
        
        lines = status.get("lines", [])
        
        if line_number >= len(lines):
            raise ValidationError(f"Line number {line_number} out of range")
        
        line = lines[line_number]
        
        return BaseResponse(
            message=f"Retrieved versions for line {line_number}",
            details={
                "line_number": line_number,
                "speaker_filename": line.get("speaker_filename"),
                "text": line.get("text"),
                "versions": line.get("versions", [])
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to get line versions: {str(e)}")


@router.post("/{conversation_id}/line/{line_number}/select-version")
async def select_line_version(
    conversation_id: str,
    line_number: int,
    version_index: int,
    request: Request
):
    """
    Select a specific version for a line.
    
    Args:
        conversation_id: ID of the conversation
        line_number: Line number (0-based)
        version_index: Version index to select
        request: HTTP request object
        
    Returns:
        BaseResponse: Selection result
    """
    try:
        conversation_service = get_conversation_service(request)
        
        # Get conversation status from service
        status = conversation_service.get_conversation_status(conversation_id)
        
        if status["status"] != "completed":
            raise ValidationError(f"Conversation not completed: {conversation_id}")
        
        lines = status.get("lines", [])
        
        if line_number >= len(lines):
            raise ValidationError(f"Line number {line_number} out of range")
        
        line = lines[line_number]
        versions = line.get("versions", [])
        
        if version_index >= len(versions):
            raise ValidationError(f"Version index {version_index} out of range")
        
        # Update selection
        for i, version in enumerate(versions):
            version["is_selected"] = (i == version_index)
        
        return BaseResponse(
            message=f"Selected version {version_index} for line {line_number}",
            details={
                "line_number": line_number,
                "selected_version_index": version_index,
                "selected_audio_path": versions[version_index].get("audio_path")
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to select line version: {str(e)}")


@router.post("/{conversation_id}/concatenate", response_model=BaseResponse)
async def concatenate_conversation(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Concatenate selected versions to create final conversation audio.
    
    Args:
        conversation_id: ID of the conversation
        background_tasks: FastAPI background tasks
        request: HTTP request object
        
    Returns:
        BaseResponse: Concatenation result
    """
    try:
        conversation_service = get_conversation_service(request)
        
        # Get conversation status from service
        status = conversation_service.get_conversation_status(conversation_id)
        
        if status["status"] != "completed":
            raise ValidationError(f"Conversation not completed: {conversation_id}")
        
        # Start concatenation in background
        background_tasks.add_task(
            concatenate_conversation_background,
            conversation_id,
            status.get("lines", []),
            conversation_service
        )
        
        return BaseResponse(
            message="Conversation concatenation started",
            details={"conversation_id": conversation_id}
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to start concatenation: {str(e)}")


async def concatenate_conversation_background(conversation_id: str, lines: List[dict], conversation_service: ConversationService):
    """
    Background task for conversation concatenation.
    
    Args:
        conversation_id: ID of the conversation
        lines: Conversation lines
        conversation_service: Conversation service instance
    """
    try:
        # Import here to avoid circular imports
        from pydub import AudioSegment
        
        # Check if pydub is available
        try:
            from pydub import AudioSegment
            PYDUB_AVAILABLE = True
        except ImportError:
            PYDUB_AVAILABLE = False
        
        if not PYDUB_AVAILABLE:
            # Update task with error
            status = conversation_service.get_conversation_status(conversation_id)
            if status:
                # Store error in service
                if hasattr(conversation_service.active_conversations, conversation_id):
                    conversation_service.active_conversations[conversation_id]["concatenation_error"] = "pydub not available for audio concatenation"
            return
        
        # Collect selected audio segments
        audio_segments = []
        
        for line_data in lines:
            versions = line_data.get("versions", [])
            
            # Find selected version or best version
            selected_version = None
            best_version = None
            best_score = -1.0
            
            for version in versions:
                if version.get("is_selected"):
                    selected_version = version
                    break
                
                # Track best version as fallback
                quality_score = version.get("quality_score", -1.0)
                if quality_score > best_score:
                    best_score = quality_score
                    best_version = version
            
            # Use selected version or best version
            version_to_use = selected_version or best_version
            
            if version_to_use:
                audio_path = version_to_use.get("audio_path")
                if audio_path and Path(audio_path).exists():
                    try:
                        audio_segment = AudioSegment.from_wav(audio_path)
                        audio_segments.append(audio_segment)
                    except Exception as e:
                        print(f"Error loading audio {audio_path}: {e}")
                        continue
        
        if not audio_segments:
            raise AudioProcessingError("No valid audio segments found for concatenation")
        
        # Concatenate audio segments
        final_audio = sum(audio_segments)
        
        # Save final conversation
        temp_dir = TEMP_CONVERSATION_SEGMENTS_DIR
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        final_filename = f"final_conversation_{conversation_id[:8]}_{uuid.uuid4().hex[:8]}.wav"
        final_path = temp_dir / final_filename
        
        final_audio.export(str(final_path), format="wav")
        
        # Update task with result
        if hasattr(conversation_service.active_conversations, conversation_id):
            conversation_service.active_conversations[conversation_id]["concatenated_audio_path"] = str(final_path)
            conversation_service.active_conversations[conversation_id]["concatenation_completed"] = True
        
    except Exception as e:
        # Update task with error
        if hasattr(conversation_service.active_conversations, conversation_id):
            conversation_service.active_conversations[conversation_id]["concatenation_error"] = str(e)


@router.get("/{conversation_id}/download")
async def download_conversation(conversation_id: str, request: Request):
    """
    Download the concatenated conversation audio.
    
    Args:
        conversation_id: ID of the conversation
        request: HTTP request object
        
    Returns:
        FileResponse: Audio file download
    """
    try:
        conversation_service = get_conversation_service(request)
        
        # Get conversation status from service
        status = conversation_service.get_conversation_status(conversation_id)
        
        if not status.get("concatenation_completed"):
            raise ValidationError(f"Conversation not concatenated: {conversation_id}")
        
        audio_path = status.get("concatenated_audio_path")
        
        if not audio_path or not Path(audio_path).exists():
            raise FileNotFoundError(f"Concatenated audio file not found: {conversation_id}")
        
        return FileResponse(
            path=audio_path,
            filename=f"conversation_{conversation_id[:8]}.wav",
            media_type="audio/wav"
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to download conversation: {str(e)}")


@router.get("/{conversation_id}/line/{line_number}/version/{version_index}/download")
async def download_line_version(
    conversation_id: str,
    line_number: int,
    version_index: int,
    request: Request
):
    """
    Download a specific version of a conversation line.
    
    Args:
        conversation_id: ID of the conversation
        line_number: Line number (0-based)
        version_index: Version index
        request: HTTP request object
        
    Returns:
        FileResponse: Audio file download
    """
    try:
        conversation_service = get_conversation_service(request)
        
        # Get conversation status from service
        status = conversation_service.get_conversation_status(conversation_id)
        
        if status["status"] != "completed":
            raise ValidationError(f"Conversation not completed: {conversation_id}")
        
        lines = status.get("lines", [])
        
        if line_number >= len(lines):
            raise ValidationError(f"Line number {line_number} out of range")
        
        line = lines[line_number]
        versions = line.get("versions", [])
        
        if version_index >= len(versions):
            raise ValidationError(f"Version index {version_index} out of range")
        
        version = versions[version_index]
        audio_path = version.get("audio_path")
        
        if not audio_path or not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        return FileResponse(
            path=audio_path,
            filename=f"line_{line_number:03d}_version_{version_index + 1}.wav",
            media_type="audio/wav"
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to download line version: {str(e)}")


@router.delete("/{conversation_id}/cleanup", response_model=BaseResponse)
async def cleanup_conversation_files(conversation_id: str, request: Request):
    """
    Clean up all generated files for a conversation.
    
    Args:
        conversation_id: ID of the conversation
        request: HTTP request object
        
    Returns:
        BaseResponse: Cleanup result
    """
    try:
        conversation_service = get_conversation_service(request)
        
        # Delete conversation using service
        result = conversation_service.delete_conversation(conversation_id, cleanup_files=True)
        
        return BaseResponse(
            message=result["message"],
            details={
                "deleted_files_count": len(result.get("deleted_files", []))
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to cleanup conversation files: {str(e)}")


@router.get("/{conversation_id}/concatenation-status")
async def get_concatenation_status(conversation_id: str, request: Request):
    """
    Get the concatenation status of a conversation.
    
    Args:
        conversation_id: ID of the conversation
        request: HTTP request object
        
    Returns:
        BaseResponse: Concatenation status
    """
    try:
        conversation_service = get_conversation_service(request)
        
        # Get conversation status from service
        status = conversation_service.get_conversation_status(conversation_id)
        
        return BaseResponse(
            message=f"Retrieved concatenation status for conversation: {conversation_id}",
            details={
                "conversation_id": conversation_id,
                "concatenation_completed": status.get("concatenation_completed", False),
                "concatenation_error": status.get("concatenation_error"),
                "concatenated_audio_path": status.get("concatenated_audio_path")
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to get concatenation status: {str(e)}")


@router.post("/{conversation_id}/line/{line_number}/regenerate")
async def regenerate_line(
    conversation_id: str,
    line_number: int,
    request_body: LineRegenerationRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Regenerate a specific line in a conversation with new versions.
    
    Args:
        conversation_id: ID of the conversation
        line_number: Line number (0-based)
        request_body: Line regeneration request body
        background_tasks: FastAPI background tasks
        request: HTTP request object
        
    Returns:
        BaseResponse: Regeneration result
    """
    regen_count = request_body.regen_count
    regen_mode = request_body.mode.value if hasattr(request_body.mode, "value") else str(request_body.mode)
    edited_text = request_body.edited_text
    manual_similarity_threshold = request_body.manual_similarity_threshold
    
    print(f"DEBUG: API router received regen_count={regen_count}, mode={regen_mode}")
    
    try:
        conversation_service = get_conversation_service(request)
        
        # Get conversation status from service
        status = conversation_service.get_conversation_status(conversation_id)
        
        if status["status"] != "completed":
            raise ValidationError(f"Conversation not completed: {conversation_id}")
        
        lines = status.get("lines", [])
        
        if line_number >= len(lines):
            raise ValidationError(f"Line number {line_number} out of range")
        
        line = lines[line_number]
        generation_params = conversation_service.active_conversations.get(conversation_id, {}).get("generation_params", {})

        if regen_mode == LineRegenerationMode.BELOW_THRESHOLD.value and manual_similarity_threshold is None:
            raise ValidationError("Manual similarity threshold is required for threshold regeneration")
        if regen_mode == LineRegenerationMode.BELOW_THRESHOLD.value and request_body.max_manual_attempts is None:
            max_manual_attempts = max(1, int(generation_params.get("auto_regen_attempts", 1) or 1))
        else:
            max_manual_attempts = request_body.max_manual_attempts
        
        # Create regeneration task ID
        regen_task_id = f"regen_{conversation_id}_{line_number}_{uuid.uuid4().hex[:8]}"
        
        # Initialize regeneration task
        conversation_service.active_conversations[regen_task_id] = {
            "status": "pending",
            "progress": 0.0,
            "current_step": "Initializing regeneration",
            "line_number": line_number,
            "regen_count": regen_count,
            "conversation_id": conversation_id,
            "mode": regen_mode,
            "edited_text": edited_text,
            "manual_similarity_threshold": manual_similarity_threshold,
            "max_manual_attempts": max_manual_attempts,
            "error": None,
            "start_time": time.time(),
            "end_time": None,
            "new_versions": []
        }
        
        # Start regeneration in background
        background_tasks.add_task(
            regenerate_line_background,
            regen_task_id,
            conversation_id,
            line_number,
            regen_count,
            line,
            conversation_service
        )
        
        return BaseResponse(
            message="Line regeneration started",
            details={
                "regen_task_id": regen_task_id,
                "conversation_id": conversation_id,
                "line_number": line_number,
                "regen_count": regen_count,
                "mode": regen_mode
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to start line regeneration: {str(e)}")


@router.get("/{conversation_id}/line/{line_number}/regenerate/status")
async def get_regeneration_status(conversation_id: str, line_number: int, request: Request):
    """
    Get the status of a line regeneration task.
    
    Args:
        conversation_id: ID of the conversation
        line_number: Line number (0-based)
        request: HTTP request object
        
    Returns:
        BaseResponse: Regeneration status
    """
    try:
        conversation_service = get_conversation_service(request)
        
        # Find regeneration task
        regen_task_id = None
        for task_id, task in reversed(list(conversation_service.active_conversations.items())):
            if (task_id.startswith("regen_") and
                task.get("conversation_id") == conversation_id and
                task.get("line_number") == line_number):
                regen_task_id = task_id
                break
        
        if not regen_task_id:
            raise ValidationError(f"Regeneration task not found for line {line_number}")
        
        task = conversation_service.active_conversations[regen_task_id]
        
        return BaseResponse(
            message=f"Retrieved regeneration status for line {line_number}",
            details={
                "regen_task_id": regen_task_id,
                "conversation_id": conversation_id,
                "line_number": line_number,
                "status": task["status"],
                "progress_percent": task["progress"],
                "current_step": task["current_step"],
                "error": task.get("error"),
                "new_versions": task.get("new_versions", [])
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise AudioProcessingError(f"Failed to get regeneration status: {str(e)}")


async def regenerate_line_background(
    regen_task_id: str,
    conversation_id: str,
    line_number: int,
    regen_count: int,
    line_data: dict,
    conversation_service
):
    """
    Background task for line regeneration.
    
    Args:
        regen_task_id: ID of the regeneration task
        conversation_id: ID of the conversation
        line_number: Line number (0-based)
        regen_count: Number of new versions to generate
        line_data: Original line data
        conversation_service: Conversation service instance
    """
    task = conversation_service.active_conversations[regen_task_id]

    try:
        # Update status to running
        task["status"] = "running"
        task["current_step"] = "Starting regeneration"
        
        # Get conversation manager
        conversation_manager = conversation_service.conversation_manager
        if not conversation_manager:
            task["status"] = "failed"
            task["error"] = "Conversation manager not available"
            task["end_time"] = time.time()
            return
        
        # Get the original task so regeneration can reuse line-level metadata
        # such as per-line emotion vectors from the parsed script.
        conversation_task = conversation_service.active_conversations.get(conversation_id, {})
        generation_params = conversation_task.get("generation_params", {})
        parsed_script = conversation_task.get("parsed_script", [])
        current_lines = conversation_task.get("lines", [])

        if line_number < len(parsed_script):
            source_line = dict(parsed_script[line_number])
            source_line.setdefault("speaker_filename", line_data.get("speaker_filename"))
            source_line.setdefault("text", line_data.get("text"))
            source_line.setdefault("line_number", line_number)
            line_data = source_line

        edited_text = task.get("edited_text")
        if edited_text is not None:
            edited_text = edited_text.strip()
            if not edited_text:
                raise ValidationError("Edited text cannot be empty")
            line_data = dict(line_data)
            line_data["text"] = edited_text
        else:
            edited_text = line_data.get("text")
        
        # Convert emotion control method to integer
        emo_control_method_map = {
            "from_speaker": 0,
            "from_reference": 1,
            "from_vectors": 2,
            "from_text": 3
        }
        emo_control_method_int = emo_control_method_map.get(generation_params.get("emotion_control_method", "from_speaker"), 0)
        
        # Process emotion vectors
        vec_args = [0.0] * 8
        emotion_vectors = generation_params.get("emotion_vectors", [])
        if emotion_vectors:
            for i, vec in enumerate(emotion_vectors[:8]):
                vec_args[i] = vec
        
        # Prepare regeneration parameters for ConversationManager
        regen_params = {
            "similarity_threshold": generation_params.get("similarity_threshold", 0.60),
            "robotic_threshold": generation_params.get("robotic_threshold", 0.70),
            "auto_regen_attempts": generation_params.get("auto_regen_attempts", 1),
            "emo_control_method": emo_control_method_int,
            "emo_ref_path": generation_params.get("emotion_reference_filename"),
            "emo_weight": generation_params.get("emotion_weight", 1.0),
            "emo_random": generation_params.get("use_random_sampling", False),
            "vec1": vec_args[0], "vec2": vec_args[1], "vec3": vec_args[2], "vec4": vec_args[3],
            "vec5": vec_args[4], "vec6": vec_args[5], "vec7": vec_args[6], "vec8": vec_args[7],
            "emo_text": generation_params.get("emotion_text"),
            "do_sample_convo": generation_params.get("do_sample", True),
            "top_p_convo": generation_params.get("top_p", 0.8),
            "top_k_convo": generation_params.get("top_k", 30),
            "temperature_convo": generation_params.get("temperature", 0.8),
            "length_penalty_convo": generation_params.get("length_penalty", 0.0),
            "num_beams_convo": generation_params.get("num_beams", 3),
            "repetition_penalty_convo": generation_params.get("repetition_penalty", 10.0),
            "max_mel_tokens_convo": generation_params.get("max_mel_tokens", 1500),
            "max_text_tokens_per_segment_convo": generation_params.get("max_text_tokens_per_segment", 120),
        }

        mode = task.get("mode", LineRegenerationMode.REPLACE_ALL.value)
        updated_versions: List[Dict[str, Any]] = []

        if mode == LineRegenerationMode.BELOW_THRESHOLD.value:
            manual_similarity_threshold = task.get("manual_similarity_threshold")
            if manual_similarity_threshold is None:
                raise ValidationError("Manual similarity threshold is required for threshold regeneration")

            existing_line = deepcopy(current_lines[line_number]) if line_number < len(current_lines) else {}
            existing_versions = deepcopy(existing_line.get("versions", []))
            target_slot_count = len(existing_versions) or max(
                int(generation_params.get("versions_per_line", regen_count) or regen_count or 1),
                regen_count,
                1,
            )
            slots_to_regenerate = []
            for slot_index in range(target_slot_count):
                version = existing_versions[slot_index] if slot_index < len(existing_versions) else None
                similarity_score = version.get("similarity_score") if isinstance(version, dict) else None
                if similarity_score is None or similarity_score < manual_similarity_threshold:
                    slots_to_regenerate.append(slot_index)

            if slots_to_regenerate:
                attempts = task.get("max_manual_attempts")
                if attempts is None:
                    attempts = max(1, int(generation_params.get("auto_regen_attempts", 1) or 1))

                threshold_regen_params = dict(regen_params)
                threshold_regen_params["similarity_threshold"] = manual_similarity_threshold
                threshold_regen_params["auto_regen_attempts"] = max(int(attempts) - 1, 0)

                print(
                    "DEBUG: API router calling threshold regenerate_line "
                    f"with slot_count={len(slots_to_regenerate)}, threshold={manual_similarity_threshold}"
                )
                regen_generator = conversation_manager.regenerate_line(
                    line_number,
                    line_data,
                    len(slots_to_regenerate),
                    **threshold_regen_params,
                )

                regenerated_versions: List[Dict[str, Any]] = []
                for status_log, progress_html, progress_value, versions in regen_generator:
                    task["progress"] = progress_value
                    task["current_step"] = f"Regenerating low-threshold slots for line {line_number + 1}"
                    if versions:
                        regenerated_versions = versions

                updated_versions = _merge_threshold_versions(
                    conversation_service,
                    existing_versions,
                    regenerated_versions,
                    slots_to_regenerate,
                )
            else:
                updated_versions = [
                    _normalize_version_payload(
                        conversation_service,
                        version,
                        is_selected=bool(version.get("is_selected")) if isinstance(version, dict) else False,
                    )
                    for version in existing_versions
                ]
                updated_versions = [version for version in updated_versions if version]
                updated_versions = _ensure_single_selection(updated_versions)
                task["current_step"] = "No versions were below the manual threshold"

        else:
            print(f"DEBUG: API router calling regenerate_line with regen_count={regen_count}")
            regen_generator = conversation_manager.regenerate_line(
                line_number,
                line_data,
                regen_count,
                **regen_params
            )

            regenerated_versions: List[Dict[str, Any]] = []
            for status_log, progress_html, progress_value, versions in regen_generator:
                task["progress"] = progress_value
                task["current_step"] = f"Regenerating line {line_number + 1}"
                if versions:
                    regenerated_versions = versions

            updated_versions = [
                _normalize_version_payload(conversation_service, version)
                for version in regenerated_versions
            ]
            updated_versions = [version for version in updated_versions if version]
            updated_versions = _ensure_single_selection(updated_versions)

        task["new_versions"] = updated_versions

        if line_number < len(current_lines):
            current_line = deepcopy(current_lines[line_number])
            if updated_versions:
                current_line["versions"] = updated_versions
            current_line["text"] = edited_text
            current_line["edited_text"] = edited_text
            current_line["line_number"] = current_line.get("line_number", line_number)
            current_lines[line_number] = current_line
            conversation_task["lines"] = current_lines

        if line_number < len(parsed_script):
            parsed_script[line_number]["text"] = edited_text
            conversation_task["parsed_script"] = parsed_script

        # Mark as completed
        task["status"] = "completed"
        task["progress"] = 100.0
        task["current_step"] = "Regeneration completed"
        task["end_time"] = time.time()
        
    except Exception as e:
        # Mark as failed
        task["status"] = "failed"
        task["error"] = str(e)
        task["end_time"] = time.time()
