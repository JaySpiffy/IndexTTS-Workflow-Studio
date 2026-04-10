"""
Conversation generation endpoints for IndexTTS2 API.
"""

import uuid
import time
import asyncio
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request, Body
from fastapi.responses import StreamingResponse, FileResponse

from ..models import (
    ConversationGenerationRequest, ConversationGenerationResponse,
    ConversationScript, ConversationLine, GenerationProgress,
    BaseResponse, TaskStatusResponse, TTSGenerationRequest, TTSGenerationResponse,
    LineRegenerationRequest, LineRegenerationResponse, LineRegenerationStatusResponse,
    ConversationProjectSaveRequest, ConversationConcatenationRequest
)
from ..exceptions import (
    IndexTTSException, ModelNotLoadedError, ValidationError,
    ConversationError, TTSError
)
from ..config import settings
from ..core.app_paths import OUTPUT_DIR, TEMP_CONVERSATION_SEGMENTS_DIR, TEMP_DIR
from ..services import TTSService, ConversationService
from .conversation_results import regenerate_line_background

router = APIRouter()

# In-memory storage for conversation tasks (in production, use Redis or database)
conversation_tasks: Dict[str, Dict[str, Any]] = {}


def _guess_audio_media_type(audio_path: str) -> str:
    guessed_type, _ = mimetypes.guess_type(audio_path)
    return guessed_type or "application/octet-stream"


def summarize_line_selection_state(lines: List[dict]) -> Dict[str, Any]:
    """Summarize whether each line has exactly one selected version."""
    summary = {
        "total_lines": len(lines or []),
        "selected_line_count": 0,
        "missing_line_numbers": [],
        "multi_selected_line_numbers": [],
        "can_export": False,
    }

    for index, line in enumerate(lines or []):
        line_number = int(line.get("line_number", index) or index) + 1
        selected_versions = [
            version for version in line.get("versions", [])
            if version.get("is_selected", False)
        ]

        if len(selected_versions) == 1:
            summary["selected_line_count"] += 1
        elif len(selected_versions) == 0:
            summary["missing_line_numbers"].append(line_number)
        else:
            summary["multi_selected_line_numbers"].append(line_number)

    summary["can_export"] = (
        summary["total_lines"] > 0
        and not summary["missing_line_numbers"]
        and not summary["multi_selected_line_numbers"]
    )
    return summary


def build_selection_gating_message(selection_summary: Dict[str, Any]) -> str:
    """Build a user-facing export gating message."""
    issues: List[str] = []
    missing = selection_summary.get("missing_line_numbers", [])
    multi = selection_summary.get("multi_selected_line_numbers", [])

    if missing:
        issues.append(
            f"missing selections on lines {', '.join(str(number) for number in missing)}"
        )
    if multi:
        issues.append(
            f"multiple selections on lines {', '.join(str(number) for number in multi)}"
        )

    if not issues:
        return "Choose one version for every line before export."

    return f"Choose exactly one version for every line before export: {'; '.join(issues)}."


def get_tts_service(request: Request) -> TTSService:
    """Get TTS service from app state."""
    app = request.app
    print(f"DEBUG: get_tts_service - model_loaded: {getattr(app.state, 'model_loaded', False)}")
    print(f"DEBUG: get_tts_service - hasattr tts: {hasattr(app.state, 'tts')}")
    print(f"DEBUG: get_tts_service - hasattr tts_core: {hasattr(app.state, 'tts_core')}")
    print(f"DEBUG: get_tts_service - hasattr tts_service: {hasattr(app.state, 'tts_service')}")
    
    if not getattr(app.state, 'model_loaded', False):
        print("DEBUG: Model not loaded, raising ModelNotLoadedError")
        raise ModelNotLoadedError()
    
    if not hasattr(app.state, 'tts_service'):
        print("DEBUG: Creating new TTS service")
        tts_core = getattr(app.state, 'tts_core', None) or getattr(app.state, 'tts', None)
        print(f"DEBUG: Using tts_core: {tts_core is not None}")
        app.state.tts_service = TTSService(
            tts_core=tts_core,
            cmd_args=app.state.cmd_args if hasattr(app.state, 'cmd_args') else None
        )
    
    return app.state.tts_service


def get_conversation_service(request: Request) -> ConversationService:
    """Get conversation service from app state."""
    app = request.app
    if not getattr(app.state, 'model_loaded', False):
        raise ModelNotLoadedError()
    
    if not hasattr(app.state, 'conversation_service'):
        app.state.conversation_service = ConversationService(
            conversation_manager=app.state.conversation_manager
        )
    
    return app.state.conversation_service


def get_tts_core(request: Request):
    """Get TTS core from app state (legacy compatibility)."""
    app = request.app
    if not getattr(app.state, 'model_loaded', False):
        raise ModelNotLoadedError()
    return app.state.tts_core


def get_conversation_manager(request: Request):
    """Get conversation manager from app state (legacy compatibility)."""
    app = request.app
    if not getattr(app.state, 'model_loaded', False):
        raise ModelNotLoadedError()
    return app.state.conversation_manager


@router.post("/generate-single", response_model=TTSGenerationResponse)
async def generate_single_tts(
    request: TTSGenerationRequest,
    http_request: Request
):
    """
    Generate single TTS output with emotion control.
    
    Args:
        request: TTS generation request
        http_request: HTTP request object
        
    Returns:
        TTSGenerationResponse: Generation result
    """
    try:
        tts_service = get_tts_service(http_request)
        
        # Generate TTS
        result = tts_service.generate_single_tts(
            speaker_filename=request.speaker_filename,
            text=request.text,
            emotion_control_method=request.emotion_control_method.value,
            emotion_reference_filename=request.emotion_reference_filename,
            emotion_weight=request.emotion_weight,
            emotion_vectors=request.emotion_vectors,
            emotion_text=request.emotion_text,
            use_random_sampling=request.use_random_sampling,
            max_text_tokens_per_segment=request.max_text_tokens_per_segment,
            do_sample=request.do_sample,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            length_penalty=request.length_penalty,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            max_mel_tokens=request.max_mel_tokens,
            seed=request.seed,
            delivery_rate=request.delivery_rate,
        )
        
        return TTSGenerationResponse(
            audio_filename=result["audio_filename"],
            audio_path=result["audio_path"],
            message="TTS generation completed successfully"
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TTSError(f"Failed to generate TTS: {str(e)}")


@router.post("/generate", response_model=ConversationGenerationResponse)
async def generate_conversation(
    request: ConversationGenerationRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """
    Generate a multi-speaker conversation.
    
    Args:
        request: Conversation generation request
        background_tasks: FastAPI background tasks
        http_request: HTTP request object
        
    Returns:
        ConversationGenerationResponse: Generation result
    """
    try:
        conversation_service = get_conversation_service(http_request)
        
        # Parse script for TTS processing
        parsed_script = []
        for i, line in enumerate(request.script.lines):
            parsed_line = {
                'speaker_filename': line.speaker_filename,
                'text': line.text,
                'line_number': line.line_number if line.line_number is not None else i
            }

            if line.emotion_vectors:
                parsed_line['emotion_vectors'] = line.emotion_vectors
            if line.emotion_control_method is not None:
                parsed_line['emotion_control_method'] = line.emotion_control_method.value
            if line.emotion_reference_filename:
                parsed_line['emotion_reference_filename'] = line.emotion_reference_filename
            if line.emotion_weight is not None:
                parsed_line['emotion_weight'] = line.emotion_weight
            if line.emotion_text:
                parsed_line['emotion_text'] = line.emotion_text

            parsed_script.append(parsed_line)
        
        # Start generation using service
        task_info = conversation_service.start_conversation_generation(
            parsed_script=parsed_script,
            versions_per_line=request.versions_per_line,
            similarity_threshold=request.similarity_threshold,
            robotic_threshold=request.robotic_threshold,
            auto_regen_attempts=request.auto_regen_attempts,
            emotion_control_method=request.emotion_control_method.value,
            emotion_reference_filename=request.emotion_reference_filename,
            emotion_weight=request.emotion_weight,
            emotion_vectors=request.emotion_vectors,
            emotion_text=request.emotion_text,
            use_random_sampling=request.use_random_sampling,
            seed_strategy=request.seed_strategy.value,
            fixed_base_seed=request.fixed_base_seed,
            pacing_preset=request.pacing_preset.value,
            scene_pacing_profile=request.scene_pacing_profile.value,
            scene_gap_ms=request.scene_gap_ms,
            respect_punctuation_pauses=request.respect_punctuation_pauses,
            speaker_pacing=[setting.model_dump() for setting in request.speaker_pacing],
            do_sample=request.do_sample,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            length_penalty=request.length_penalty,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            max_mel_tokens=request.max_mel_tokens,
            max_text_tokens_per_segment=request.max_text_tokens_per_segment
        )
        
        # Start async generation in background
        background_tasks.add_task(
            conversation_service.generate_conversation_async,
            task_info["conversation_id"]
        )
        
        return ConversationGenerationResponse(
            conversation_id=task_info["conversation_id"],
            lines=[],  # Will be populated during generation
            total_versions=0,
            message="Conversation generation started"
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to start conversation generation: {str(e)}")


async def generate_conversation_background(
    conversation_id: str,
    parsed_script: list,
    request: ConversationGenerationRequest,
    tts_core,
    conversation_manager
):
    """
    Background task for conversation generation.
    """
    try:
        # Initialize task status
        conversation_tasks[conversation_id] = {
            "status": "running",
            "progress": 0.0,
            "current_step": "Initializing generation",
            "lines": [],
            "error": None,
            "start_time": time.time()
        }
        
        # Prepare generation parameters
        generation_params = {
            "versions_per_line": request.versions_per_line,
            "similarity_threshold": request.similarity_threshold,
            "robotic_threshold": request.robotic_threshold,
            "auto_regen_attempts": request.auto_regen_attempts,
            "emo_control_method": request.emotion_control_method.value,
            "emo_ref_path": request.emotion_reference_filename,
            "emo_weight": request.emotion_weight,
            "emo_random": request.use_random_sampling,
            "vec1": request.emotion_vectors[0] if len(request.emotion_vectors) > 0 else 0.0,
            "vec2": request.emotion_vectors[1] if len(request.emotion_vectors) > 1 else 0.0,
            "vec3": request.emotion_vectors[2] if len(request.emotion_vectors) > 2 else 0.0,
            "vec4": request.emotion_vectors[3] if len(request.emotion_vectors) > 3 else 0.0,
            "vec5": request.emotion_vectors[4] if len(request.emotion_vectors) > 4 else 0.0,
            "vec6": request.emotion_vectors[5] if len(request.emotion_vectors) > 5 else 0.0,
            "vec7": request.emotion_vectors[6] if len(request.emotion_vectors) > 6 else 0.0,
            "vec8": request.emotion_vectors[7] if len(request.emotion_vectors) > 7 else 0.0,
            "emo_text": request.emotion_text,
            "do_sample_convo": request.do_sample,
            "top_p_convo": request.top_p,
            "top_k_convo": request.top_k,
            "temperature_convo": request.temperature,
            "length_penalty_convo": request.length_penalty,
            "num_beams_convo": request.num_beams,
            "repetition_penalty_convo": request.repetition_penalty,
            "max_mel_tokens_convo": request.max_mel_tokens,
            "max_text_tokens_per_segment_convo": request.max_text_tokens_per_segment,
        }
        
        # Process emotion control method
        emo_control_method_map = {
            "from_speaker": 0,
            "from_reference": 1,
            "from_vectors": 2,
            "from_text": 3
        }
        generation_params["emo_control_method"] = emo_control_method_map.get(
            request.emotion_control_method.value, 0
        )
        
        # Generate conversation
        results_generator = conversation_manager.generate_conversation(
            parsed_script,
            **generation_params
        )
        
        # Process results
        line_results = []
        total_versions = 0
        last_status_log = ""
        
        for status_log, progress_html, progress_value, line_results_batch, final_output_path in results_generator:
            last_status_log = status_log or ""

            if isinstance(last_status_log, str):
                status_tail = last_status_log.strip().splitlines()[-1] if last_status_log.strip() else ""
                if status_tail.startswith("Error:"):
                    raise ConversationError(status_tail)

            # Update task status
            conversation_tasks[conversation_id]["progress"] = progress_value
            conversation_tasks[conversation_id]["current_step"] = f"Processing line {int(progress_value * len(parsed_script) / 100) + 1}/{len(parsed_script)}"
            
            if line_results_batch:
                line_results = line_results_batch
                total_versions = sum(len(line['versions']) for line in line_results)
                conversation_tasks[conversation_id]["lines"] = line_results
                
                # DEBUG: Print line results
                print(f"DEBUG: Updated task with {len(line_results)} lines")
                for i, line in enumerate(line_results):
                    print(f"DEBUG: Line {i}: {line.get('speaker_filename', 'unknown')} - {len(line.get('versions', []))} versions")

        if not line_results:
            error_message = "Conversation generation produced no line results"
            if isinstance(last_status_log, str) and last_status_log.strip():
                status_tail = last_status_log.strip().splitlines()[-1]
                if "error" in status_tail.lower() or "failed" in status_tail.lower():
                    error_message = status_tail
            raise ConversationError(error_message)
        
        # Mark as completed
        conversation_tasks[conversation_id]["status"] = "completed"
        conversation_tasks[conversation_id]["progress"] = 100.0
        conversation_tasks[conversation_id]["current_step"] = "Generation completed"
        conversation_tasks[conversation_id]["end_time"] = time.time()
        
    except Exception as e:
        # Mark as failed
        conversation_tasks[conversation_id]["status"] = "failed"
        conversation_tasks[conversation_id]["error"] = str(e)
        conversation_tasks[conversation_id]["end_time"] = time.time()


@router.get("/status/{conversation_id}", response_model=TaskStatusResponse)
async def get_conversation_status(conversation_id: str, http_request: Request):
    """
    Get the status of a conversation generation task.
    
    Args:
        conversation_id: ID of the conversation task
        http_request: HTTP request object
        
    Returns:
        TaskStatusResponse: Task status
    """
    try:
        conversation_service = get_conversation_service(http_request)
        
        # Get status from service
        status = conversation_service.get_conversation_status(conversation_id)
        
        # Check for concatenation status
        concatenation_completed = status.get("concatenation_completed", False)
        concatenation_error = status.get("concatenation_error", None)
        
        return TaskStatusResponse(
            task=GenerationProgress(
                task_id=status["conversation_id"],
                status=status["status"],
                progress_percent=status["progress_percent"],
                current_step=status["current_step"],
                queue_position=status.get("queue_position"),
                queued_jobs_ahead=status.get("queued_jobs_ahead"),
                active_generation_slots=status.get("active_generation_slots"),
                queued_generation_tasks=status.get("queued_generation_tasks"),
                estimated_time_remaining=status.get("estimated_time_remaining"),
                result={
                    "lines": status.get("lines", []),
                    "concatenation_completed": concatenation_completed,
                    "concatenation_output_path": status.get("concatenation_output_path"),
                    "concatenation_error": concatenation_error
                } if status.get("lines") or concatenation_completed else None,
                error=status.get("error")
            )
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to get conversation status: {str(e)}")


@router.post("/stop/{conversation_id}", response_model=BaseResponse)
async def stop_conversation_generation(conversation_id: str, http_request: Request):
    """
    Stop a running conversation generation task.
    
    Args:
        conversation_id: ID of the conversation task to stop
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Stop result
    """
    try:
        conversation_service = get_conversation_service(http_request)
        
        # Stop generation using service
        result = conversation_service.stop_conversation_generation(conversation_id)
        
        return BaseResponse(message=result["message"])
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to stop conversation generation: {str(e)}")


@router.delete("/{conversation_id}", response_model=BaseResponse)
async def delete_conversation(conversation_id: str, http_request: Request):
    """
    Delete a conversation task and its generated files.
    
    Args:
        conversation_id: ID of the conversation task to delete
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Deletion result
    """
    try:
        conversation_service = get_conversation_service(http_request)
        
        # Delete conversation using service
        result = conversation_service.delete_conversation(conversation_id, cleanup_files=True)
        
        return BaseResponse(
            message=result["message"],
            details={
                "cleanup_files": result["cleanup_files"],
                "deleted_files_count": len(result.get("deleted_files", []))
            }
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to delete conversation: {str(e)}")


@router.get("/list", response_model=BaseResponse)
async def list_conversations(http_request: Request):
    """
    List all conversation tasks.

    Args:
        http_request: HTTP request object

    Returns:
        BaseResponse: List of conversation tasks
    """
    try:
        conversation_service = get_conversation_service(http_request)

        # Get conversations from service
        conversations = conversation_service.list_conversations()

        # Also check global conversation tasks (for backward compatibility)
        global_conversations = []
        for conv_id, task in conversation_tasks.items():
            # Check if this conversation is already in the service list
            if not any(conv["conversation_id"] == conv_id for conv in conversations):
                global_conversations.append({
                    "conversation_id": conv_id,
                    "status": task.get("status", "unknown"),
                    "progress": task.get("progress", 0),
                    "current_step": task.get("current_step", ""),
                    "start_time": task.get("start_time"),
                    "end_time": task.get("end_time"),
                    "has_error": task.get("error") is not None,
                    "total_lines": len(task.get("parsed_script", [])),
                    "total_versions": sum(len(line.get("versions", [])) for line in task.get("lines", [])),
                    "lines": task.get("lines", [])
                })

        # Merge both sources
        all_conversations = conversations + global_conversations

        return BaseResponse(
            message=f"Found {len(all_conversations)} conversation tasks",
            details={"conversations": all_conversations}
        )

    except Exception as e:
        print(f"DEBUG: Error in list_conversations: {str(e)}")
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to list conversations: {str(e)}")


@router.get("/projects", response_model=BaseResponse)
async def list_saved_projects(http_request: Request):
    """List saved conversation workflow projects."""
    try:
        conversation_service = get_conversation_service(http_request)
        projects = conversation_service.list_project_saves()
        return BaseResponse(
            message=f"Found {len(projects)} saved projects",
            details={"projects": projects}
        )
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to list saved projects: {str(e)}")


@router.post("/projects/save", response_model=BaseResponse)
async def save_project_state(request: ConversationProjectSaveRequest, http_request: Request):
    """Save the current conversation workflow project state."""
    try:
        conversation_service = get_conversation_service(http_request)
        result = conversation_service.save_project_state(
            ui_state=request.project_data,
            requested_name=request.save_name
        )
        return BaseResponse(
            message=f"Project saved as {result['save_name']}",
            details=result
        )
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to save project: {str(e)}")


@router.get("/projects/{save_name}", response_model=BaseResponse)
async def load_project_state(save_name: str, http_request: Request):
    """Load a saved conversation workflow project state."""
    try:
        conversation_service = get_conversation_service(http_request)
        result = conversation_service.load_project_state(save_name)
        return BaseResponse(
            message=f"Project loaded from {result['save_name']}",
            details=result
        )
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to load project: {str(e)}")


@router.delete("/projects/{save_name}", response_model=BaseResponse)
async def delete_project_state(save_name: str, http_request: Request):
    """Delete a saved conversation workflow project."""
    try:
        conversation_service = get_conversation_service(http_request)
        result = conversation_service.delete_project_state(save_name)
        return BaseResponse(
            message=result["message"],
            details={"save_name": result["save_name"]}
        )
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to delete project: {str(e)}")


@router.get("/results/{conversation_id}", response_model=BaseResponse)
async def get_conversation_results(conversation_id: str, http_request: Request):
    """
    Get detailed results for a specific conversation.

    Args:
        conversation_id: ID of the conversation
        http_request: HTTP request object

    Returns:
        BaseResponse: Conversation results with detailed line information
    """
    try:
        print(f"DEBUG: get_conversation_results called with conversation_id: {conversation_id}")
        
        # Check both conversation service and global tasks
        conversation_service = get_conversation_service(http_request)
        conversations = conversation_service.list_conversations()
        print(f"DEBUG: Found {len(conversations)} conversations in service")

        # Find the conversation
        conversation_data = None
        for conv in conversations:
            if conv["conversation_id"] == conversation_id:
                conversation_data = conv
                print(f"DEBUG: Found conversation in service: {conversation_id}")
                # Get detailed conversation data including lines
                conv_status = conversation_service.get_conversation_status(conversation_id)
                conversation_data["lines"] = conv_status.get("lines", [])
                conversation_data["generation_params"] = conv_status.get("generation_params", {})
                conversation_data["seed_runtime_metadata"] = conv_status.get("seed_runtime_metadata")
                conversation_data["concatenation_output_path"] = conv_status.get("concatenation_output_path")
                conversation_data["concatenation_completed"] = conv_status.get("concatenation_completed", False)
                conversation_data["concatenation_error"] = conv_status.get("concatenation_error")
                print(f"DEBUG: Service conversation has {len(conversation_data.get('lines', []))} lines")
                break

        # Also check global conversation tasks
        if conversation_id in conversation_tasks:
            task = conversation_tasks[conversation_id]
            print(f"DEBUG: Found conversation in global tasks: {conversation_id}")
            print(f"DEBUG: Task status: {task.get('status')}, lines count: {len(task.get('lines', []))}")
            if not conversation_data:  # Only use global task if not found in service
                conversation_data = {
                    "conversation_id": conversation_id,
                    "status": task.get("status", "unknown"),
                    "progress": task.get("progress", 0),
                    "current_step": task.get("current_step", ""),
                    "start_time": task.get("start_time"),
                    "end_time": task.get("end_time"),
                    "has_error": task.get("error") is not None,
                    "total_lines": len(task.get("parsed_script", [])),
                    "total_versions": sum(len(line.get("versions", [])) for line in task.get("lines", [])),
                    "lines": task.get("lines", []),
                    "generation_params": task.get("generation_params", {}),
                    "seed_runtime_metadata": task.get("seed_runtime_metadata"),
                    "concatenation_output_path": task.get("concatenation_output_path"),
                    "concatenation_completed": task.get("concatenation_completed", False),
                    "concatenation_error": task.get("concatenation_error"),
                }
            else:
                print(f"DEBUG: Using service data, ignoring global task data")

        if not conversation_data:
            print(f"DEBUG: Conversation not found: {conversation_id}")
            raise ValidationError(f"Conversation not found: {conversation_id}")

        # Format lines for frontend
        formatted_lines = []
        lines = conversation_data.get("lines", [])
        print(f"DEBUG: Processing {len(lines)} lines for frontend")

        for i, line in enumerate(lines):
            if isinstance(line, dict) and "versions" in line:
                # Add audio URLs to all versions
                versions_with_urls = []
                for version in line.get("versions", []):
                    audio_path = version.get("audio_path")
                    if audio_path and Path(audio_path).exists():
                        # Extract just the filename for the URL
                        filename = Path(audio_path).name
                        version.setdefault("audio_filename", filename)
                        version["audio_url"] = f"/api/conversation/assets/audio/{filename}"
                        print(f"DEBUG: Added audio URL for {audio_path}: {version['audio_url']}")
                        print(f"DEBUG: Filename extracted: {filename}")
                    else:
                        print(f"DEBUG: Audio file not found or missing: {audio_path}")
                    versions_with_urls.append(version)
                
                formatted_line = {
                    "line_number": i,
                    "speaker_filename": line.get("speaker_filename", "Unknown"),
                    "text": line.get("text", ""),
                    "versions": versions_with_urls
                }
                formatted_lines.append(formatted_line)
                print(f"DEBUG: Formatted line {i} with {len(versions_with_urls)} versions")

        result = BaseResponse(
            message="Conversation results retrieved",
            details={
                "conversation_id": conversation_id,
                "status": conversation_data.get("status"),
                "lines": formatted_lines,
                "total_lines": len(formatted_lines),
                "total_versions": sum(len(line.get("versions", [])) for line in formatted_lines),
                "generation_params": conversation_data.get("generation_params", {}),
                "seed_runtime_metadata": conversation_data.get("seed_runtime_metadata"),
                "concatenation_completed": bool(conversation_data.get("concatenation_completed")),
                "concatenation_output_path": conversation_data.get("concatenation_output_path"),
                "concatenation_output_filename": (
                    Path(conversation_data["concatenation_output_path"]).name
                    if conversation_data.get("concatenation_output_path")
                    else None
                ),
                "concatenation_error": conversation_data.get("concatenation_error"),
            }
        )
        
        print(f"DEBUG: Returning result with {len(formatted_lines)} lines")
        return result

    except Exception as e:
        print(f"DEBUG: Error in get_conversation_results: {str(e)}")
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to get conversation results: {str(e)}")


@router.get("/results/{conversation_id}/line/{line_index}/version/{version_index}/download")
async def download_conversation_audio_version(conversation_id: str, line_index: int, version_index: int, http_request: Request):
    """
    Download a specific version of a conversation line.

    Args:
        conversation_id: ID of the conversation
        line_index: Index of the line
        version_index: Index of the version
        http_request: HTTP request object

    Returns:
        FileResponse: Audio file
    """
    try:
        print(f"DEBUG: Download request - conversation_id: {conversation_id}, line_index: {line_index}, version_index: {version_index}")
        
        # Find the conversation
        if conversation_id not in conversation_tasks:
            raise ValidationError(f"Conversation not found: {conversation_id}")

        task = conversation_tasks[conversation_id]
        lines = task.get("lines", [])
        print(f"DEBUG: Found {len(lines)} lines in conversation")

        if line_index >= len(lines):
            raise ValidationError(f"Line index out of range: {line_index}")

        line = lines[line_index]
        versions = line.get("versions", [])
        print(f"DEBUG: Line {line_index} has {len(versions)} versions")

        if version_index >= len(versions):
            raise ValidationError(f"Version index out of range: {version_index}")

        version = versions[version_index]
        audio_path = version.get("audio_path")
        print(f"DEBUG: Audio path for version {version_index}: {audio_path}")

        if not audio_path or not Path(audio_path).exists():
            print(f"DEBUG: Audio file does not exist: {audio_path}")
            raise ValidationError(f"Audio file not found: {audio_path}")

        print(f"DEBUG: Serving audio file: {audio_path}")
        return FileResponse(
            path=audio_path,
            media_type='audio/wav',
            filename=version.get("audio_filename", f"line_{line_index}_v{version_index}.wav")
        )

    except Exception as e:
        print(f"DEBUG: Error in download_conversation_audio_version: {str(e)}")
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to download audio: {str(e)}")


@router.post("/results/{conversation_id}/concatenate", response_model=BaseResponse)
async def concatenate_conversation_audio(
    conversation_id: str,
    mix_request: Optional[ConversationConcatenationRequest] = Body(default=None),
    http_request: Request = None
):
    """
    Concatenate selected audio versions from a conversation.

    Args:
        conversation_id: ID of the conversation
        http_request: HTTP request object

    Returns:
        BaseResponse: Concatenation result
    """
    try:
        # Find the conversation in both service and global tasks
        conversation_service = get_conversation_service(http_request)
        found_conversation = False
        lines = []
        
        # First check in service
        conversations = conversation_service.list_conversations()
        for conv in conversations:
            if conv["conversation_id"] == conversation_id:
                found_conversation = True
                # Get detailed conversation data including lines
                conv_data = conversation_service.get_conversation_status(conversation_id)
                lines = conv_data.get("lines", [])
                break
        
        # If not found in service, check global tasks
        if not found_conversation and conversation_id in conversation_tasks:
            task = conversation_tasks[conversation_id]
            lines = task.get("lines", [])
            found_conversation = True
        
        if not found_conversation:
            raise ValidationError(f"Conversation not found: {conversation_id}")

        if not lines:
            raise ValidationError("No lines found in conversation")

        conversation_generation_params: Dict[str, Any] = {}
        try:
            conversation_generation_params = conversation_service.get_conversation_status(conversation_id).get("generation_params", {})
        except Exception:
            conversation_generation_params = {}

        selection_summary = summarize_line_selection_state(lines)
        if not selection_summary["can_export"]:
            raise ValidationError(build_selection_gating_message(selection_summary))

        # Collect selected audio files
        audio_files = []
        print(f"DEBUG: Checking {len(lines)} lines for selected versions")
        for i, line in enumerate(lines):
            versions = line.get("versions", [])
            print(f"DEBUG: Line {i} has {len(versions)} versions")
            selected_version = None
            for j, version in enumerate(versions):
                is_selected = version.get("is_selected", False)
                print(f"DEBUG: Version {j} is_selected: {is_selected}")
                if is_selected:
                    selected_version = version
                    print(f"DEBUG: Selected version {j} chosen for line {i}")
                    break

            if not selected_version:
                continue

            audio_path = selected_version.get("audio_path")
            print(f"DEBUG: Selected audio path for line {i}: {audio_path}")
            if audio_path and Path(audio_path).exists():
                audio_files.append(audio_path)
                selected_version["audio_url"] = f"/api/conversation/assets/audio/{Path(audio_path).name}"
                print(f"DEBUG: Added audio file to concatenation list: {audio_path}")
            else:
                print(f"DEBUG: Audio file does not exist: {audio_path}")

        if not audio_files:
            raise ValidationError("No audio versions selected for concatenation")

        # Import audio concatenation function from standalone implementation
        from api.core.audio_processing import concatenate_audio_files

        # Generate output filename
        output_format = mix_request.output_format if mix_request else "wav"
        output_filename = f"conversation_{conversation_id}_concatenated.{output_format}"
        output_path = str(TEMP_CONVERSATION_SEGMENTS_DIR / output_filename)

        overlap_plan_text = mix_request.overlap_plan_text if mix_request else None
        output_bitrate_kbps = mix_request.output_bitrate_kbps if mix_request else 192
        normalize_segments = mix_request.normalize_segments if mix_request else True
        target_level_dbfs = mix_request.target_level_dbfs if mix_request else -19.0
        peak_limit_dbfs = mix_request.peak_limit_dbfs if mix_request else -1.0
        normalize_final_mix = mix_request.normalize_final_mix if mix_request else True
        trim_leading_silence = mix_request.trim_leading_silence if mix_request else True
        trim_trailing_silence = mix_request.trim_trailing_silence if mix_request else True
        trim_silence_threshold_dbfs = mix_request.trim_silence_threshold_dbfs if mix_request else -42.0
        trim_min_silence_len_ms = mix_request.trim_min_silence_len_ms if mix_request else 120
        fade_in_ms = mix_request.fade_in_ms if mix_request else 0
        fade_out_ms = mix_request.fade_out_ms if mix_request else 60
        scene_pacing_profile = (
            mix_request.scene_pacing_profile.value
            if mix_request and getattr(mix_request, "scene_pacing_profile", None) is not None
            else conversation_generation_params.get("scene_pacing_profile", "balanced")
        )
        scene_gap_ms = (
            mix_request.scene_gap_ms
            if mix_request and getattr(mix_request, "scene_gap_ms", None) is not None
            else int(conversation_generation_params.get("scene_gap_ms", 140) or 140)
        )
        respect_punctuation_pauses = (
            mix_request.respect_punctuation_pauses
            if mix_request and getattr(mix_request, "respect_punctuation_pauses", None) is not None
            else bool(conversation_generation_params.get("respect_punctuation_pauses", True))
        )
        selected_line_texts = [
            str(line.get("text") or "")
            for line in lines
            if any(version.get("is_selected", False) for version in line.get("versions", []))
        ]

        # Concatenate audio files
        mix_result = concatenate_audio_files(
            audio_files,
            output_path,
            overlap_plan_text=overlap_plan_text,
            output_format=output_format,
            output_bitrate_kbps=output_bitrate_kbps,
            normalize_segments=normalize_segments,
            target_level_dbfs=target_level_dbfs,
            peak_limit_dbfs=peak_limit_dbfs,
            normalize_final_mix=normalize_final_mix,
            trim_leading_silence=trim_leading_silence,
            trim_trailing_silence=trim_trailing_silence,
            trim_silence_threshold_dbfs=trim_silence_threshold_dbfs,
            trim_min_silence_len_ms=trim_min_silence_len_ms,
            fade_in_ms=fade_in_ms,
            fade_out_ms=fade_out_ms,
            scene_pacing_profile=scene_pacing_profile,
            scene_gap_ms=scene_gap_ms,
            respect_punctuation_pauses=respect_punctuation_pauses,
            line_texts=selected_line_texts,
        )
        if not mix_result.get("success"):
            raise ConversationError(mix_result.get("error", "Failed to create concatenated audio"))

        # Update conversation task with concatenation info
        if conversation_id in conversation_tasks:
            conversation_tasks[conversation_id]["concatenation_completed"] = True
            conversation_tasks[conversation_id]["concatenation_output_path"] = output_path
            print(f"DEBUG: Updated conversation task {conversation_id} with concatenation info")
        
        # Also update in conversation service if available
        try:
            conversation_service = get_conversation_service(http_request)
            conversation_service.update_concatenation_status(conversation_id, True, output_path)
            print(f"DEBUG: Updated concatenation status in service")
        except Exception as e:
            print(f"DEBUG: Failed to update concatenation status in service: {str(e)}")

        return BaseResponse(
            message="Audio files concatenated successfully",
            details={
                "output_filename": output_filename,
                "output_path": output_path,
                "output_url": f"/api/conversation/results/{conversation_id}/download",
                "concatenated_files": len(audio_files),
                "concatenation_completed": True,
                "overlap_plan_applied": bool(overlap_plan_text and overlap_plan_text.strip()),
                "segment_leveling_applied": bool(normalize_segments),
                "target_level_dbfs": target_level_dbfs,
                "peak_limit_dbfs": peak_limit_dbfs,
                "final_mix_peak_protection": bool(normalize_final_mix),
                "normalization": mix_result.get("normalization", {}),
                "output_format": mix_result.get("output_format", output_format),
                "output_bitrate_kbps": mix_result.get("finishing", {}).get("output_bitrate_kbps"),
                "scene_pacing_profile": scene_pacing_profile,
                "scene_gap_ms": scene_gap_ms,
                "respect_punctuation_pauses": respect_punctuation_pauses,
                "finishing": mix_result.get("finishing", {}),
                "selection_summary": selection_summary,
            }
        )

    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to concatenate audio: {str(e)}")


@router.get("/assets/audio/{filename}")
async def get_audio_file(filename: str):
    """
    Serve audio files for playback in the frontend.

    Args:
        filename: Name of the audio file

    Returns:
        FileResponse: The audio file
    """
    from fastapi.responses import FileResponse
    
    print(f"DEBUG: Audio file requested: {filename}")
    
    # Try different possible locations for the audio file
    possible_paths = [
        str(TEMP_CONVERSATION_SEGMENTS_DIR / filename),
        f"./{filename}",
        str(TEMP_DIR / filename),
        str(OUTPUT_DIR / filename),
    ]
    
    for path in possible_paths:
        print(f"DEBUG: Checking path: {path}")
        if Path(path).exists():
            print(f"DEBUG: Found audio file at: {path}")
            file_response = FileResponse(path, media_type="audio/wav")
            print(f"DEBUG: FileResponse created with media_type: audio/wav")
            print(f"DEBUG: File path: {file_response.path}")
            return file_response
    
    print(f"DEBUG: Audio file not found in any location: {filename}")
    print(f"DEBUG: Searched paths: {possible_paths}")
    raise ValidationError(f"Audio file not found: {filename}")


@router.get("/results/{conversation_id}/download")
async def download_concatenated_audio(conversation_id: str, http_request: Request):
    """
    Download the concatenated conversation audio.

    Args:
        conversation_id: ID of the conversation
        http_request: HTTP request object

    Returns:
        FileResponse: Concatenated audio file
    """
    try:
        print(f"DEBUG: Download concatenated audio requested for conversation {conversation_id}")
        
        # Check in both conversation service and global tasks
        found_conversation = False
        output_path = None
        
        # First check in conversation service
        try:
            conversation_service = get_conversation_service(http_request)
            conversations = conversation_service.list_conversations()
            
            for conv in conversations:
                if conv["conversation_id"] == conversation_id:
                    found_conversation = True
                    # Get detailed conversation data including concatenation info
                    conv_status = conversation_service.get_conversation_status(conversation_id)
                    output_path = conv_status.get("concatenation_output_path")
                    print(f"DEBUG: Found conversation in service, output_path: {output_path}")
                    break
        except Exception as e:
            print(f"DEBUG: Error checking conversation service: {str(e)}")
        
        # If not found in service, check global tasks
        if not found_conversation and conversation_id in conversation_tasks:
            task = conversation_tasks[conversation_id]
            output_path = task.get("concatenation_output_path")
            found_conversation = True
            print(f"DEBUG: Found conversation in global tasks, output_path: {output_path}")
        
        if not found_conversation:
            print(f"DEBUG: Conversation not found: {conversation_id}")
            raise ValidationError(f"Conversation not found: {conversation_id}")
        
        # If no output path, try the default location
        if not output_path:
            matching_files = sorted(TEMP_CONVERSATION_SEGMENTS_DIR.glob(f"conversation_{conversation_id}_concatenated.*"))
            output_path = str(matching_files[-1]) if matching_files else None
            print(f"DEBUG: Using default output path: {output_path}")

        if not output_path:
            print(f"DEBUG: No output path found for conversation {conversation_id}")
            raise ValidationError(f"Concatenated audio not available for conversation: {conversation_id}")

        if not Path(output_path).exists():
            print(f"DEBUG: Concatenated audio file does not exist: {output_path}")
            raise ValidationError(f"Concatenated audio file not found: {output_path}")

        print(f"DEBUG: Serving concatenated audio file: {output_path}")
        return FileResponse(
            path=output_path,
            media_type=_guess_audio_media_type(output_path),
            filename=Path(output_path).name
        )

    except Exception as e:
        print(f"DEBUG: Error in download_concatenated_audio: {str(e)}")
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to download concatenated audio: {str(e)}")


@router.post("/parse-script", response_model=BaseResponse)
async def parse_conversation_script(script: ConversationScript, http_request: Request):
    """
    Parse and validate a conversation script.
    
    Args:
        script: Conversation script to parse
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Parsed script validation
    """
    try:
        conversation_service = get_conversation_service(http_request)
        
        # Convert script to text format for parsing
        script_text = "\n".join([
            f"{line.speaker_filename}: {line.text}"
            for line in script.lines
        ])
        
        # Parse script using service
        result = conversation_service.parse_conversation_script(script_text)
        
        if not result["success"]:
            raise ValidationError(f"Script parsing failed: {', '.join(result['errors'])}")
        
        return BaseResponse(
            message="Script parsed successfully",
            details=result["statistics"]
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise ValidationError(f"Failed to parse script: {str(e)}")


@router.get("/emotion-methods", response_model=BaseResponse)
async def get_emotion_methods(http_request: Request):
    """
    Get available emotion control methods.
    
    Args:
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Available emotion methods
    """
    try:
        tts_service = get_tts_service(http_request)
        
        # Get emotion methods from service
        methods = tts_service.get_supported_emotion_methods()
        
        return BaseResponse(
            message="Emotion control methods retrieved",
            details={"methods": methods}
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TTSError(f"Failed to get emotion methods: {str(e)}")


@router.get("/default-parameters", response_model=BaseResponse)
async def get_default_parameters(http_request: Request):
    """
    Get default TTS generation parameters.
    
    Args:
        http_request: HTTP request object
        
    Returns:
        BaseResponse: Default parameters
    """
    try:
        tts_service = get_tts_service(http_request)
        
        # Get default parameters from service
        params = tts_service.get_default_generation_parameters()
        
        return BaseResponse(
            message="Default parameters retrieved",
            details={"parameters": params}
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise TTSError(f"Failed to get default parameters: {str(e)}")


@router.post("/results/{conversation_id}/line/{line_index}/regenerate", response_model=LineRegenerationResponse)
async def regenerate_conversation_line(
    conversation_id: str,
    line_index: int,
    request: LineRegenerationRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """
    Regenerate a specific line in a conversation.
    
    Args:
        conversation_id: ID of the conversation
        line_index: Index of the line to regenerate
        request: Line regeneration request
        background_tasks: FastAPI background tasks
        http_request: HTTP request object
        
    Returns:
        LineRegenerationResponse: Regeneration result
    """
    try:
        # Validate inputs
        if not conversation_id or not conversation_id.strip():
            raise ValidationError("Conversation ID cannot be empty")
        
        if line_index < 0:
            raise ValidationError("Line index must be non-negative")
        
        if request.regen_count < 1 or request.regen_count > 5:
            raise ValidationError("Regeneration count must be between 1 and 5")
        
        conversation_service = get_conversation_service(http_request)
        
        # Check if conversation exists and is in a valid state
        try:
            conv_status = conversation_service.get_conversation_status(conversation_id)
        except ValidationError:
            raise ValidationError(f"Conversation not found: {conversation_id}")
        
        if conv_status["status"] not in ["completed", "failed"]:
            raise ValidationError(f"Cannot regenerate line for conversation in '{conv_status['status']}' state. Conversation must be completed or failed.")
        
        # Check if line exists
        lines = conv_status.get("lines", [])
        if line_index >= len(lines):
            raise ValidationError(f"Line index {line_index} out of range. Conversation has {len(lines)} lines.")

        generation_params = conversation_service.active_conversations.get(conversation_id, {}).get("generation_params", {})
        regen_mode = request.mode.value if hasattr(request.mode, "value") else str(request.mode)
        if regen_mode == "below_threshold" and request.manual_similarity_threshold is None:
            raise ValidationError("Manual similarity threshold is required for threshold regeneration")

        if regen_mode == "below_threshold" and request.max_manual_attempts is None:
            max_manual_attempts = max(1, int(generation_params.get("auto_regen_attempts", 1) or 1))
        else:
            max_manual_attempts = request.max_manual_attempts

        regen_task_id = f"regen_{conversation_id}_{line_index}_{uuid.uuid4().hex[:8]}"
        conversation_service.active_conversations[regen_task_id] = {
            "status": "pending",
            "progress": 0.0,
            "current_step": "Initializing regeneration",
            "line_number": line_index,
            "regen_count": request.regen_count,
            "conversation_id": conversation_id,
            "mode": regen_mode,
            "edited_text": request.edited_text,
            "manual_similarity_threshold": request.manual_similarity_threshold,
            "max_manual_attempts": max_manual_attempts,
            "error": None,
            "start_time": time.time(),
            "end_time": None,
            "new_versions": []
        }

        background_tasks.add_task(
            regenerate_line_background,
            regen_task_id,
            conversation_id,
            line_index,
            request.regen_count,
            lines[line_index],
            conversation_service
        )

        return LineRegenerationResponse(
            regeneration_id=regen_task_id,
            conversation_id=conversation_id,
            line_index=line_index,
            regen_count=request.regen_count,
            message="Line regeneration started"
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to start line regeneration: {str(e)}")


@router.get("/results/{conversation_id}/line/{line_index}/regenerate/status", response_model=LineRegenerationStatusResponse)
async def get_line_regeneration_status(
    conversation_id: str,
    line_index: int,
    regeneration_id: Optional[str] = None,
    http_request: Request = None
):
    """
    Get the status of a line regeneration task.
    
    Args:
        conversation_id: ID of the conversation
        line_index: Index of the line
        regeneration_id: ID of the regeneration task (optional, will find latest if not provided)
        http_request: HTTP request object
        
    Returns:
        LineRegenerationStatusResponse: Regeneration status
    """
    try:
        # Validate inputs
        if not conversation_id or not conversation_id.strip():
            raise ValidationError("Conversation ID cannot be empty")
        
        if line_index < 0:
            raise ValidationError("Line index must be non-negative")
        
        conversation_service = get_conversation_service(http_request)
        
        # Check if conversation exists
        try:
            conv_status = conversation_service.get_conversation_status(conversation_id)
        except ValidationError:
            raise ValidationError(f"Conversation not found: {conversation_id}")
        
        # If regeneration_id is not provided, try to find the latest regeneration for this line
        if not regeneration_id:
            latest_regen_id = None
            for task_id, regen_task in reversed(list(conversation_service.active_conversations.items())):
                if (
                    task_id.startswith("regen_")
                    and regen_task.get("conversation_id") == conversation_id
                    and regen_task.get("line_number") == line_index
                ):
                    latest_regen_id = task_id
                    break

            if not latest_regen_id:
                raise ValidationError(f"No regeneration task found for conversation {conversation_id}, line {line_index}")
            
            regeneration_id = latest_regen_id

        if regeneration_id not in conversation_service.active_conversations:
            raise ValidationError(f"Regeneration task not found: {regeneration_id}")

        status = conversation_service.active_conversations[regeneration_id]
        if (
            status.get("conversation_id") != conversation_id
            or status.get("line_number") != line_index
        ):
            raise ValidationError(f"Regeneration task {regeneration_id} does not match the requested conversation and line")

        new_versions_with_urls = []
        for version in status.get("new_versions", []):
            version_copy = version.copy()
            audio_path = version_copy.get("audio_path")
            if audio_path:
                filename = Path(audio_path).name
                version_copy["audio_url"] = f"/api/conversation/assets/audio/{filename}"
                version_copy.setdefault("audio_filename", filename)
            new_versions_with_urls.append(version_copy)
        
        return LineRegenerationStatusResponse(
            regeneration_id=regeneration_id,
            conversation_id=conversation_id,
            line_index=line_index,
            status=status["status"],
            progress_percent=status.get("progress", 0.0),
            current_step=status["current_step"],
            new_versions=new_versions_with_urls,
            error=status.get("error")
        )
        
    except Exception as e:
        if isinstance(e, IndexTTSException):
            raise
        raise ConversationError(f"Failed to get line regeneration status: {str(e)}")
