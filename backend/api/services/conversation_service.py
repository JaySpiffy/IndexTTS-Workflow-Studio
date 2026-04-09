"""
Conversation Service for IndexTTS2 API.
Handles multi-speaker conversation generation and management.
"""

import json
import re
import uuid
import time
import asyncio
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Tuple

from ..exceptions import ConversationError, ValidationError, ModelNotLoadedError
from ..config import settings
from ..core.app_paths import TEMP_CONVERSATION_SEGMENTS_DIR
from ..core.file_utils import SAVE_DIR


SEED_MODULUS = 2 ** 32
DEFAULT_FIXED_BASE_SEED = 1234


class ConversationService:
    """Service for handling conversation generation operations."""
    
    def __init__(self, conversation_manager=None):
        """
        Initialize conversation service.
        
        Args:
            conversation_manager: ConversationManager instance from webui
        """
        self.conversation_manager = conversation_manager
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.active_regenerations: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _extract_progress_step(status_log: Optional[str], fallback: str) -> str:
        """Return the most useful single-line progress message from a status log."""
        if not status_log:
            return fallback

        for raw_line in reversed(status_log.splitlines()):
            line = raw_line.strip()
            if not line:
                continue
            return line

        return fallback

    @staticmethod
    def _dispatch_progress_callback(
        loop: Optional[asyncio.AbstractEventLoop],
        progress_callback,
        *callback_args
    ) -> None:
        """Schedule an async progress callback from a worker thread without blocking it."""
        if not progress_callback or loop is None:
            return

        future = asyncio.run_coroutine_threadsafe(progress_callback(*callback_args), loop)

        def _log_future_exception(done_future):
            try:
                done_future.result()
            except Exception as callback_error:
                print(f"DEBUG: Progress callback failed: {callback_error}")

        future.add_done_callback(_log_future_exception)

    @staticmethod
    def _clamp_progress_value(progress_value: float, upper_bound: float = 99.5) -> float:
        """Clamp externally-reported progress so live updates never overshoot completion."""
        try:
            numeric_progress = float(progress_value)
        except (TypeError, ValueError):
            numeric_progress = 0.0
        return max(0.0, min(float(upper_bound), numeric_progress))

    @staticmethod
    def _normalize_seed_strategy(seed_strategy: Optional[str]) -> str:
        """Normalize unknown seed strategies back to the safest default."""
        allowed = {
            "fully_random",
            "random_base_sequential",
            "fixed_base_sequential",
            "fixed_base_reused_list",
            "random_base_reused_list",
        }
        normalized = str(seed_strategy or "fully_random").strip().lower()
        return normalized if normalized in allowed else "fully_random"

    @staticmethod
    def _normalize_seed_value(seed_value: Optional[Any], fallback: int = DEFAULT_FIXED_BASE_SEED) -> int:
        """Coerce arbitrary seed input into an unsigned 32-bit integer."""
        try:
            numeric_seed = int(seed_value)
        except (TypeError, ValueError):
            numeric_seed = int(fallback)
        return numeric_seed % SEED_MODULUS

    def _resolve_seed_runtime(
        self,
        *,
        seed_strategy: Optional[str],
        fixed_base_seed: Optional[Any],
        versions_per_line: int,
    ) -> Dict[str, Any]:
        """Resolve the runtime seed context for a generation run."""
        normalized_strategy = self._normalize_seed_strategy(seed_strategy)
        normalized_fixed_seed = self._normalize_seed_value(fixed_base_seed)
        resolved_base_seed: Optional[int] = None
        reused_seed_list: List[int] = []

        if normalized_strategy in {"fixed_base_sequential", "fixed_base_reused_list"}:
            resolved_base_seed = normalized_fixed_seed
        elif normalized_strategy in {"random_base_sequential", "random_base_reused_list"}:
            resolved_base_seed = random.SystemRandom().randrange(SEED_MODULUS)

        if normalized_strategy == "fixed_base_reused_list":
            reused_seed_list = [
                (resolved_base_seed + index) % SEED_MODULUS
                for index in range(max(1, int(versions_per_line)))
            ]
        elif normalized_strategy == "random_base_reused_list":
            rng = random.Random(resolved_base_seed)
            reused_seed_list = [
                rng.randrange(SEED_MODULUS)
                for _ in range(max(1, int(versions_per_line)))
            ]

        return {
            "seed_strategy": normalized_strategy,
            "fixed_base_seed": normalized_fixed_seed,
            "resolved_base_seed": resolved_base_seed,
            "reused_seed_list": reused_seed_list,
        }
        
    def parse_conversation_script(self, script_text: str) -> Dict[str, Any]:
        """
        Parse conversation script with SpeakerFile.ext: Text format.
        
        Args:
            script_text: Conversation script text
        
        Returns:
            Dict: Parsed script with validation results
        """
        if not script_text or not script_text.strip():
            raise ValidationError("Script text cannot be empty")
        
        # Import here to avoid circular imports
        from backend.api.core.file_utils import parse_conversation_script
        
        try:
            parsed_script, errors = parse_conversation_script(script_text)
            
            # Calculate script statistics
            total_chars = sum(len(line['text']) for line in parsed_script)
            total_words = sum(len(line['text'].split()) for line in parsed_script)
            unique_speakers = list(set(line['speaker_filename'] for line in parsed_script))
            
            return {
                "success": len(errors) == 0,
                "parsed_script": parsed_script,
                "errors": errors,
                "statistics": {
                    "total_lines": len(parsed_script),
                    "total_characters": total_chars,
                    "total_words": total_words,
                    "unique_speakers": unique_speakers,
                    "estimated_duration_minutes": total_words / 150  # Rough estimate
                }
            }
            
        except Exception as e:
            raise ConversationError(f"Failed to parse conversation script: {str(e)}")

    def _get_source_line_data(self, conversation_id: str, line_index: int, fallback_line: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get the most complete source line data for a conversation line.

        We prefer the original parsed script because it can carry per-line
        emotion metadata that is not preserved in result summaries.
        """
        conversation_task = self.active_conversations.get(conversation_id, {})
        parsed_script = conversation_task.get("parsed_script", [])

        if 0 <= line_index < len(parsed_script):
            source_line = dict(parsed_script[line_index])
        else:
            source_line = dict(fallback_line or {})

        if fallback_line:
            source_line.setdefault("speaker_filename", fallback_line.get("speaker_filename"))
            source_line.setdefault("text", fallback_line.get("text"))
            if fallback_line.get("line_number") is not None:
                source_line.setdefault("line_number", fallback_line.get("line_number"))

        source_line.setdefault("line_number", line_index)
        return source_line

    def _normalize_script_lines(self, script_lines: Any) -> List[Dict[str, Any]]:
        """Normalize script-like payloads into the backend parsed_script shape."""
        if isinstance(script_lines, dict):
            script_lines = script_lines.get("lines", [])

        normalized_lines: List[Dict[str, Any]] = []
        for index, line in enumerate(script_lines or []):
            if not isinstance(line, dict):
                continue

            speaker_filename = line.get("speaker_filename") or line.get("speaker")
            text = line.get("text")
            if not speaker_filename or not text:
                continue

            normalized_line = {
                "speaker_filename": speaker_filename,
                "text": text,
                "line_number": line.get("line_number", line.get("line", index))
            }

            emotion_vectors = line.get("emotion_vectors")
            if emotion_vectors is None:
                emotion_vectors = line.get("emo_vector")
            if emotion_vectors:
                normalized_line["emotion_vectors"] = list(emotion_vectors)

            emotion_control_method = line.get("emotion_control_method")
            if emotion_control_method:
                normalized_line["emotion_control_method"] = (
                    emotion_control_method.value
                    if hasattr(emotion_control_method, "value")
                    else emotion_control_method
                )

            if line.get("emotion_reference_filename"):
                normalized_line["emotion_reference_filename"] = line["emotion_reference_filename"]
            if line.get("emotion_weight") is not None:
                normalized_line["emotion_weight"] = line["emotion_weight"]
            if line.get("emotion_text"):
                normalized_line["emotion_text"] = line["emotion_text"]

            normalized_lines.append(normalized_line)

        return normalized_lines

    def _build_audio_url(self, audio_path: Optional[str]) -> Optional[str]:
        """Build a frontend audio URL for a stored audio path."""
        if not audio_path:
            return None
        return f"/api/conversation/assets/audio/{Path(audio_path).name}"

    def _build_conversation_snapshot_from_ui_state(self, ui_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build a restorable conversation snapshot from frontend state."""
        current_conversation_data = ui_state.get("currentConversationData") or {}
        current_conversation_id = (
            ui_state.get("currentConversationId")
            or current_conversation_data.get("conversation_id")
        )
        current_lines = deepcopy(current_conversation_data.get("lines", []))

        if not current_conversation_id or not current_lines:
            return None

        parsed_script = self._normalize_script_lines(
            ui_state.get("parsedScript")
            or ui_state.get("conversationScript")
            or current_lines
        )

        if not parsed_script:
            return None

        timestamp = time.time()
        return {
            "conversation_id": current_conversation_id,
            "status": current_conversation_data.get("status", "completed"),
            "progress": 100.0 if current_lines else 0.0,
            "current_step": current_conversation_data.get("status", "Loaded from saved project"),
            "lines": current_lines,
            "error": None,
            "start_time": timestamp,
            "end_time": timestamp,
            "parsed_script": parsed_script,
            "generation_params": deepcopy(ui_state.get("generationSettings", {})),
            "concatenation_completed": bool(current_conversation_data.get("concatenated_audio_path")),
            "concatenated_audio_path": current_conversation_data.get("concatenated_audio_path"),
            "concatenation_error": current_conversation_data.get("concatenation_error"),
        }

    def _sanitize_save_name(self, requested_name: Optional[str], ui_state: Dict[str, Any]) -> str:
        """Convert user input into a safe JSON filename."""
        base_name = (
            requested_name
            or ui_state.get("conversationTitle")
            or (ui_state.get("parsedScript") or {}).get("title")
            or "project"
        )
        base_name = str(base_name).strip()
        if not base_name:
            base_name = "project"

        base_name = re.sub(r'[\\/*?:"<>|]+', "_", base_name)
        base_name = re.sub(r"\s+", "_", base_name)
        base_name = base_name.strip("._")
        if not base_name:
            base_name = "project"

        if not base_name.lower().endswith(".json"):
            base_name = f"{base_name}.json"

        if requested_name:
            return base_name

        save_path = SAVE_DIR / base_name
        if save_path.exists():
            stem = Path(base_name).stem
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            return f"{stem}_{timestamp}.json"

        return base_name

    def _normalize_loaded_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a saved conversation snapshot before restoring it."""
        restored_snapshot = deepcopy(snapshot)
        restored_snapshot.setdefault("status", "completed")
        restored_snapshot.setdefault("progress", 100.0 if restored_snapshot.get("lines") else 0.0)
        restored_snapshot.setdefault("current_step", "Loaded from saved project")
        restored_snapshot.setdefault("error", None)
        restored_snapshot.setdefault("start_time", time.time())
        restored_snapshot.setdefault("end_time", time.time())
        restored_snapshot.setdefault("generation_params", {})

        parsed_script = restored_snapshot.get("parsed_script")
        if not parsed_script:
            restored_snapshot["parsed_script"] = self._normalize_script_lines(restored_snapshot.get("lines", []))
        else:
            restored_snapshot["parsed_script"] = self._normalize_script_lines(parsed_script)

        if restored_snapshot["status"] in {"pending", "running"}:
            restored_snapshot["status"] = "stopped"
            restored_snapshot["current_step"] = "Loaded from saved project"
            restored_snapshot["progress"] = 100.0 if restored_snapshot.get("lines") else 0.0

        for line_index, line in enumerate(restored_snapshot.get("lines", [])):
            if not isinstance(line, dict):
                continue
            line.setdefault("line_number", line_index)
            line.setdefault("speaker_filename", "Unknown")
            line.setdefault("text", "")
            for version in line.get("versions", []):
                if not isinstance(version, dict):
                    continue
                version.setdefault("audio_url", self._build_audio_url(version.get("audio_path")))

        return restored_snapshot

    def list_project_saves(self) -> List[Dict[str, Any]]:
        """List available saved project files with lightweight metadata."""
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        project_saves: List[Dict[str, Any]] = []

        for save_path in SAVE_DIR.glob("*.json"):
            metadata = {
                "save_name": save_path.name,
                "file_size": save_path.stat().st_size,
                "modified_at": datetime.fromtimestamp(save_path.stat().st_mtime, tz=timezone.utc).isoformat()
            }

            try:
                with open(save_path, "r", encoding="utf-8") as save_file:
                    saved_project = json.load(save_file)

                project_data = saved_project.get("project_data", {})
                ui_state = project_data.get("ui_state", {})
                parsed_script = self._normalize_script_lines(
                    ui_state.get("parsedScript") or ui_state.get("conversationScript")
                )

                metadata.update(
                    {
                        "saved_at": saved_project.get("saved_at", metadata["modified_at"]),
                        "title": (
                            ui_state.get("conversationTitle")
                            or (ui_state.get("parsedScript") or {}).get("title")
                            or save_path.stem
                        ),
                        "total_lines": len(parsed_script),
                        "has_conversation": bool(project_data.get("conversation_snapshot")),
                        "conversation_id": ui_state.get("currentConversationId"),
                    }
                )
            except Exception as error:
                metadata.update(
                    {
                        "saved_at": metadata["modified_at"],
                        "title": save_path.stem,
                        "total_lines": 0,
                        "has_conversation": False,
                        "corrupted": True,
                        "error": str(error),
                    }
                )

            project_saves.append(metadata)

        project_saves.sort(key=lambda item: item.get("saved_at", item["modified_at"]), reverse=True)
        return project_saves

    def save_project_state(self, ui_state: Dict[str, Any], requested_name: Optional[str] = None) -> Dict[str, Any]:
        """Persist the current frontend state and any active conversation snapshot."""
        if not isinstance(ui_state, dict):
            raise ValidationError("Project save payload must be an object")

        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        save_name = self._sanitize_save_name(requested_name, ui_state)
        save_path = SAVE_DIR / save_name
        saved_at = datetime.now(timezone.utc).isoformat()

        conversation_id = ui_state.get("currentConversationId")
        if conversation_id and conversation_id in self.active_conversations:
            conversation_snapshot = deepcopy(self.active_conversations[conversation_id])
            conversation_snapshot["conversation_id"] = conversation_id
            current_conversation_data = ui_state.get("currentConversationData") or {}
            if current_conversation_data.get("lines"):
                conversation_snapshot["lines"] = deepcopy(current_conversation_data["lines"])
            normalized_script = self._normalize_script_lines(
                ui_state.get("parsedScript") or ui_state.get("conversationScript")
            )
            if normalized_script:
                conversation_snapshot["parsed_script"] = normalized_script
            if current_conversation_data.get("concatenated_audio_path"):
                conversation_snapshot["concatenated_audio_path"] = current_conversation_data["concatenated_audio_path"]
                conversation_snapshot["concatenation_completed"] = True
            if current_conversation_data.get("concatenation_error"):
                conversation_snapshot["concatenation_error"] = current_conversation_data["concatenation_error"]
        else:
            conversation_snapshot = self._build_conversation_snapshot_from_ui_state(ui_state)

        saved_project = {
            "schema_version": 1,
            "save_name": save_name,
            "saved_at": saved_at,
            "project_data": {
                "ui_state": deepcopy(ui_state),
                "conversation_snapshot": conversation_snapshot,
            },
        }

        with open(save_path, "w", encoding="utf-8") as save_file:
            json.dump(saved_project, save_file, indent=2, ensure_ascii=False)

        summary = {
            "save_name": save_name,
            "saved_at": saved_at,
            "file_path": str(save_path),
            "total_lines": len(
                self._normalize_script_lines(
                    ui_state.get("parsedScript") or ui_state.get("conversationScript")
                )
            ),
            "has_conversation": bool(conversation_snapshot),
            "conversation_id": conversation_id,
        }

        return summary

    def load_project_state(self, save_name: str) -> Dict[str, Any]:
        """Load a saved project file and restore its conversation snapshot."""
        if not save_name or not isinstance(save_name, str):
            raise ValidationError("A saved project filename is required")

        save_name = self._sanitize_save_name(save_name, {})
        save_path = SAVE_DIR / save_name
        if not save_path.is_file():
            raise ValidationError(f"Saved project not found: {save_name}")

        with open(save_path, "r", encoding="utf-8") as save_file:
            saved_project = json.load(save_file)

        project_data = deepcopy(saved_project.get("project_data", {}))
        ui_state = project_data.get("ui_state", {})
        conversation_snapshot = project_data.get("conversation_snapshot")

        restored_conversation_id = None
        if conversation_snapshot:
            normalized_snapshot = self._normalize_loaded_snapshot(conversation_snapshot)
            restored_conversation_id = (
                normalized_snapshot.get("conversation_id")
                or ui_state.get("currentConversationId")
                or str(uuid.uuid4())
            )
            normalized_snapshot["conversation_id"] = restored_conversation_id
            self.active_conversations[restored_conversation_id] = normalized_snapshot

        if restored_conversation_id:
            ui_state["currentConversationId"] = restored_conversation_id

        project_data["ui_state"] = ui_state

        return {
            "save_name": save_name,
            "saved_at": saved_project.get("saved_at"),
            "project_data": project_data,
            "restored_conversation_id": restored_conversation_id,
        }

    def delete_project_state(self, save_name: str) -> Dict[str, Any]:
        """Delete a saved project file."""
        if not save_name or not isinstance(save_name, str):
            raise ValidationError("A saved project filename is required")

        save_name = self._sanitize_save_name(save_name, {})
        save_path = SAVE_DIR / save_name
        if not save_path.is_file():
            raise ValidationError(f"Saved project not found: {save_name}")

        save_path.unlink()
        return {
            "save_name": save_name,
            "message": "Saved project deleted"
        }
    
    def validate_conversation_script(self, parsed_script: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a parsed conversation script.
        
        Args:
            parsed_script: Parsed conversation script
        
        Returns:
            Dict: Validation result
        """
        if not parsed_script:
            raise ValidationError("Parsed script cannot be empty")
        
        # Import here to avoid circular imports
        from backend.api.core.file_utils import validate_speaker_files
        
        try:
            errors = validate_speaker_files(parsed_script)
            
            # Additional validation
            if len(parsed_script) > settings.max_conversation_length:
                errors.append(f"Conversation too long (max {settings.max_conversation_length} lines)")
            
            # Check for empty text
            for i, line in enumerate(parsed_script):
                if not line.get('text', '').strip():
                    errors.append(f"Line {i+1} has empty text")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "total_lines": len(parsed_script)
            }
            
        except Exception as e:
            raise ConversationError(f"Failed to validate conversation script: {str(e)}")
    
    def start_conversation_generation(
        self,
        parsed_script: List[Dict[str, Any]],
        versions_per_line: int = 3,
        similarity_threshold: float = 0.60,
        robotic_threshold: float = 0.70,
        auto_regen_attempts: int = 1,
        emotion_control_method: str = "from_speaker",
        emotion_reference_filename: Optional[str] = None,
        emotion_weight: float = 1.0,
        emotion_vectors: List[float] = None,
        emotion_text: Optional[str] = None,
        use_random_sampling: bool = False,
        seed_strategy: str = "fully_random",
        fixed_base_seed: Optional[int] = DEFAULT_FIXED_BASE_SEED,
        pacing_preset: str = "natural",
        scene_pacing_profile: str = "balanced",
        scene_gap_ms: int = 140,
        respect_punctuation_pauses: bool = True,
        speaker_pacing: Optional[List[Dict[str, Any]]] = None,
        **generation_params
    ) -> Dict[str, Any]:
        """
        Start a conversation generation task.
        
        Args:
            parsed_script: Parsed conversation script
            versions_per_line: Number of versions per line
            similarity_threshold: Similarity threshold for auto-regeneration
            robotic_threshold: Robotic speech threshold
            auto_regen_attempts: Number of auto-regeneration attempts
            emotion_control_method: Emotion control method
            emotion_reference_filename: Emotion reference audio filename
            emotion_weight: Emotion weight
            emotion_vectors: Emotion vector components
            emotion_text: Emotion description text
            use_random_sampling: Whether to use random sampling
            seed_strategy: Strategy for assigning seeds to versions
            fixed_base_seed: Fixed base seed used by fixed-seed strategies
            **generation_params: Additional generation parameters
        
        Returns:
            Dict: Generation task information
        """
        if not self.conversation_manager:
            raise ModelNotLoadedError("Conversation manager not available")
        
        # Validate script
        validation_result = self.validate_conversation_script(parsed_script)
        if not validation_result["valid"]:
            raise ValidationError(f"Script validation failed: {', '.join(validation_result['errors'])}")
        
        # Generate unique conversation ID
        conversation_id = str(uuid.uuid4())
        
        # Initialize task
        self.active_conversations[conversation_id] = {
            "status": "pending",
            "progress": 0.0,
            "current_step": "Initializing generation",
            "lines": [],
            "error": None,
            "start_time": time.time(),
            "end_time": None,
            "parsed_script": parsed_script,
            "generation_params": {
                "versions_per_line": versions_per_line,
                "similarity_threshold": similarity_threshold,
                "robotic_threshold": robotic_threshold,
                "auto_regen_attempts": auto_regen_attempts,
                "emotion_control_method": emotion_control_method,
                "emotion_reference_filename": emotion_reference_filename,
                "emotion_weight": emotion_weight,
                "emotion_vectors": emotion_vectors or [],
                "emotion_text": emotion_text,
                "use_random_sampling": use_random_sampling,
                "seed_strategy": self._normalize_seed_strategy(seed_strategy),
                "fixed_base_seed": self._normalize_seed_value(fixed_base_seed),
                "pacing_preset": str(pacing_preset or "natural"),
                "scene_pacing_profile": str(scene_pacing_profile or "balanced"),
                "scene_gap_ms": int(scene_gap_ms or 0),
                "respect_punctuation_pauses": bool(respect_punctuation_pauses),
                "speaker_pacing": deepcopy(speaker_pacing or []),
                **generation_params
            },
            "seed_runtime_metadata": None,
        }
        
        return {
            "conversation_id": conversation_id,
            "status": "pending",
            "message": "Conversation generation started",
            "total_lines": len(parsed_script)
        }
    
    async def generate_conversation_async(
        self,
        conversation_id: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Generate conversation asynchronously.
        
        Args:
            conversation_id: ID of the conversation task
            progress_callback: Optional progress callback function
        
        Returns:
            Dict: Generation result
        """
        if conversation_id not in self.active_conversations:
            raise ValidationError(f"Conversation task not found: {conversation_id}")
        
        task = self.active_conversations[conversation_id]
        
        if task["status"] != "pending":
            raise ValidationError(f"Conversation task not in pending state: {conversation_id}")

        loop = asyncio.get_running_loop()
        return await asyncio.to_thread(
            self._generate_conversation_sync,
            conversation_id,
            progress_callback,
            loop
        )

    def _generate_conversation_sync(
        self,
        conversation_id: str,
        progress_callback: Optional[callable] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> Dict[str, Any]:
        """Run conversation generation in a worker thread so the API remains responsive."""
        if conversation_id not in self.active_conversations:
            raise ValidationError(f"Conversation task not found: {conversation_id}")

        task = self.active_conversations[conversation_id]

        if task["status"] != "pending":
            raise ValidationError(f"Conversation task not in pending state: {conversation_id}")

        try:
            # Update status to running
            task["status"] = "running"
            task["current_step"] = "Preparing conversation generation"
            
            # Prepare generation parameters
            params = task["generation_params"]
            seed_runtime = self._resolve_seed_runtime(
                seed_strategy=params.get("seed_strategy"),
                fixed_base_seed=params.get("fixed_base_seed"),
                versions_per_line=params["versions_per_line"],
            )
            task["seed_runtime_metadata"] = deepcopy(seed_runtime)
            
            # Convert emotion control method to integer
            emo_control_method_map = {
                "from_speaker": 0,
                "from_reference": 1,
                "from_vectors": 2,
                "from_text": 3
            }
            emo_control_method_int = emo_control_method_map.get(params["emotion_control_method"], 0)
            
            # Process emotion vectors
            vec_args = [0.0] * 8
            if params["emotion_vectors"]:
                for i, vec in enumerate(params["emotion_vectors"][:8]):
                    vec_args[i] = vec
            
            # Prepare generation parameters for ConversationManager
            generation_params = {
                "versions_per_line": params["versions_per_line"],
                "similarity_threshold": params["similarity_threshold"],
                "robotic_threshold": params["robotic_threshold"],
                "auto_regen_attempts": params["auto_regen_attempts"],
                "emo_control_method": emo_control_method_int,
                "emo_ref_path": params["emotion_reference_filename"],
                "emo_weight": params["emotion_weight"],
                "emo_random": params["use_random_sampling"],
                "vec1": vec_args[0], "vec2": vec_args[1], "vec3": vec_args[2], "vec4": vec_args[3],
                "vec5": vec_args[4], "vec6": vec_args[5], "vec7": vec_args[6], "vec8": vec_args[7],
                "emo_text": params["emotion_text"],  # Pass None instead of empty string for from_text method
                "do_sample_convo": params.get("do_sample", True),
                "top_p_convo": params.get("top_p", 0.8),
                "top_k_convo": params.get("top_k", 30),
                "temperature_convo": params.get("temperature", 0.8),
                "length_penalty_convo": params.get("length_penalty", 0.0),
                "num_beams_convo": params.get("num_beams", 3),
                "repetition_penalty_convo": params.get("repetition_penalty", 10.0),
                "max_mel_tokens_convo": params.get("max_mel_tokens", 1500),
                "max_text_tokens_per_segment_convo": params.get("max_text_tokens_per_segment", 120),
                "speaker_pacing": deepcopy(params.get("speaker_pacing", [])),
                "seed_strategy": seed_runtime["seed_strategy"],
                "fixed_base_seed": seed_runtime["fixed_base_seed"],
                "resolved_base_seed": seed_runtime.get("resolved_base_seed"),
                "reused_seed_list": deepcopy(seed_runtime.get("reused_seed_list", [])),
            }
            
            def live_progress_update(progress_value: float, current_step: str) -> None:
                task["progress"] = max(
                    task.get("progress", 0.0),
                    self._clamp_progress_value(progress_value)
                )
                if current_step:
                    task["current_step"] = current_step

                self._dispatch_progress_callback(
                    loop,
                    progress_callback,
                    conversation_id,
                    task["progress"],
                    task["current_step"]
                )

            # Generate conversation
            results_generator = self.conversation_manager.generate_conversation(
                task["parsed_script"],
                progress=live_progress_update,
                **generation_params
            )
            
            # Process results
            line_results = []
            total_versions = 0
            last_status_log = ""
            
            for status_log, progress_html, progress_value, line_results_batch, final_output_path in results_generator:
                last_status_log = status_log or ""

                # Surface generator-level failures instead of silently turning
                # empty runs into "completed" tasks.
                if isinstance(last_status_log, str):
                    status_tail = last_status_log.strip().splitlines()[-1] if last_status_log.strip() else ""
                    if status_tail.startswith("Error:"):
                        raise ConversationError(status_tail)

                # Update task status
                task["progress"] = max(task.get("progress", 0.0), float(progress_value))
                task["current_step"] = self._extract_progress_step(
                    last_status_log,
                    f"Processing {len(task['parsed_script'])} conversation lines"
                )
                
                if line_results_batch:
                    line_results = line_results_batch
                    total_versions = sum(len(line['versions']) for line in line_results)
                    task["lines"] = line_results
                    print(f"DEBUG: ConversationService - Updated task {conversation_id} with {len(line_results)} lines")
                    for i, line in enumerate(line_results):
                        print(f"DEBUG: ConversationService - Line {i}: {line.get('speaker_filename', 'unknown')} - {len(line.get('versions', []))} versions")
                        for j, version in enumerate(line.get('versions', [])):
                            print(f"DEBUG: ConversationService - Version {j}: {version.get('audio_path', 'no path')}")
                
                # Call progress callback if provided
                self._dispatch_progress_callback(
                    loop,
                    progress_callback,
                    conversation_id,
                    task["progress"],
                    task["current_step"]
                )

            if not line_results:
                error_message = "Conversation generation produced no line results"
                if isinstance(last_status_log, str) and last_status_log.strip():
                    status_tail = last_status_log.strip().splitlines()[-1]
                    if "error" in status_tail.lower() or "failed" in status_tail.lower():
                        error_message = status_tail
                raise ConversationError(error_message)
            
            # Mark as completed
            task["status"] = "completed"
            task["progress"] = 100.0
            task["current_step"] = "Generation completed"
            task["end_time"] = time.time()
            
            return {
                "conversation_id": conversation_id,
                "status": "completed",
                "lines": line_results,
                "total_versions": total_versions,
                "generation_time": task["end_time"] - task["start_time"]
            }
            
        except Exception as e:
            # Mark as failed
            task["status"] = "failed"
            task["error"] = str(e)
            task["end_time"] = time.time()
            raise ConversationError(f"Conversation generation failed: {str(e)}")
    
    def stop_conversation_generation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Stop a running conversation generation task.
        
        Args:
            conversation_id: ID of the conversation task to stop
        
        Returns:
            Dict: Stop result
        """
        if conversation_id not in self.active_conversations:
            raise ValidationError(f"Conversation task not found: {conversation_id}")
        
        task = self.active_conversations[conversation_id]
        
        if task["status"] != "running":
            raise ValidationError(f"Conversation is not running: {conversation_id}")
        
        try:
            # Stop the generation
            if self.conversation_manager:
                self.conversation_manager.stop_generation()
            
            # Update task status
            task["status"] = "stopped"
            task["current_step"] = "Generation stopped by user"
            task["end_time"] = time.time()
            
            return {
                "conversation_id": conversation_id,
                "status": "stopped",
                "message": "Conversation generation stopped"
            }
            
        except Exception as e:
            raise ConversationError(f"Failed to stop conversation generation: {str(e)}")
    
    def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get the status of a conversation generation task.
        
        Args:
            conversation_id: ID of the conversation task
        
        Returns:
            Dict: Task status
        """
        if conversation_id not in self.active_conversations:
            raise ValidationError(f"Conversation task not found: {conversation_id}")
        
        task = self.active_conversations[conversation_id]
        
        # DEBUG: Print task info
        print(f"DEBUG: ConversationService.get_conversation_status for {conversation_id}")
        print(f"DEBUG: Task status: {task.get('status')}")
        print(f"DEBUG: Task lines count: {len(task.get('lines', []))}")
        
        # Calculate estimated time remaining
        estimated_time_remaining = None
        if task["status"] == "running" and task.get("start_time"):
            elapsed = time.time() - task["start_time"]
            if task["progress"] > 0:
                total_estimated = elapsed / (task["progress"] / 100)
                estimated_time_remaining = int(total_estimated - elapsed)
        
        return {
            "conversation_id": conversation_id,
            "status": task["status"],
            "progress_percent": task["progress"],
            "current_step": task["current_step"],
            "estimated_time_remaining": estimated_time_remaining,
            "start_time": task.get("start_time"),
            "end_time": task.get("end_time"),
            "error": task.get("error"),
            "lines": task.get("lines", []),
            "total_versions": sum(len(line.get("versions", [])) for line in task.get("lines", [])),
            "generation_params": deepcopy(task.get("generation_params", {})),
            "seed_runtime_metadata": deepcopy(task.get("seed_runtime_metadata")),
            "concatenation_completed": task.get("concatenation_completed", False),
            "concatenation_output_path": task.get("concatenation_output_path", None),
            "concatenation_error": task.get("concatenation_error", None)
        }
    
    def delete_conversation(self, conversation_id: str, cleanup_files: bool = True) -> Dict[str, Any]:
        """
        Delete a conversation task and optionally cleanup generated files.
        
        Args:
            conversation_id: ID of the conversation task to delete
            cleanup_files: Whether to cleanup generated files
        
        Returns:
            Dict: Deletion result
        """
        if conversation_id not in self.active_conversations:
            raise ValidationError(f"Conversation task not found: {conversation_id}")
        
        task = self.active_conversations[conversation_id]
        
        # Cleanup generated files if requested
        deleted_files = []
        if cleanup_files and task.get("lines"):
            temp_dir = TEMP_CONVERSATION_SEGMENTS_DIR
            for line in task["lines"]:
                for version in line.get("versions", []):
                    audio_path = version.get("audio_path")
                    if audio_path and Path(audio_path).exists():
                        try:
                            Path(audio_path).unlink()
                            deleted_files.append(audio_path)
                        except Exception:
                            pass  # Ignore file deletion errors
        
        # Remove task from memory
        del self.active_conversations[conversation_id]
        
        return {
            "conversation_id": conversation_id,
            "message": "Conversation deleted",
            "cleanup_files": cleanup_files,
            "deleted_files": deleted_files
        }
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all conversation tasks.
        
        Returns:
            List: Conversation tasks
        """
        conversations = []
        for conversation_id, task in self.active_conversations.items():
            conversations.append({
                "conversation_id": conversation_id,
                "status": task["status"],
                "progress": task["progress"],
                "current_step": task["current_step"],
                "start_time": task.get("start_time"),
                "end_time": task.get("end_time"),
                "has_error": task.get("error") is not None,
                "total_lines": len(task.get("parsed_script", [])),
                "total_versions": sum(len(line.get("versions", [])) for line in task.get("lines", []))
            })
        
        # Sort by start time (newest first)
        conversations.sort(key=lambda x: x.get("start_time", 0), reverse=True)
        return conversations
    
    def cleanup_conversation_resources(self, conversation_id: str) -> Dict[str, Any]:
        """
        Clean up resources for a conversation.
        
        Args:
            conversation_id: ID of the conversation
        
        Returns:
            Dict: Cleanup result
        """
        if conversation_id not in self.active_conversations:
            raise ValidationError(f"Conversation task not found: {conversation_id}")
        
        try:
            if self.conversation_manager:
                self.conversation_manager.cleanup_generated_files()
                self.conversation_manager.cleanup_memory()
            
            return {
                "conversation_id": conversation_id,
                "message": "Conversation resources cleaned up"
            }
            
        except Exception as e:
            raise ConversationError(f"Failed to cleanup conversation resources: {str(e)}")
    
    def update_concatenation_status(self, conversation_id: str, completed: bool, output_path: str = None, error: str = None) -> Dict[str, Any]:
        """
        Update concatenation status for a conversation.
        
        Args:
            conversation_id: ID of the conversation
            completed: Whether concatenation is completed
            output_path: Path to concatenated audio file (if completed)
            error: Error message (if failed)
        
        Returns:
            Dict: Update result
        """
        if conversation_id not in self.active_conversations:
            raise ValidationError(f"Conversation task not found: {conversation_id}")
        
        task = self.active_conversations[conversation_id]
        
        # Update concatenation status
        task["concatenation_completed"] = completed
        if output_path:
            task["concatenation_output_path"] = output_path
        if error:
            task["concatenation_error"] = error
        
        print(f"DEBUG: Updated concatenation status for conversation {conversation_id}: completed={completed}")
        
        return {
            "conversation_id": conversation_id,
            "concatenation_completed": completed,
            "message": "Concatenation status updated"
        }
    
    def start_line_regeneration(
        self,
        conversation_id: str,
        line_index: int,
        regen_count: int = 1,
        **generation_params
    ) -> Dict[str, Any]:
        """
        Start regenerating a specific line in a conversation.
        
        Args:
            conversation_id: ID of the conversation
            line_index: Index of the line to regenerate
            regen_count: Number of new versions to generate
            **generation_params: Additional generation parameters
        
        Returns:
            Dict: Regeneration task information
        """
        if conversation_id not in self.active_conversations:
            raise ValidationError(f"Conversation task not found: {conversation_id}")
        
        task = self.active_conversations[conversation_id]
        lines = task.get("lines", [])
        
        if line_index >= len(lines):
            raise ValidationError(f"Line index out of range: {line_index}")
        
        line = lines[line_index]
        if not line:
            raise ValidationError(f"Line not found at index: {line_index}")
        
        # Generate unique regeneration ID
        regeneration_id = str(uuid.uuid4())
        
        # Initialize regeneration task
        self.active_regenerations[regeneration_id] = {
            "regeneration_id": regeneration_id,
            "conversation_id": conversation_id,
            "line_index": line_index,
            "status": "pending",
            "progress": 0.0,
            "current_step": "Initializing regeneration",
            "new_versions": [],
            "error": None,
            "start_time": time.time(),
            "end_time": None,
            "regen_count": regen_count,
            "original_line": line,
            "generation_params": generation_params
        }
        
        return {
            "regeneration_id": regeneration_id,
            "conversation_id": conversation_id,
            "line_index": line_index,
            "regen_count": regen_count,
            "status": "pending",
            "message": "Line regeneration started"
        }
    
    async def regenerate_line_async(
        self,
        regeneration_id: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Regenerate a line asynchronously.
        
        Args:
            regeneration_id: ID of the regeneration task
            progress_callback: Optional progress callback function
        
        Returns:
            Dict: Regeneration result
        """
        if regeneration_id not in self.active_regenerations:
            raise ValidationError(f"Regeneration task not found: {regeneration_id}")
        
        regen_task = self.active_regenerations[regeneration_id]
        
        if regen_task["status"] != "pending":
            raise ValidationError(f"Regeneration task not in pending state: {regeneration_id}")

        loop = asyncio.get_running_loop()
        return await asyncio.to_thread(
            self._regenerate_line_sync,
            regeneration_id,
            progress_callback,
            loop
        )

    def _regenerate_line_sync(
        self,
        regeneration_id: str,
        progress_callback: Optional[callable] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> Dict[str, Any]:
        """Run line regeneration in a worker thread so status polling stays live."""
        if regeneration_id not in self.active_regenerations:
            raise ValidationError(f"Regeneration task not found: {regeneration_id}")

        regen_task = self.active_regenerations[regeneration_id]

        if regen_task["status"] != "pending":
            raise ValidationError(f"Regeneration task not in pending state: {regeneration_id}")

        try:
            # Update status to running
            regen_task["status"] = "running"
            regen_task["current_step"] = "Preparing line regeneration"
            
            # Get original line data
            original_line = regen_task["original_line"]
            conversation_id = regen_task["conversation_id"]
            line_index = regen_task["line_index"]
            
            # Get conversation task to access generation parameters
            conversation_task = self.active_conversations[conversation_id]
            conversation_params = conversation_task.get("generation_params", {})
            
            # Prepare line for regeneration
            line_data = self._get_source_line_data(
                conversation_id,
                line_index,
                fallback_line={
                    'speaker_filename': original_line.get("speaker_filename"),
                    'text': original_line.get("text"),
                    'line_number': line_index
                }
            )
            
            # Merge generation parameters
            generation_params = {
                "versions_per_line": regen_task["regen_count"],
                "similarity_threshold": conversation_params.get("similarity_threshold", 0.60),
                "robotic_threshold": conversation_params.get("robotic_threshold", 0.70),
                "auto_regen_attempts": conversation_params.get("auto_regen_attempts", 1),
                "emotion_control_method": conversation_params.get("emotion_control_method", "from_speaker"),
                "emotion_reference_filename": conversation_params.get("emotion_reference_filename"),
                "emotion_weight": conversation_params.get("emotion_weight", 1.0),
                "emotion_vectors": conversation_params.get("emotion_vectors", []),
                "emotion_text": conversation_params.get("emotion_text"),
                "use_random_sampling": conversation_params.get("use_random_sampling", False),
                "seed_strategy": conversation_params.get("seed_strategy", "fully_random"),
                "fixed_base_seed": conversation_params.get("fixed_base_seed", DEFAULT_FIXED_BASE_SEED),
                "do_sample": conversation_params.get("do_sample", True),
                "top_p": conversation_params.get("top_p", 0.8),
                "top_k": conversation_params.get("top_k", 30),
                "temperature": conversation_params.get("temperature", 0.8),
                "length_penalty": conversation_params.get("length_penalty", 0.0),
                "num_beams": conversation_params.get("num_beams", 3),
                "repetition_penalty": conversation_params.get("repetition_penalty", 10.0),
                "max_mel_tokens": conversation_params.get("max_mel_tokens", 1500),
                "max_text_tokens_per_segment": conversation_params.get("max_text_tokens_per_segment", 120),
            }
            
            # Override with regeneration-specific params
            generation_params.update(regen_task.get("generation_params", {}))

            if regen_task.get("mode") == "below_threshold":
                generation_params["seed_strategy"] = "fully_random"

            seed_runtime = self._resolve_seed_runtime(
                seed_strategy=generation_params.get("seed_strategy"),
                fixed_base_seed=generation_params.get("fixed_base_seed"),
                versions_per_line=generation_params["versions_per_line"],
            )
            regen_task["seed_runtime_metadata"] = deepcopy(seed_runtime)
            
            # Convert emotion control method to integer
            emo_control_method_map = {
                "from_speaker": 0,
                "from_reference": 1,
                "from_vectors": 2,
                "from_text": 3
            }
            emo_control_method_int = emo_control_method_map.get(generation_params["emotion_control_method"], 0)
            
            # Process emotion vectors
            vec_args = [0.0] * 8
            if generation_params["emotion_vectors"]:
                for i, vec in enumerate(generation_params["emotion_vectors"][:8]):
                    vec_args[i] = vec
            
            # Prepare generation parameters for ConversationManager
            final_generation_params = {
                "versions_per_line": generation_params["versions_per_line"],
                "similarity_threshold": generation_params["similarity_threshold"],
                "robotic_threshold": generation_params["robotic_threshold"],
                "auto_regen_attempts": generation_params["auto_regen_attempts"],
                "emo_control_method": emo_control_method_int,
                "emo_ref_path": generation_params["emotion_reference_filename"],
                "emo_weight": generation_params["emotion_weight"],
                "emo_random": generation_params["use_random_sampling"],
                "vec1": vec_args[0], "vec2": vec_args[1], "vec3": vec_args[2], "vec4": vec_args[3],
                "vec5": vec_args[4], "vec6": vec_args[5], "vec7": vec_args[6], "vec8": vec_args[7],
                "emo_text": generation_params["emotion_text"],
                "do_sample_convo": generation_params["do_sample"],
                "top_p_convo": generation_params["top_p"],
                "top_k_convo": generation_params["top_k"],
                "temperature_convo": generation_params["temperature"],
                "length_penalty_convo": generation_params["length_penalty"],
                "num_beams_convo": generation_params["num_beams"],
                "repetition_penalty_convo": generation_params["repetition_penalty"],
                "max_mel_tokens_convo": generation_params["max_mel_tokens"],
                "max_text_tokens_per_segment_convo": generation_params["max_text_tokens_per_segment"],
                "speaker_pacing": deepcopy(generation_params.get("speaker_pacing", [])),
                "seed_strategy": seed_runtime["seed_strategy"],
                "fixed_base_seed": seed_runtime["fixed_base_seed"],
                "resolved_base_seed": seed_runtime.get("resolved_base_seed"),
                "reused_seed_list": deepcopy(seed_runtime.get("reused_seed_list", [])),
            }
            
            # Update progress
            regen_task["progress"] = 10.0
            regen_task["current_step"] = "Generating new versions"
            
            def live_regeneration_progress(progress_value: float, current_step: str) -> None:
                regen_task["progress"] = max(
                    regen_task.get("progress", 10.0),
                    self._clamp_progress_value(progress_value)
                )
                if current_step:
                    regen_task["current_step"] = current_step

                self._dispatch_progress_callback(
                    loop,
                    progress_callback,
                    regeneration_id,
                    regen_task["progress"],
                    regen_task["current_step"]
                )

            # Generate new versions for the line
            results_generator = self.conversation_manager.generate_conversation(
                [line_data],  # Single line
                progress=live_regeneration_progress,
                **final_generation_params
            )
            
            # Process results
            new_versions = []
            
            for status_log, progress_html, progress_value, line_results_batch, final_output_path in results_generator:
                # Update task status
                regen_task["progress"] = max(regen_task.get("progress", 10.0), 10.0 + (progress_value * 0.8))  # Scale to 10-90%
                regen_task["current_step"] = self._extract_progress_step(
                    status_log,
                    f"Processing regeneration ({int(progress_value)}%)"
                )
                
                if line_results_batch and len(line_results_batch) > 0:
                    line_result = line_results_batch[0]  # Get the first (and only) line
                    new_versions = line_result.get("versions", [])
                    regen_task["new_versions"] = new_versions
                    print(f"DEBUG: RegenerationService - Generated {len(new_versions)} new versions for line {line_index}")
                    for i, version in enumerate(new_versions):
                        print(f"DEBUG: RegenerationService - Version {i}: {version.get('audio_path', 'no path')}")
                
                # Call progress callback if provided
                self._dispatch_progress_callback(
                    loop,
                    progress_callback,
                    regeneration_id,
                    regen_task["progress"],
                    regen_task["current_step"]
                )
            
            # Update the original conversation line with new versions
            if new_versions:
                # Append new versions to existing versions
                existing_versions = original_line.get("versions", [])
                original_line["versions"] = existing_versions + new_versions
                
                # Update the conversation task
                conversation_task["lines"][line_index] = original_line
                print(f"DEBUG: RegenerationService - Updated conversation line {line_index} with {len(new_versions)} new versions")
            
            # Mark as completed
            regen_task["status"] = "completed"
            regen_task["progress"] = 100.0
            regen_task["current_step"] = "Line regeneration completed"
            regen_task["end_time"] = time.time()
            
            return {
                "regeneration_id": regeneration_id,
                "conversation_id": conversation_id,
                "line_index": line_index,
                "status": "completed",
                "new_versions": new_versions,
                "total_new_versions": len(new_versions),
                "regeneration_time": regen_task["end_time"] - regen_task["start_time"]
            }
            
        except Exception as e:
            # Mark as failed
            regen_task["status"] = "failed"
            regen_task["error"] = str(e)
            regen_task["end_time"] = time.time()
            raise ConversationError(f"Line regeneration failed: {str(e)}")
    
    def get_line_regeneration_status(self, regeneration_id: str) -> Dict[str, Any]:
        """
        Get the status of a line regeneration task.
        
        Args:
            regeneration_id: ID of the regeneration task
        
        Returns:
            Dict: Regeneration status
        """
        if regeneration_id not in self.active_regenerations:
            raise ValidationError(f"Regeneration task not found: {regeneration_id}")
        
        regen_task = self.active_regenerations[regeneration_id]
        
        # Calculate estimated time remaining
        estimated_time_remaining = None
        if regen_task["status"] == "running" and regen_task.get("start_time"):
            elapsed = time.time() - regen_task["start_time"]
            if regen_task["progress"] > 10:  # Exclude initial setup time
                total_estimated = elapsed / ((regen_task["progress"] - 10) / 90)  # Adjust for 10-100% range
                estimated_time_remaining = int(total_estimated - elapsed)
        
        return {
            "regeneration_id": regeneration_id,
            "conversation_id": regen_task["conversation_id"],
            "line_index": regen_task["line_index"],
            "status": regen_task["status"],
            "progress_percent": regen_task["progress"],
            "current_step": regen_task["current_step"],
            "estimated_time_remaining": estimated_time_remaining,
            "start_time": regen_task.get("start_time"),
            "end_time": regen_task.get("end_time"),
            "error": regen_task.get("error"),
            "new_versions": regen_task.get("new_versions", []),
            "regen_count": regen_task.get("regen_count", 0)
        }
    
    def cleanup_regeneration_task(self, regeneration_id: str) -> Dict[str, Any]:
        """
        Clean up a completed regeneration task.
        
        Args:
            regeneration_id: ID of the regeneration task to clean up
        
        Returns:
            Dict: Cleanup result
        """
        if regeneration_id not in self.active_regenerations:
            raise ValidationError(f"Regeneration task not found: {regeneration_id}")
        
        # Remove task from memory
        del self.active_regenerations[regeneration_id]
        
        return {
            "regeneration_id": regeneration_id,
            "message": "Regeneration task cleaned up"
        }
