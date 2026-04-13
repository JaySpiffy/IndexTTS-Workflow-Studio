"""
Pydantic models for IndexTTS2 API request/response data structures.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, ConfigDict, Field, validator
from enum import Enum


# Base models
class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error_code: str
    details: Optional[Dict[str, Any]] = None


# Health check models
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool
    requested_device: str
    runtime_device: Optional[str] = None
    using_gpu: bool = False
    using_deepspeed: bool = False


# Speaker models
class SpeakerInfo(BaseModel):
    """Speaker information."""
    filename: str
    name: str
    size_bytes: int
    size_kb: float
    duration_seconds: Optional[float] = None


class SpeakerListResponse(BaseResponse):
    """Speaker list response."""
    speakers: List[SpeakerInfo]
    total_count: int


class SpeakerUploadResponse(BaseResponse):
    """Speaker upload response."""
    speaker_info: SpeakerInfo


# Audio processing models
class AudioProcessingOptions(BaseModel):
    """Audio processing options."""
    use_noise_reduction: bool = False
    use_vocal_separation: bool = False
    normalization_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    noise_reduction_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    noise_reduction_backend: str = Field(default="auto", pattern="^(auto|classic|deepfilter)$")


class VideoExtractionRequest(BaseModel):
    """Video to audio extraction request."""
    video_filename: str
    output_name: str
    
    @validator('output_name')
    def validate_output_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Output name cannot be empty')
        return v.strip()


class AudioTrimRequest(BaseModel):
    """Audio trimming request."""
    original_filename: str
    output_name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @validator('output_name')
    def validate_output_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Output name cannot be empty')
        return v.strip()


class BatchProcessingRequest(BaseModel):
    """Batch processing request."""
    options: AudioProcessingOptions


class SourceClipPreparationRequest(BaseModel):
    """Request to prepare a source clip for cloning."""
    source_filename: str
    output_name: Optional[str] = None
    target_category: str = Field(default="speakers", pattern="^(source_clips|speakers)$")
    start_time: Optional[float] = Field(default=None, ge=0.0)
    end_time: Optional[float] = Field(default=None, ge=0.0)
    convert_to_mono: bool = True
    normalize_audio: bool = True
    target_peak_dbfs: float = Field(default=-1.0, ge=-12.0, le=0.0)
    use_noise_reduction: bool = False
    noise_reduction_strength: float = Field(default=0.35, ge=0.0, le=1.0)
    noise_reduction_backend: str = Field(default="auto", pattern="^(auto|classic|deepfilter)$")
    use_vocal_separation: bool = False

    @validator("output_name")
    def validate_output_name(cls, v):
        if v is None:
            return v
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Output name cannot be empty")
        return cleaned

    @validator("end_time")
    def validate_trim_window(cls, v, values):
        start_time = values.get("start_time")
        if v is not None and start_time is not None and v <= start_time:
            raise ValueError("end_time must be greater than start_time")
        return v


class ProcessingProgress(BaseModel):
    """Processing progress information."""
    current_file: str
    progress_percent: float
    status_message: str
    files_processed: int
    total_files: int


# TTS generation models
class EmotionControlMethod(str, Enum):
    """Emotion control methods."""
    FROM_SPEAKER = "from_speaker"
    FROM_REFERENCE = "from_reference"
    FROM_VECTORS = "from_vectors"
    FROM_TEXT = "from_text"


class EmotionInterpolationType(str, Enum):
    """Emotion interpolation types."""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"


class SeedStrategy(str, Enum):
    """Seed strategies carried over from v1 for reproducible generation."""
    FULLY_RANDOM = "fully_random"
    RANDOM_BASE_SEQUENTIAL = "random_base_sequential"
    FIXED_BASE_SEQUENTIAL = "fixed_base_sequential"
    FIXED_BASE_REUSED_LIST = "fixed_base_reused_list"
    RANDOM_BASE_REUSED_LIST = "random_base_reused_list"


class ScenePacingProfile(str, Enum):
    """Scene-level pacing presets for conversation rendering."""
    RELAXED = "relaxed"
    BALANCED = "balanced"
    SNAPPY = "snappy"
    TENSE = "tense"


class DialoguePacingPreset(str, Enum):
    """Named conversation pacing presets exposed in the workflow UI."""
    NATURAL = "natural"
    CALM = "calm"
    ARGUMENT = "argument"
    PANIC = "panic"


class SpeakerPacingSetting(BaseModel):
    """Per-speaker delivery-rate override applied after generation."""
    speaker_filename: str
    delivery_rate: float = Field(default=1.0, ge=0.85, le=1.15)


class EmotionKeyframe(BaseModel):
    """Emotion keyframe for timeline-based emotion control."""
    keyframe_id: str
    timestamp: float = Field(ge=0.0, description="Timestamp in seconds from segment start")
    emotion_vectors: List[float] = Field(default=[], max_items=8, description="8-dimensional emotion vector")
    interpolation_type: EmotionInterpolationType = EmotionInterpolationType.LINEAR
    transition_duration: float = Field(default=0.5, ge=0.0, description="Transition duration in seconds")
    
    @validator('emotion_vectors')
    def validate_emotion_vectors(cls, v):
        if len(v) > 8:
            raise ValueError('Maximum 8 emotion vector components allowed')
        if sum(abs(x) for x in v) > 1.5:
            raise ValueError('Sum of emotion vector components cannot exceed 1.5')
        return v


class EmotionTransitionSettings(BaseModel):
    """Settings for emotion transitions."""
    default_interpolation_type: EmotionInterpolationType = EmotionInterpolationType.LINEAR
    default_transition_duration: float = Field(default=0.5, ge=0.0, le=5.0)
    smooth_transitions: bool = True
    preserve_emotion_intensity: bool = True
    transition_curve_strength: float = Field(default=1.0, ge=0.1, le=3.0)


class TTSGenerationRequest(BaseModel):
    """TTS generation request."""
    speaker_filename: str
    text: str
    emotion_control_method: EmotionControlMethod = EmotionControlMethod.FROM_SPEAKER
    emotion_reference_filename: Optional[str] = None
    emotion_weight: float = Field(default=1.0, ge=0.0, le=2.0)
    emotion_vectors: List[float] = Field(default=[], max_items=8)
    emotion_text: Optional[str] = Field(None, description="Emotion description text. Optional when using from_text method.")
    use_random_sampling: bool = False
    
    # Generation parameters
    max_text_tokens_per_segment: int = Field(default=120, ge=1, le=500)
    do_sample: bool = True
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=30, ge=0)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    length_penalty: float = Field(default=0.0, ge=-5.0, le=5.0)
    num_beams: int = Field(default=3, ge=1, le=10)
    repetition_penalty: float = Field(default=10.0, ge=1.0, le=50.0)
    max_mel_tokens: int = Field(default=1500, ge=100, le=5000)
    seed: Optional[int] = Field(default=None, ge=0, le=4294967295)
    delivery_rate: float = Field(default=1.0, ge=0.85, le=1.15)
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v) > 10000:
            raise ValueError('Text too long (max 10000 characters)')
        return v.strip()
    
    @validator('emotion_vectors')
    def validate_emotion_vectors(cls, v):
        if len(v) > 8:
            raise ValueError('Maximum 8 emotion vector components allowed')
        if sum(abs(x) for x in v) > 1.5:
            raise ValueError('Sum of emotion vector components cannot exceed 1.5')
        return v


class TTSGenerationResponse(BaseResponse):
    """TTS generation response."""
    audio_filename: str
    audio_path: str
    generation_time_seconds: Optional[float] = None


# Conversation models
class ConversationLine(BaseModel):
    """Single line in a conversation."""
    model_config = ConfigDict(populate_by_name=True)

    speaker_filename: str
    text: str
    line_number: Optional[int] = None
    # Accept the frontend's existing `emo_vector` payload while exposing a
    # clearer backend field name.
    emotion_vectors: List[float] = Field(default_factory=list, alias="emo_vector", max_items=8)
    emotion_control_method: Optional[EmotionControlMethod] = None
    emotion_reference_filename: Optional[str] = None
    emotion_weight: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    emotion_text: Optional[str] = None

    @validator('emotion_vectors')
    def validate_emotion_vectors(cls, v):
        if len(v) > 8:
            raise ValueError('Maximum 8 emotion vector components allowed')
        if sum(abs(x) for x in v) > 1.5:
            raise ValueError('Sum of emotion vector components cannot exceed 1.5')
        return v


class ConversationScript(BaseModel):
    """Conversation script."""
    lines: List[ConversationLine]
    title: Optional[str] = None
    
    @validator('lines')
    def validate_lines(cls, v):
        if not v:
            raise ValueError('Conversation must have at least one line')
        if len(v) > 100:
            raise ValueError('Conversation too long (max 100 lines)')
        return v


class ConversationGenerationRequest(BaseModel):
    """Conversation generation request."""
    script: ConversationScript
    versions_per_line: int = Field(default=3, ge=1, le=5)
    similarity_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    robotic_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    auto_regen_attempts: int = Field(default=1, ge=0, le=5)
    seed_strategy: SeedStrategy = SeedStrategy.FULLY_RANDOM
    fixed_base_seed: Optional[int] = Field(default=1234, ge=0, le=4294967295)
    pacing_preset: DialoguePacingPreset = DialoguePacingPreset.NATURAL
    scene_pacing_profile: ScenePacingProfile = ScenePacingProfile.BALANCED
    scene_gap_ms: int = Field(default=140, ge=0, le=1200)
    respect_punctuation_pauses: bool = True
    speaker_pacing: List[SpeakerPacingSetting] = Field(default_factory=list)
    
    # Inherit TTS parameters
    emotion_control_method: EmotionControlMethod = EmotionControlMethod.FROM_SPEAKER
    emotion_reference_filename: Optional[str] = None
    emotion_weight: float = Field(default=1.0, ge=0.0, le=2.0)
    emotion_vectors: List[float] = Field(default=[], max_items=8)
    emotion_text: Optional[str] = Field(None, description="Emotion description text. Optional when using from_text method.")
    use_random_sampling: bool = False
    
    # Generation parameters
    max_text_tokens_per_segment: int = Field(default=120, ge=1, le=500)
    do_sample: bool = True
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=30, ge=0)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    length_penalty: float = Field(default=0.0, ge=-5.0, le=5.0)
    num_beams: int = Field(default=3, ge=1, le=10)
    repetition_penalty: float = Field(default=10.0, ge=1.0, le=50.0)
    max_mel_tokens: int = Field(default=1500, ge=100, le=5000)


class LineVersion(BaseModel):
    """Single version of a conversation line."""
    audio_filename: str
    audio_path: str
    similarity_score: float
    robotic_score: float
    quality_score: float
    is_selected: bool = False
    meets_quality_gate: Optional[bool] = None
    quality_gate_failures: List[str] = Field(default_factory=list)
    seed: Optional[int] = None
    seed_origin: Optional[str] = None
    seed_strategy: Optional[str] = None
    delivery_rate: float = 1.0
    duration_seconds: Optional[float] = None
    expected_duration_seconds: Optional[float] = None
    pacing_score: Optional[float] = None
    pacing_label: Optional[str] = None
    pacing_notes: List[str] = Field(default_factory=list)
    review_score: Optional[float] = None


class ConversationLineResult(BaseModel):
    """Result for a single conversation line."""
    line_number: int
    speaker_filename: str
    text: str
    versions: List[LineVersion]
    best_version_index: int


class ConversationGenerationResponse(BaseResponse):
    """Conversation generation response."""
    conversation_id: str
    lines: List[ConversationLineResult]
    total_versions: int
    generation_time_seconds: Optional[float] = None


class ConversationConcatenationRequest(BaseModel):
    """Optional export-time overlap plan for conversation mixing."""
    overlap_plan_text: Optional[str] = Field(default=None, max_length=50000)
    output_format: str = Field(default="wav", pattern="^(wav|mp3|ogg)$")
    output_bitrate_kbps: int = Field(default=192, ge=64, le=320)
    normalize_segments: bool = True
    target_level_dbfs: float = Field(default=-19.0, ge=-40.0, le=-3.0)
    peak_limit_dbfs: float = Field(default=-1.0, ge=-12.0, le=0.0)
    normalize_final_mix: bool = True
    trim_leading_silence: bool = True
    trim_trailing_silence: bool = True
    trim_silence_threshold_dbfs: float = Field(default=-42.0, ge=-70.0, le=-12.0)
    trim_min_silence_len_ms: int = Field(default=120, ge=20, le=2000)
    fade_in_ms: int = Field(default=0, ge=0, le=4000)
    fade_out_ms: int = Field(default=60, ge=0, le=4000)
    scene_pacing_profile: ScenePacingProfile = ScenePacingProfile.BALANCED
    scene_gap_ms: int = Field(default=140, ge=0, le=1200)
    respect_punctuation_pauses: bool = True


# Similarity analysis models
class SimilarityAnalysisRequest(BaseModel):
    """Speaker similarity analysis request."""
    reference_filename: str
    generated_filename: str
    similarity_backend: Optional[str] = Field(default=None, pattern="^(auto|speechbrain|campplus|fusion)$")


class SimilarityAnalysisResponse(BaseResponse):
    """Speaker similarity analysis response."""
    similarity_score: float
    robotic_score: float
    quality_score: float
    analysis_details: Optional[Dict[str, Any]] = None


class BatchSimilarityRequest(BaseModel):
    """Batch similarity analysis request."""
    reference_filename: str
    generated_filenames: List[str]
    similarity_backend: Optional[str] = Field(default=None, pattern="^(auto|speechbrain|campplus|fusion)$")
    
    @validator('generated_filenames')
    def validate_filenames(cls, v):
        if not v:
            raise ValueError('At least one generated filename must be provided')
        if len(v) > 50:
            raise ValueError('Too many files for batch analysis (max 50)')
        return v


class BatchSimilarityResponse(BaseResponse):
    """Batch similarity analysis response."""
    results: List[SimilarityAnalysisResponse]


# File management models
class FileUploadResponse(BaseResponse):
    """File upload response."""
    filename: str
    file_path: str
    size_bytes: int
    content_type: str


class FileListResponse(BaseResponse):
    """File list response."""
    files: List[Dict[str, Any]]
    total_count: int


# Model management models
class ModelLoadRequest(BaseModel):
    """Model load request."""
    model_path: Optional[str] = None
    use_gpu: Optional[bool] = None


class ModelLoadResponse(BaseResponse):
    """Model load response."""
    model_loaded: bool
    model_info: Optional[Dict[str, Any]] = None


# Progress tracking models
class GenerationProgress(BaseModel):
    """Generation progress information."""
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress_percent: float
    current_step: str
    queue_position: Optional[int] = None
    queued_jobs_ahead: Optional[int] = None
    active_generation_slots: Optional[int] = None
    queued_generation_tasks: Optional[int] = None
    estimated_time_remaining: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TaskStatusResponse(BaseResponse):
    """Task status response."""
    task: GenerationProgress


# Line regeneration models
class LineRegenerationMode(str, Enum):
    """Supported review-time regeneration flows."""
    REPLACE_ALL = "replace_all"
    BELOW_THRESHOLD = "below_threshold"


class LineRegenerationRequest(BaseModel):
    """Line regeneration request."""
    regen_count: int = Field(default=1, ge=1, le=5, description="Number of new versions to generate")
    mode: LineRegenerationMode = LineRegenerationMode.REPLACE_ALL
    edited_text: Optional[str] = None
    manual_similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_manual_attempts: Optional[int] = Field(default=None, ge=0, le=10)

    @validator("edited_text")
    def validate_edited_text(cls, v):
        if v is None:
            return v
        normalized = v.strip()
        if not normalized:
            raise ValueError("Edited text cannot be empty")
        if len(normalized) > 10000:
            raise ValueError("Edited text too long (max 10000 characters)")
        return normalized


class LineRegenerationResponse(BaseResponse):
    """Line regeneration response."""
    regeneration_id: str
    conversation_id: str
    line_index: int
    regen_count: int
    message: str


class LineRegenerationStatusResponse(BaseResponse):
    """Line regeneration status response."""
    regeneration_id: str
    conversation_id: str
    line_index: int
    status: str  # "pending", "running", "completed", "failed"
    progress_percent: float
    current_step: str
    new_versions: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class ConversationProjectSaveRequest(BaseModel):
    """Request to save a conversation workflow project."""
    save_name: Optional[str] = None
    project_data: Dict[str, Any]


# Timeline models
class TimelineSegment(BaseModel):
    """Single segment in a timeline track."""
    segment_id: str
    text: str
    speaker_filename: str
    start_time: float  # Position in timeline (seconds)
    duration: float  # Duration of segment (seconds)
    audio_filename: Optional[str] = None  # Generated audio file
    emotion_control_method: EmotionControlMethod = EmotionControlMethod.FROM_SPEAKER
    emotion_reference_filename: Optional[str] = None
    emotion_weight: float = Field(default=1.0, ge=0.0, le=2.0)
    emotion_vectors: List[float] = Field(default=[], max_items=8)
    emotion_text: Optional[str] = None
    use_random_sampling: bool = False
    
    # Emotion timeline fields
    emotion_keyframes: List[EmotionKeyframe] = Field(default=[], description="Emotion keyframes for this segment")
    emotion_interpolation_type: EmotionInterpolationType = EmotionInterpolationType.LINEAR
    emotion_transition_duration: float = Field(default=0.5, ge=0.0, le=5.0, description="Default transition duration in seconds")
    emotion_timeline_enabled: bool = False
    
    # TTS generation parameters
    max_text_tokens_per_segment: int = Field(default=120, ge=1, le=500)
    do_sample: bool = True
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=30, ge=0)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    length_penalty: float = Field(default=0.0, ge=-5.0, le=5.0)
    num_beams: int = Field(default=3, ge=1, le=10)
    repetition_penalty: float = Field(default=10.0, ge=1.0, le=50.0)
    max_mel_tokens: int = Field(default=1500, ge=100, le=5000)
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        if len(v) > 10000:
            raise ValueError('Text too long (max 10000 characters)')
        return v.strip()
    
    @validator('start_time', 'duration')
    def validate_time(cls, v):
        if v < 0:
            raise ValueError('Time values must be non-negative')
        return v
    
    @validator('emotion_keyframes')
    def validate_emotion_keyframes(cls, v):
        # Sort keyframes by timestamp
        if v:
            v.sort(key=lambda x: x.timestamp)
        return v


class TimelineTrack(BaseModel):
    """Single track in a timeline project."""
    track_id: str
    track_name: str
    speaker_filename: str  # Default speaker for this track
    segments: List[TimelineSegment] = []
    volume: float = Field(default=1.0, ge=0.0, le=2.0)
    muted: bool = False
    solo: bool = False
    
    @validator('track_name')
    def validate_track_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Track name cannot be empty')
        return v.strip()


class TimelineProject(BaseModel):
    """Complete timeline project with multiple tracks."""
    project_id: str
    project_name: str
    description: Optional[str] = None
    conversation_id: Optional[str] = None  # Link to conversation if created from one
    tracks: List[TimelineTrack] = []
    total_duration: float = 0.0  # Total timeline duration in seconds
    sample_rate: int = 22050  # Audio sample rate
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    @validator('project_name')
    def validate_project_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Project name cannot be empty')
        return v.strip()


class TimelineProjectRequest(BaseModel):
    """Request to create a new timeline project."""
    project_name: str
    description: Optional[str] = None
    conversation_id: Optional[str] = None


class TimelineProjectResponse(BaseResponse):
    """Response for timeline project creation/retrieval."""
    project: TimelineProject


class TimelineTrackRequest(BaseModel):
    """Request to add a track to a timeline project."""
    track_name: str
    speaker_filename: str


class TimelineTrackResponse(BaseResponse):
    """Response for track creation/retrieval."""
    track: TimelineTrack


class TimelineTrackVolumeRequest(BaseModel):
    """Request to update a track's playback level."""
    volume: float = Field(default=1.0, ge=0.0, le=2.0)


class TimelineSegmentRequest(BaseModel):
    """Request to add a segment to a timeline track."""
    track_id: str
    text: str
    start_time: float
    duration: float
    emotion_control_method: EmotionControlMethod = EmotionControlMethod.FROM_SPEAKER
    emotion_reference_filename: Optional[str] = None
    emotion_weight: float = Field(default=1.0, ge=0.0, le=2.0)
    emotion_vectors: List[float] = Field(default=[], max_items=8)
    emotion_text: Optional[str] = None
    use_random_sampling: bool = False
    
    # TTS generation parameters
    max_text_tokens_per_segment: int = Field(default=120, ge=1, le=500)
    do_sample: bool = True
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=30, ge=0)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    length_penalty: float = Field(default=0.0, ge=-5.0, le=5.0)
    num_beams: int = Field(default=3, ge=1, le=10)
    repetition_penalty: float = Field(default=10.0, ge=1.0, le=50.0)
    max_mel_tokens: int = Field(default=1500, ge=100, le=5000)


class TimelineSegmentResponse(BaseResponse):
    """Response for segment creation/retrieval."""
    segment: TimelineSegment


class TimelineSegmentUpdateRequest(BaseModel):
    """Request to update segment timing."""
    start_time: float
    duration: float
    
    @validator('start_time', 'duration')
    def validate_time(cls, v):
        if v < 0:
            raise ValueError('Time values must be non-negative')
        return v


class TimelineSegmentSplitRequest(BaseModel):
    """Request to split a timeline segment into two pieces."""
    split_offset: float = Field(..., gt=0.0, description="Seconds from the segment start where the split should happen")
    first_text: Optional[str] = None
    second_text: Optional[str] = None

    @validator('first_text', 'second_text')
    def validate_split_text(cls, v):
        if v is None:
            return v
        cleaned = v.strip()
        return cleaned or None


class TimelineSegmentSplitResponse(BaseResponse):
    """Response for a segment split operation."""
    updated_segment: TimelineSegment
    new_segment: TimelineSegment


class TimelineExportRequest(BaseModel):
    """Request to export timeline as audio."""
    project_id: str
    output_filename: str
    format: str = Field(default="wav", pattern="^(wav|mp3|ogg)$")
    output_bitrate_kbps: int = Field(default=192, ge=64, le=320)
    sample_rate: int = Field(default=22050, ge=16000, le=48000)
    duck_overlaps: bool = True
    duck_amount_db: float = Field(default=6.0, ge=0.0, le=24.0)
    duck_fade_ms: int = Field(default=120, ge=0, le=2000)
    normalize_segments: bool = True
    target_level_dbfs: float = Field(default=-19.0, ge=-40.0, le=-3.0)
    peak_limit_dbfs: float = Field(default=-1.0, ge=-12.0, le=0.0)
    normalize_final_mix: bool = True
    trim_leading_silence: bool = True
    trim_trailing_silence: bool = True
    trim_silence_threshold_dbfs: float = Field(default=-42.0, ge=-70.0, le=-12.0)
    trim_min_silence_len_ms: int = Field(default=120, ge=20, le=2000)
    fade_in_ms: int = Field(default=0, ge=0, le=4000)
    fade_out_ms: int = Field(default=60, ge=0, le=4000)
    
    @validator('output_filename')
    def validate_output_filename(cls, v):
        if not v or not v.strip():
            raise ValueError('Output filename cannot be empty')
        return v.strip()


class TimelineExportResponse(BaseResponse):
    """Response for timeline export."""
    output_filename: str
    output_path: str
    file_size: int
    duration: float
    export_time_seconds: Optional[float] = None


class TimelineWaveformResponse(BaseResponse):
    """Waveform preview payload for a generated timeline segment."""
    project_id: str
    track_id: str
    segment_id: str
    audio_filename: str
    duration_ms: int
    bar_count: int
    peaks: List[float]
