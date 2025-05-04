# constants.py
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Iterator, Union, cast # Make sure Tuple is imported

# --- Directory and File Constants ---
SPEAKER_DIR = Path("./speakers")
TEMP_CONVO_MULTI_DIR = Path(tempfile.gettempdir()) / "indextts_webui_convo_multi_segments"
FINAL_OUTPUT_DIR = Path("./conversation_outputs")
SAVE_DIR = Path("./project_saves") # Directory for saving states
SELECTED_SEEDS_FILENAME = Path("./selected_seeds.json") # Save file location
SPEAKER_MODEL_SAVEDIR = "pretrained_models/spkrec-ecapa-voxceleb" # Cache directory

# --- UI Constants ---
NO_SPEAKER_OPTION = "[No Speaker Selected]"
MAX_VERSIONS_ALLOWED = 5
VERSION_PREFIX = "Version "

# --- Audio Processing Defaults ---
DEFAULT_TRIM_MIN_SILENCE_LEN_MS = 250
DEFAULT_TRIM_SILENCE_THRESH_DBFS = -40
DEFAULT_TRIM_KEEP_SILENCE_MS = 50
DEFAULT_SEGMENT_NORM_TARGET_DBFS = -1.0 # Target for per-segment normalization
DEFAULT_FINAL_NORM_TARGET_DBFS = -0.5 # Target for final peak normalization
DEFAULT_OUTPUT_FORMAT = "wav"
OUTPUT_FORMAT_CHOICES = ["wav", "mp3"]
DEFAULT_MP3_BITRATE = "192"
MP3_BITRATE_CHOICES = ["128", "160", "192", "256", "320"]

# --- Model and Generation Constants ---
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SIMILARITY_THRESHOLD = 0.60 # Global baseline threshold for auto-regen during initial generation
AUTO_REGEN_ATTEMPTS = 1 # Limit initial auto-regeneration attempts
MANUAL_SIMILARITY_MIN = 0.60 # Minimum value for the manual slider
MANUAL_SIMILARITY_MAX = 1.00 # Maximum value for the manual slider
MANUAL_SIMILARITY_STEP = 0.01
DEFAULT_MANUAL_REGEN_ATTEMPTS = '5' # Default value for the new dropdown

# Seed Strategy Constants
SEED_STRATEGY_FULLY_RANDOM = "Fully Random (Unique Seed per Segment)"
SEED_STRATEGY_RANDOM_BASE_SEQUENTIAL = "Random Base + Per-Line Sequential Offset"
SEED_STRATEGY_FIXED_BASE_SEQUENTIAL = "Fixed Base + Per-Line Sequential Offset"
SEED_STRATEGY_FIXED_BASE_REUSED_LIST = "Fixed Base + Reused Sequential List (Old Sequential)"
SEED_STRATEGY_RANDOM_BASE_REUSED_LIST = "Random Base + Reused Random List (Old Random)"
SEED_STRATEGY_CHOICES = [
    SEED_STRATEGY_FULLY_RANDOM,
    SEED_STRATEGY_RANDOM_BASE_SEQUENTIAL,
    SEED_STRATEGY_FIXED_BASE_SEQUENTIAL,
    SEED_STRATEGY_FIXED_BASE_REUSED_LIST,
    SEED_STRATEGY_RANDOM_BASE_REUSED_LIST,
]
DEFAULT_SEED_STRATEGY = SEED_STRATEGY_FULLY_RANDOM
DEFAULT_FIXED_BASE_SEED = 1234

# --- Type definitions for state ---
ParsedScript = List[Dict[str, str]]
AllOptionsState = List[List[Optional[str]]]
SelectionsState = Dict[int, int]
EditedTextsState = Dict[int, str]
ConvoGenYield = Tuple[str, ParsedScript, AllOptionsState, EditedTextsState, int, dict, dict, dict]
ReviewYield = Tuple[str, str, str, str, int, dict, dict, *([dict]*MAX_VERSIONS_ALLOWED), dict]
RegenYield = Tuple[str, AllOptionsState, SelectionsState, EditedTextsState]
ManualRegenYield = Tuple[str, AllOptionsState, SelectionsState, EditedTextsState]
