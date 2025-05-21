# -*- coding: utf-8 -*-
import gradio as gr
import os
import tempfile
import uuid
import re
import time
import random # Ensure random is imported
import json # Import json for save/load
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Iterator, Union, cast # Make sure Tuple is imported
import traceback # Keep for error logging
import math # For checking -inf dBFS
import torch # Added for SpeechBrain / device check

# --- Attempt to import pydub ---
try:
    from pydub import AudioSegment
    from pydub import silence as _silence_Imported # Check for silence module
    from pydub.effects import compress_dynamic_range as _compress_dynamic_range_Imported # Check for effects
    PYDUB_AVAILABLE = True
    PYDUB_SILENCE_AVAILABLE = _silence_Imported is not None
    PYDUB_COMPRESS_AVAILABLE = _compress_dynamic_range_Imported is not None
    print("pydub library found.")
    if not PYDUB_SILENCE_AVAILABLE: print("  Warning: pydub 'silence' module not found. Trimming disabled.")
    if not PYDUB_COMPRESS_AVAILABLE: print("  Warning: pydub 'compress_dynamic_range' not found. Compression disabled.")
except ImportError:
    print("Warning: pydub library not found (pip install pydub). Concatenation/Processing features will be disabled.")
    PYDUB_AVAILABLE = False
    PYDUB_SILENCE_AVAILABLE = False
    PYDUB_COMPRESS_AVAILABLE = False
    AudioSegment = None # type: ignore

# --- Attempt to import Scipy/Numpy/Noisereduce ---
try:
    import numpy as np
    import scipy.signal
    SCIPY_AVAILABLE = True
    print("scipy and numpy found. EQ enabled.")
    try:
        import noisereduce as nr
        NOISEREDUCE_AVAILABLE = True
        print("noisereduce found. Noise Reduction enabled.")
    except ImportError:
        print("Warning: noisereduce not found (pip install noisereduce). Noise Reduction disabled.")
        NOISEREDUCE_AVAILABLE = False
        nr = None # type: ignore
except ImportError:
    print("Warning: scipy or numpy not found (pip install scipy numpy). EQ and Noise Reduction disabled.")
    SCIPY_AVAILABLE = False
    NOISEREDUCE_AVAILABLE = False
    np = None # type: ignore
    scipy = None # type: ignore
    nr = None # type: ignore


# --- Import TTS ---
try:
    from indextts.infer import IndexTTS
except ImportError as e:
    print(f"Error importing IndexTTS: {e}. Ensure indextts package is correctly installed.")
    tts = None
except Exception as e: # Catch other potential init errors within the class
     print(f"Error during IndexTTS internal import or setup: {e}")
     tts = None


# --- Import from Utility Files ---
try:
    # Import the new function from audio_utils
    from audio_utils import (
        apply_eq, apply_noise_reduction, apply_reverb,
        change_pitch, change_speed, analyze_speaker_similarity,
        SPEECHBRAIN_AVAILABLE # Also import the flag
    )
except ImportError as e:
    print(f"ERROR importing from audio_utils.py: {e}. Ensure file exists and dependencies are met.")
    def apply_eq(*args, **kwargs): raise NotImplementedError("audio_utils import failed")
    def apply_noise_reduction(*args, **kwargs): raise NotImplementedError("audio_utils import failed")
    def apply_reverb(*args, **kwargs): raise NotImplementedError("audio_utils import failed")
    def change_pitch(*args, **kwargs): raise NotImplementedError("audio_utils import failed")
    def change_speed(*args, **kwargs): raise NotImplementedError("audio_utils import failed")
    def analyze_speaker_similarity(*args, **kwargs): print("Dummy analyze_speaker_similarity call"); return -1.0 # Placeholder
    SPEECHBRAIN_AVAILABLE = False # Assume false if import fails

try:
    from general_utils import (
        list_speaker_files, prepare_temp_dir, split_text_simple,
        check_all_selections_valid, enable_concatenation_buttons,
        list_save_files # Import new function if added for save/load later
    )
except ImportError as e:
    print(f"ERROR importing from general_utils.py: {e}. Ensure file exists.")
    def list_speaker_files(*args, **kwargs): return ['[No Speaker Selected]'], []
    def prepare_temp_dir(*args, **kwargs): print("ERROR: prepare_temp_dir not loaded from utils!"); return False
    def split_text_simple(*args, **kwargs): return []
    def check_all_selections_valid(*args, **kwargs): return False
    def enable_concatenation_buttons(*args, **kwargs): return gr.update(interactive=False), gr.update(interactive=False)
    def list_save_files(*args, **kwargs): return []

# --- Import from UI Layout --- (Added for Timeline Tab)
try:
    from ui_layout import create_timeline_tab
except ImportError as e:
    print(f"ERROR importing from ui_layout.py: {e}. Timeline tab will be unavailable.")
    def create_timeline_tab(): return (None,) * 9 # Return dummy tuple matching expected output

# --- Import AppContext ---
try:
    from app_context import AppContext
except ImportError as e:
    print(f"ERROR importing AppContext from app_context.py: {e}")
    AppContext = None

# --- Import from UI Logic ---
try:
    from ui_logic import update_timeline_with_selection
except ImportError as e:
    print(f"ERROR importing from ui_logic.py: {e}")
    # Define a dummy if not found, to prevent crashes if ui_logic.py is not ready
    def update_timeline_with_selection(*args, **kwargs):
        print("Warning: ui_logic.update_timeline_with_selection not found!")
        return (gr.update(),) * 7


# --- Define Constants ---
SPEAKER_DIR = Path("./speakers")
NO_SPEAKER_OPTION = "[No Speaker Selected]"
MAX_VERSIONS_ALLOWED = 5
TEMP_CONVO_MULTI_DIR = Path(tempfile.gettempdir()) / "indextts_webui_convo_multi_segments"
VERSION_PREFIX = "Version "
FINAL_OUTPUT_DIR = Path("./conversation_outputs")
SAVE_DIR = Path("./project_saves") # Directory for saving states
DEFAULT_TRIM_MIN_SILENCE_LEN_MS = 250
DEFAULT_TRIM_SILENCE_THRESH_DBFS = -40
DEFAULT_TRIM_KEEP_SILENCE_MS = 50
DEFAULT_SEGMENT_NORM_TARGET_DBFS = -1.0 # Target for per-segment normalization
DEFAULT_FINAL_NORM_TARGET_DBFS = -0.5 # Target for final peak normalization
DEFAULT_OUTPUT_FORMAT = "wav"
OUTPUT_FORMAT_CHOICES = ["wav", "mp3"]
DEFAULT_MP3_BITRATE = "192"
MP3_BITRATE_CHOICES = ["128", "160", "192", "256", "320"]
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_MODEL_SAVEDIR = "pretrained_models/spkrec-ecapa-voxceleb" # Cache directory
SIMILARITY_THRESHOLD = 0.60 # Global baseline threshold for auto-regen during initial generation
AUTO_REGEN_ATTEMPTS = 1 # Limit initial auto-regeneration attempts
MANUAL_SIMILARITY_MIN = 0.60 # Minimum value for the manual slider
MANUAL_SIMILARITY_MAX = 1.00 # Maximum value for the manual slider
MANUAL_SIMILARITY_STEP = 0.01
DEFAULT_MANUAL_REGEN_ATTEMPTS = '5' # Default value for the new dropdown
SELECTED_SEEDS_FILENAME = Path("./selected_seeds.json") # Save file location

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

# --- Type definitions for state (Defined AFTER imports) ---
ParsedScript = List[Dict[str, str]]
AllOptionsState = List[List[Optional[str]]]
SelectionsState = Dict[int, int]
EditedTextsState = Dict[int, str]
ConvoGenYield = Tuple[str, ParsedScript, AllOptionsState, EditedTextsState, int, dict, dict, dict]
ReviewYield = Tuple[str, str, str, str, int, dict, dict, dict, dict, dict, dict, dict, dict]
RegenYield = Tuple[str, AllOptionsState, SelectionsState, EditedTextsState]
ManualRegenYield = Tuple[str, AllOptionsState, SelectionsState, EditedTextsState]

# --- Determine Device ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("CUDA device found. Using GPU for speaker similarity analysis.")
else:
    DEVICE = torch.device("cpu")
    print("CUDA device not found. Using CPU for speaker similarity analysis.")

# --- Initialize TTS ---
try:
    if 'IndexTTS' in globals() and IndexTTS is not None:
        tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml", is_fp16=(DEVICE.type=='cuda'))
        print("IndexTTS Initialized Successfully.")
    else:
        tts = None
except Exception as e:
    print(f"CRITICAL Error initializing IndexTTS: {e}")
    traceback.print_exc()
    tts = None

# --- Initialize Speaker Model (as a global variable) ---
speaker_similarity_model = None
if SPEECHBRAIN_AVAILABLE:
    try:
        from speechbrain.pretrained import EncoderClassifier
        print(f"Loading speaker embedding model ({SPEAKER_MODEL_SOURCE})...")
        Path(SPEAKER_MODEL_SAVEDIR).mkdir(parents=True, exist_ok=True)
        speaker_similarity_model = EncoderClassifier.from_hparams(
            source=SPEAKER_MODEL_SOURCE,
            savedir=SPEAKER_MODEL_SAVEDIR,
            run_opts={"device": DEVICE}
        )
        speaker_similarity_model.eval()
        print("Speaker embedding model loaded successfully.")
    except ImportError:
        print("SpeechBrain installed but EncoderClassifier could not be imported. Speaker similarity disabled.")
        SPEECHBRAIN_AVAILABLE = False
    except Exception as e:
        print(f"CRITICAL Error loading SpeechBrain speaker model ({SPEAKER_MODEL_SOURCE}): {e}")
        traceback.print_exc()
        speaker_similarity_model = None
        SPEECHBRAIN_AVAILABLE = False

# --- Initialize AppContext ---
app_context = None
if AppContext is not None:
    app_context = AppContext()
    app_context.tts = tts
    app_context.speaker_similarity_model = speaker_similarity_model
    app_context.device = DEVICE
    app_context.pydub_available = PYDUB_AVAILABLE
    app_context.pydub_silence_available = PYDUB_SILENCE_AVAILABLE
    app_context.pydub_compress_available = PYDUB_COMPRESS_AVAILABLE
    app_context.scipy_available = SCIPY_AVAILABLE
    app_context.noisereduce_available = NOISEREDUCE_AVAILABLE
    app_context.speechbrain_available = SPEECHBRAIN_AVAILABLE
    print("AppContext initialized and populated.")
else:
    print("AppContext class not available. Skipping initialization.")


# --- Gradio UI Functions ---
def infer_single(selected_speaker_filename, text, temperature_val, top_p_val, top_k_val):
    if tts is None: return "Error: TTS model failed to initialize."
    speaker_path = None
    if selected_speaker_filename and selected_speaker_filename != NO_SPEAKER_OPTION:
        potential_path = SPEAKER_DIR / selected_speaker_filename
        if potential_path.is_file(): speaker_path = str(potential_path)
        else: return f"Error: Speaker file '{selected_speaker_filename}' not found."
    else: return "Error: Please select a speaker file from Dropdown."
    if not text or not text.strip(): return "Error: Please provide text."
    output_path = os.path.join(tempfile.gettempdir(), f"index_tts_single_{uuid.uuid4()}.wav")
    print(f"Generating single segment for '{text[:30]}...' with speaker '{selected_speaker_filename}'...")
    try:
        tts.infer( speaker_path, text, output_path, temperature=temperature_val, top_p=top_p_val, top_k=int(top_k_val), seed=-1 )
        print(f"Single segment saved: {output_path}")
        return output_path
    except Exception as e:
        err_msg = f"Error during single TTS inference ({type(e).__name__}: {e})"; print(f"{err_msg}"); traceback.print_exc(); return err_msg
def gen_single(selected_speaker_filename, text, temperature_val, top_p_val, top_k_val):
     if tts is None: return "Error: TTS model failed to initialize."
     output_path = infer_single(selected_speaker_filename, text, temperature_val, top_p_val, top_k_val)
     if isinstance(output_path, str) and output_path.startswith("Error:"): return None
     else: return output_path

def parse_validate_and_start_convo(
    script_text: str,
    num_versions_str: str,
    temperature_val: float, top_p_val: float, top_k_val: int,
    seed_strategy: str,
    fixed_base_seed: int,
    progress=gr.Progress(track_tqdm=True)
) -> Iterator[ConvoGenYield]:
    global speaker_similarity_model
    status_log = ["Starting Conversation Multi-Version Generation..."]; yield "\n".join(status_log), [], [], {}, -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
    if tts is None: status_log.append("Error: TTS model not initialized."); yield "\n".join(status_log), [], [], {}, -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
    try:
        num_versions = int(num_versions_str);
        if not (1 <= num_versions <= MAX_VERSIONS_ALLOWED): raise ValueError(f"Versions must be 1-{MAX_VERSIONS_ALLOWED}.")
    except (ValueError, TypeError): status_log.append(f"Error: Invalid number of versions '{num_versions_str}'."); yield "\n".join(status_log), [], [], {}, -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
    if not script_text or not script_text.strip(): status_log.append("Error: Input script text cannot be empty."); yield "\n".join(status_log), [], [], {}, -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
    if not prepare_temp_dir(TEMP_CONVO_MULTI_DIR): status_log.append(f"Error: Failed to prepare temp directory {TEMP_CONVO_MULTI_DIR}."); yield "\n".join(status_log), [], [], {}, -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
    status_log.append(f"Using temp directory: {TEMP_CONVO_MULTI_DIR}")
    status_log.append("Parsing script and validating speakers..."); yield "\n".join(status_log), [], [], {}, -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
    lines = script_text.strip().split('\n'); parsed_script: ParsedScript = []; available_speaker_files = list_speaker_files()[1]
    for i, line in enumerate(lines):
        line_num = i + 1; line = line.strip();
        if not line: continue
        match = re.match(r'^([^:]+):\s*(.*)$', line)
        if match:
            speaker_filename = match.group(1).strip(); line_text = match.group(2).strip();
            if not line_text: status_log.append(f"Warning: Line {line_num} skipped (no text)."); continue
            if speaker_filename not in available_speaker_files: err = f"Error: Speaker file '{speaker_filename}' on line {line_num} not found in ./speakers/. Cannot proceed."; status_log.append(err); print(err); yield "\n".join(status_log), [], [], {}, -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
            parsed_script.append({'speaker_filename': speaker_filename, 'text': line_text})
        else: err = f"Error: Invalid format on line {line_num}. Expected 'SpeakerFile.ext: Text'. Cannot proceed."; status_log.append(err); print(err); yield "\n".join(status_log), [], [], {}, -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
    if not parsed_script: status_log.append("Error: No valid lines found in the script after parsing."); yield "\n".join(status_log), [], [], {}, -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
    total_lines = len(parsed_script); status_log.append(f"Script parsed: {total_lines} valid lines found."); yield "\n".join(status_log), parsed_script, [], {}, -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore

    base_seed = 0
    random_seed_list = []
    sequential_seed_list = []
    status_log.append(f"Selected Seed Strategy: {seed_strategy}")
    print(f"Selected Seed Strategy: {seed_strategy}")
    if seed_strategy == SEED_STRATEGY_FIXED_BASE_SEQUENTIAL or seed_strategy == SEED_STRATEGY_FIXED_BASE_REUSED_LIST:
        try:
            base_seed = int(fixed_base_seed)
            status_log.append(f"Using Fixed Base Seed: {base_seed}")
            print(f"Using Fixed Base Seed: {base_seed}")
        except (ValueError, TypeError):
            base_seed = DEFAULT_FIXED_BASE_SEED
            status_log.append(f"Warning: Invalid Fixed Base Seed input, using default: {base_seed}")
            print(f"Warning: Invalid Fixed Base Seed input, using default: {base_seed}")
    elif seed_strategy == SEED_STRATEGY_RANDOM_BASE_SEQUENTIAL or seed_strategy == SEED_STRATEGY_RANDOM_BASE_REUSED_LIST:
        base_seed = random.randint(0, 2**32 - 1)
        status_log.append(f"Using Random Base Seed: {base_seed}")
        print(f"Using Random Base Seed: {base_seed}")
    if seed_strategy == SEED_STRATEGY_RANDOM_BASE_REUSED_LIST:
        random_seed_list = random.sample(range(2**32), num_versions)
        seed_info_str = ', '.join([f'V{k+1}=R({s})' for k, s in enumerate(random_seed_list)])
        status_log.append(f"Reused Random Seed List: [{seed_info_str}]")
        print(f"Reused Random Seed List: [{seed_info_str}]")
    elif seed_strategy == SEED_STRATEGY_FIXED_BASE_REUSED_LIST:
        sequential_seed_list = [(base_seed + j) % (2**32) for j in range(num_versions)]
        seed_info_str = ', '.join([f'V{k+1}=S({s})' for k, s in enumerate(sequential_seed_list)])
        status_log.append(f"Reused Sequential Seed List: [{seed_info_str}]")
        print(f"Reused Sequential Seed List: [{seed_info_str}]")
    yield "\n".join(status_log), parsed_script, [], {}, -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore

    all_options: AllOptionsState = []; generation_successful = True
    for i, line_info in enumerate(progress.tqdm(parsed_script, desc="Generating Lines")):
        line_idx_0_based = i
        speaker_filename = line_info['speaker_filename']; line_text = line_info['text'];
        speaker_path_str = str(SPEAKER_DIR / speaker_filename)
        ref_speaker_path_exists = Path(speaker_path_str).is_file()
        line_version_seeds = [0] * num_versions
        seeds_generated_this_line = False
        if seed_strategy == SEED_STRATEGY_FIXED_BASE_SEQUENTIAL or seed_strategy == SEED_STRATEGY_RANDOM_BASE_SEQUENTIAL:
            line_offset = i * num_versions * 10
            line_version_seeds = [(base_seed + line_offset + j) % (2**32) for j in range(num_versions)]
            seeds_generated_this_line = True
        elif seed_strategy == SEED_STRATEGY_FIXED_BASE_REUSED_LIST:
            line_version_seeds = sequential_seed_list
            seeds_generated_this_line = False
        elif seed_strategy == SEED_STRATEGY_RANDOM_BASE_REUSED_LIST:
            line_version_seeds = random_seed_list
            seeds_generated_this_line = False
        if seeds_generated_this_line:
             current_seed_info = ', '.join([f'V{k+1}=S({s})' for k, s in enumerate(line_version_seeds)])
             status_current = f"\nGenerating Line {line_idx_0_based + 1}/{total_lines} ({speaker_filename}). Seeds: [{current_seed_info}]";
             print(status_current); status_log.append(status_current)
             yield "\n".join(status_log), parsed_script, all_options, {}, -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
        elif i == 0 and not seeds_generated_this_line and seed_strategy != SEED_STRATEGY_FULLY_RANDOM:
             status_log.append(f"-> Applying reused seed list for Line {line_idx_0_based + 1}...")
             print(f"-> Applying reused seed list for Line {line_idx_0_based + 1}...")
             yield "\n".join(status_log), parsed_script, all_options, {}, -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
        line_options_generated: List[Optional[str]] = [None] * num_versions
        for j in range(num_versions):
            version_idx_1_based = j + 1
            if seed_strategy == SEED_STRATEGY_FULLY_RANDOM:
                current_seed = random.randint(0, 2**32 - 1)
                print(f"  Generating V{version_idx_1_based} (Seed: R({current_seed}))... Text: '{line_text[:30]}...'")
            else:
                current_seed = line_version_seeds[j]
                print(f"  Generating V{version_idx_1_based} (Seed: {current_seed})... Text: '{line_text[:30]}...'")
            segment_filename = f"line{line_idx_0_based:03d}_spk-{Path(speaker_filename).stem}_v{j+1:02d}_s{current_seed}.wav";
            output_path = str(TEMP_CONVO_MULTI_DIR / segment_filename)
            try:
                tts.infer(speaker_path_str, line_text, output_path, temperature=temperature_val, top_p=top_p_val, top_k=int(top_k_val), seed=current_seed)
                if Path(output_path).is_file() and Path(output_path).stat().st_size > 0:
                     line_options_generated[j] = output_path
                     print(f"    V{version_idx_1_based} OK: {segment_filename}")
                else:
                     print(f"    V{version_idx_1_based} FAILED (file not found or empty).")
                     status_log.append(f"  Warning: Line {line_idx_0_based + 1}, V{version_idx_1_based} initial generation failed (file missing/empty).")
            except Exception as e:
                err_msg = f"Error L{line_idx_0_based + 1} V{version_idx_1_based}: Failed during initial TTS inference: {type(e).__name__}: {e}"; print(f"    {err_msg}"); status_log.append(f"\n❌ {err_msg}"); traceback.print_exc(); generation_successful = False; break
        if generation_successful and SPEECHBRAIN_AVAILABLE and speaker_similarity_model is not None and ref_speaker_path_exists:
            status_log.append(f"  Checking speaker similarity for Line {line_idx_0_based + 1} (Auto-Regen Threshold: {SIMILARITY_THRESHOLD:.2f})...")
            yield "\n".join(status_log), parsed_script, all_options, {}, -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
            print(f"  Checking speaker similarity for Line {line_idx_0_based + 1} (Threshold: {SIMILARITY_THRESHOLD:.2f})...")
            for j, initial_path in enumerate(line_options_generated):
                 if initial_path and Path(initial_path).is_file():
                     print(f"    Analyzing V{j+1} ({Path(initial_path).name})...")
                     score = analyze_speaker_similarity(speaker_similarity_model, speaker_path_str, initial_path, device=DEVICE.type)
                     if score != -1.0 and score < SIMILARITY_THRESHOLD:
                         status_log.append(f"    🔄 Low similarity on V{j+1} (Score: {score:.2f} < {SIMILARITY_THRESHOLD:.2f}). Triggering auto-regeneration...")
                         print(status_log[-1])
                         yield "\n".join(status_log), parsed_script, all_options, {}, -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
                         regen_success = False
                         for attempt in range(AUTO_REGEN_ATTEMPTS):
                             new_seed = random.randint(0, 2**32 - 1)
                             new_segment_filename = f"line{line_idx_0_based:03d}_spk-{Path(speaker_filename).stem}_v{j+1:02d}_autoregen{attempt+1}_s{new_seed}.wav";
                             new_output_path = str(TEMP_CONVO_MULTI_DIR / new_segment_filename)
                             print(f"      Attempt {attempt+1}/{AUTO_REGEN_ATTEMPTS}: Auto-Regenerating V{j+1} with new seed {new_seed} -> {new_segment_filename}...")
                             try:
                                 tts.infer(speaker_path_str, line_text, new_output_path, temperature=temperature_val, top_p=top_p_val, top_k=int(top_k_val), seed=new_seed)
                                 if Path(new_output_path).is_file() and Path(new_output_path).stat().st_size > 0:
                                     status_log.append(f"        ✅ Auto-Regen V{j+1} (Attempt {attempt+1}) successful. Replacing original.")
                                     print(status_log[-1])
                                     line_options_generated[j] = new_output_path
                                     regen_success = True
                                     break
                                 else:
                                     status_log.append(f"        ❌ Auto-Regen V{j+1} (Attempt {attempt+1}) failed (file missing/empty).")
                                     print(status_log[-1])
                             except Exception as e:
                                 err_msg = f"Error during Auto-Regen L{line_idx_0_based+1} V{j+1} (Attempt {attempt+1}): {type(e).__name__}: {e}"; print(f"      {err_msg}"); status_log.append(f"\n      ❌ {err_msg}"); traceback.print_exc()
                         if not regen_success:
                             status_log.append(f"    ⚠️ Failed to auto-regenerate V{j+1} above threshold after {AUTO_REGEN_ATTEMPTS} attempts. Keeping original low-scoring version.")
                             print(status_log[-1])
                         yield "\n".join(status_log), parsed_script, all_options, {}, -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
                     elif score == -1.0:
                         print(f"    ⚠️ Similarity analysis failed for V{j+1}. Skipping auto-regeneration.")
                         status_log.append(f"    ⚠️ Similarity analysis failed for V{j+1}. Skipping.")
                         yield "\n".join(status_log), parsed_script, all_options, {}, -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
        all_options.append(line_options_generated)
        yield "\n".join(status_log), parsed_script, all_options, {}, -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
        if not generation_successful: break
    if generation_successful: status_log.append(f"\n✅ Generation & Auto-Regen Check Complete for {total_lines} lines."); status_log.append("Proceed to 'Review & Select Lines' tab."); review_interactive = True if total_lines > 0 else False
    else: status_log.append(f"\n❌ Generation stopped due to critical error."); review_interactive = False
    print(status_log[-2]);
    if len(status_log) > 1: print(status_log[-1])
    yield "\n".join(status_log), parsed_script, all_options, {}, 0, gr.update(interactive=True), gr.update(interactive=review_interactive), gr.update(interactive=False) # type: ignore

def display_line_for_review(
    target_index: int,
    parsed_script: ParsedScript,
    all_options_state: AllOptionsState,
    selections_state: SelectionsState,
    edited_texts_state: EditedTextsState,
) -> ReviewYield:
    global speaker_similarity_model
    status = "";
    audio_updates = [gr.update(value=None, interactive=False, label=f"{VERSION_PREFIX}{i+1} (No Audio)") for i in range(MAX_VERSIONS_ALLOWED)];
    radio_update = gr.update(choices=[], value=None, interactive=False) # type: ignore
    nav_display = "Line ? / ?"; line_text_display = "N/A"; editable_text_display = ""
    prev_interactive = False; next_interactive = False
    if not isinstance(parsed_script, list) or not parsed_script:
        status = "Error: No script loaded."; print(status)
        return status, nav_display, line_text_display, editable_text_display, target_index, gr.update(interactive=False), gr.update(interactive=False), *audio_updates, radio_update # type: ignore
    num_lines = len(parsed_script)
    if not isinstance(target_index, int) or not (0 <= target_index < num_lines):
        status = f"Error: Invalid index {target_index}. Resetting to 0."; print(status); target_index = 0
    if not isinstance(all_options_state, list) or target_index >= len(all_options_state):
        status = f"Error: Options data missing for line {target_index+1}."
        line_info = parsed_script[target_index]; original_speaker = line_info['speaker_filename']; original_text = line_info['text']
        nav_display = f"Line {target_index + 1} / {num_lines}"; line_text_display = f"Speaker: {original_speaker}\nText: {original_text}"; editable_text_display = original_text
        return status, nav_display, line_text_display, editable_text_display, target_index, gr.update(interactive=False), gr.update(interactive=False), *audio_updates, radio_update # type: ignore
    if not isinstance(selections_state, dict):
        selections_state = {}; print("Warning: selections_state was invalid, reset.")
    if not isinstance(edited_texts_state, dict):
        edited_texts_state = {}; print("Warning: edited_texts_state was invalid, reset.")
    line_info = parsed_script[target_index]; original_speaker = line_info['speaker_filename']; original_text = line_info['text']
    nav_display = f"Line {target_index + 1} / {num_lines}"; line_text_display = f"Speaker: {original_speaker}\nText: {original_text}";
    editable_text_display = edited_texts_state.get(target_index, original_text)
    current_line_options = all_options_state[target_index] if target_index < len(all_options_state) and isinstance(all_options_state[target_index], list) else []
    similarities: List[Tuple[int, float]] = []
    ref_speaker_path = str(SPEAKER_DIR / original_speaker)
    can_analyze = SPEECHBRAIN_AVAILABLE and speaker_similarity_model is not None and Path(ref_speaker_path).is_file()
    if not can_analyze:
         status += " (Warn: Sim analysis unavailable)"
    valid_audio_paths_with_indices: List[Tuple[int, str]] = []
    for i in range(MAX_VERSIONS_ALLOWED):
        path = current_line_options[i] if i < len(current_line_options) else None
        if path and isinstance(path, str):
            try:
                p = Path(path)
                if p.is_file() and p.stat().st_size > 0:
                    valid_audio_paths_with_indices.append((i, str(p)))
                else:
                    audio_updates[i] = gr.update(value=None, interactive=False, label=f"{VERSION_PREFIX}{i+1} (Not Found)")
            except Exception as e:
                print(f"Error checking file for Line {target_index+1} V{i+1} ({path}): {e}")
                audio_updates[i] = gr.update(value=None, interactive=False, label=f"{VERSION_PREFIX}{i+1} (Error)")
        else:
             is_option_expected = i < len(current_line_options)
             audio_updates[i] = gr.update(value=None, interactive=False, label=f"{VERSION_PREFIX}{i+1} ({'Not Found' if is_option_expected else 'Not Generated'})")
    if can_analyze and valid_audio_paths_with_indices:
        for i, audio_path_str in valid_audio_paths_with_indices:
            score = analyze_speaker_similarity(speaker_similarity_model, ref_speaker_path, audio_path_str, device=DEVICE.type)
            if score != -1.0:
                 similarities.append((i, score))
            else:
                 audio_updates[i] = gr.update(value=audio_path_str, interactive=True, label=f"{VERSION_PREFIX}{i+1} (Analysis Error)")
    radio_choices = []
    highest_score = -2.0
    best_version_index = -1
    best_version_label = None
    for i, audio_path_str in valid_audio_paths_with_indices:
        score_info = ""
        current_score = -1.0
        for idx, score in similarities:
            if idx == i:
                score_info = f" (Sim: {score:.2f})"
                current_score = score
                break
        label_text = f"{VERSION_PREFIX}{i+1}{score_info}"
        radio_choices.append(label_text)
        audio_updates[i] = gr.update(value=audio_path_str, interactive=True, label=label_text)
        if current_score > highest_score:
             highest_score = current_score
             best_version_index = i
             best_version_label = label_text
    current_selection_index = selections_state.get(target_index, -1)
    current_selection_label = None
    radio_interactive = bool(radio_choices)
    is_current_selection_valid = False
    if 0 <= current_selection_index < MAX_VERSIONS_ALLOWED:
        expected_label_prefix = f"{VERSION_PREFIX}{current_selection_index + 1}"
        for choice_label in radio_choices:
            if choice_label.startswith(expected_label_prefix):
                current_selection_label = choice_label
                is_current_selection_valid = True
                break
    if not is_current_selection_valid:
        if best_version_label:
            current_selection_label = best_version_label
            if selections_state.get(target_index) != best_version_index:
                 selections_state[target_index] = best_version_index
        elif radio_choices:
            current_selection_label = radio_choices[0]
            try:
                fallback_index = int(radio_choices[0].split(" ")[1]) - 1
                if selections_state.get(target_index) != fallback_index:
                    selections_state[target_index] = fallback_index
            except (ValueError, IndexError): pass
        else:
            current_selection_label = None
            if target_index in selections_state: del selections_state[target_index]
    radio_update = gr.update(choices=radio_choices, value=current_selection_label, interactive=radio_interactive)
    prev_interactive = (target_index > 0)
    next_interactive = (target_index < num_lines - 1)
    if not status: status = f"Displaying {len(valid_audio_paths_with_indices)} versions for Line {target_index + 1}."
    if not radio_choices: status += " (Warn: No valid audio!)"
    elif can_analyze and not similarities and valid_audio_paths_with_indices: status += " (Warn: Sim analysis failed)"
    return status, nav_display, line_text_display, editable_text_display, target_index, gr.update(interactive=prev_interactive), gr.update(interactive=next_interactive), *audio_updates, radio_update # type: ignore

def save_seed_data(seed_data_dict):
    """Saves the selected seed data dictionary to a JSON file."""
    try:
        with open(SELECTED_SEEDS_FILENAME, 'w', encoding='utf-8') as f:
             json.dump(seed_data_dict, f, indent=4, ensure_ascii=False)
        print(f"Selected seed data saved to {SELECTED_SEEDS_FILENAME}")
        return f"Seed data saved for {len(seed_data_dict)} lines."
    except Exception as e:
        error_msg = f"Error saving seed data: {e}"
        print(error_msg)
        traceback.print_exc()
        return error_msg

def update_convo_selection(
    choice_input: Optional[str],
    current_line_index: int,
    selections_state: SelectionsState,
    parsed_script: ParsedScript,
    all_options_state: AllOptionsState,
    selected_seed_data_state: dict
) -> Tuple[SelectionsState, dict, str]: # Return updated states AND status
    status = ""
    if not isinstance(current_line_index, int) or current_line_index < 0:
        print(f"Warn update_sel: invalid index {current_line_index}")
        return selections_state, selected_seed_data_state, "Error: Invalid line index."
    if choice_input is None:
        return selections_state, selected_seed_data_state, ""
    if not isinstance(selections_state, dict): selections_state = {}
    if not isinstance(selected_seed_data_state, dict): selected_seed_data_state = {}
    choice_index: int = -1
    seed_value: Optional[int] = None
    selected_file_path: Optional[str] = None
    if isinstance(choice_input, str) and choice_input.startswith(VERSION_PREFIX):
        try:
            version_num_str = choice_input[len(VERSION_PREFIX):].split(" ")[0]
            choice_index = int(version_num_str) - 1
        except (ValueError, IndexError):
            error_msg = f"Error: Cannot parse choice index from '{choice_input}'"
            print(error_msg); return selections_state, selected_seed_data_state, error_msg
    else:
        error_msg = f"Error: Invalid choice format '{choice_input}'"
        print(error_msg); return selections_state, selected_seed_data_state, error_msg
    if not (0 <= choice_index < MAX_VERSIONS_ALLOWED):
        error_msg = f"Error: Parsed choice index {choice_index} out of bounds"
        print(error_msg); return selections_state, selected_seed_data_state, error_msg
    try:
        if current_line_index < len(all_options_state):
            line_options = all_options_state[current_line_index]
            if isinstance(line_options, list) and choice_index < len(line_options):
                selected_file_path = line_options[choice_index]
                if selected_file_path and isinstance(selected_file_path, str):
                    filename = Path(selected_file_path).name
                    match = re.search(r'_s(\d+)\.wav$', filename, re.IGNORECASE)
                    if match:
                        seed_value = int(match.group(1))
                        print(f"  Extracted seed {seed_value} from {filename}")
                    else:
                        print(f"  Warning: Could not parse seed from filename: {filename}")
                        status = "Warning: Seed not found in filename."
                else:
                     print(f"  Warning: Invalid file path data for selected option L{current_line_index+1} V{choice_index+1}.")
                     status = "Warning: Invalid file path."
            else:
                 print(f"  Warning: Options list invalid or index out of bounds for L{current_line_index+1}.")
                 status = "Warning: Options data error."
        else:
            print(f"  Warning: Line index {current_line_index} out of bounds for all_options_state.")
            status = "Warning: Line index error."
    except Exception as e:
        print(f"Error extracting seed: {e}")
        status = "Error extracting seed."
        traceback.print_exc()
    updated_selections = selections_state.copy()
    updated_selections[current_line_index] = choice_index
    print(f"Selection state updated for Line {current_line_index + 1}: Now V{choice_index + 1}")
    updated_seed_data = selected_seed_data_state.copy()
    if seed_value is not None and current_line_index < len(parsed_script):
        line_data = parsed_script[current_line_index]
        updated_seed_data[str(current_line_index)] = { # Use string key for JSON compatibility
            "line_index": current_line_index,
            "speaker": line_data.get('speaker_filename', 'Unknown'),
            "text": line_data.get('text', 'Unknown'),
            "selected_version_index": choice_index,
            "selected_version_path": selected_file_path,
            "seed": seed_value
        }
        print(f"  Stored seed data for Line {current_line_index + 1}")
        if not status: status = f"Selected V{choice_index+1} (Seed: {seed_value})."
    elif seed_value is None and not status:
        status = f"Selected V{choice_index+1}. (Could not save seed)."
    return updated_selections, updated_seed_data, status

def regenerate_below_threshold(
    current_line_index: int,
    manual_threshold: float,
    editable_text: str,
    parsed_script: ParsedScript,
    all_options_state: AllOptionsState,
    selections_state: SelectionsState,
    edited_texts_state: EditedTextsState,
    temperature_val: float, top_p_val: float, top_k_val: int,
    max_manual_attempts_str: str,
    progress=gr.Progress(track_tqdm=True)
) -> Iterator[ManualRegenYield]:
    global speaker_similarity_model
    line_idx_0_based = current_line_index
    try:
        max_manual_attempts_int = int(max_manual_attempts_str)
        if not (max_manual_attempts_int >= 0):
             raise ValueError("Max attempts must be 0 or greater.")
    except (ValueError, TypeError):
        status = f"Error: Invalid 'Max Manual Regen Attempts' value ('{max_manual_attempts_str}') selected on Tab 1. Defaulting to 5."
        print(status)
        max_manual_attempts_int = 5
        all_options_copy = [list(opts) if isinstance(opts, list) else [] for opts in all_options_state]
        selections_copy = selections_state.copy(); edited_texts_copy = edited_texts_state.copy()
        yield status, all_options_copy, selections_copy, edited_texts_copy; return
    status = f"Starting threshold process for Line {line_idx_0_based + 1} (Threshold: {manual_threshold:.2f}, Max Regen Attempts: {max_manual_attempts_int})..."
    all_options_copy = [list(opts) if isinstance(opts, list) else [] for opts in all_options_state]
    selections_copy = selections_state.copy()
    edited_texts_copy = edited_texts_state.copy()
    yield status, all_options_copy, selections_copy, edited_texts_copy
    if tts is None: yield "Error: TTS model not initialized.", all_options_copy, selections_copy, edited_texts_copy; return
    if not SPEECHBRAIN_AVAILABLE or speaker_similarity_model is None: yield "Error: Speaker model not available for similarity check.", all_options_copy, selections_copy, edited_texts_copy; return
    if not isinstance(parsed_script, list) or not (0 <= line_idx_0_based < len(parsed_script)): yield f"Error: Invalid script data or line index ({line_idx_0_based}).", all_options_copy, selections_copy, edited_texts_copy; return
    if not isinstance(all_options_copy, list) or line_idx_0_based >= len(all_options_copy): yield f"Error: Options state invalid for line index ({line_idx_0_based}).", all_options_copy, selections_copy, edited_texts_copy; return
    if not editable_text or not editable_text.strip(): yield f"Error: Editable text for Line {line_idx_0_based + 1} cannot be empty.", all_options_copy, selections_copy, edited_texts_copy; return
    line_info = parsed_script[line_idx_0_based]; speaker_filename = line_info['speaker_filename'];
    ref_speaker_path = str(SPEAKER_DIR / speaker_filename)
    if not Path(ref_speaker_path).is_file(): yield f"Error: Reference speaker '{speaker_filename}' not found.", all_options_copy, selections_copy, edited_texts_copy; return
    speaker_path_str = ref_speaker_path
    attempts_per_slot = {j: 0 for j in range(MAX_VERSIONS_ALLOWED)}
    slot_results = {}
    slots_to_process_initially = []
    initial_options = all_options_copy[line_idx_0_based]
    status += "\nAnalyzing initial versions..."
    yield status, all_options_copy, selections_copy, edited_texts_copy
    print(f"Analyzing Line {line_idx_0_based + 1} against manual threshold {manual_threshold:.2f}...")
    for j, audio_path in enumerate(initial_options):
        current_best_score = -2.0
        current_best_path = audio_path
        generated_paths_for_slot = []
        if audio_path and Path(audio_path).is_file():
            print(f"  Checking initial V{j+1} ({Path(audio_path).name})...")
            score = analyze_speaker_similarity(speaker_similarity_model, ref_speaker_path, audio_path, device=DEVICE.type)
            if score != -1.0:
                 current_best_score = score
                 generated_paths_for_slot.append(audio_path)
                 if score < manual_threshold:
                     slots_to_process_initially.append(j)
                     print(f"    Marked V{j+1} for regeneration (Initial Score: {score:.2f} < {manual_threshold:.2f})")
                 else:
                      print(f"    V{j+1} meets threshold (Initial Score: {score:.2f})")
            else:
                 status += f"\n  ⚠️ Analysis failed for initial V{j+1}. Cannot evaluate."
                 print(f"    Analysis failed for initial V{j+1}.")
                 current_best_path = None
        else:
             if max_manual_attempts_int > 0:
                print(f"    Marked empty/invalid slot V{j+1} for regeneration attempt.")
                slots_to_process_initially.append(j)
                current_best_path = None
             else:
                print(f"    Skipping empty/invalid slot V{j+1} as Max Attempts is 0.")
        slot_results[j] = {
            'best_score': current_best_score,
            'best_path': current_best_path,
            'generated_paths': generated_paths_for_slot
        }
    if not slots_to_process_initially and max_manual_attempts_int > 0 :
        status += "\nNo versions initially below threshold. No regeneration needed."
        print(f"No versions initially below threshold for Line {line_idx_0_based + 1}.")
        yield status, all_options_copy, selections_copy, edited_texts_copy
        return
    elif max_manual_attempts_int == 0:
        status += "\nMax attempts set to 0. Skipping regeneration, proceeding to cleanup."
        print(f"Max attempts set to 0. Skipping regeneration for Line {line_idx_0_based + 1}.")
    slots_still_processing = list(slots_to_process_initially)
    overall_regenerated_count = 0 # Initialize counter for successful regens
    if max_manual_attempts_int > 0:
        status += f"\nAttempting regeneration for {len(slots_still_processing)} slot(s)..."
        yield status, all_options_copy, selections_copy, edited_texts_copy
        for attempt in range(max_manual_attempts_int):
            current_attempt_num = attempt + 1
            if not slots_still_processing: break
            status += f"\n--- Regeneration Pass {current_attempt_num}/{max_manual_attempts_int} ---"
            yield status, all_options_copy, selections_copy, edited_texts_copy
            print(f"--- Starting Regen Pass {current_attempt_num}/{max_manual_attempts_int} for Line {line_idx_0_based + 1} ---")
            slots_processed_this_pass = list(slots_still_processing)
            pass_regenerated_count = 0
            for j in progress.tqdm(slots_processed_this_pass, desc=f"Regen Pass {current_attempt_num}"):
                attempts_per_slot[j] = attempts_per_slot.get(j, 0) + 1
                version_idx_1_based = j + 1
                new_seed = random.randint(0, 2**32 - 1)
                segment_filename = f"line{line_idx_0_based:03d}_spk-{Path(speaker_filename).stem}_v{j+1:02d}_mregen{current_attempt_num}_s{new_seed}.wav";
                new_output_path = str(TEMP_CONVO_MULTI_DIR / segment_filename)
                print(f"  Regenerating V{version_idx_1_based} (Attempt {attempts_per_slot[j]}, Seed: {new_seed})...")
                try:
                    tts.infer(speaker_path_str, editable_text, output_path=new_output_path, temperature=temperature_val, top_p=top_p_val, top_k=int(top_k_val), seed=new_seed)
                    if Path(new_output_path).is_file() and Path(new_output_path).stat().st_size > 0:
                        print(f"    V{version_idx_1_based} (Attempt {current_attempt_num}) generated OK.")
                        slot_results[j]['generated_paths'].append(new_output_path)
                        new_score = analyze_speaker_similarity(speaker_similarity_model, ref_speaker_path, new_output_path, device=DEVICE.type)
                        if new_score != -1.0:
                             print(f"      New Score: {new_score:.2f}")
                             if new_score > slot_results[j]['best_score']:
                                 slot_results[j]['best_score'] = new_score
                                 slot_results[j]['best_path'] = new_output_path
                                 print(f"      * New best score for V{version_idx_1_based} *")
                             if new_score >= manual_threshold:
                                 print(f"      Threshold {manual_threshold:.2f} met/exceeded. Stopping retries for V{version_idx_1_based}.")
                                 if j in slots_still_processing: slots_still_processing.remove(j)
                        else:
                            print(f"      ⚠️ Analysis failed for regenerated V{version_idx_1_based} (Attempt {current_attempt_num}).")
                            status += f"\n  ⚠️ Analysis failed for regen V{version_idx_1_based} (Attempt {current_attempt_num})."
                        pass_regenerated_count += 1
                        overall_regenerated_count +=1 # Count successful generations
                        edited_texts_copy[line_idx_0_based] = editable_text
                    else:
                         status += f"\n  ❌ Regen V{version_idx_1_based} (Attempt {attempts_per_slot[j]}) failed (file missing/empty)."
                         print(f"    V{version_idx_1_based} (Attempt {attempts_per_slot[j]}) FAILED.")
                except Exception as e:
                     err_msg = f"Error during Regen L{line_idx_0_based + 1} V{version_idx_1_based} (Attempt {attempts_per_slot[j]}): {type(e).__name__}: {e}"; print(f"    {err_msg}"); status += f"\n❌ {err_msg}"; traceback.print_exc()
            yield status, all_options_copy, selections_copy, edited_texts_copy # Update UI after each version attempt within a pass
        status += f"\n Pass {max(attempts_per_slot.values())} finished. ({pass_regenerated_count}/{len(slots_processed_this_pass)} regenerated this pass)."
        yield status, all_options_copy, selections_copy, edited_texts_copy # Update UI after each full pass
    status += f"\n--- Finalizing results for Line {line_idx_0_based + 1} ---"
    print(f"--- Finalizing results for Line {line_idx_0_based + 1} ---")
    successfully_updated_count = 0
    for j in range(MAX_VERSIONS_ALLOWED):
        if j in slot_results:
            final_best_path = slot_results[j]['best_path']
            final_best_score = slot_results[j]['best_score']
            # Update the main state AFTER cleanup, using the determined best path
            all_options_copy[line_idx_0_based][j] = final_best_path # This assignment happens here now
            if final_best_path is not None:
                successfully_updated_count +=1
                print(f"  Final selection for V{j+1}: {Path(final_best_path).name} (Score: {final_best_score:.2f})")
                # Only update edited text if we actually tried to regenerate this slot
                if j in slots_to_process_initially and max_manual_attempts_int > 0:
                    edited_texts_copy[line_idx_0_based] = editable_text
                # Clean up other generated files for this slot
                for generated_path in slot_results[j]['generated_paths']:
                    if generated_path != final_best_path: # Don't delete the best one!
                        try:
                            p = Path(generated_path)
                            if p.is_file(): # Check if it exists before deleting
                                print(f"    Cleaning up: {p.name}")
                                p.unlink()
                        except OSError as del_err:
                            print(f"    Warning: Failed to delete temp file {generated_path}: {del_err}")
            else:
                 print(f"  V{j+1}: No valid version could be generated or kept.")
                 all_options_copy[line_idx_0_based][j] = None # Ensure slot is None if all failed
    status += f"\n✅ Persistent regeneration/cleanup finished. Kept best version for each slot."
    print(f"Persistent regeneration/cleanup finished for Line {line_idx_0_based + 1}.")
    yield status, all_options_copy, selections_copy, edited_texts_copy

RegenLineYield = Tuple[str, AllOptionsState, SelectionsState, EditedTextsState]
def regenerate_single_line(
    current_line_index: int, parsed_script: ParsedScript, all_options_state: AllOptionsState, selections_state: SelectionsState, edited_texts_state: EditedTextsState,
    editable_text: str, num_versions_str: str,
    temperature_val: float, top_p_val: float, top_k_val: int,
    # <<< UPDATED: Use seed strategy inputs >>>
    seed_strategy: str,
    fixed_base_seed: int,
    progress=gr.Progress(track_tqdm=True)
) -> Iterator[RegenLineYield]:
    line_idx_0_based = current_line_index
    status = f"Starting regeneration for Line {line_idx_0_based + 1} using strategy: {seed_strategy}..."
    all_options_copy = [list(opts) if opts else [] for opts in all_options_state]
    selections_copy = selections_state.copy()
    edited_texts_copy = edited_texts_state.copy()
    yield status, all_options_copy, selections_copy, edited_texts_copy
    if tts is None: yield "Error: TTS model not initialized.", all_options_copy, selections_copy, edited_texts_copy; return
    if not isinstance(parsed_script, list) or not (0 <= line_idx_0_based < len(parsed_script)): yield f"Error: Invalid script data or line index ({line_idx_0_based}).", all_options_copy, selections_copy, edited_texts_copy; return
    if not isinstance(all_options_copy, list) or line_idx_0_based >= len(all_options_copy): yield f"Error: Options state invalid for line index ({line_idx_0_based}).", all_options_copy, selections_copy, edited_texts_copy; return
    if not editable_text or not editable_text.strip(): yield f"Error: Editable text for Line {line_idx_0_based + 1} cannot be empty.", all_options_copy, selections_copy, edited_texts_copy; return
    line_info = parsed_script[line_idx_0_based]; speaker_filename = line_info['speaker_filename']; speaker_path = SPEAKER_DIR / speaker_filename
    if not speaker_path.is_file(): yield f"Error: Speaker file '{speaker_filename}' not found for Line {line_idx_0_based + 1}.", all_options_copy, selections_copy, edited_texts_copy; return
    speaker_path_str = str(speaker_path)
    try:
        num_versions = int(num_versions_str);
        if not (1 <= num_versions <= MAX_VERSIONS_ALLOWED): raise ValueError(f"Versions must be 1-{MAX_VERSIONS_ALLOWED}.")
    except (ValueError, TypeError):
        yield f"Error: Invalid number of versions '{num_versions_str}' selected on Tab 1.", all_options_copy, selections_copy, edited_texts_copy; return

    base_seed_for_line = 0
    version_seeds = []
    seed_log_info = ""
    if seed_strategy == SEED_STRATEGY_FIXED_BASE_SEQUENTIAL or seed_strategy == SEED_STRATEGY_FIXED_BASE_REUSED_LIST:
         try: base_seed_for_line = int(fixed_base_seed)
         except: base_seed_for_line = DEFAULT_FIXED_BASE_SEED; print("Warning: Using default fixed base seed for regen.")
    elif seed_strategy == SEED_STRATEGY_RANDOM_BASE_SEQUENTIAL or seed_strategy == SEED_STRATEGY_RANDOM_BASE_REUSED_LIST:
         base_seed_for_line = random.randint(0, 2**32 - 1)
    if seed_strategy == SEED_STRATEGY_FULLY_RANDOM:
        version_seeds = random.sample(range(2**32), num_versions)
        seed_log_info = "Fully Random Seeds: " + ', '.join([f'R({s})' for s in version_seeds])
    elif seed_strategy == SEED_STRATEGY_RANDOM_BASE_SEQUENTIAL or seed_strategy == SEED_STRATEGY_FIXED_BASE_SEQUENTIAL:
        line_offset = line_idx_0_based * num_versions * 10
        version_seeds = [(base_seed_for_line + line_offset + j) % (2**32) for j in range(num_versions)]
        seed_log_info = f"{'Random' if seed_strategy == SEED_STRATEGY_RANDOM_BASE_SEQUENTIAL else 'Fixed'} Base + Seq Offset Seeds: " + ', '.join([f'S({s})' for s in version_seeds])
    elif seed_strategy == SEED_STRATEGY_RANDOM_BASE_REUSED_LIST:
        version_seeds = random.sample(range(2**32), num_versions)
        seed_log_info = "New Random List Seeds: " + ', '.join([f'R({s})' for s in version_seeds])
    elif seed_strategy == SEED_STRATEGY_FIXED_BASE_REUSED_LIST:
        version_seeds = [(base_seed_for_line + j) % (2**32) for j in range(num_versions)]
        seed_log_info = "Fixed Base + Reused Seq List Seeds: " + ', '.join([f'S({s})' for s in version_seeds])
    status += f"\nUsing seeds: {seed_log_info}"; print(f"Regenerating Line {line_idx_0_based + 1} - Seeds: {seed_log_info}"); yield status, all_options_copy, selections_copy, edited_texts_copy

    new_line_options: List[Optional[str]] = [None] * num_versions; generation_successful = True;
    line_status = f"\nRegenerating {num_versions} versions for Line {line_idx_0_based + 1} ({speaker_filename})..."; status += line_status; print(line_status); yield status, all_options_copy, selections_copy, edited_texts_copy
    for j in progress.tqdm(range(num_versions), desc=f"Regen Line {line_idx_0_based+1}"):
        version_idx_1_based = j + 1; current_seed = version_seeds[j];
        segment_filename = f"line{line_idx_0_based:03d}_spk-{Path(speaker_filename).stem}_v{j+1:02d}_regen_s{current_seed}.wav";
        output_path = str(TEMP_CONVO_MULTI_DIR / segment_filename)
        print(f"  Regenerating V{version_idx_1_based} (Seed: {current_seed})... Text: '{editable_text[:30]}...'")
        try:
            if Path(output_path).exists():
                 try: Path(output_path).unlink()
                 except OSError as del_e: print(f"    Warn: Could not delete existing file {output_path}: {del_e}")
            tts.infer(speaker_path_str, editable_text, output_path, temperature=temperature_val, top_p=top_p_val, top_k=int(top_k_val), seed=current_seed)
            if Path(output_path).is_file() and Path(output_path).stat().st_size > 0: new_line_options[j] = output_path; print(f"    V{version_idx_1_based} OK: {segment_filename}")
            else: print(f"    V{version_idx_1_based} FAILED (file not found or empty after generation)."); status += f"\n  Warning: Line {line_idx_0_based + 1}, V{version_idx_1_based} failed (file missing/empty)."
        except Exception as e: err_msg = f"Regen L{line_idx_0_based+1} V{version_idx_1_based}: Failed during TTS inference: {type(e).__name__}: {e}"; print(f"    {err_msg}"); status += f"\n❌ {err_msg}"; traceback.print_exc(); generation_successful = False

    if any(p is not None for p in new_line_options):
         all_options_copy[line_idx_0_based] = new_line_options
         first_valid_index = -1
         for idx, opt_path in enumerate(new_line_options):
             if opt_path and Path(opt_path).is_file() and Path(opt_path).stat().st_size > 0:
                 first_valid_index = idx
                 break
         if first_valid_index != -1:
             selections_copy[line_idx_0_based] = first_valid_index
             edited_texts_copy[line_idx_0_based] = editable_text
         else:
            if line_idx_0_based in selections_copy: del selections_copy[line_idx_0_based]
            if line_idx_0_based in edited_texts_copy: del edited_texts_copy[line_idx_0_based]
         status += f"\n✅ Regeneration finished for Line {line_idx_0_based + 1}."
         if not all(p is not None and Path(p).is_file() and Path(p).stat().st_size > 0 for p in new_line_options):
             status += " (with errors)"
             generation_successful = False
         print(f"Regeneration finished for Line {line_idx_0_based + 1}.")
         yield status, all_options_copy, selections_copy, edited_texts_copy
    else:
        status += f"\n❌ Regeneration completely failed for Line {line_idx_0_based + 1}. No files generated.";
        print(f"Regeneration completely failed for Line {line_idx_0_based + 1}.")
        yield status, all_options_state, selections_state, edited_texts_state

def concatenate_conversation_versions(
    parsed_script: ParsedScript, all_options_state: AllOptionsState, selections_state: SelectionsState,
    output_format: str, mp3_bitrate: str,
    normalize_segments_flag: bool,
    trim_silence_flag: bool, trim_thresh_dbfs_val: float, trim_min_silence_len_ms_val: int,
    apply_compression_flag: bool, compress_thresh_db: float, compress_ratio_val: float, compress_attack_ms: float, compress_release_ms: float,
    apply_nr_flag: bool, nr_strength_val: float,
    eq_low_gain: float, eq_mid_gain: float, eq_high_gain: float,
    apply_peak_norm_flag: bool, peak_norm_target_dbfs: float,
    reverb_amount: float,
    pitch_shift: float,
    speed_factor: float,
    progress=gr.Progress(track_tqdm=True)
) -> Iterator[Tuple[str, Optional[str]]]:
    status = "Starting concatenation & processing..."; final_output_path = None; yield status, None
    if not PYDUB_AVAILABLE or AudioSegment is None: yield "Error: Pydub not available.", None; return
    if not isinstance(parsed_script, list): yield "Error: No script data.", None; return
    if not isinstance(all_options_state, list): yield "Error: No options data.", None; return
    if not isinstance(selections_state, dict): yield "Error: Invalid selection data.", None; return
    try:
        export_format = output_format.lower().strip();
        if export_format not in OUTPUT_FORMAT_CHOICES: raise ValueError("Invalid output format")
        bitrate_str = f"{mp3_bitrate}k" if export_format == "mp3" else None
        _trim_thresh = float(trim_thresh_dbfs_val)
        _trim_len = int(trim_min_silence_len_ms_val)
        _compress_thresh = float(compress_thresh_db)
        _compress_ratio = float(compress_ratio_val)
        _compress_attack = float(compress_attack_ms)
        _compress_release = float(compress_release_ms)
        _nr_strength = float(nr_strength_val)
        _eq_low = float(eq_low_gain)
        _eq_mid = float(eq_mid_gain)
        _eq_high = float(eq_high_gain)
        _final_norm_target = float(peak_norm_target_dbfs)
        _segment_norm_target = DEFAULT_SEGMENT_NORM_TARGET_DBFS
        _reverb_amount = float(reverb_amount)
        _pitch_shift = float(pitch_shift)
        _speed_factor = float(speed_factor)
    except (ValueError, TypeError) as e:
        yield f"Error: Invalid parameter settings ({type(e).__name__}: {e})", None; return
    selected_files: List[str] = []; warnings: List[str] = []; print("Collecting selected files for concatenation:")
    for i in range(len(parsed_script)):
        selected_version_index = selections_state.get(i, -1); path_found = False
        if selected_version_index == -1: warnings.append(f"L{i+1}: No version selected."); continue
        line_options = []
        if i < len(all_options_state) and isinstance(all_options_state[i], list):
             line_options = all_options_state[i]
        else:
             warnings.append(f"L{i+1}: Options data missing or invalid.")
             continue
        if 0 <= selected_version_index < len(line_options):
             selected_path = line_options[selected_version_index];
             if selected_path and isinstance(selected_path, str):
                 try:
                     p = Path(selected_path);
                     if p.is_file() and p.stat().st_size > 0:
                         selected_files.append(str(p));
                         print(f"  Line {i+1}: Selected V{selected_version_index+1} ({p.name})");
                         path_found = True
                     elif p.exists(): warnings.append(f"L{i+1},V{selected_version_index+1}: File empty ('{p.name}').")
                     else: warnings.append(f"L{i+1},V{selected_version_index+1}: File not found ('{p.name}').")
                 except Exception as e: warnings.append(f"L{i+1},V{selected_version_index+1}: Error checking file '{selected_path}' ({type(e).__name__}: {e})")
             else: warnings.append(f"L{i+1},V{selected_version_index+1}: Invalid path data.")
        else: warnings.append(f"L{i+1}: Invalid selection index ({selected_version_index} for {len(line_options)} options).")
        if not path_found: warnings.append(f"L{i+1}: Could not find valid file for selected V{selected_version_index+1}.")
    if warnings: status += "\nWarnings during file collection:\n" + "\n".join([f"  - {w}" for w in warnings]); print("Warnings:", warnings)
    if not selected_files: status += "\n\nError: No valid audio files were selected or found."; yield status, None; return
    status += f"\nLoading, normalizing (if enabled), and trimming {len(selected_files)} segments...";
    print(status.split('\n')[-1]); yield status, None
    processed_segments: List[AudioSegment] = []; _trim_keep = DEFAULT_TRIM_KEEP_SILENCE_MS;
    loading_desc = "Loading/Norm/Trim" if normalize_segments_flag and trim_silence_flag \
                   else "Loading/Norm" if normalize_segments_flag \
                   else "Loading/Trim" if trim_silence_flag \
                   else "Loading Segments"
    if trim_silence_flag and not PYDUB_SILENCE_AVAILABLE:
        status += "\nWarning: Silence trimming enabled but pydub.silence not available. Skipping trim.";
        print(" Warning: Skipping trim (module unavailable)."); trim_silence_flag = False
    for i, path_str in enumerate(progress.tqdm(selected_files, desc=loading_desc)):
        segment = None
        try:
            path = Path(path_str);
            segment = AudioSegment.from_file(path);
            if normalize_segments_flag:
                current_peak = segment.max_dBFS
                if not math.isinf(current_peak):
                    gain_needed = _segment_norm_target - current_peak
                    segment = segment.apply_gain(gain_needed)
            if trim_silence_flag:
                nonsilent_ranges = _silence_Imported.detect_nonsilent(segment, min_silence_len=_trim_len, silence_thresh=_trim_thresh);
                if nonsilent_ranges:
                    start_trim = max(0, nonsilent_ranges[0][0] - _trim_keep);
                    end_trim = min(len(segment), nonsilent_ranges[-1][1] + _trim_keep)
                else: start_trim, end_trim = 0, 0
                if end_trim > start_trim: segment = segment[start_trim:end_trim]
            processed_segments.append(segment)
        except Exception as e:
            warn = f"Warning L{i+1}: Error loading/processing '{os.path.basename(path_str)}' ({type(e).__name__}: {e}). Skipping segment.";
            status += f"\n  {warn}"; print(f"    {warn}"); traceback.print_exc()
    if not processed_segments: status += "\n\nError: Failed to load any valid segments after processing."; yield status, None; return
    status += f"\nConcatenating {len(processed_segments)} segments..."; print(status.split('\n')[-1]); yield status, None; combined = AudioSegment.empty(); silence_between = AudioSegment.silent(duration=300)
    try:
        crossfade_duration = 10
        combined = processed_segments[0]
        for i in range(1, len(processed_segments)):
            combined = combined.append(silence_between, crossfade=crossfade_duration)
            combined = combined.append(processed_segments[i], crossfade=crossfade_duration)
    except Exception as e: status += f"\n\nError during concatenation ({type(e).__name__}: {e})"; yield status, None; return
    if len(combined) == 0: status += "\n\nError: Concatenation resulted in empty audio."; yield status, None; return
    processed_audio = combined
    status += "\nApplying post-processing effects to combined audio..."; print(status.split('\n')[-1]); yield status, None; processing_errors = []
    if _pitch_shift is not None and abs(_pitch_shift) > 0.01:
        try:
            processed_audio = change_pitch(processed_audio, _pitch_shift)
            status += f"\n  Applied Pitch Shift ({_pitch_shift:+.2f} semitones)"
        except Exception as e:
            err_msg = f"Pitch Shift failed ({type(e).__name__}: {e})"
            processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status += f"\n  ❌ Error during Pitch Shift: {err_msg}"; traceback.print_exc()
    if _speed_factor is not None and abs(_speed_factor - 1.0) > 0.01:
        try:
            processed_audio = change_speed(processed_audio, _speed_factor)
            status += f"\n  Applied Speed Change (x{_speed_factor:.2f})"
        except Exception as e:
            err_msg = f"Speed Change failed ({type(e).__name__}: {e})"
            processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status += f"\n  ❌ Error during Speed Change: {err_msg}"; traceback.print_exc()
    if apply_nr_flag:
        if NOISEREDUCE_AVAILABLE:
            try:
                processed_audio = apply_noise_reduction(processed_audio, _nr_strength)
                status += f"\n  Applied Noise Reduction (Strength: {_nr_strength:.2f})"
            except Exception as e:
                err_msg = f"Noise Reduction failed ({type(e).__name__}: {e})"; processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status += f"\n  ❌ Error during Noise Reduction: {err_msg}"; traceback.print_exc()
        else: status += "\n  Skipped Noise Reduction (Library not available)"
    if _eq_low != 0 or _eq_mid != 0 or _eq_high != 0:
        if SCIPY_AVAILABLE:
            try:
                processed_audio = apply_eq(processed_audio, _eq_low, _eq_mid, _eq_high)
                status += f"\n  Applied EQ (L:{_eq_low}, M:{_eq_mid}, H:{_eq_high} dB)"
            except Exception as e:
                err_msg = f"EQ failed ({type(e).__name__}: {e})"; processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status += f"\n  ❌ Error during EQ: {err_msg}"; traceback.print_exc()
        else: status += "\n  Skipped EQ (Library not available)"
    if apply_compression_flag:
        if PYDUB_COMPRESS_AVAILABLE and _compress_dynamic_range_Imported:
            try:
                processed_audio = _compress_dynamic_range_Imported( processed_audio, threshold=_compress_thresh, ratio=_compress_ratio, attack=_compress_attack, release=_compress_release );
                status += f"\n  Applied Compression (Thresh:{_compress_thresh}, Ratio:{_compress_ratio:.1f})"
            except Exception as e:
                err_msg = f"Compression failed ({type(e).__name__}: {e})"; processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status += f"\n  ❌ Error during Compression: {err_msg}"; traceback.print_exc()
        else: status += "\n  Skipped Compression (pydub effects component not available)"
    if _reverb_amount is not None and _reverb_amount > 0.0:
        try:
            processed_audio = apply_reverb(processed_audio, _reverb_amount)
            status += f"\n  Applied Reverb (Amount: {_reverb_amount:.2f})"
        except Exception as e:
            err_msg = f"Reverb failed ({type(e).__name__}: {e})"
            processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status += f"\n  ❌ Error during Reverb: {err_msg}"; traceback.print_exc()
    if apply_peak_norm_flag:
        if PYDUB_AVAILABLE:
            try:
                current_peak_dbfs = processed_audio.max_dBFS
                if not math.isinf(current_peak_dbfs):
                    gain_to_apply = _final_norm_target - current_peak_dbfs;
                    if gain_to_apply < -0.01:
                        processed_audio = processed_audio.apply_gain(gain_to_apply);
                        status += f"\n  Applied Final Peak Normalization (Target: {_final_norm_target}dBFS, Applied Gain: {gain_to_apply:.2f}dB)"
            except Exception as e:
                err_msg = f"Peak Normalization failed ({type(e).__name__}: {e})"; processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status += f"\n  ❌ Error during Peak Normalization: {err_msg}"; traceback.print_exc()
        else: status += "\n  Skipped Final Peak Normalization (pydub not available)"
    yield status, None
    try: FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e: status += f"\nError creating output directory '{FINAL_OUTPUT_DIR}' ({type(e).__name__}: {e})"; yield status, None; return
    final_filename = f"conversation_processed_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{export_format}"; final_output_path = str(FINAL_OUTPUT_DIR / final_filename)
    export_args: Dict[str, Any] = {"format": export_format};
    if bitrate_str: export_args["bitrate"] = bitrate_str
    status += f"\n\nExporting final audio as {export_format.upper()} to:\n'{final_output_path}'..."; print(status.split('\n')[-2]); print(status.split('\n')[-1]); yield status, None
    try:
        processed_audio.export(final_output_path, **export_args);
        final_summary = "\n\n✅ Processing & Export successful!";
        if processing_errors: final_summary += f"\n  Encountered {len(processing_errors)} non-fatal error(s) during post-processing:";
        for proc_err in processing_errors: final_summary += f"\n    - {proc_err}"
        status += final_summary; print(final_summary); yield status, final_output_path
    except Exception as e:
        err_type = type(e).__name__; err = f"\n\n❌ Error exporting final audio ({err_type}: {e})";
        status += err; print(err); traceback.print_exc(); yield status, None


# --- Build Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# IndexTTS Web Demo (Conversation Workflow)")
    parsed_script_state = gr.State([]); all_options_state = gr.State([]); selections_state = gr.State({}); current_line_index_state = gr.State(0); edited_texts_state = gr.State({})
    selected_seed_data_state = gr.State({}) # State for selected seed data

    with gr.Accordion("Generation Parameters (Used by All Generators)", open=True):
        with gr.Row():
            temperature_slider = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=1.0, label="Temperature")
            top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=0.8, label="Top-P")
            top_k_slider = gr.Slider(minimum=1, maximum=100, step=1, value=30, label="Top-K")

    with gr.Tabs() as tabs:
        with gr.TabItem("1. Generate Conversation Lines", id="tab_convo_gen"):
             gr.Markdown("Enter script using `SpeakerFile.ext: Text` format. Set versions/seed, then Generate.")
             with gr.Row():
                 with gr.Column(scale=2):
                     script_input_convo = gr.Textbox(label="Conversation Script", lines=15, placeholder="speaker1.wav: Line 1\nspeaker2.wav: Line 2...")
                     list_speakers_btn = gr.Button("List Available Speakers")
                     available_speakers_display = gr.Textbox(label="Available Speaker Files", interactive=False, lines=5)
                 with gr.Column(scale=1):
                     num_versions_convo_radio = gr.Radio(label="Versions per Line", choices=[str(i) for i in range(1, MAX_VERSIONS_ALLOWED + 1)], value=str(MAX_VERSIONS_ALLOWED), interactive=True)
                     manual_regen_attempts_dd = gr.Dropdown(
                         label="Max Manual Regen Attempts (Tab 2)",
                         info="Max times to retry regenerating a single version slot when using the 'Regenerate Below Manual Threshold' button.",
                         choices=[str(i) for i in range(1, 21)], # 1 to 20 attempts
                         value=DEFAULT_MANUAL_REGEN_ATTEMPTS,
                         interactive=True
                     )
                     with gr.Accordion("Seed Control (Initial Generation)", open=False):
                         seed_strategy_dd = gr.Dropdown(
                             label="Seed Strategy",
                             choices=SEED_STRATEGY_CHOICES,
                             value=DEFAULT_SEED_STRATEGY,
                             interactive=True
                         )
                         fixed_base_seed_input = gr.Number(
                             label="Fixed Base Seed (if applicable)",
                             value=DEFAULT_FIXED_BASE_SEED,
                             visible=(DEFAULT_SEED_STRATEGY == SEED_STRATEGY_FIXED_BASE_SEQUENTIAL or DEFAULT_SEED_STRATEGY == SEED_STRATEGY_FIXED_BASE_REUSED_LIST),
                             interactive=True,
                             precision=0
                         )
                 generate_convo_button = gr.Button("Generate All Lines & Versions", variant="primary")
             convo_gen_status_output = gr.Textbox(label="Generation Status", lines=8, interactive=False, max_lines=20)
             gr.Markdown("<small>*(During generation, a 'Cancel' button will appear next to the progress bar)*</small>", visible=True)

        with gr.TabItem("2. Review & Select Lines", id="tab_review", interactive=False) as review_tab:
             gr.Markdown("### Review and Select Best Version for Each Line");
             if SPEECHBRAIN_AVAILABLE and speaker_similarity_model is not None:
                 gr.Markdown(f"<small>*(Auto-regen below {SIMILARITY_THRESHOLD:.2f} triggered during initial generation ({AUTO_REGEN_ATTEMPTS} attempt(s)). Manual regen below threshold retries up to the limit set on Tab 1. Higher Sim score is closer to reference speaker.)*</small>")
             else:
                 gr.Markdown("<small>*(Speaker similarity analysis disabled. Check logs.)*</small>")
             with gr.Row(): prev_line_button = gr.Button("<< Previous Line", interactive=False); line_nav_display = gr.Markdown("Line 0 / 0"); next_line_button = gr.Button("Next Line >>", interactive=False)
             current_line_display_review = gr.Textbox(label="Current Line Info (Original)", interactive=False, lines=4); editable_line_text_review = gr.Textbox( label="Editable Text for Regeneration", lines=4, interactive=True, placeholder="Edit the text here before clicking Regenerate Current Line..." );
             with gr.Row():
                 regenerate_current_line_button = gr.Button("🔄 Regenerate All (Uses Tab 1 Seed Strategy)")
                 with gr.Column(visible=SPEECHBRAIN_AVAILABLE and speaker_similarity_model is not None):
                    manual_regen_threshold_slider = gr.Slider(
                        label="Manual Regen Similarity Threshold",
                        minimum=MANUAL_SIMILARITY_MIN, maximum=MANUAL_SIMILARITY_MAX,
                        step=MANUAL_SIMILARITY_STEP, value=MANUAL_SIMILARITY_MIN,
                        interactive=True
                    )
                    threshold_regen_button = gr.Button("🔄 Regenerate Below Manual Threshold")
             gr.Markdown(f"Listen to the versions below and select the best one:")
             review_audio_outputs = [];
             with gr.Column():
                 for i in range(MAX_VERSIONS_ALLOWED): audio_player = gr.Audio( label=f"Version {i+1}", type="filepath", interactive=False, visible=True, elem_id=f"review_audio_{i}" ); review_audio_outputs.append(audio_player)
             line_choice_radio = gr.Radio(label="Select Best Version", choices=[], interactive=False, value=None); review_status_output = gr.Textbox(label="Review Status", lines=1, interactive=False); proceed_to_concat_button = gr.Button("Proceed to Concatenate Tab ->", interactive=False)

        with gr.TabItem("3. Concatenate & Export", id="tab_concat", interactive=False, visible=PYDUB_AVAILABLE) as concat_tab:
             gr.Markdown("### Concatenate Selected Lines & Apply Post-Processing");
             with gr.Row():
                 with gr.Column(scale=1):
                     with gr.Accordion("Output Format", open=True): output_format_dropdown = gr.Dropdown(label="Output Format", choices=OUTPUT_FORMAT_CHOICES, value=DEFAULT_OUTPUT_FORMAT, interactive=True); mp3_bitrate_dropdown = gr.Dropdown(label="MP3 Bitrate (kbps)", choices=MP3_BITRATE_CHOICES, value=DEFAULT_MP3_BITRATE, interactive=False, visible=(DEFAULT_OUTPUT_FORMAT=="mp3"))
                     with gr.Accordion("Per-Segment Normalization (Applied BEFORE Concat/Trim)", open=True):
                         normalize_segments_checkbox = gr.Checkbox(label=f"Enable (Normalize each line to {DEFAULT_SEGMENT_NORM_TARGET_DBFS}dBFS peak)", value=True, interactive=PYDUB_AVAILABLE)
                         if not PYDUB_AVAILABLE: gr.Markdown("<small>*(Requires pydub)*</small>")
                     with gr.Accordion("Silence Trimming (Applied AFTER Segment Norm, BEFORE Concat)", open=False): trim_silence_checkbox = gr.Checkbox( label=f"Enable", value=False, interactive=PYDUB_SILENCE_AVAILABLE); trim_threshold_input = gr.Number(label="Trim Threshold (dBFS, lower is stricter)", value=DEFAULT_TRIM_SILENCE_THRESH_DBFS, interactive=PYDUB_SILENCE_AVAILABLE); trim_length_input = gr.Number(label="Trim Min Silence (ms)", value=DEFAULT_TRIM_MIN_SILENCE_LEN_MS, minimum=50, step=50, precision=0, interactive=PYDUB_SILENCE_AVAILABLE);
                     if not PYDUB_SILENCE_AVAILABLE: gr.Markdown("<small>*(Requires pydub silence component)*</small>")
                     with gr.Accordion("Pitch & Speed (Applied AFTER Concat)", open=False):
                         pitch_shift_slider = gr.Slider(label="Pitch Shift (Semitones)", minimum=-12, maximum=12, value=0, step=0.1, interactive=True, info="-12 = one octave down, +12 = one octave up")
                         speed_slider = gr.Slider(label="Speed (Factor)", minimum=0.5, maximum=2.0, value=1.0, step=0.01, interactive=True, info="0.5 = half speed, 2.0 = double speed")
                     with gr.Accordion("Noise Reduction (Applied AFTER Pitch/Speed)", open=False): apply_noise_reduction_checkbox = gr.Checkbox( label="Enable", value=False, interactive=NOISEREDUCE_AVAILABLE); noise_reduction_strength_slider = gr.Slider( label="Strength (0=off, 1=max)", minimum=0.0, maximum=1.0, value=0.85, step=0.05, interactive=NOISEREDUCE_AVAILABLE);
                     if not NOISEREDUCE_AVAILABLE: gr.Markdown("<small>*(Requires noisereduce, scipy, numpy)*</small>")
                     with gr.Accordion("Equalization (EQ) (Applied AFTER NR)", open=False): eq_low_gain_input = gr.Slider(label="Low Gain (Shelf)", minimum=-12, maximum=12, value=0, step=0.5, interactive=SCIPY_AVAILABLE); eq_mid_gain_input = gr.Slider(label="Mid Gain (Peak)", minimum=-12, maximum=12, value=0, step=0.5, interactive=SCIPY_AVAILABLE); eq_high_gain_input = gr.Slider(label="High Gain (Shelf)", minimum=-12, maximum=12, value=0, step=0.5, interactive=SCIPY_AVAILABLE);
                     if not SCIPY_AVAILABLE: gr.Markdown("<small>*(Requires scipy & numpy)*</small>")
                     with gr.Accordion("Compression (Applied AFTER EQ)", open=False): apply_compression_checkbox = gr.Checkbox( label="Enable", value=False, interactive=PYDUB_COMPRESS_AVAILABLE); compress_threshold_input = gr.Slider(label="Threshold (dBFS)", minimum=-60, maximum=0, value=-20, step=1, interactive=PYDUB_COMPRESS_AVAILABLE); compress_ratio_input = gr.Slider(label="Ratio (N:1)", minimum=1.0, maximum=20.0, value=4.0, step=0.1, interactive=PYDUB_COMPRESS_AVAILABLE); compress_attack_input = gr.Slider(label="Attack (ms)", minimum=1, maximum=200, value=5, step=1, interactive=PYDUB_COMPRESS_AVAILABLE); compress_release_input = gr.Slider(label="Release (ms)", minimum=20, maximum=1000, value=100, step=10, interactive=PYDUB_COMPRESS_AVAILABLE);
                     if not PYDUB_COMPRESS_AVAILABLE: gr.Markdown("<small>*(Requires pydub effects component)*</small>")
                     with gr.Accordion("Reverb (Applied AFTER Compression)", open=False):
                         reverb_amount_slider = gr.Slider(label="Reverb Amount", minimum=0.0, maximum=1.0, value=0.0, step=0.05, interactive=PYDUB_AVAILABLE, info="0 = no reverb, 1 = max reverb")
                     with gr.Accordion("Final Peak Normalization (Applied LAST)", open=False): apply_peak_norm_checkbox = gr.Checkbox( label="Enable", value=True, interactive=PYDUB_AVAILABLE); peak_norm_target_input = gr.Number( label="Target Peak (dBFS)", value=DEFAULT_FINAL_NORM_TARGET_DBFS, minimum=-12.0, maximum=-0.1, step=0.1, interactive=PYDUB_AVAILABLE);
                     if not PYDUB_AVAILABLE: gr.Markdown("<small>*(Requires pydub)*</small>")
                 with gr.Column(scale=1): concatenate_convo_button = gr.Button("Concatenate & Process Selected Lines", variant="primary", interactive=False); concat_status_output = gr.Textbox(label="Concatenation & Processing Status", lines=15, interactive=False, max_lines=30); final_conversation_audio = gr.Audio(label="Final Output Audio", type="filepath", interactive=False)
        if not PYDUB_AVAILABLE:
             with gr.TabItem("Concatenate & Export (Disabled)"): gr.Markdown("### Feature Disabled\nRequires `pydub` library (`pip install pydub`) and `ffmpeg`.")
        with gr.TabItem("4. Advanced Audio", id="tab_advanced_audio"):
            gr.Markdown("## Advanced Audio Effects (Preview Only)")
            test_audio_upload = gr.Audio(label="Upload Test Audio (optional)", type="filepath", interactive=True)
            with gr.Accordion("Pitch Correction (Auto-Tune) [Not Implemented]", open=False):
                enable_pitch_correction = gr.Checkbox(label="Enable Pitch Correction", value=False, interactive=False)
                pitch_correction_strength = gr.Slider(label="Correction Strength", minimum=0.0, maximum=1.0, value=1.0, step=0.05, interactive=False)
                pitch_correction_mode = gr.Dropdown(label="Snap Mode", choices=["chromatic"], value="chromatic", interactive=False)
            with gr.Accordion("Chorus Effect", open=False):
                enable_chorus = gr.Checkbox(label="Enable Chorus", value=False)
                chorus_depth = gr.Slider(label="Depth (ms)", minimum=1.0, maximum=30.0, value=15.0, step=0.1)
                chorus_rate = gr.Slider(label="Rate (Hz)", minimum=0.1, maximum=5.0, value=1.5, step=0.01)
                chorus_mix = gr.Slider(label="Mix", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
            with gr.Accordion("Flanger Effect", open=False):
                enable_flanger = gr.Checkbox(label="Enable Flanger", value=False)
                flanger_depth = gr.Slider(label="Depth (ms)", minimum=0.1, maximum=10.0, value=3.0, step=0.01)
                flanger_rate = gr.Slider(label="Rate (Hz)", minimum=0.05, maximum=2.0, value=0.5, step=0.01)
                flanger_feedback = gr.Slider(label="Feedback", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
                flanger_mix = gr.Slider(label="Mix", minimum=0.0, maximum=1.0, value=0.5, step=0.01)
            with gr.Accordion("Noise Gate [Not Implemented]", open=False):
                enable_noise_gate = gr.Checkbox(label="Enable Noise Gate", value=False, interactive=False)
                noise_gate_threshold = gr.Slider(label="Threshold (dBFS)", minimum=-80.0, maximum=0.0, value=-40.0, step=0.5, interactive=False)
                noise_gate_attack = gr.Slider(label="Attack (ms)", minimum=1.0, maximum=100.0, value=10.0, step=1.0, interactive=False)
                noise_gate_release = gr.Slider(label="Release (ms)", minimum=10.0, maximum=500.0, value=100.0, step=1.0, interactive=False)
            with gr.Accordion("Gain Adjustment (Volume)", open=True):
                 gain_slider_advanced = gr.Slider(
                     label="Gain (dB)", minimum=-24.0, maximum=6.0, step=0.5, value=0.0, interactive=True
                 )
            with gr.Accordion("10-Band Graphical Equalizer", open=False):
                enable_graphical_eq = gr.Checkbox(label="Enable 10-Band EQ", value=False)
                eq_band_labels = ["31Hz", "62Hz", "125Hz", "250Hz", "500Hz", "1kHz", "2kHz", "4kHz", "8kHz", "16kHz"]
                eq_band_sliders = []
                for i, label in enumerate(eq_band_labels):
                    slider = gr.Slider(label=f"{label}", minimum=-12.0, maximum=12.0, value=0.0, step=0.5)
                    eq_band_sliders.append(slider)
            gr.Markdown("You can preview the effect of these settings on a test audio sample before applying them to your final export.")
            preview_button = gr.Button("Preview Advanced Effects on Test Audio")
            preview_audio = gr.Audio(label="Preview Output", type="filepath", interactive=False)
            import shutil # Correctly placed within Advanced Audio tab
            def process_advanced_effects_preview( # Correctly placed
                test_audio_path,
                enable_pitch_correction, pitch_correction_strength, pitch_correction_mode,
                enable_chorus, chorus_depth, chorus_rate, chorus_mix,
                enable_flanger, flanger_depth, flanger_rate, flanger_feedback, flanger_mix,
                enable_noise_gate, noise_gate_threshold, noise_gate_attack, noise_gate_release,
                gain_db_advanced, # Added gain input
                enable_graphical_eq, *eq_band_gains
            ):
                try: from audio_utils import (apply_pitch_correction, apply_chorus, apply_flanger, apply_noise_gate, apply_graphical_eq)
                except ImportError:
                    def apply_pitch_correction(seg, *a, **k): print("WARN: audio_utils not loaded, pitch correction skipped"); return seg
                    def apply_chorus(seg, *a, **k): print("WARN: audio_utils not loaded, chorus skipped"); return seg
                    def apply_flanger(seg, *a, **k): print("WARN: audio_utils not loaded, flanger skipped"); return seg
                    def apply_noise_gate(seg, *a, **k): print("WARN: audio_utils not loaded, noise gate skipped"); return seg
                    def apply_graphical_eq(seg, *a, **k): print("WARN: audio_utils not loaded, graphical EQ skipped"); return seg
                if test_audio_path and os.path.isfile(test_audio_path): audio_path = test_audio_path
                else:
                    default_sample = os.path.join(tempfile.gettempdir(), "indextts_default_sample.wav")
                    if not os.path.isfile(default_sample):
                        try: from pydub import AudioSegment; AudioSegment.silent(duration=1000).export(default_sample, format="wav")
                        except Exception: return None
                    audio_path = default_sample
                try: from pydub import AudioSegment; segment = AudioSegment.from_file(audio_path)
                except Exception as e: print(f"Error loading audio for preview: {e}"); return None
                if enable_pitch_correction: segment = apply_pitch_correction(segment, strength=pitch_correction_strength, snap_mode=pitch_correction_mode)
                if enable_chorus: segment = apply_chorus(segment, depth_ms=chorus_depth, rate_hz=chorus_rate, mix=chorus_mix)
                if enable_flanger: segment = apply_flanger(segment, depth_ms=flanger_depth, rate_hz=flanger_rate, feedback=flanger_feedback, mix=flanger_mix)
                if enable_noise_gate: segment = apply_noise_gate(segment, threshold_db=noise_gate_threshold, attack_ms=noise_gate_attack, release_ms=noise_gate_release)
                if enable_graphical_eq: segment = apply_graphical_eq(segment, list(eq_band_gains))
                if gain_db_advanced != 0.0: # Apply Gain
                    print(f"  Applying Advanced Gain: {gain_db_advanced:+.1f} dB")
                    segment = segment + gain_db_advanced
                out_path = os.path.join(tempfile.gettempdir(), f"indextts_adv_preview_{uuid.uuid4().hex[:8]}.wav")
                try: segment.export(out_path, format="wav")
                except Exception as e: print(f"Error exporting preview: {e}"); return None
                return out_path
            preview_button.click(
                fn=process_advanced_effects_preview,
                inputs=[
                    test_audio_upload,
                    enable_pitch_correction, pitch_correction_strength, pitch_correction_mode,
                    enable_chorus, chorus_depth, chorus_rate, chorus_mix,
                    enable_flanger, flanger_depth, flanger_rate, flanger_feedback, flanger_mix,
                    enable_noise_gate, noise_gate_threshold, noise_gate_attack, noise_gate_release,
                    gain_slider_advanced, # Added gain slider input
                    enable_graphical_eq, *eq_band_sliders
                ], outputs=preview_audio
            )
        # End of "4. Advanced Audio" TabItem context

        # --- Timeline / Final Edit Tab --- (Correctly Added from ui_layout.py)
        (timeline_tab_component, timeline_line_selector_dd_component,
         timeline_original_speaker_text_component, timeline_original_text_display_component,
         timeline_editable_text_input_component, timeline_selected_audio_player_component,
         timeline_selected_audio_seed_text_component, timeline_regenerate_button_component,
         timeline_status_text_component) = create_timeline_tab()

    # --- Event Handlers ---
    list_speakers_btn.click(lambda: "\n".join(list_speaker_files()[1]), inputs=None, outputs=available_speakers_display)
    display_line_inputs = [ current_line_index_state, parsed_script_state, all_options_state, selections_state, edited_texts_state, app_context ] # Added app_context
    review_display_outputs_base = [ review_status_output, line_nav_display, current_line_display_review, editable_line_text_review, current_line_index_state, prev_line_button, next_line_button, *review_audio_outputs, line_choice_radio ]
    review_display_outputs_nav_full = review_display_outputs_base + [proceed_to_concat_button, concatenate_convo_button]
    generate_convo_inputs = [
        script_input_convo, num_versions_convo_radio, temperature_slider, top_p_slider, top_k_slider,
        seed_strategy_dd, fixed_base_seed_input, app_context # Added app_context
    ]
    convo_gen_outputs = [ convo_gen_status_output, parsed_script_state, all_options_state, edited_texts_state, current_line_index_state, generate_convo_button, review_tab, concat_tab ]
    generate_convo_button.click( fn=prepare_temp_dir, inputs=gr.State(TEMP_CONVO_MULTI_DIR), outputs=None, queue=False ).then( fn=parse_validate_and_start_convo, inputs=generate_convo_inputs, outputs=convo_gen_outputs, show_progress="full" ).then( fn=display_line_for_review, inputs=display_line_inputs, outputs=review_display_outputs_base ).then( fn=enable_concatenation_buttons, inputs=[parsed_script_state, all_options_state, selections_state], outputs=[proceed_to_concat_button, concatenate_convo_button] )
    def navigate_and_display( direction: int, current_index: int, parsed_script: ParsedScript, all_options_state: AllOptionsState, selections_state: SelectionsState, edited_texts: EditedTextsState, app_context_param: AppContext ): # Added app_context_param
        new_index = current_index + direction
        review_yield_tuple = display_line_for_review(new_index, parsed_script, all_options_state, selections_state, edited_texts, app_context_param) # Pass app_context_param
        can_proceed_update, can_concat_btn_update = enable_concatenation_buttons(parsed_script, all_options_state, selections_state)
        return review_yield_tuple + (can_proceed_update, can_concat_btn_update)
    prev_line_button.click( fn=navigate_and_display, inputs=[gr.State(-1), current_line_index_state, parsed_script_state, all_options_state, selections_state, edited_texts_state, app_context], outputs=review_display_outputs_nav_full, queue=False ) # Added app_context
    next_line_button.click( fn=navigate_and_display, inputs=[gr.State(1), current_line_index_state, parsed_script_state, all_options_state, selections_state, edited_texts_state, app_context], outputs=review_display_outputs_nav_full, queue=False ) # Added app_context
    line_choice_radio_update_convo_inputs = [ # Defined new input list for update_convo_selection
        line_choice_radio, current_line_index_state, selections_state,
        parsed_script_state, all_options_state, selected_seed_data_state, app_context # Added app_context
    ]
    line_choice_radio.change(
        fn=update_convo_selection,
        inputs=line_choice_radio_update_convo_inputs, # Used new input list
        outputs=[ selections_state, selected_seed_data_state, review_status_output ],
        queue=False
    ).then(
        fn=save_seed_data,
        inputs=[selected_seed_data_state],
        outputs=[review_status_output]
    ).then(
        fn=enable_concatenation_buttons,
        inputs=[parsed_script_state, all_options_state, selections_state],
        outputs=[proceed_to_concat_button, concatenate_convo_button]
    ).then(
        fn=update_timeline_with_selection,
        inputs=[
            current_line_index_state,
            parsed_script_state,
            selections_state,
            all_options_state,
            selected_seed_data_state,
            edited_texts_state,
            app_context # Added app_context
        ],
        outputs=[
            timeline_line_selector_dd_component,
            timeline_original_speaker_text_component, 
            timeline_original_text_display_component, 
            timeline_editable_text_input_component,   
            timeline_selected_audio_player_component, 
            timeline_selected_audio_seed_text_component, 
            timeline_status_text_component            
        ],
        queue=False
    )
    regenerate_inputs = [
        current_line_index_state, parsed_script_state, all_options_state, selections_state, edited_texts_state,
        editable_line_text_review, num_versions_convo_radio,
        temperature_slider, top_p_slider, top_k_slider,
        seed_strategy_dd, fixed_base_seed_input, app_context # Added app_context
    ]
    regenerate_outputs = [ review_status_output, all_options_state, selections_state, edited_texts_state ]
    regenerate_current_line_button.click( fn=lambda: gr.update(interactive=False), inputs=None, outputs=regenerate_current_line_button, queue=False ).then( fn=regenerate_single_line, inputs=regenerate_inputs, outputs=regenerate_outputs, show_progress="full" ).then( fn=display_line_for_review, inputs=display_line_inputs, outputs=review_display_outputs_base ).then( fn=enable_concatenation_buttons, inputs=[ parsed_script_state, all_options_state, selections_state ], outputs=[proceed_to_concat_button, concatenate_convo_button] ).then( fn=lambda: gr.update(interactive=True), inputs=None, outputs=regenerate_current_line_button, queue=False )
    threshold_regen_inputs = [
        current_line_index_state,
        manual_regen_threshold_slider,
        editable_line_text_review,
        parsed_script_state,
        all_options_state,
        selections_state,
        edited_texts_state,
        temperature_slider, top_p_slider, top_k_slider,
        manual_regen_attempts_dd, app_context # Added app_context
    ]
    threshold_regen_outputs = [ review_status_output, all_options_state, selections_state, edited_texts_state ]
    if 'threshold_regen_button' in locals():
        threshold_regen_button.click( fn=lambda: gr.update(interactive=False), inputs=None, outputs=threshold_regen_button, queue=False ).then( fn=regenerate_below_threshold, inputs=threshold_regen_inputs, outputs=threshold_regen_outputs, show_progress="full" ).then( fn=display_line_for_review, inputs=display_line_inputs, outputs=review_display_outputs_base ).then( fn=enable_concatenation_buttons, inputs=[ parsed_script_state, all_options_state, selections_state ], outputs=[proceed_to_concat_button, concatenate_convo_button] ).then( fn=lambda: gr.update(interactive=True), inputs=None, outputs=threshold_regen_button, queue=False )
    # Seed strategy dropdown visibility handler
    def update_fixed_seed_visibility(strategy):
        needs_fixed_seed = (strategy == SEED_STRATEGY_FIXED_BASE_SEQUENTIAL or \
                            strategy == SEED_STRATEGY_FIXED_BASE_REUSED_LIST)
        return gr.update(visible=needs_fixed_seed)
    if 'seed_strategy_dd' in locals() and 'fixed_base_seed_input' in locals():
        seed_strategy_dd.change(
            fn=update_fixed_seed_visibility,
            inputs=[seed_strategy_dd],
            outputs=[fixed_base_seed_input],
            queue=False
        )
    proceed_to_concat_button.click( fn=lambda: (gr.update(interactive=True), gr.update(selected="tab_concat")), inputs=None, outputs=[concat_tab, tabs], queue=False )
    concat_inputs = [
        parsed_script_state, all_options_state, selections_state,
        output_format_dropdown, mp3_bitrate_dropdown,
        normalize_segments_checkbox,
        trim_silence_checkbox, trim_threshold_input, trim_length_input,
        apply_compression_checkbox, compress_threshold_input, compress_ratio_input, compress_attack_input, compress_release_input,
        apply_noise_reduction_checkbox, noise_reduction_strength_slider,
        eq_low_gain_input, eq_mid_gain_input, eq_high_gain_input,
        apply_peak_norm_checkbox, peak_norm_target_input,
        reverb_amount_slider,
        pitch_shift_slider,
        speed_slider,
        app_context # Added app_context
    ]
    if PYDUB_AVAILABLE:
        concatenate_convo_button.click( fn=lambda: gr.update(interactive=False), inputs=None, outputs=concatenate_convo_button, queue=False ).then( fn=concatenate_conversation_versions, inputs=concat_inputs, outputs=[concat_status_output, final_conversation_audio], show_progress="full" ).then( fn=lambda: gr.update(interactive=True), inputs=None, outputs=concatenate_convo_button, queue=False )
        output_format_dropdown.change( fn=lambda fmt: gr.update(visible=(fmt.lower() == 'mp3') if isinstance(fmt, str) else False), inputs=output_format_dropdown, outputs=mp3_bitrate_dropdown, queue=False )

# --- Launch the App ---
if __name__ == "__main__":
    if tts is None:
        print("\n" + "="*50);
        print("❌ ERROR: TTS model failed to initialize. Gradio UI cannot launch.");
        print("   Please check the console output above for specific errors during TTS setup.");
        print("   Common issues include missing model files or incorrect paths in config.yaml.");
        print("="*50 + "\n")
    elif SPEECHBRAIN_AVAILABLE and speaker_similarity_model is None:
         print("\n" + "="*50);
         print("⚠️ WARNING: TTS model initialized, but SpeechBrain Speaker Similarity model failed to load.");
         print("   The UI will launch, but speaker similarity scoring and auto-regeneration will be disabled.");
         print("   Check console logs for errors related to loading 'speechbrain/spkrec-ecapa-voxceleb'.")
         print("="*50 + "\n")
         print("Launching Gradio UI (Speaker Similarity Disabled)...");
         SAVE_DIR.mkdir(parents=True, exist_ok=True);
         demo.launch(share=False, inbrowser=True)
    else:
        print("TTS and Speaker Similarity Model (if applicable) Initialized.")
        print("Launching Gradio UI...");
        SAVE_DIR.mkdir(parents=True, exist_ok=True);
        demo.launch(share=False, inbrowser=True)