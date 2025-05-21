# ui_logic.py
import gradio as gr
import os
import tempfile
import uuid
import re
import time
import random
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Iterator, Union, cast
import traceback
import datetime
import math
import torch

# --- Import Constants and Types ---
# Assuming constants.py is in the same directory
from constants import *
from app_context import AppContext # Import the context class

# --- Attempt to import libraries (needed for function bodies) ---
try:
    from pydub import AudioSegment
    from pydub import silence as _silence_Imported
    from pydub.effects import compress_dynamic_range as _compress_dynamic_range_Imported
    # Define constants based on imports if needed elsewhere in this file
    DEFAULT_SILENCE_MS = 300 # Example, adjust if needed
    DEFAULT_CROSSFADE_MS = 10 # Example, adjust if needed
except ImportError:
    AudioSegment = None
    _silence_Imported = None
    _compress_dynamic_range_Imported = None
    DEFAULT_SILENCE_MS = 300
    DEFAULT_CROSSFADE_MS = 10


try:
    import numpy as np
    import scipy.signal
    import noisereduce as nr
except ImportError:
    np = None
    scipy = None
    nr = None

try:
    from indextts.infer import IndexTTS
except ImportError:
    IndexTTS = None # type: ignore

try:
    from audio_utils import (
        apply_eq, apply_noise_reduction, apply_reverb,
        change_pitch, change_speed, analyze_speaker_similarity,
        SPEECHBRAIN_AVAILABLE # Keep this specific import for now if audio_utils uses it internally
    )
except ImportError:
    def apply_eq(*args, **kwargs): raise NotImplementedError("audio_utils import failed")
    def apply_noise_reduction(*args, **kwargs): raise NotImplementedError("audio_utils import failed")
    def apply_reverb(*args, **kwargs): raise NotImplementedError("audio_utils import failed")
    def change_pitch(*args, **kwargs): raise NotImplementedError("audio_utils import failed")
    def change_speed(*args, **kwargs): raise NotImplementedError("audio_utils import failed")
    def analyze_speaker_similarity(*args, **kwargs): print("Dummy analyze_speaker_similarity call"); return -1.0
    # SPEECHBRAIN_AVAILABLE is now managed via AppContext

try:
    from general_utils import (
        list_speaker_files, prepare_temp_dir, split_text_simple,
        check_all_selections_valid, enable_concatenation_buttons,
        list_save_files
    )
except ImportError:
    # Define dummy functions if general_utils fails to import
    def list_speaker_files(*args, **kwargs): return ['[No Speaker Selected]'], []
    def prepare_temp_dir(*args, **kwargs): print("ERROR: prepare_temp_dir not loaded from utils!"); return False
    def split_text_simple(*args, **kwargs): return []
    def check_all_selections_valid(*args, **kwargs): return False
    def enable_concatenation_buttons(*args, **kwargs): return gr.update(interactive=False), gr.update(interactive=False)
    def list_save_files(*args, **kwargs): return []

# --- Removed Placeholder Global Variables ---
# These are now passed via the AppContext object


# --- Save/Load Functions ---
def save_project_state(
    filename: str,
    parsed_script: ParsedScript,
    all_options: AllOptionsState,
    selections: SelectionsState,
    edited_texts: EditedTextsState,
    selected_seeds: dict,
) -> Tuple[str, List[str], Optional[str]]:
    """Saves the current project state to a JSON file."""
    status = ""
    original_filename_input = filename
    if not filename or not isinstance(filename, str):
        status = "Error: Invalid filename provided."
        print(status)
        return status, list_save_files(), None

    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
    if not filename.lower().endswith(".json"):
        filename += ".json"

    save_path = SAVE_DIR / filename
    print(f"Attempting to save project state to: {save_path}")

    try:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        status = f"Error: Could not create save directory '{SAVE_DIR}': {e}"
        print(status)
        traceback.print_exc()
        return status, list_save_files(), None

    project_state = {
        "parsed_script": parsed_script,
        "all_options": all_options,
        "selections": selections,
        "edited_texts": edited_texts,
        "selected_seeds": selected_seeds,
        "save_timestamp": datetime.datetime.now().isoformat()
    }

    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(project_state, f, indent=4, ensure_ascii=False)
        status = f"✅ Project state successfully saved to '{filename}'."
        print(status)
    except TypeError as e:
        status = f"Error: Failed to serialize project state to JSON. Check data types. ({e})"
        print(status)
        traceback.print_exc()
    except OSError as e:
        status = f"Error: Failed to write save file '{save_path}': {e}"
        print(status)
        traceback.print_exc()
    except Exception as e:
        status = f"Error: An unexpected error occurred during saving: {e}"
        print(status)
        traceback.print_exc()
        return status, list_save_files(), None

    updated_save_files = list_save_files()
    return status, updated_save_files, filename

def load_project_state(
    filename: str,
    app_context: AppContext # Add context for checking PYDUB_AVAILABLE
) -> Tuple[str, ParsedScript, AllOptionsState, SelectionsState, EditedTextsState, dict, int, dict, dict, dict]:
    """Loads project state from a JSON file with improved error handling."""
    status = ""
    # Default return values in case of error - ensure they match expected types
    empty_state = ([], [], {}, {}, {}, 0) # script, options, selections, edited, seeds, index
    ui_updates = (gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)) # generate_btn, review_tab, concat_tab

    if not filename or not isinstance(filename, str):
        status = "❌ Error: No filename selected for loading."
        print(status)
        return status, *empty_state, *ui_updates

    load_path = SAVE_DIR / filename
    print(f"Attempting to load project state from: {load_path}")

    # 1. Check if file exists
    if not load_path.is_file():
        status = f"❌ Error: Save file not found: '{filename}'"
        print(status)
        return status, *empty_state, *ui_updates

    # 2. Read and Parse JSON
    try:
        with open(load_path, 'r', encoding='utf-8') as f:
            project_state = json.load(f)
    except json.JSONDecodeError as e:
        status = f"❌ Error: Failed to parse JSON file '{filename}'. File might be corrupted. ({e})"
        print(status)
        traceback.print_exc()
        return status, *empty_state, *ui_updates
    except OSError as e:
        status = f"❌ Error: Could not read save file '{load_path}': {e}"
        print(status)
        traceback.print_exc()
        return status, *empty_state, *ui_updates
    except Exception as e:
        status = f"❌ Error: An unexpected error occurred reading file '{filename}': {e}"
        print(status)
        traceback.print_exc()
        return status, *empty_state, *ui_updates

    # 3. Validate loaded data structure and types
    try:
        required_keys = ["parsed_script", "all_options", "selections", "edited_texts", "selected_seeds"]
        if not all(key in project_state for key in required_keys):
            missing_keys = [key for key in required_keys if key not in project_state]
            status = f"❌ Error: Save file '{filename}' is missing required data fields: {', '.join(missing_keys)}."
            print(status)
            return status, *empty_state, *ui_updates

        # Extract state variables with type checks and conversions
        loaded_parsed_script = project_state["parsed_script"]
        loaded_all_options = project_state["all_options"]
        # Convert keys back to int, handle potential errors
        loaded_selections = {int(k): v for k, v in project_state["selections"].items()}
        loaded_edited_texts = {int(k): v for k, v in project_state["edited_texts"].items()}
        loaded_selected_seeds = project_state["selected_seeds"]

        # Basic validation of loaded types
        if not isinstance(loaded_parsed_script, list) or \
           not isinstance(loaded_all_options, list) or \
           not isinstance(loaded_selections, dict) or \
           not isinstance(loaded_edited_texts, dict) or \
           not isinstance(loaded_selected_seeds, dict):
            status = f"❌ Error: Data types in save file '{filename}' are incorrect."
            print(status)
            return status, *empty_state, *ui_updates

        # Optional: More detailed validation (e.g., check list item types) could go here

    except (ValueError, TypeError, KeyError, Exception) as e:
        status = f"❌ Error: Invalid data structure or type in save file '{filename}'. ({type(e).__name__}: {e})"
        print(status)
        traceback.print_exc()
        return status, *empty_state, *ui_updates


    # 4. Check if loaded state allows enabling review/concat tabs
    can_review = bool(loaded_parsed_script)
    # Use the imported check_all_selections_valid function
    can_concat = check_all_selections_valid(loaded_parsed_script, loaded_all_options, loaded_selections)

    status = f"✅ Project state successfully loaded from '{filename}'."
    print(status)

    # 5. Return loaded state and UI updates
    loaded_state_tuple = (
        loaded_parsed_script,
        loaded_all_options,
        loaded_selections,
        loaded_edited_texts,
        loaded_selected_seeds,
        0 # current_line_index_state reset to 0
    )
    final_ui_updates = (
        gr.update(interactive=True), # generate_convo_button
        gr.update(interactive=can_review), # review_tab
        gr.update(interactive=(can_concat and app_context.pydub_available)) # concat_tab
    )

    return status, *loaded_state_tuple, *final_ui_updates


# --- Gradio UI Functions ---
def infer_single(
    selected_speaker_filename: str,
    text: str,
    temperature_val: float,
    top_p_val: float,
    top_k_val: int,
    app_context: AppContext # Add context
):
    """Generates a single audio segment using the TTS model from the context."""
    if app_context.tts is None: return "Error: TTS model failed to initialize."
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
        app_context.tts.infer( speaker_path, text, output_path, temperature=temperature_val, top_p=top_p_val, top_k=int(top_k_val), seed=-1 )
        print(f"Single segment saved: {output_path}")
        return output_path
    except Exception as e:
        err_msg = f"Error during single TTS inference ({type(e).__name__}: {e})"; print(f"{err_msg}"); traceback.print_exc(); return err_msg

def gen_single(
    selected_speaker_filename: str,
    text: str,
    temperature_val: float,
    top_p_val: float,
    top_k_val: int,
    app_context: AppContext # Add context
):
     """Wrapper for infer_single, providing the context."""
     # No need to check tts here, infer_single does it
     output_path = infer_single(selected_speaker_filename, text, temperature_val, top_p_val, top_k_val, app_context)
     if isinstance(output_path, str) and output_path.startswith("Error:"): return None
     else: return output_path


# --- Helper Functions for parse_validate_and_start_convo ---

def _validate_convo_gen_inputs(
    script_text: str,
    num_versions_str: str,
    app_context: AppContext # Add context
) -> Tuple[Optional[str], Optional[int]]:
    """Validate basic inputs for conversation generation."""
    if app_context.tts is None:
        return "Error: TTS model not initialized.", None
    try:
        num_versions = int(num_versions_str)
        if not (1 <= num_versions <= MAX_VERSIONS_ALLOWED):
            raise ValueError(f"Versions must be 1-{MAX_VERSIONS_ALLOWED}.")
    except (ValueError, TypeError):
        return f"Error: Invalid number of versions '{num_versions_str}'.", None
    if not script_text or not script_text.strip():
        return "Error: Input script text cannot be empty.", None
    # Removed prepare_temp_dir call from here
    return None, num_versions

def _parse_script_and_validate_speakers(
    script_text: str
) -> Tuple[Optional[str], Optional[ParsedScript]]:
    """Parse script lines and validate speaker files exist."""
    lines = script_text.strip().split('\n')
    parsed_script: ParsedScript = []
    available_speaker_files = list_speaker_files()[1]
    validation_errors = []
    for i, line in enumerate(lines):
        line_num = i + 1
        line = line.strip()
        if not line: continue
        match = re.match(r'^([^:]+):\s*(.*)$', line)
        if match:
            speaker_filename = match.group(1).strip()
            line_text = match.group(2).strip()
            if not line_text:
                print(f"Warning: Line {line_num} skipped (no text).")
                continue
            if speaker_filename not in available_speaker_files:
                err = f"Error: Speaker file '{speaker_filename}' on line {line_num} not found in ./speakers/."
                validation_errors.append(err)
                continue # Continue checking other lines
            parsed_script.append({'speaker_filename': speaker_filename, 'text': line_text})
        else:
            err = f"Error: Invalid format on line {line_num}. Expected 'SpeakerFile.ext: Text'."
            validation_errors.append(err)
            continue # Continue checking other lines

    if validation_errors:
        return "\n".join(validation_errors) + "\nCannot proceed.", None
    if not parsed_script:
        return "Error: No valid lines found in the script after parsing.", None
    return None, parsed_script


def _determine_line_seeds(
    seed_strategy: str,
    fixed_base_seed_input: int,
    num_versions: int,
    line_index: int, # 0-based index of the current line
    total_lines: int, # Total number of lines in the script
    base_seed: int # Pass base seed in
) -> Tuple[List[int], str]: # Return only seeds and log info
    """Determine the seeds to use for a specific line based on the strategy."""
    # Base seed is now determined outside and passed in
    seeds_generated_this_line = False
    seed_log_info = ""

    # Determine version seeds for the *current* line
    line_version_seeds = [0] * num_versions
    if seed_strategy == SEED_STRATEGY_FULLY_RANDOM:
        line_version_seeds = random.sample(range(2**32), num_versions)
        seed_log_info = "Fully Random Seeds: " + ', '.join([f'R({s})' for s in line_version_seeds])
        seeds_generated_this_line = True
    elif seed_strategy == SEED_STRATEGY_FIXED_BASE_SEQUENTIAL or seed_strategy == SEED_STRATEGY_RANDOM_BASE_SEQUENTIAL:
        line_offset = line_index * num_versions * 10 # Simple offset based on line index
        line_version_seeds = [(base_seed + line_offset + j) % (2**32) for j in range(num_versions)]
        seed_log_info = f"{'Random' if seed_strategy == SEED_STRATEGY_RANDOM_BASE_SEQUENTIAL else 'Fixed'} Base + Seq Offset Seeds: " + ', '.join([f'S({s})' for s in line_version_seeds])
        seeds_generated_this_line = True
    elif seed_strategy == SEED_STRATEGY_RANDOM_BASE_REUSED_LIST:
        # This case requires the list to be passed in or handled differently.
        # For now, let's assume it's handled by the caller using a pre-generated list.
        # This function shouldn't generate the list itself if it's meant to be reused.
        # We'll modify the main function to handle this.
        # Placeholder: Return empty seeds, caller must provide.
        seed_log_info = "(Using externally provided reused random list)"
        line_version_seeds = [] # Indicate caller needs to provide
    elif seed_strategy == SEED_STRATEGY_FIXED_BASE_REUSED_LIST:
        line_version_seeds = [(base_seed + j) % (2**32) for j in range(num_versions)]
        seed_log_info = "Fixed Base + Reused Seq List Seeds: " + ', '.join([f'S({s})' for s in line_version_seeds])
        # This list is the same for all lines, generated here based on base_seed.

    return line_version_seeds, seed_log_info


def _generate_versions_for_line(
    line_idx_0_based: int,
    line_info: Dict[str, str],
    num_versions: int,
    version_seeds: List[int],
    temperature_val: float, top_p_val: float, top_k_val: int,
    status_log: List[str],
    progress, # Gradio progress object
    app_context: AppContext # Add context
) -> Tuple[List[Optional[str]], bool]:
    """Generate audio files for all versions of a single line."""
    # Removed global tts access
    line_options_generated: List[Optional[str]] = [None] * num_versions
    generation_successful = True
    speaker_filename = line_info['speaker_filename']
    line_text = line_info['text']
    speaker_path_str = str(SPEAKER_DIR / speaker_filename)

    if len(version_seeds) != num_versions:
         status_log.append(f"❌ Internal Error: Seed list length mismatch for Line {line_idx_0_based + 1}.")
         return [None] * num_versions, False # Return failure

    for j in range(num_versions):
        version_idx_1_based = j + 1
        current_seed = version_seeds[j]
        print(f"  Generating V{version_idx_1_based} (Seed: {current_seed})... Text: '{line_text[:30]}...'")

        segment_filename = f"line{line_idx_0_based:03d}_spk-{Path(speaker_filename).stem}_v{j+1:02d}_s{current_seed}.wav"
        output_path = str(TEMP_CONVO_MULTI_DIR / segment_filename)

        try:
            # Optional: Delete existing file if overwriting
            # if Path(output_path).exists(): Path(output_path).unlink()

            app_context.tts.infer(speaker_path_str, line_text, output_path,
                                  temperature=temperature_val, top_p=top_p_val, top_k=int(top_k_val), seed=current_seed)

            if Path(output_path).is_file() and Path(output_path).stat().st_size > 0:
                line_options_generated[j] = output_path
                print(f"    V{version_idx_1_based} OK: {segment_filename}")
            else:
                print(f"    V{version_idx_1_based} FAILED (file not found or empty).")
                status_log.append(f"  Warning: Line {line_idx_0_based + 1}, V{version_idx_1_based} initial generation failed (file missing/empty).")
        except Exception as e:
            err_msg = f"Error L{line_idx_0_based + 1} V{version_idx_1_based}: Failed during initial TTS inference: {type(e).__name__}: {e}"
            print(f"    {err_msg}")
            status_log.append(f"\n❌ {err_msg}")
            traceback.print_exc()
            generation_successful = False
            break # Stop generating versions for this line on critical error

    return line_options_generated, generation_successful


def _check_similarity_and_autoregen_line(
    line_idx_0_based: int,
    line_info: Dict[str, str],
    initial_paths: List[Optional[str]],
    num_versions: int,
    temperature_val: float, top_p_val: float, top_k_val: int,
    status_log: List[str],
    progress, # Gradio progress object
    app_context: AppContext # Add context
) -> List[Optional[str]]:
    """Check similarity and perform auto-regeneration for a single line's versions."""
    # Removed global access
    speaker_filename = line_info['speaker_filename']
    line_text = line_info['text']
    speaker_path_str = str(SPEAKER_DIR / speaker_filename)
    ref_speaker_path_exists = Path(speaker_path_str).is_file()

    final_line_options = list(initial_paths) # Start with the initial paths

    if not (app_context.speechbrain_available and app_context.speaker_similarity_model is not None and ref_speaker_path_exists):
        print(f"  Skipping similarity check/auto-regen for Line {line_idx_0_based + 1} (unavailable/ref missing).")
        return final_line_options # Return initial paths if cannot check

    status_log.append(f"  Checking speaker similarity for Line {line_idx_0_based + 1} (Auto-Regen Threshold: {SIMILARITY_THRESHOLD:.2f})...")
    # yield status update here if needed from the main function
    print(f"  Checking speaker similarity for Line {line_idx_0_based + 1} (Threshold: {SIMILARITY_THRESHOLD:.2f})...")

    for j, initial_path in enumerate(initial_paths):
        # Correct indentation for the 'if' block below
        if initial_path and Path(initial_path).is_file():
            print(f"    Analyzing V{j+1} ({Path(initial_path).name})...")
            score = analyze_speaker_similarity(app_context.speaker_similarity_model, speaker_path_str, initial_path, device=app_context.device.type)

            if score != -1.0 and score < SIMILARITY_THRESHOLD:
                status_log.append(f"    🔄 Low similarity on V{j+1} (Score: {score:.2f} < {SIMILARITY_THRESHOLD:.2f}). Triggering auto-regeneration...")
                print(status_log[-1])
                # yield status update here if needed

                regen_success = False
                for attempt in range(AUTO_REGEN_ATTEMPTS):
                    new_seed = random.randint(0, 2**32 - 1)
                    new_segment_filename = f"line{line_idx_0_based:03d}_spk-{Path(speaker_filename).stem}_v{j+1:02d}_autoregen{attempt+1}_s{new_seed}.wav"
                    new_output_path = str(TEMP_CONVO_MULTI_DIR / new_segment_filename)
                    print(f"      Attempt {attempt+1}/{AUTO_REGEN_ATTEMPTS}: Auto-Regenerating V{j+1} with new seed {new_seed} -> {new_segment_filename}...")

                    try:
                        app_context.tts.infer(speaker_path_str, line_text, new_output_path,
                                              temperature=temperature_val, top_p=top_p_val, top_k=int(top_k_val), seed=new_seed)

                        if Path(new_output_path).is_file() and Path(new_output_path).stat().st_size > 0:
                            status_log.append(f"        ✅ Auto-Regen V{j+1} (Attempt {attempt+1}) successful. Replacing original.")
                            print(status_log[-1])
                            final_line_options[j] = new_output_path # Update the list with the new path
                            regen_success = True
                            # Optional: Clean up the replaced file
                            # try: Path(initial_path).unlink() except OSError: pass
                            break # Success, stop retrying for this version
                        else:
                            status_log.append(f"        ❌ Auto-Regen V{j+1} (Attempt {attempt+1}) failed (file missing/empty).")
                            print(status_log[-1])
                    except Exception as e:
                        err_msg = f"Error during Auto-Regen L{line_idx_0_based+1} V{j+1} (Attempt {attempt+1}): {type(e).__name__}: {e}"
                        print(f"      {err_msg}")
                        status_log.append(f"\n      ❌ {err_msg}")
                        traceback.print_exc()
                        # Continue to next attempt even if one fails

                if not regen_success:
                    status_log.append(f"    ⚠️ Failed to auto-regenerate V{j+1} above threshold after {AUTO_REGEN_ATTEMPTS} attempts. Keeping original low-scoring version.")
                    print(status_log[-1])
                # yield status update here if needed

            elif score == -1.0:
                print(f"    ⚠️ Similarity analysis failed for V{j+1}. Skipping auto-regeneration.")
                status_log.append(f"    ⚠️ Similarity analysis failed for V{j+1}. Skipping.")
                # yield status update here if needed

    return final_line_options


# --- Main Conversation Generation Function (Refactored) ---

def parse_validate_and_start_convo(
    script_text: str,
    num_versions_str: str,
    temperature_val: float, top_p_val: float, top_k_val: int,
    seed_strategy: str,
    fixed_base_seed_input: int, # Renamed from fixed_base_seed
    app_context: AppContext, # Add context
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[str, ParsedScript, AllOptionsState, EditedTextsState, int, Any, Any, Any]: # Returns final status, states, and UI updates
    """
    Parses script, validates inputs, generates multiple versions for each line
    using specified parameters and seed strategy, performs auto-regeneration
    based on speaker similarity, and yields progress updates.
    """
    status_log = ["Starting Conversation Multi-Version Generation..."]
    # Initialize states
    parsed_script: ParsedScript = []
    all_options: AllOptionsState = []
    final_edited_texts: EditedTextsState = {}
    final_line_index: int = -1

    # 0. Prepare Temp Directory
    if not prepare_temp_dir(TEMP_CONVO_MULTI_DIR):
        status_log.append(f"Error: Failed to prepare temp directory {TEMP_CONVO_MULTI_DIR}.")
        # Return error status and initial empty states/UI updates
        return "\n".join(status_log), [], [], {}, -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)
    status_log.append(f"Using temp directory: {TEMP_CONVO_MULTI_DIR}")
    # Don't yield here

    # 1. Validate Initial Inputs
    error_msg, num_versions = _validate_convo_gen_inputs(script_text, num_versions_str, app_context)
    if error_msg:
        status_log.append(error_msg)
        return "\n".join(status_log), [], [], {}, -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)
    status_log.append("Input parameters validated.")
    # Don't yield here

    # 2. Parse Script and Validate Speakers
    status_log.append("Parsing script and validating speakers...")
    # Don't yield here
    error_msg, parsed_script_result = _parse_script_and_validate_speakers(script_text)
    if error_msg:
        status_log.append(error_msg)
        return "\n".join(status_log), [], [], {}, -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False)
    parsed_script = parsed_script_result # Assign if successful
    total_lines = len(parsed_script)
    status_log.append(f"Script parsed: {total_lines} valid lines found.")
    # Don't yield here

    # 3. Determine Base Seed and Reused Seed Lists (if applicable)
    base_seed = 0
    reused_random_seed_list = None
    reused_sequential_seed_list = None
    status_log.append(f"Selected Seed Strategy: {seed_strategy}")
    print(f"Selected Seed Strategy: {seed_strategy}")

    if seed_strategy in [SEED_STRATEGY_FIXED_BASE_SEQUENTIAL, SEED_STRATEGY_FIXED_BASE_REUSED_LIST]:
        try:
            base_seed = int(fixed_base_seed_input)
            status_log.append(f"Using Fixed Base Seed: {base_seed}")
            print(f"Using Fixed Base Seed: {base_seed}")
        except (ValueError, TypeError):
            base_seed = DEFAULT_FIXED_BASE_SEED
            status_log.append(f"Warning: Invalid Fixed Base Seed input, using default: {base_seed}")
            print(f"Warning: Invalid Fixed Base Seed input, using default: {base_seed}")
    elif seed_strategy in [SEED_STRATEGY_RANDOM_BASE_SEQUENTIAL, SEED_STRATEGY_RANDOM_BASE_REUSED_LIST]:
        base_seed = random.randint(0, 2**32 - 1)
        status_log.append(f"Using Random Base Seed: {base_seed}")
        print(f"Using Random Base Seed: {base_seed}")

    if seed_strategy == SEED_STRATEGY_RANDOM_BASE_REUSED_LIST:
        reused_random_seed_list = random.sample(range(2**32), num_versions)
        seed_info_str = ', '.join([f'V{k+1}=R({s})' for k, s in enumerate(reused_random_seed_list)])
        status_log.append(f"Reused Random Seed List: [{seed_info_str}]")
        print(f"Reused Random Seed List: [{seed_info_str}]")
    elif seed_strategy == SEED_STRATEGY_FIXED_BASE_REUSED_LIST:
        reused_sequential_seed_list = [(base_seed + j) % (2**32) for j in range(num_versions)]
        seed_info_str = ', '.join([f'V{k+1}=S({s})' for k, s in enumerate(reused_sequential_seed_list)])
        status_log.append(f"Reused Sequential Seed List: [{seed_info_str}]")
        print(f"Reused Sequential Seed List: [{seed_info_str}]")

    # Don't yield here

    # 4. Main Generation Loop
    all_options: AllOptionsState = []
    generation_successful = True
    for i, line_info in enumerate(progress.tqdm(parsed_script, desc="Generating Lines")):
        line_idx_0_based = i

        # 4a. Determine Seeds for this specific line
        if seed_strategy == SEED_STRATEGY_RANDOM_BASE_REUSED_LIST:
            line_version_seeds = reused_random_seed_list
            seed_log_info = f"(Using reused random list for Line {i+1})" if i == 0 else ""
        elif seed_strategy == SEED_STRATEGY_FIXED_BASE_REUSED_LIST:
            line_version_seeds = reused_sequential_seed_list
            seed_log_info = f"(Using reused sequential list for Line {i+1})" if i == 0 else ""
        else:
            # Calculate seeds specifically for this line
            line_version_seeds, seed_log_info = _determine_line_seeds(
                seed_strategy, base_seed, num_versions, line_idx_0_based, total_lines, base_seed
            )
            # base_seed is not updated per line for sequential offset strategies

        if seed_log_info: # Only log if info was generated (i.e., not reused list after first line)
            status_current = f"\nGenerating Line {line_idx_0_based + 1}/{total_lines} ({line_info['speaker_filename']}). Seeds: {seed_log_info}"
            print(status_current)
            status_log.append(status_current)
            # Don't yield here

        # 4b. Generate initial versions for the line
        initial_line_options, line_gen_success = _generate_versions_for_line(
            line_idx_0_based, line_info, num_versions, line_version_seeds,
            temperature_val, top_p_val, top_k_val, status_log, progress, app_context # Pass context
        )
        if not line_gen_success:
            generation_successful = False
            status_log.append(f"❌ Critical error generating Line {line_idx_0_based + 1}. Stopping.")
            # Don't yield here
            # Break loop and proceed to final return
            break

        # 4c. Check similarity and auto-regenerate
        final_line_options = _check_similarity_and_autoregen_line(
            line_idx_0_based, line_info, initial_line_options, num_versions,
            temperature_val, top_p_val, top_k_val, status_log, progress, app_context # Pass context
        )

        all_options.append(final_line_options)
        # Don't yield progress per line


    # 5. Final Status Update & Return all values
    final_edited_texts = {} # This function doesn't edit texts during initial gen
    final_line_index = 0 if generation_successful and total_lines > 0 else -1 # Start review at 0 if successful
    review_interactive = generation_successful and total_lines > 0
    # Check concat readiness based on final state (assuming no selections yet)
    can_concat = check_all_selections_valid(parsed_script, all_options, {}) and app_context.pydub_available

    if generation_successful:
        status_log.append(f"\n✅ Generation & Auto-Regen Check Complete for {total_lines} lines.")
        if review_interactive:
             status_log.append("Proceed to 'Review & Select Lines' tab.")
    else:
        # Error message already added to log during loop
        status_log.append(f"\n❌ Generation finished with errors.")

    print(status_log[-2])
    if len(status_log) > 1: print(status_log[-1])

    # Return all necessary values for the Gradio output components
    return (
        "\n".join(status_log),
        parsed_script,
        all_options,
        final_edited_texts,
        final_line_index,
        gr.update(interactive=True), # generate_convo_button
        gr.update(interactive=review_interactive), # review_tab
        gr.update(interactive=can_concat) # concat_tab
    )


def display_line_for_review(
    target_index: int,
    parsed_script: ParsedScript,
    all_options_state: AllOptionsState,
    selections_state: SelectionsState,
    edited_texts_state: EditedTextsState,
    app_context: AppContext # Add context
) -> ReviewYield:
    # Removed global access
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
    # Ensure all_options_state has enough entries, pad if necessary
    while target_index >= len(all_options_state):
        all_options_state.append([None] * MAX_VERSIONS_ALLOWED)
        print(f"Warning: Padded all_options_state for index {len(all_options_state)-1}")

    if not isinstance(selections_state, dict):
        selections_state = {}; print("Warning: selections_state was invalid, reset.")
    if not isinstance(edited_texts_state, dict):
        edited_texts_state = {}; print("Warning: edited_texts_state was invalid, reset.")

    line_info = parsed_script[target_index]; original_speaker = line_info['speaker_filename']; original_text = line_info['text']
    nav_display = f"Line {target_index + 1} / {num_lines}"; line_text_display = f"Speaker: {original_speaker}\nText: {original_text}";
    editable_text_display = edited_texts_state.get(target_index, original_text)
    current_line_options = all_options_state[target_index] if isinstance(all_options_state[target_index], list) else []
    # Ensure current_line_options has the correct length
    while len(current_line_options) < MAX_VERSIONS_ALLOWED:
        current_line_options.append(None)

    similarities: List[Tuple[int, float]] = []
    ref_speaker_path = str(SPEAKER_DIR / original_speaker)
    # Use context for checks
    can_analyze = app_context.speechbrain_available and app_context.speaker_similarity_model is not None and Path(ref_speaker_path).is_file()
    if not can_analyze:
         status += " (Warn: Sim analysis unavailable)"
    valid_audio_paths_with_indices: List[Tuple[int, str]] = []
    for i in range(MAX_VERSIONS_ALLOWED):
        path = current_line_options[i] # Already padded
        if path and isinstance(path, str):
            try:
                p = Path(path)
                if p.is_file() and p.stat().st_size > 0:
                    valid_audio_paths_with_indices.append((i, str(p)))
                else:
                    audio_updates[i] = gr.update(value=None, interactive=False, label=f"{VERSION_PREFIX}{i+1} (Not Found/Empty)")
            except Exception as e:
                print(f"Error checking file for Line {target_index+1} V{i+1} ({path}): {e}")
                audio_updates[i] = gr.update(value=None, interactive=False, label=f"{VERSION_PREFIX}{i+1} (Error)")
        else:
             audio_updates[i] = gr.update(value=None, interactive=False, label=f"{VERSION_PREFIX}{i+1} (Not Generated)")

    if can_analyze and valid_audio_paths_with_indices:
        for i, audio_path_str in valid_audio_paths_with_indices:
            # Use context for model and device
            score = analyze_speaker_similarity(app_context.speaker_similarity_model, ref_speaker_path, audio_path_str, device=app_context.device.type)
            if score != -1.0:
                 similarities.append((i, score))
            else:
                 # Update label even if analysis failed, keep audio playable
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
        # Update audio player only if not already updated due to analysis error
        if not audio_updates[i] or audio_updates[i].get('label') != f"{VERSION_PREFIX}{i+1} (Analysis Error)":
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
                 status += f" (Auto-selected V{best_version_index+1})"
        elif radio_choices:
            current_selection_label = radio_choices[0]
            try:
                fallback_index = int(radio_choices[0].split(" ")[1]) - 1
                if selections_state.get(target_index) != fallback_index:
                    selections_state[target_index] = fallback_index
                    status += f" (Defaulted to V{fallback_index+1})"
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


# --- Helper Functions for regenerate_below_threshold ---

def _validate_threshold_regen_inputs(
    line_idx_0_based: int,
    editable_text: str,
    parsed_script: ParsedScript,
    all_options_state: AllOptionsState,
    max_manual_attempts_str: str,
    app_context: AppContext # Add context
) -> Tuple[Optional[str], Optional[int], Optional[str], Optional[Dict]]:
    """Validate inputs for the threshold regeneration process."""
    # Removed global access
    status = ""
    max_manual_attempts_int = 5 # Default

    try:
        max_manual_attempts_int = int(max_manual_attempts_str)
        if not (max_manual_attempts_int >= 0):
             raise ValueError("Max attempts must be 0 or greater.")
    except (ValueError, TypeError):
        status = f"Error: Invalid 'Max Manual Regen Attempts' value ('{max_manual_attempts_str}') selected on Tab 1. Defaulting to 5."
        print(status)
        # Return status and default attempts, but indicate failure with None for others
        return status, max_manual_attempts_int, None, None

    if app_context.tts is None:
        return "Error: TTS model not initialized.", max_manual_attempts_int, None, None
    if not app_context.speechbrain_available or app_context.speaker_similarity_model is None:
        return "Error: Speaker model not available for similarity check.", max_manual_attempts_int, None, None
    if not isinstance(parsed_script, list) or not (0 <= line_idx_0_based < len(parsed_script)):
        return f"Error: Invalid script data or line index ({line_idx_0_based}).", max_manual_attempts_int, None, None
    if not isinstance(all_options_state, list) or line_idx_0_based >= len(all_options_state):
         # Check if the index is exactly the length, meaning the list is too short
         if isinstance(all_options_state, list) and line_idx_0_based == len(all_options_state):
             # This might happen if generation failed for later lines. Allow proceeding but warn.
             print(f"Warning: Options state only has {len(all_options_state)} entries, requested index {line_idx_0_based}. Proceeding with empty options for this line.")
             # We'll handle the empty options list later.
             pass
         else:
            return f"Error: Options state invalid for line index ({line_idx_0_based}).", max_manual_attempts_int, None, None

    if not editable_text or not editable_text.strip():
        return f"Error: Editable text for Line {line_idx_0_based + 1} cannot be empty.", max_manual_attempts_int, None, None

    line_info = parsed_script[line_idx_0_based]
    speaker_filename = line_info['speaker_filename']
    ref_speaker_path = str(SPEAKER_DIR / speaker_filename)
    if not Path(ref_speaker_path).is_file():
        return f"Error: Reference speaker '{speaker_filename}' not found.", max_manual_attempts_int, None, None

    return None, max_manual_attempts_int, ref_speaker_path, line_info


def _analyze_initial_versions_for_threshold(
    line_idx_0_based: int,
    initial_options: List[Optional[str]],
    manual_threshold: float,
    ref_speaker_path: str,
    status_log: List[str],
    app_context: AppContext # Add context
) -> Tuple[Dict[int, Dict], List[int]]:
    """Analyze initial versions against threshold and return slots needing regen."""
    # Removed global access
    slot_results = {}
    slots_to_process_initially = []

    print(f"Analyzing Line {line_idx_0_based + 1} against manual threshold {manual_threshold:.2f}...")
    status_log.append("\nAnalyzing initial versions...")

    for j, audio_path in enumerate(initial_options):
        current_best_score = -2.0
        current_best_path = audio_path
        generated_paths_for_slot = []

        if audio_path and Path(audio_path).is_file():
            print(f"  Checking initial V{j+1} ({Path(audio_path).name})...")
            # Use context for model and device
            score = analyze_speaker_similarity(app_context.speaker_similarity_model, ref_speaker_path, audio_path, device=app_context.device.type)
            if score != -1.0:
                current_best_score = score
                generated_paths_for_slot.append(audio_path)
                if score < manual_threshold:
                    slots_to_process_initially.append(j)
                    print(f"    Marked V{j+1} for regeneration (Initial Score: {score:.2f} < {manual_threshold:.2f})")
                else:
                    print(f"    V{j+1} meets threshold (Initial Score: {score:.2f})")
            else:
                status_log.append(f"\n  ⚠️ Analysis failed for initial V{j+1}. Cannot evaluate.")
                print(f"    Analysis failed for initial V{j+1}.")
                current_best_path = None # Treat analysis failure as invalid
        else:
            # If the slot is empty/invalid initially, mark it for potential regeneration
            print(f"    Marked empty/invalid slot V{j+1} for regeneration attempt.")
            slots_to_process_initially.append(j)
            current_best_path = None

        slot_results[j] = {
            'best_score': current_best_score,
            'best_path': current_best_path,
            'generated_paths': generated_paths_for_slot
        }
    return slot_results, slots_to_process_initially


def _regenerate_slot_attempt(
    line_idx_0_based: int,
    slot_index: int, # j
    attempt_num_total: int, # Overall attempt number for this slot
    pass_num: int, # Current regeneration pass number
    speaker_path_str: str,
    editable_text: str,
    temperature_val: float, top_p_val: float, top_k_val: int,
    slot_results: Dict[int, Dict],
    manual_threshold: float,
    status_log: List[str],
    app_context: AppContext # Add context
) -> bool:
    """Perform one regeneration attempt for a slot and update results."""
    # Removed global access
    version_idx_1_based = slot_index + 1
    new_seed = random.randint(0, 2**32 - 1)
    speaker_filename_stem = Path(speaker_path_str).stem # Get stem from path
    segment_filename = f"line{line_idx_0_based:03d}_spk-{speaker_filename_stem}_v{version_idx_1_based:02d}_mregen{pass_num}_s{new_seed}.wav"
    new_output_path = str(TEMP_CONVO_MULTI_DIR / segment_filename)

    print(f"  Regenerating V{version_idx_1_based} (Attempt {attempt_num_total}, Pass {pass_num}, Seed: {new_seed})...")
    threshold_met = False
    try:
        app_context.tts.infer(speaker_path_str, editable_text, output_path=new_output_path,
                              temperature=temperature_val, top_p=top_p_val, top_k=int(top_k_val), seed=new_seed)

        if Path(new_output_path).is_file() and Path(new_output_path).stat().st_size > 0:
            print(f"    V{version_idx_1_based} (Attempt {attempt_num_total}) generated OK.")
            slot_results[slot_index]['generated_paths'].append(new_output_path)
            # Use context for model and device
            new_score = analyze_speaker_similarity(app_context.speaker_similarity_model, speaker_path_str, new_output_path, device=app_context.device.type)

            if new_score != -1.0:
                print(f"      New Score: {new_score:.2f}")
                if new_score > slot_results[slot_index]['best_score']:
                    slot_results[slot_index]['best_score'] = new_score
                    slot_results[slot_index]['best_path'] = new_output_path
                    print(f"      * New best score for V{version_idx_1_based} *")
                if new_score >= manual_threshold:
                    print(f"      Threshold {manual_threshold:.2f} met/exceeded.")
                    threshold_met = True
            else:
                print(f"      ⚠️ Analysis failed for regenerated V{version_idx_1_based} (Attempt {attempt_num_total}).")
                status_log.append(f"\n  ⚠️ Analysis failed for regen V{version_idx_1_based} (Attempt {attempt_num_total}).")
        else:
            status_log.append(f"\n  ❌ Regen V{version_idx_1_based} (Attempt {attempt_num_total}) failed (file missing/empty).")
            print(f"    V{version_idx_1_based} (Attempt {attempt_num_total}) FAILED.")
    except Exception as e:
        err_msg = f"Error during Regen L{line_idx_0_based + 1} V{version_idx_1_based} (Attempt {attempt_num_total}): {type(e).__name__}: {e}"
        print(f"    {err_msg}")
        status_log.append(f"\n❌ {err_msg}")
        traceback.print_exc()

    return threshold_met


def _finalize_threshold_regen_slot(
    slot_index: int,
    slot_results: Dict[int, Dict],
    all_options_copy_line: List[Optional[str]] # The specific line's options list
):
    """Update the options list with the best path and clean up others."""
    final_best_path = slot_results[slot_index]['best_path']
    final_best_score = slot_results[slot_index]['best_score']

    # Update the main state list for this line
    all_options_copy_line[slot_index] = final_best_path

    if final_best_path is not None:
        print(f"  Final selection for V{slot_index+1}: {Path(final_best_path).name} (Score: {final_best_score:.2f})")
        # Clean up other generated files for this slot
        for generated_path in slot_results[slot_index]['generated_paths']:
            if generated_path != final_best_path: # Don't delete the best one!
                try:
                    p = Path(generated_path)
                    if p.is_file(): # Check if it exists before deleting
                        print(f"    Cleaning up: {p.name}")
                        p.unlink()
                except OSError as del_err:
                    print(f"    Warning: Failed to delete temp file {generated_path}: {del_err}")
    else:
        print(f"  V{slot_index+1}: No valid version could be generated or kept.")
        all_options_copy_line[slot_index] = None # Ensure slot is None if all failed


# --- Refactored Function ---
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
    app_context: AppContext, # Add context
    progress=gr.Progress(track_tqdm=True)
) -> Iterator[ManualRegenYield]:
    """
    Regenerates audio versions for a specific line that fall below a manual
    similarity threshold, using helper functions for validation, analysis,
    regeneration attempts, and finalization.
    """
    line_idx_0_based = current_line_index
    status_log = [f"Starting threshold process for Line {line_idx_0_based + 1} (Threshold: {manual_threshold:.2f})..."]

    # Make copies of state to modify during the process
    # Ensure deep copies if necessary, though list/dict copies might suffice here
    all_options_copy = [list(opts) if isinstance(opts, list) else [] for opts in all_options_state]
    selections_copy = selections_state.copy()
    edited_texts_copy = edited_texts_state.copy()

    # Yield initial status
    yield "\n".join(status_log), all_options_copy, selections_copy, edited_texts_copy

    # 1. Validate Inputs
    error_msg, max_attempts, ref_speaker_path, line_info = _validate_threshold_regen_inputs(
        line_idx_0_based, editable_text, parsed_script, all_options_state, max_manual_attempts_str, app_context # Pass context
    )
    if error_msg:
        status_log.append(error_msg)
        yield "\n".join(status_log), all_options_copy, selections_copy, edited_texts_copy
        return

    status_log.append(f"Max Regen Attempts: {max_attempts}")
    yield "\n".join(status_log), all_options_copy, selections_copy, edited_texts_copy

    # Ensure the options list for the current line exists and has the correct length
    if line_idx_0_based >= len(all_options_copy):
         # Pad with None if the line index was out of bounds but valid (e.g., last line failed generation)
         while line_idx_0_based >= len(all_options_copy):
             all_options_copy.append([None] * MAX_VERSIONS_ALLOWED)
         print(f"Warning: Padded options state for line {line_idx_0_based + 1}.")
    # Ensure the specific line's options list has the correct number of slots
    current_line_options = all_options_copy[line_idx_0_based]
    while len(current_line_options) < MAX_VERSIONS_ALLOWED:
        current_line_options.append(None)


    # 2. Analyze Initial Versions
    slot_results, slots_to_process_initially = _analyze_initial_versions_for_threshold(
        line_idx_0_based, current_line_options, manual_threshold, ref_speaker_path, status_log, app_context # Pass context
    )
    yield "\n".join(status_log), all_options_copy, selections_copy, edited_texts_copy

    if not slots_to_process_initially and max_attempts > 0:
        status_log.append("\nNo versions initially below threshold or needing generation. No regeneration needed.")
        print(f"No versions initially below threshold for Line {line_idx_0_based + 1}.")
        yield "\n".join(status_log), all_options_copy, selections_copy, edited_texts_copy
        return
    elif max_attempts == 0:
        status_log.append("\nMax attempts set to 0. Skipping regeneration, proceeding to cleanup.")
        print(f"Max attempts set to 0. Skipping regeneration for Line {line_idx_0_based + 1}.")
        # Proceed directly to finalization/cleanup below
    else:
        # 3. Regeneration Loop
        status_log.append(f"\nAttempting regeneration for {len(slots_to_process_initially)} slot(s)...")
        yield "\n".join(status_log), all_options_copy, selections_copy, edited_texts_copy

        slots_still_processing = list(slots_to_process_initially)
        attempts_per_slot = {j: 0 for j in slots_to_process_initially}

        for pass_num in range(1, max_attempts + 1):
            if not slots_still_processing: break # Stop if all slots met threshold

            status_log.append(f"\n--- Regeneration Pass {pass_num}/{max_attempts} ---")
            yield "\n".join(status_log), all_options_copy, selections_copy, edited_texts_copy
            print(f"--- Starting Regen Pass {pass_num}/{max_attempts} for Line {line_idx_0_based + 1} ---")

            slots_processed_this_pass = list(slots_still_processing) # Iterate over copy
            pass_regenerated_count = 0

            for slot_index in progress.tqdm(slots_processed_this_pass, desc=f"Regen Pass {pass_num}"):
                attempts_per_slot[slot_index] += 1
                threshold_met = _regenerate_slot_attempt(
                    line_idx_0_based, slot_index, attempts_per_slot[slot_index], pass_num,
                    ref_speaker_path, editable_text,
                    temperature_val, top_p_val, top_k_val,
                    slot_results, manual_threshold, status_log, app_context # Pass context
                )
                pass_regenerated_count += 1 # Count attempt

                if threshold_met:
                    if slot_index in slots_still_processing:
                        slots_still_processing.remove(slot_index) # Stop processing this slot

                # Yield intermediate status after each attempt? Maybe too noisy.
                # yield "\n".join(status_log), all_options_copy, selections_copy, edited_texts_copy

            status_log.append(f"\n Pass {pass_num} finished. ({pass_regenerated_count}/{len(slots_processed_this_pass)} slots attempted this pass).")
            yield "\n".join(status_log), all_options_copy, selections_copy, edited_texts_copy # Update UI after each full pass

    # 4. Finalize and Clean Up
    status_log.append(f"\n--- Finalizing results for Line {line_idx_0_based + 1} ---")
    print(f"--- Finalizing results for Line {line_idx_0_based + 1} ---")
    yield "\n".join(status_log), all_options_copy, selections_copy, edited_texts_copy

    for j in range(MAX_VERSIONS_ALLOWED):
        if j in slot_results: # Only process slots that were analyzed/attempted
             _finalize_threshold_regen_slot(j, slot_results, current_line_options) # Pass the specific line's list

    # Update edited text state only if regeneration was attempted and successful for at least one slot
    if any(slot_results[j]['best_path'] is not None and j in slots_to_process_initially for j in slot_results if j in slots_to_process_initially):
         edited_texts_copy[line_idx_0_based] = editable_text

    status_log.append(f"\n✅ Threshold regeneration/cleanup finished. Kept best version for each slot.")
    print(f"Threshold regeneration/cleanup finished for Line {line_idx_0_based + 1}.")
    yield status_log[-1], all_options_copy, selections_copy, edited_texts_copy # Final yield with updated state


def regenerate_single_line(
    current_line_index: int, parsed_script: ParsedScript, all_options_state: AllOptionsState, selections_state: SelectionsState, edited_texts_state: EditedTextsState,
    editable_text: str, num_versions_str: str,
    temperature_val: float, top_p_val: float, top_k_val: int,
    seed_strategy: str,
    fixed_base_seed: int,
    app_context: AppContext, # Add context
    progress=gr.Progress(track_tqdm=True)
) -> Iterator[RegenYield]: # Corrected type hint
    # Removed global tts access
    line_idx_0_based = current_line_index
    status = f"Starting regeneration for Line {line_idx_0_based + 1} using strategy: {seed_strategy}..."
    all_options_copy = [list(opts) if opts else [] for opts in all_options_state]
    selections_copy = selections_state.copy()
    edited_texts_copy = edited_texts_state.copy()
    yield status, all_options_copy, selections_copy, edited_texts_copy
    if app_context.tts is None: yield "Error: TTS model not initialized.", all_options_copy, selections_copy, edited_texts_copy; return
    if not isinstance(parsed_script, list) or not (0 <= line_idx_0_based < len(parsed_script)): yield f"Error: Invalid script data or line index ({line_idx_0_based}).", all_options_copy, selections_copy, edited_texts_copy; return
    # Ensure all_options_copy has an entry for the current line index
    while line_idx_0_based >= len(all_options_copy):
        all_options_copy.append([None] * MAX_VERSIONS_ALLOWED) # Pad if needed
    # Ensure the specific line's options list has the correct number of slots
    current_line_options_list = all_options_copy[line_idx_0_based]
    while len(current_line_options_list) < MAX_VERSIONS_ALLOWED:
        current_line_options_list.append(None)

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
            app_context.tts.infer(speaker_path_str, editable_text, output_path, temperature=temperature_val, top_p=top_p_val, top_k=int(top_k_val), seed=current_seed)
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
        # Return original state if all versions failed
        yield status, all_options_state, selections_state, edited_texts_state


# --- Helper Functions for concatenate_conversation_versions ---

def _validate_concat_params(
    output_format: str, mp3_bitrate: str,
    trim_thresh_dbfs_val: float, trim_min_silence_len_ms_val: int,
    compress_thresh_db: float, compress_ratio_val: float, compress_attack_ms: float, compress_release_ms: float,
    nr_strength_val: float,
    eq_low_gain: float, eq_mid_gain: float, eq_high_gain: float,
    peak_norm_target_dbfs: float,
    reverb_amount: float, pitch_shift: float, speed_factor: float
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Validate parameters for concatenation and processing."""
    params = {}
    try:
        params['export_format'] = output_format.lower().strip()
        if params['export_format'] not in OUTPUT_FORMAT_CHOICES:
            raise ValueError("Invalid output format")
        params['bitrate_str'] = f"{mp3_bitrate}k" if params['export_format'] == "mp3" else None
        params['trim_thresh'] = float(trim_thresh_dbfs_val)
        params['trim_len'] = int(trim_min_silence_len_ms_val)
        params['compress_thresh'] = float(compress_thresh_db)
        params['compress_ratio'] = float(compress_ratio_val)
        params['compress_attack'] = float(compress_attack_ms)
        params['compress_release'] = float(compress_release_ms)
        params['nr_strength'] = float(nr_strength_val)
        params['eq_low'] = float(eq_low_gain)
        params['eq_mid'] = float(eq_mid_gain)
        params['eq_high'] = float(eq_high_gain)
        params['final_norm_target'] = float(peak_norm_target_dbfs)
        params['segment_norm_target'] = DEFAULT_SEGMENT_NORM_TARGET_DBFS # Constant
        params['reverb_amount'] = float(reverb_amount)
        params['pitch_shift'] = float(pitch_shift)
        params['speed_factor'] = float(speed_factor)
        params['trim_keep_silence'] = DEFAULT_TRIM_KEEP_SILENCE_MS # Constant
    except (ValueError, TypeError) as e:
        return f"Error: Invalid parameter settings ({type(e).__name__}: {e})", {}
    return None, params

def _collect_selected_audio_files(
    parsed_script: ParsedScript,
    all_options_state: AllOptionsState,
    selections_state: SelectionsState
) -> Tuple[List[str], List[str]]:
    """Collect valid file paths based on user selections."""
    selected_files: List[str] = []
    warnings: List[str] = []
    print("Collecting selected files for concatenation:")

    if not isinstance(parsed_script, list) or \
       not isinstance(all_options_state, list) or \
       not isinstance(selections_state, dict):
        warnings.append("Invalid script, options, or selections state.")
        return [], warnings

    for i in range(len(parsed_script)):
        selected_version_index = selections_state.get(i, -1)
        path_found = False

        if selected_version_index == -1:
            warnings.append(f"L{i+1}: No version selected.")
            continue

        line_options = []
        if i < len(all_options_state) and isinstance(all_options_state[i], list):
            line_options = all_options_state[i]
        else:
            warnings.append(f"L{i+1}: Options data missing or invalid.")
            continue

        if 0 <= selected_version_index < len(line_options):
            selected_path = line_options[selected_version_index]
            if selected_path and isinstance(selected_path, str):
                try:
                    p = Path(selected_path)
                    if p.is_file() and p.stat().st_size > 0:
                        selected_files.append(str(p))
                        print(f"  Line {i+1}: Selected V{selected_version_index+1} ({p.name})")
                        path_found = True
                    elif p.exists():
                        warnings.append(f"L{i+1},V{selected_version_index+1}: File empty ('{p.name}').")
                    else:
                        warnings.append(f"L{i+1},V{selected_version_index+1}: File not found ('{p.name}').")
                except Exception as e:
                    warnings.append(f"L{i+1},V{selected_version_index+1}: Error checking file '{selected_path}' ({type(e).__name__}: {e})")
            else:
                warnings.append(f"L{i+1},V{selected_version_index+1}: Invalid path data.")
        else:
            warnings.append(f"L{i+1}: Invalid selection index ({selected_version_index} for {len(line_options)} options).")

        if not path_found:
            # This warning might be redundant if previous checks caught the issue
            # warnings.append(f"L{i+1}: Could not find valid file for selected V{selected_version_index+1}.")
            pass

    return selected_files, warnings


def _load_and_process_segment(
    path_str: str,
    line_num: int, # 1-based for logging
    normalize_segments_flag: bool,
    trim_silence_flag: bool,
    params: Dict[str, Any],
    status_log: List[str],
    app_context: AppContext # Add context
) -> Optional[AudioSegment]:
    """Load a single segment and apply pre-concat processing."""
    # Removed global access
    segment = None
    try:
        path = Path(path_str)
        segment = AudioSegment.from_file(path)

        # 1. Segment Normalization
        if normalize_segments_flag:
            current_peak = segment.max_dBFS
            # Check for silence or invalid dBFS
            if not math.isinf(current_peak) and current_peak > -96.0: # Avoid boosting pure silence
                gain_needed = params['segment_norm_target'] - current_peak
                segment = segment.apply_gain(gain_needed)
            elif math.isinf(current_peak):
                 print(f"    L{line_num}: Skipping norm (silent segment).")


        # 2. Silence Trimming
        if trim_silence_flag:
            # Use context flag
            if app_context.pydub_silence_available and _silence_Imported:
                nonsilent_ranges = _silence_Imported.detect_nonsilent(
                    segment,
                    min_silence_len=params['trim_len'],
                    silence_thresh=params['trim_thresh']
                )
                if nonsilent_ranges:
                    start_trim = max(0, nonsilent_ranges[0][0] - params['trim_keep_silence'])
                    end_trim = min(len(segment), nonsilent_ranges[-1][1] + params['trim_keep_silence'])
                    if end_trim > start_trim:
                        segment = segment[start_trim:end_trim]
                    else:
                        print(f"    L{line_num}: Skipping trim (no non-silent audio detected).")
                        # Return None or empty segment if trimming results in nothing?
                        # Let's return the (potentially empty) trimmed segment for now.
                else: # No non-silent ranges found
                    print(f"    L{line_num}: Skipping trim (entire segment detected as silent).")
                    # Return None or empty segment? Let's return the original silent segment.
            else:
                # This warning should ideally be handled before the loop
                # status_log.append(f"\n  Warning: Skipping trim for L{line_num} (module unavailable).")
                pass # Already warned outside the loop

        return segment

    except Exception as e:
        warn = f"Warning L{line_num}: Error loading/processing '{os.path.basename(path_str)}' ({type(e).__name__}: {e}). Skipping segment."
        status_log.append(f"\n  {warn}")
        print(f"    {warn}")
        traceback.print_exc()
        return None


def _concatenate_segments(
    processed_segments: List[AudioSegment]
) -> Optional[AudioSegment]:
    """Concatenate processed segments with silence and crossfade."""
    if not processed_segments:
        return None

    combined = AudioSegment.empty()
    # Use constants for silence and crossfade
    silence_between = AudioSegment.silent(duration=DEFAULT_SILENCE_MS)
    crossfade_duration = DEFAULT_CROSSFADE_MS

    try:
        combined = processed_segments[0]
        for i in range(1, len(processed_segments)):
            # Add silence only if the next segment is not empty
            if len(processed_segments[i]) > 0:
                 combined = combined.append(silence_between, crossfade=0) # No crossfade for silence
            combined = combined.append(processed_segments[i], crossfade=crossfade_duration)
    except Exception as e:
        print(f"Error during concatenation: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    if len(combined) == 0:
        print("Error: Concatenation resulted in empty audio.")
        return None

    return combined


def _apply_global_effects(
    audio: AudioSegment,
    params: Dict[str, Any],
    apply_nr_flag: bool,
    apply_compression_flag: bool,
    status_log: List[str],
    app_context: AppContext # Add context
) -> Tuple[AudioSegment, List[str]]:
    """Apply post-concatenation effects."""
    # Removed global access
    processed_audio = audio
    processing_errors = []

    # Order: Pitch -> Speed -> NR -> EQ -> Compression -> Reverb
    if params['pitch_shift'] is not None and abs(params['pitch_shift']) > 0.01:
        try:
            processed_audio = change_pitch(processed_audio, params['pitch_shift'])
            status_log.append(f"\n  Applied Pitch Shift ({params['pitch_shift']:+.2f} semitones)")
        except Exception as e:
            err_msg = f"Pitch Shift failed ({type(e).__name__}: {e})"
            processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status_log.append(f"\n  ❌ Error during Pitch Shift: {err_msg}"); traceback.print_exc()

    if params['speed_factor'] is not None and abs(params['speed_factor'] - 1.0) > 0.01:
        try:
            processed_audio = change_speed(processed_audio, params['speed_factor'])
            status_log.append(f"\n  Applied Speed Change (x{params['speed_factor']:.2f})")
        except Exception as e:
            err_msg = f"Speed Change failed ({type(e).__name__}: {e})"
            processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status_log.append(f"\n  ❌ Error during Speed Change: {err_msg}"); traceback.print_exc()

    if apply_nr_flag:
        # Use context flag
        if app_context.noisereduce_available:
            try:
                processed_audio = apply_noise_reduction(processed_audio, params['nr_strength'])
                status_log.append(f"\n  Applied Noise Reduction (Strength: {params['nr_strength']:.2f})")
            except Exception as e:
                err_msg = f"Noise Reduction failed ({type(e).__name__}: {e})"; processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status_log.append(f"\n  ❌ Error during Noise Reduction: {err_msg}"); traceback.print_exc()
        else: status_log.append("\n  Skipped Noise Reduction (Library not available)")

    if params['eq_low'] != 0 or params['eq_mid'] != 0 or params['eq_high'] != 0:
        # Use context flag
        if app_context.scipy_available:
            try:
                processed_audio = apply_eq(processed_audio, params['eq_low'], params['eq_mid'], params['eq_high'])
                status_log.append(f"\n  Applied EQ (L:{params['eq_low']}, M:{params['eq_mid']}, H:{params['eq_high']} dB)")
            except Exception as e:
                err_msg = f"EQ failed ({type(e).__name__}: {e})"; processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status_log.append(f"\n  ❌ Error during EQ: {err_msg}"); traceback.print_exc()
        else: status_log.append("\n  Skipped EQ (Library not available)")

    if apply_compression_flag:
        # Use context flag
        if app_context.pydub_compress_available and _compress_dynamic_range_Imported:
            try:
                processed_audio = _compress_dynamic_range_Imported(
                    processed_audio,
                    threshold=params['compress_thresh'],
                    ratio=params['compress_ratio'],
                    attack=params['compress_attack'],
                    release=params['compress_release']
                )
                status_log.append(f"\n  Applied Compression (Thresh:{params['compress_thresh']}, Ratio:{params['compress_ratio']:.1f})")
            except Exception as e:
                err_msg = f"Compression failed ({type(e).__name__}: {e})"; processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status_log.append(f"\n  ❌ Error during Compression: {err_msg}"); traceback.print_exc()
        else: status_log.append("\n  Skipped Compression (pydub effects component not available)")

    if params['reverb_amount'] is not None and params['reverb_amount'] > 0.0:
         # Use context flag
         if app_context.pydub_available: # Reverb might depend on basic pydub
            try:
                processed_audio = apply_reverb(processed_audio, params['reverb_amount'])
                status_log.append(f"\n  Applied Reverb (Amount: {params['reverb_amount']:.2f})")
            except Exception as e:
                err_msg = f"Reverb failed ({type(e).__name__}: {e})"
                processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status_log.append(f"\n  ❌ Error during Reverb: {err_msg}"); traceback.print_exc()
         else: status_log.append("\n  Skipped Reverb (pydub not available)")


    return processed_audio, processing_errors


def _apply_final_normalization(
    audio: AudioSegment,
    apply_peak_norm_flag: bool,
    params: Dict[str, Any],
    status_log: List[str],
    app_context: AppContext # Add context
) -> Tuple[AudioSegment, List[str]]:
    """Apply final peak normalization if enabled."""
    # Removed global access
    processed_audio = audio
    norm_errors = []
    if apply_peak_norm_flag:
        # Use context flag
        if app_context.pydub_available:
            try:
                current_peak_dbfs = processed_audio.max_dBFS
                if not math.isinf(current_peak_dbfs) and current_peak_dbfs > -96.0: # Avoid normalizing pure silence
                    gain_to_apply = params['final_norm_target'] - current_peak_dbfs
                    # Only apply gain if it's actually needed (avoids tiny adjustments)
                    if gain_to_apply < -0.01:
                        processed_audio = processed_audio.apply_gain(gain_to_apply)
                        status_log.append(f"\n  Applied Final Peak Normalization (Target: {params['final_norm_target']}dBFS, Applied Gain: {gain_to_apply:.2f}dB)")
                    else:
                         status_log.append(f"\n  Skipped Final Peak Normalization (Already at or above target)")

            except Exception as e:
                err_msg = f"Peak Normalization failed ({type(e).__name__}: {e})"
                norm_errors.append(err_msg); print(f"    Error: {err_msg}"); status_log.append(f"\n  ❌ Error during Peak Normalization: {err_msg}"); traceback.print_exc()
        else:
            status_log.append("\n  Skipped Final Peak Normalization (pydub not available)")
    return processed_audio, norm_errors


def _export_final_audio(
    audio: AudioSegment,
    params: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str]]:
    """Export the final audio segment to a file."""
    try:
        FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        error_msg = f"Error creating output directory '{FINAL_OUTPUT_DIR}' ({type(e).__name__}: {e})"
        return None, error_msg

    final_filename = f"conversation_processed_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.{params['export_format']}"
    final_output_path = str(FINAL_OUTPUT_DIR / final_filename)

    export_args: Dict[str, Any] = {"format": params['export_format']}
    if params['bitrate_str']:
        export_args["bitrate"] = params['bitrate_str']

    print(f"Exporting final audio as {params['export_format'].upper()} to:\n'{final_output_path}'...")
    try:
        audio.export(final_output_path, **export_args)
        return final_output_path, None # Success
    except Exception as e:
        err_type = type(e).__name__
        error_msg = f"Error exporting final audio ({err_type}: {e})"
        print(error_msg)
        traceback.print_exc()
        return None, error_msg


# --- Refactored Concatenation Function ---
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
    app_context: AppContext, # Add context
    progress=gr.Progress(track_tqdm=True)
) -> Iterator[Tuple[str, Optional[str]]]:
    """
    Concatenates selected audio versions, applies various processing steps,
    and exports the final audio file, using helper functions.
    """
    status_log = ["Starting concatenation & processing..."]
    final_output_path = None
    yield "\n".join(status_log), final_output_path

    # Check pydub availability (essential) via context
    if not app_context.pydub_available or AudioSegment is None:
        status_log.append("Error: Pydub library not available. Cannot proceed.")
        yield "\n".join(status_log), None
        return

    # 1. Validate Parameters
    error_msg, params = _validate_concat_params(
        output_format, mp3_bitrate,
        trim_thresh_dbfs_val, trim_min_silence_len_ms_val,
        compress_thresh_db, compress_ratio_val, compress_attack_ms, compress_release_ms,
        nr_strength_val,
        eq_low_gain, eq_mid_gain, eq_high_gain,
        peak_norm_target_dbfs,
        reverb_amount, pitch_shift, speed_factor
    )
    if error_msg:
        status_log.append(error_msg)
        yield "\n".join(status_log), None
        return
    status_log.append("Parameters validated.")
    yield "\n".join(status_log), None

    # 2. Collect Selected Files
    selected_files, warnings = _collect_selected_audio_files(parsed_script, all_options_state, selections_state)
    if warnings:
        status_log.append("\nWarnings during file collection:")
        status_log.extend([f"  - {w}" for w in warnings])
        print("Warnings:", warnings)
        yield "\n".join(status_log), None # Yield warnings
    if not selected_files:
        status_log.append("\n\nError: No valid audio files were selected or found.")
        yield "\n".join(status_log), None
        return
    status_log.append(f"\nCollected {len(selected_files)} valid audio files.")
    yield "\n".join(status_log), None

    # 3. Load and Process Segments (Normalization, Trimming)
    status_log.append(f"\nLoading and processing {len(selected_files)} segments...")
    print(status_log[-1])
    yield "\n".join(status_log), None
    processed_segments: List[AudioSegment] = []
    loading_desc = "Loading/Processing Segments" # Simplified description
    # Use context flag
    if trim_silence_flag and not app_context.pydub_silence_available:
        status_log.append("\nWarning: Silence trimming enabled but pydub.silence not available. Skipping trim.")
        print(" Warning: Skipping trim (module unavailable).")
        trim_silence_flag = False # Disable flag if module missing

    for i, path_str in enumerate(progress.tqdm(selected_files, desc=loading_desc)):
        segment = _load_and_process_segment(
            path_str, i + 1, normalize_segments_flag, trim_silence_flag, params, status_log, app_context # Pass context
        )
        if segment is not None:
            processed_segments.append(segment)
        # Yield status updates periodically? Maybe not necessary per segment.

    if not processed_segments:
        status_log.append("\n\nError: Failed to load any valid segments after processing.")
        yield "\n".join(status_log), None
        return
    status_log.append(f"\nSuccessfully loaded and processed {len(processed_segments)} segments.")
    yield "\n".join(status_log), None

    # 4. Concatenate Segments
    status_log.append(f"\nConcatenating {len(processed_segments)} segments...")
    print(status_log[-1])
    yield "\n".join(status_log), None
    combined_audio = _concatenate_segments(processed_segments)
    if combined_audio is None:
        status_log.append("\n\nError during concatenation or result was empty.")
        yield "\n".join(status_log), None
        return
    status_log.append("\nConcatenation successful.")
    yield "\n".join(status_log), None

    # 5. Apply Global Effects
    status_log.append("\nApplying post-processing effects to combined audio...")
    print(status_log[-1])
    yield "\n".join(status_log), None
    processed_audio, effect_errors = _apply_global_effects(
        combined_audio, params, apply_nr_flag, apply_compression_flag, status_log, app_context # Pass context
    )
    # status_log is updated inside _apply_global_effects
    yield "\n".join(status_log), None

    # 6. Apply Final Normalization
    processed_audio, norm_errors = _apply_final_normalization(
        processed_audio, apply_peak_norm_flag, params, status_log, app_context # Pass context
    )
    processing_errors = effect_errors + norm_errors
    # status_log is updated inside _apply_final_normalization
    yield "\n".join(status_log), None

    # 7. Export Final Audio
    status_log.append(f"\n\nExporting final audio as {params['export_format'].upper()}...")
    yield "\n".join(status_log), None
    final_output_path, export_error = _export_final_audio(processed_audio, params)

    if export_error:
        status_log.append(f"\n\n❌ {export_error}")
        yield "\n".join(status_log), None
    else:
        final_summary = "\n\n✅ Processing & Export successful!";
        if processing_errors:
            final_summary += f"\n  Encountered {len(processing_errors)} non-fatal error(s) during post-processing:"
            for proc_err in processing_errors:
                final_summary += f"\n    - {proc_err}"
        status_log.append(final_summary)
        print(final_summary)
        yield "\n".join(status_log), final_output_path


# --- Helper functions for UI event handlers ---
def navigate_and_display(
    direction: int,
    current_index: int,
    parsed_script: ParsedScript,
    all_options_state: AllOptionsState,
    selections_state: SelectionsState,
    edited_texts: EditedTextsState,
    app_context: AppContext # Add context
):
    """Navigates lines and updates the review display, passing context."""
    new_index = current_index + direction
    # Pass context to display_line_for_review
    review_yield_tuple = display_line_for_review(new_index, parsed_script, all_options_state, selections_state, edited_texts, app_context)
    can_proceed_update, can_concat_btn_update = enable_concatenation_buttons(parsed_script, all_options_state, selections_state)
    return review_yield_tuple + (can_proceed_update, can_concat_btn_update)

def update_fixed_seed_visibility(strategy):
    needs_fixed_seed = (strategy == SEED_STRATEGY_FIXED_BASE_SEQUENTIAL or \
                        strategy == SEED_STRATEGY_FIXED_BASE_REUSED_LIST)
    return gr.update(visible=needs_fixed_seed)

# --- Advanced Audio Preview Function ---
def process_advanced_effects_preview(
    test_audio_path,
    enable_pitch_correction, pitch_correction_strength, pitch_correction_mode,
    enable_chorus, chorus_depth, chorus_rate, chorus_mix,
    enable_flanger, flanger_depth, flanger_rate, flanger_feedback, flanger_mix,
    enable_noise_gate, noise_gate_threshold, noise_gate_attack, noise_gate_release,
    gain_db_advanced, # Added gain input
    enable_graphical_eq, *eq_band_gains
):
    # This function doesn't seem to rely on the shared context (TTS, models, etc.)
    # It primarily uses audio_utils and pydub directly.
    # If audio_utils functions were refactored to take context, this would need updating.
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

# --- Timeline Tab Functions ---

def prepare_timeline_data(
    parsed_script: ParsedScript,
    selections_state: SelectionsState,
    edited_texts_state: EditedTextsState
) -> Tuple[List[str], Optional[str]]:
    """
    Prepares a list of strings for the timeline dropdown, representing each line
    with its speaker, selected version, and a preview of its text.
    """
    if not parsed_script:
        return [], None

    timeline_display_strings = []
    for idx, line_info in enumerate(parsed_script):
        speaker = line_info.get('speaker_filename', 'Unknown Speaker')
        # Prefer edited text if available, otherwise use original
        text_to_display = edited_texts_state.get(idx, line_info.get('text', ''))
        text_preview = text_to_display[:30].strip() + ("..." if len(text_to_display) > 30 else "")

        selected_version_idx = selections_state.get(idx, -1)
        version_display = f"V{selected_version_idx + 1}" if selected_version_idx != -1 else "N/A"

        display_string = f"L{idx + 1}: {speaker} ({version_display}) - '{text_preview}'"
        timeline_display_strings.append(display_string)

    default_selection = timeline_display_strings[0] if timeline_display_strings else None
    return timeline_display_strings, default_selection

def load_timeline_line_details(
    selected_line_str: Optional[str],
    parsed_script: ParsedScript,
    all_options_state: AllOptionsState,
    selections_state: SelectionsState,
    edited_texts_state: EditedTextsState,
    selected_seed_data_state: dict,
    app_context: AppContext # Required by signature, though not used in this specific logic yet
) -> Tuple[str, str, str, Optional[str], str, str]:
    """
    Loads details for a selected line from the timeline_line_selector_dd.
    Outputs: original_speaker, original_text, editable_text, audio_path, seed_text, status_text
    """
    default_return = "", "", "", None, "N/A", "No line selected or script empty."
    if not selected_line_str or not parsed_script:
        return default_return

    try:
        # Example: "L1: Speaker (V1) - 'Text...'" -> Extract "1"
        match = re.match(r"L(\d+):", selected_line_str)
        if not match:
            return "", "", "", None, "N/A", "Error parsing line identifier from selection."
        line_idx = int(match.group(1)) - 1 # Convert to 0-based index
    except (ValueError, TypeError):
        return "", "", "", None, "N/A", "Error parsing line index from selection."

    if not (0 <= line_idx < len(parsed_script)):
        return "", "", "", None, "N/A", f"Error: Parsed line index {line_idx+1} is out of bounds."

    line_info = parsed_script[line_idx]
    original_speaker = line_info.get('speaker_filename', 'Unknown Speaker')
    original_text = line_info.get('text', '')
    editable_text = edited_texts_state.get(line_idx, original_text)

    selected_version_idx = selections_state.get(line_idx, -1)
    audio_path: Optional[str] = None

    if selected_version_idx != -1:
        if line_idx < len(all_options_state) and \
           isinstance(all_options_state[line_idx], list) and \
           selected_version_idx < len(all_options_state[line_idx]):
            potential_path = all_options_state[line_idx][selected_version_idx]
            if potential_path and isinstance(potential_path, str) and Path(potential_path).is_file():
                audio_path = potential_path
            else:
                print(f"Warning: Audio path for L{line_idx+1} V{selected_version_idx+1} ('{potential_path}') is invalid or file missing.")
        else:
            print(f"Warning: Could not retrieve audio path for L{line_idx+1} V{selected_version_idx+1} due to state mismatch.")


    seed_text = "N/A"
    if isinstance(selected_seed_data_state, dict) and str(line_idx) in selected_seed_data_state:
        seed_info = selected_seed_data_state[str(line_idx)]
        if isinstance(seed_info, dict) and 'seed' in seed_info:
            seed_text = str(seed_info['seed'])

    status_text = f"Details loaded for Line {line_idx + 1} ({original_speaker})."
    if not audio_path:
        status_text += " (Warning: Audio file not found for selected version)"

    return original_speaker, original_text, editable_text, audio_path, seed_text, status_text


def regenerate_timeline_audio_segment(
    selected_line_str: Optional[str],
    current_edited_text: str,
    parsed_script: ParsedScript,
    all_options_state: AllOptionsState,
    selections_state: SelectionsState,
    edited_texts_state: EditedTextsState, # Added to update edited text
    selected_seed_data_state: dict,
    temperature_val: float,
    top_p_val: float,
    top_k_val: int,
    app_context: AppContext,
    progress=gr.Progress(track_tqdm=True)
) -> Iterator[Tuple[str, Optional[str], AllOptionsState, SelectionsState, EditedTextsState, dict]]:
    """
    Regenerates a single audio segment for a line selected in the timeline,
    using its current edited text and potentially its original seed.
    Updates the all_options_state and selected_seed_data_state.
    Yields: status_message, new_audio_path_or_none, updated_all_options, updated_selections, updated_edited_texts, updated_selected_seeds
    """
    # Initial yield with original states
    yield ("Starting regeneration...", None, all_options_state, selections_state, edited_texts_state, selected_seed_data_state)

    if not selected_line_str:
        yield ("Error: No line selected for regeneration.", None, all_options_state, selections_state, edited_texts_state, selected_seed_data_state)
        return
    if not current_edited_text or not current_edited_text.strip():
        yield ("Error: Text for regeneration cannot be empty.", None, all_options_state, selections_state, edited_texts_state, selected_seed_data_state)
        return
    if app_context.tts is None:
        yield ("Error: TTS model not initialized.", None, all_options_state, selections_state, edited_texts_state, selected_seed_data_state)
        return

    try:
        match = re.match(r"L(\d+):", selected_line_str)
        if not match:
            yield (f"Error: Could not parse line index from '{selected_line_str}'.", None, all_options_state, selections_state, edited_texts_state, selected_seed_data_state)
            return
        line_idx = int(match.group(1)) - 1
    except (ValueError, TypeError):
        yield (f"Error: Invalid line index parsed from '{selected_line_str}'.", None, all_options_state, selections_state, edited_texts_state, selected_seed_data_state)
        return

    if not (0 <= line_idx < len(parsed_script)):
        yield (f"Error: Line index {line_idx + 1} is out of bounds.", None, all_options_state, selections_state, edited_texts_state, selected_seed_data_state)
        return

    line_info = parsed_script[line_idx]
    speaker_filename = line_info.get('speaker_filename')
    if not speaker_filename:
        yield (f"Error: Speaker filename not found for Line {line_idx + 1}.", None, all_options_state, selections_state, edited_texts_state, selected_seed_data_state)
        return

    speaker_path = SPEAKER_DIR / speaker_filename
    if not speaker_path.is_file():
        yield (f"Error: Speaker file '{speaker_path}' not found.", None, all_options_state, selections_state, edited_texts_state, selected_seed_data_state)
        return

    seed_to_use = -1 # Default to random seed
    seed_found = False
    if str(line_idx) in selected_seed_data_state and 'seed' in selected_seed_data_state[str(line_idx)]:
        try:
            seed_to_use = int(selected_seed_data_state[str(line_idx)]['seed'])
            seed_found = True
            status_msg = f"Using original seed: {seed_to_use} for Line {line_idx + 1}."
        except (ValueError, TypeError):
            status_msg = f"Warning: Could not parse seed for Line {line_idx + 1}. Using random seed."
            seed_to_use = -1 # Fallback to random if parsing fails
    else:
        status_msg = f"No seed found for Line {line_idx + 1}. Using random seed."

    yield (status_msg, None, all_options_state, selections_state, edited_texts_state, selected_seed_data_state)

    selected_version_idx = selections_state.get(line_idx, -1)
    if selected_version_idx == -1:
        # This case should ideally be prevented by UI logic if regenerating a selected line's audio.
        # If it occurs, we might need to decide how to handle it, e.g., pick the first slot or error out.
        # For now, let's assume this means we are creating a *new* version for a line that had no version,
        # or replacing the first slot if it's ambiguous.
        # This part of the logic might need refinement based on exact UI flow for "timeline regeneration".
        # If the timeline always shows a *selected* version, this index should always be valid.
        # Let's assume for now it means "replace the currently selected version slot".
        # If no version was selected, this regeneration is problematic.
        # However, the `load_timeline_line_details` implies a version is already selected to be loaded.
        # So, selected_version_idx should be valid.
        yield (f"Error: No version slot selected for Line {line_idx + 1}. Cannot determine where to save regenerated audio.", None, all_options_state, selections_state, edited_texts_state, selected_seed_data_state)
        return

    # Ensure temp directory exists
    TEMP_CONVO_MULTI_DIR.mkdir(parents=True, exist_ok=True)
    output_filename_stem = f"line{line_idx:03d}_spk-{speaker_path.stem}_timeline_regen_s{seed_to_use if seed_to_use != -1 else random.randint(0, 2**31-1)}"
    output_filename = str(TEMP_CONVO_MULTI_DIR / f"{output_filename_stem}.wav")

    try:
        progress(0.5, desc=f"Generating audio for L{line_idx+1}...")
        app_context.tts.infer(
            str(speaker_path),
            current_edited_text,
            output_filename,
            temperature=temperature_val,
            top_p=top_p_val,
            top_k=int(top_k_val),
            seed=seed_to_use
        )
        progress(1.0, desc=f"Audio generated for L{line_idx+1}")

        if Path(output_filename).is_file() and Path(output_filename).stat().st_size > 0:
            # Create copies of states to modify
            all_options_copy = [list(opts) if isinstance(opts, list) else [] for opts in all_options_state]
            # Ensure the line_idx exists and is a list
            while line_idx >= len(all_options_copy): all_options_copy.append([None] * MAX_VERSIONS_ALLOWED)
            if not isinstance(all_options_copy[line_idx], list): all_options_copy[line_idx] = [None] * MAX_VERSIONS_ALLOWED
            # Ensure the version_idx is within bounds for that line
            while selected_version_idx >= len(all_options_copy[line_idx]): all_options_copy[line_idx].append(None)

            all_options_copy[line_idx][selected_version_idx] = output_filename

            edited_texts_copy = edited_texts_state.copy()
            edited_texts_copy[line_idx] = current_edited_text # Update edited text

            selected_seed_data_copy = selected_seed_data_state.copy()
            seed_entry = selected_seed_data_copy.get(str(line_idx), {})
            seed_entry.update({
                "seed": seed_to_use,
                "selected_version_path": output_filename, # Update path
                "text": current_edited_text # Update text in seed log
            })
            selected_seed_data_copy[str(line_idx)] = seed_entry

            yield (f"Regeneration successful for Line {line_idx + 1}. Seed used: {seed_to_use}.", output_filename, all_options_copy, selections_state, edited_texts_copy, selected_seed_data_copy)
        else:
            yield (f"Regeneration failed for Line {line_idx + 1}: Output file not created or empty.", None, all_options_state, selections_state, edited_texts_state, selected_seed_data_state)

    except Exception as e:
        error_message = f"Error during TTS inference for Line {line_idx + 1}: {type(e).__name__} - {e}"
        traceback.print_exc()
        yield (error_message, None, all_options_state, selections_state, edited_texts_state, selected_seed_data_state)
