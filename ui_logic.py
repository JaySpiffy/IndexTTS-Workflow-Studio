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


# --- Timeline Tab Logic ---
def update_timeline_with_selection(
    current_line_index: int,
    parsed_script: ParsedScript,
    selections_state: SelectionsState,
    all_options_state: AllOptionsState,
    selected_seed_data_state: dict,
    edited_texts_state: EditedTextsState,
    app_context: AppContext
) -> Tuple[Any, str, str, str, Optional[str], str, str]:
    """
    Updates the Timeline tab UI components based on the currently selected line
    and its chosen version from the Review tab.
    """
    status = "Ready."
    line_selector_choices = []
    original_speaker = "N/A"
    original_text = "N/A"
    editable_text = ""
    selected_audio_path = None
    selected_seed_text = "N/A"

    if not isinstance(parsed_script, list) or not parsed_script:
        status = "Error: No script loaded to populate timeline."
        print(status)
        return (gr.update(choices=[], value=None, interactive=False), 
                original_speaker, 
                original_text, 
                editable_text, 
                selected_audio_path, 
                selected_seed_text, 
                status)

    num_lines = len(parsed_script)
    line_selector_choices = [f"Line {i+1}: {line['speaker_filename']}" for i, line in enumerate(parsed_script)]

    # Ensure current_line_index is valid, default to 0 if not
    if not isinstance(current_line_index, int) or not (0 <= current_line_index < num_lines):
        current_line_index = 0
        status = f"Warning: Invalid line index, defaulting to Line 1."

    # Get data for the current line
    line_info = parsed_script[current_line_index]
    original_speaker = line_info.get('speaker_filename', 'Unknown Speaker')
    original_text = line_info.get('text', 'No Text')
    editable_text = edited_texts_state.get(current_line_index, original_text)

    # Get selected version and its path
    selected_version_index = selections_state.get(current_line_index, -1)
    if selected_version_index != -1 and isinstance(all_options_state, list) and current_line_index < len(all_options_state):
        line_options = all_options_state[current_line_index]
        if isinstance(line_options, list) and selected_version_index < len(line_options):
            selected_audio_path = line_options[selected_version_index]

    # Get seed data
    seed_data = selected_seed_data_state.get(str(current_line_index))  # Keys are strings in JSON
    if seed_data and isinstance(seed_data, dict) and 'seed' in seed_data:
        selected_seed_text = str(seed_data['seed'])

    return (gr.update(choices=line_selector_choices, 
                     value=line_selector_choices[current_line_index], 
                     interactive=True),
            original_speaker,
            original_text,
            editable_text,
            selected_audio_path,
            selected_seed_text,
            status)


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
        line_version_seeds = [(base_seed + line_offset + j) % (2^32) for j in range(num_versions)]
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
        line_version_seeds = [(base_seed + j) % (2^32) for j in range(num_versions)]
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
            # Use context for model and device
            score = analyze_speaker_similarity(app_context.speaker_similarity_model, speaker_path_str, initial_path, device=app_context.device.type)

            if score != -1.0 and score < SIMILARITY_THRESHOLD:
                status_log.append(f"    🔄 Low similarity on V{j+1} (Score: {score:.2f} < {SIMILARITY_THRESHOLD:.2f}). Triggering auto-regeneration...")
                print(status_log[-1])
                # yield status update here if needed

                regen_success = False
                for attempt in range(AUTO_REGEN_ATTEMPTS):
                    new_seed = random.randint(0, 2^32 - 1)
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
        base_seed = random.randint(0, 2^32 - 1)
        status_log.append(f"Using Random Base Seed: {base_seed}")
        print(f"Using Random Base Seed: {base_seed}")

    if seed_strategy == SEED_STRATEGY_RANDOM_BASE_REUSED_LIST:
        reused_random_seed_list = random.sample(range(2^32), num_versions)
        seed_info_str = ', '.join([f'V{k+1}=R({s})' for k, s in enumerate(reused_random_seed_list)])
        status_log.append(f"Reused Random Seed List: [{seed_info_str}]")
        print(f"Reused Random Seed List: [{seed_info_str}]")
    elif seed_strategy == SEED_STRATEGY_FIXED_BASE_REUSED_LIST:
        reused_sequential_seed_list = [(base_seed + j) % (2^32) for j in range(num_versions)]
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
                current_se
