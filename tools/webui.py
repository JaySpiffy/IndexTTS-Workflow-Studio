# -*- coding: utf-8 -*-
import gradio as gr
import os
import tempfile
import uuid
import re
import time
import random
import json # Import json for save/load
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Iterator, Union, cast
import traceback # Keep for error logging
import math # For checking -inf dBFS

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
    from audio_utils import apply_eq, apply_noise_reduction
except ImportError as e:
    print(f"ERROR importing from audio_utils.py: {e}. Ensure file exists and dependencies are met.")
    def apply_eq(*args, **kwargs): raise NotImplementedError("audio_utils import failed")
    def apply_noise_reduction(*args, **kwargs): raise NotImplementedError("audio_utils import failed")

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

# Type definitions for state
ParsedScript = List[Dict[str, str]]
AllOptionsState = List[List[Optional[str]]]
SelectionsState = Dict[int, int]


# --- Initialize TTS ---
try:
    if 'IndexTTS' in globals() and IndexTTS is not None:
        tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml", is_fp16=True)
        print("IndexTTS Initialized Successfully.")
    else:
        tts = None
except Exception as e:
    print(f"CRITICAL Error initializing IndexTTS: {e}")
    traceback.print_exc()
    tts = None


# --- Gradio UI Functions ---

# Single Generation
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

# Multi-Version Generation
ConvoGenYield = Tuple[str, ParsedScript, AllOptionsState, int, dict, dict, dict]
def parse_validate_and_start_convo(script_text: str, num_versions_str: str, temperature_val: float, top_p_val: float, top_k_val: int, seed_val: Optional[Union[str, int, float]], randomize_seed_checkbox: bool, progress=gr.Progress(track_tqdm=True)) -> Iterator[ConvoGenYield]:
    status_log = ["Starting Conversation Multi-Version Generation..."]; yield "\n".join(status_log), [], [], -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
    if tts is None: status_log.append("Error: TTS model not initialized."); yield "\n".join(status_log), [], [], -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
    try:
        num_versions = int(num_versions_str);
        if not (1 <= num_versions <= MAX_VERSIONS_ALLOWED): raise ValueError(f"Versions must be 1-{MAX_VERSIONS_ALLOWED}.")
    except (ValueError, TypeError): status_log.append(f"Error: Invalid number of versions '{num_versions_str}'."); yield "\n".join(status_log), [], [], -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
    if not script_text or not script_text.strip(): status_log.append("Error: Input script text cannot be empty."); yield "\n".join(status_log), [], [], -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
    if not prepare_temp_dir(TEMP_CONVO_MULTI_DIR): status_log.append(f"Error: Failed to prepare temp directory {TEMP_CONVO_MULTI_DIR}."); yield "\n".join(status_log), [], [], -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
    status_log.append(f"Using temp directory: {TEMP_CONVO_MULTI_DIR}")
    status_log.append("Parsing script and validating speakers..."); yield "\n".join(status_log), [], [], -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
    lines = script_text.strip().split('\n'); parsed_script: ParsedScript = []; available_speaker_files = list_speaker_files()[1]
    for i, line in enumerate(lines):
        line_num = i + 1; line = line.strip();
        if not line: continue
        match = re.match(r'^([^:]+):\s*(.*)$', line)
        if match:
            speaker_filename = match.group(1).strip(); line_text = match.group(2).strip();
            if not line_text: status_log.append(f"Warning: Line {line_num} skipped (no text)."); continue
            if speaker_filename not in available_speaker_files: err = f"Error: Speaker file '{speaker_filename}' on line {line_num} not found in ./speakers/. Cannot proceed."; status_log.append(err); print(err); yield "\n".join(status_log), [], [], -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
            parsed_script.append({'speaker_filename': speaker_filename, 'text': line_text})
        else: err = f"Error: Invalid format on line {line_num}. Expected 'SpeakerFile.ext: Text'. Cannot proceed."; status_log.append(err); print(err); yield "\n".join(status_log), [], [], -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
    if not parsed_script: status_log.append("Error: No valid lines found in the script after parsing."); yield "\n".join(status_log), [], [], -1, gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=False); return # type: ignore
    total_lines = len(parsed_script); status_log.append(f"Script parsed: {total_lines} valid lines found."); yield "\n".join(status_log), parsed_script, [], -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
    base_seed = 0; effective_randomize = randomize_seed_checkbox; seed_status_lines = []
    try:
        if seed_val is None or str(seed_val).strip() == "" or seed_val == -1: base_seed = random.randint(0, 2**32 - 1); effective_randomize = True; seed_status_lines.append("Using random seeds.")
        else: base_seed = int(seed_val);
        if not effective_randomize: seed_status_lines.append(f"Using base seed: {base_seed} for sequential.")
        else: seed_status_lines.append(f"Using random seeds (randomize checked)."); base_seed = random.randint(0, 2**32 - 1)
    except (ValueError, TypeError): base_seed = random.randint(0, 2**32 - 1); effective_randomize = True; seed_status_lines.append(f"Invalid seed '{seed_val}', using random seeds.")
    version_seeds = []; seed_info = ""
    if effective_randomize: version_seeds = random.sample(range(2**32), num_versions); seed_info = ', '.join([f'V{i+1}={s}' for i, s in enumerate(version_seeds)])
    else: version_seeds = [(base_seed + i) % (2**32) for i in range(num_versions)]; seed_info = ', '.join([f'V{i+1}={s}' for i, s in enumerate(version_seeds)])
    seed_status_lines.append(f"Seeds: {seed_info}"); status_log.extend(seed_status_lines); print("\n".join(seed_status_lines)); yield "\n".join(status_log), parsed_script, [], -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
    all_options: AllOptionsState = []; generation_successful = True
    for i, line_info in enumerate(progress.tqdm(parsed_script, desc="Generating Lines")):
        line_idx_0_based = i
        speaker_filename = line_info['speaker_filename']; line_text = line_info['text']; speaker_path_str = str(SPEAKER_DIR / speaker_filename)
        line_options_generated: List[Optional[str]] = [None] * num_versions
        status_current = f"\nGenerating Line {line_idx_0_based + 1}/{total_lines} ({speaker_filename}, {num_versions} versions)..."; print(status_current); status_log.append(status_current)
        yield "\n".join(status_log), parsed_script, all_options, -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
        for j in range(num_versions):
            version_idx_1_based = j + 1; current_seed = version_seeds[j]
            segment_filename = f"line{line_idx_0_based:03d}_spk-{Path(speaker_filename).stem}_v{j+1:02d}_s{current_seed}.wav";
            output_path = str(TEMP_CONVO_MULTI_DIR / segment_filename)
            print(f"  Generating V{version_idx_1_based} (Seed: {current_seed})... Text: '{line_text[:30]}...'")
            try:
                tts.infer(speaker_path_str, line_text, output_path, temperature=temperature_val, top_p=top_p_val, top_k=int(top_k_val), seed=current_seed)
                if Path(output_path).is_file() and Path(output_path).stat().st_size > 0: line_options_generated[j] = output_path; print(f"    V{version_idx_1_based} OK: {segment_filename}")
                else: print(f"    V{version_idx_1_based} FAILED (file not found or empty)."); status_log.append(f"  Warning: Line {line_idx_0_based + 1}, V{version_idx_1_based} failed (file missing/empty).")
            except Exception as e: err_msg = f"Error L{line_idx_0_based + 1} V{version_idx_1_based}: Failed during TTS inference: {type(e).__name__}: {e}"; print(f"    {err_msg}"); status_log.append(f"\n❌ {err_msg}"); traceback.print_exc(); generation_successful = False; break
        all_options.append(line_options_generated); yield "\n".join(status_log), parsed_script, all_options, -1, gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False) # type: ignore
        if not generation_successful: break
    if generation_successful: status_log.append(f"\n✅ Generation Complete for {total_lines} lines."); status_log.append("Proceed to 'Review & Select Lines' tab."); review_interactive = True if total_lines > 0 else False
    else: status_log.append(f"\n❌ Generation stopped or cancelled."); review_interactive = False
    print(status_log[-2]);
    if len(status_log) > 1: print(status_log[-1])
    yield "\n".join(status_log), parsed_script, all_options, 0, gr.update(interactive=True), gr.update(interactive=review_interactive), gr.update(interactive=False) # type: ignore

# Review Tab Display
ReviewYield = Tuple[str, str, str, str, int, dict, dict, *([dict]*MAX_VERSIONS_ALLOWED), dict] # Added str for editable text
def display_line_for_review( target_index: int, parsed_script: ParsedScript, all_options_state: AllOptionsState, selections_state: SelectionsState ) -> ReviewYield:
    status = ""; audio_updates = [gr.update(value=None, interactive=False, label=f"{VERSION_PREFIX}{i+1}") for i in range(MAX_VERSIONS_ALLOWED)]; radio_update = gr.update(choices=[], value=None, interactive=False) # type: ignore
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

    line_info = parsed_script[target_index]; original_speaker = line_info['speaker_filename']; original_text = line_info['text']
    nav_display = f"Line {target_index + 1} / {num_lines}"; line_text_display = f"Speaker: {original_speaker}\nText: {original_text}"; editable_text_display = original_text
    current_line_options = all_options_state[target_index] if target_index < len(all_options_state) and isinstance(all_options_state[target_index], list) else []
    valid_labels = []; num_actual_versions = 0

    # --- Loop to check files and update UI ---
    for i in range(MAX_VERSIONS_ALLOWED):
        path = current_line_options[i] if i < len(current_line_options) else None
        label = f"{VERSION_PREFIX}{i+1}"
        if path and isinstance(path, str):
            try: # Indent Level 1
                p = Path(path) # Indent Level 2
                if p.is_file() and p.stat().st_size > 0: # Indent Level 2
                    audio_updates[i] = gr.update(value=str(p), interactive=True, label=label) # Indent Level 3
                    valid_labels.append(label) # Indent Level 3
                    num_actual_versions += 1 # Indent Level 3
                else: # Indent Level 2
                    print(f"Warning: File missing/empty for Line {target_index+1} V{i+1}: {path}") # Indent Level 3
                    audio_updates[i] = gr.update(value=None, interactive=False, label=f"{label} (Not Found)") # Indent Level 3
            except Exception as e: # Indent Level 1 (Matches the try)
                print(f"Error checking file for Line {target_index+1} V{i+1} ({path}): {e}") # Indent Level 2
                audio_updates[i] = gr.update(value=None, interactive=False, label=f"{label} (Error)") # Indent Level 2
        else: # Indent Level 1 (Matches the `if path...`)
            is_option_expected = i < len(current_line_options) # Indent Level 2
            audio_updates[i] = gr.update(value=None, interactive=False, label=f"{label} ({'Not Found' if is_option_expected else 'Not Generated'})") # Indent Level 2
    # --- End Loop ---

    current_selection_index = selections_state.get(target_index, -1); current_selection_label = None; radio_interactive = bool(valid_labels)
    if valid_labels:
        is_current_selection_valid = False
        # --- Separate Try/Except for checking the selected file ---
        if 0 <= current_selection_index < len(current_line_options): # Indent 2
            path = current_line_options[current_selection_index] # Indent 3
            if path and isinstance(path, str): # Indent 3
                try: # Indent 4
                    p = Path(path) # Indent 5
                    if p.is_file() and p.stat().st_size > 0: # Indent 5
                        is_current_selection_valid = True # Indent 6
                except Exception: # Indent 4
                    pass # Indent 5 (Ignore errors checking file here, just means selection is not valid)
        # --- End separate Try/Except ---

        if is_current_selection_valid and f"{VERSION_PREFIX}{current_selection_index + 1}" in valid_labels: # Indent 2
            current_selection_label = f"{VERSION_PREFIX}{current_selection_index + 1}" # Indent 3
        else: # Indent 2
            if valid_labels: # Indent 3
                current_selection_label = valid_labels[0] # Indent 4
                try: # Indent 4
                    new_selection_index = int(valid_labels[0][len(VERSION_PREFIX):]) - 1 # Indent 5
                    if selections_state.get(target_index) != new_selection_index: # Indent 5
                        selections_state[target_index] = new_selection_index # Indent 6
                        print(f"Info: Setting default selection for line {target_index+1} to V{new_selection_index+1}") # Indent 6
                except (ValueError, IndexError): # Indent 4
                    pass # Indent 5
            else: # Indent 3 (Matches `if valid_labels:`)
                 current_selection_label = None # Indent 4
                 if target_index in selections_state: # Indent 4 (Check before deleting)
                    del selections_state[target_index] # Indent 5

    radio_update = gr.update(choices=valid_labels, value=current_selection_label, interactive=radio_interactive); prev_interactive = (target_index > 0); next_interactive = (target_index < num_lines - 1)
    if not status: status = f"Displaying {num_actual_versions} versions for Line {target_index + 1}.";
    if not valid_labels: status += " (Warning: No valid audio files found for this line!)"; print(status)
    return status, nav_display, line_text_display, editable_text_display, target_index, gr.update(interactive=prev_interactive), gr.update(interactive=next_interactive), *audio_updates, radio_update # type: ignore


# Review Tab Selection Update
def update_convo_selection(choice_input: Optional[str], current_line_index: int, selections_state: SelectionsState) -> SelectionsState:
    if not isinstance(current_line_index, int) or current_line_index < 0: print(f"Warn update_sel: invalid index {current_line_index}"); return selections_state
    if choice_input is None: return selections_state
    if not isinstance(selections_state, dict): selections_state = {}
    choice_index: int = -1
    if isinstance(choice_input, str) and choice_input.startswith(VERSION_PREFIX):
        try: choice_index = int(choice_input[len(VERSION_PREFIX):]) - 1
        except (ValueError, IndexError): print(f"Error update_sel: Cannot parse choice '{choice_input}'"); return selections_state
    else: print(f"Warn update_sel: Invalid choice format '{choice_input}'"); return selections_state
    if not (0 <= choice_index < MAX_VERSIONS_ALLOWED): print(f"Error update_sel: Parsed index {choice_index} out of bounds"); return selections_state
    try:
        if selections_state.get(current_line_index) != choice_index: updated_selections = selections_state.copy(); updated_selections[current_line_index] = choice_index; print(f"Selection state updated for Line {current_line_index + 1}: Now V{choice_index + 1}"); return updated_selections
        else: return selections_state
    except Exception as e: print(f"Error update_sel dict: {type(e).__name__}: {e}"); return selections_state

# Function to Regenerate a Single Line
RegenLineYield = Tuple[str, AllOptionsState, SelectionsState]
def regenerate_single_line( current_line_index: int, parsed_script: ParsedScript, all_options_state: AllOptionsState, selections_state: SelectionsState, editable_text: str, num_versions_str: str, temperature_val: float, top_p_val: float, top_k_val: int, seed_val: Optional[Union[str, int, float]], randomize_seed_checkbox: bool, progress=gr.Progress(track_tqdm=True) ) -> Iterator[RegenLineYield]:
    line_idx_0_based = current_line_index; status = f"Starting regeneration for Line {line_idx_0_based + 1}..."; all_options_copy = [list(opts) if opts else [] for opts in all_options_state]; selections_copy = selections_state.copy(); yield status, all_options_copy, selections_copy
    if tts is None: yield "Error: TTS model not initialized.", all_options_copy, selections_copy; return
    if not isinstance(parsed_script, list) or not (0 <= line_idx_0_based < len(parsed_script)): yield f"Error: Invalid script data or line index ({line_idx_0_based}).", all_options_copy, selections_copy; return
    if not isinstance(all_options_copy, list) or line_idx_0_based >= len(all_options_copy): yield f"Error: Options state invalid for line index ({line_idx_0_based}).", all_options_copy, selections_copy; return
    if not editable_text or not editable_text.strip(): yield f"Error: Editable text for Line {line_idx_0_based + 1} cannot be empty.", all_options_copy, selections_copy; return
    line_info = parsed_script[line_idx_0_based]; speaker_filename = line_info['speaker_filename']; speaker_path = SPEAKER_DIR / speaker_filename
    if not speaker_path.is_file(): yield f"Error: Speaker file '{speaker_filename}' not found for Line {line_idx_0_based + 1}.", all_options_copy, selections_copy; return
    speaker_path_str = str(speaker_path)
    try: # Indent 1
        num_versions = int(num_versions_str); # Indent 2
        if not (1 <= num_versions <= MAX_VERSIONS_ALLOWED): # Indent 2
            raise ValueError(f"Versions must be 1-{MAX_VERSIONS_ALLOWED}.") # Indent 3
    except (ValueError, TypeError): # Indent 1
        yield f"Error: Invalid number of versions '{num_versions_str}' selected on Tab 1.", all_options_copy, selections_copy; return # Indent 2
    base_seed = 0; effective_randomize = randomize_seed_checkbox; seed_status_lines = []
    try:
        if seed_val is None or str(seed_val).strip() == "" or seed_val == -1: base_seed = random.randint(0, 2**32 - 1); effective_randomize = True; seed_status_lines.append("Using random seeds for regeneration.")
        else: base_seed = int(seed_val);
        if not effective_randomize: seed_status_lines.append(f"Using base seed {base_seed} (sequential) for regeneration.")
        else: seed_status_lines.append("Using random seeds (randomize checked) for regeneration."); base_seed = random.randint(0, 2**32 - 1)
    except (ValueError, TypeError): base_seed = random.randint(0, 2**32 - 1); effective_randomize = True; seed_status_lines.append(f"Invalid seed '{seed_val}' on Tab 1, using random seeds for regeneration.")
    version_seeds = []; seed_info = "";
    if effective_randomize: version_seeds = random.sample(range(2**32), num_versions); seed_info = ', '.join([f'V{i+1}={s}' for i, s in enumerate(version_seeds)])
    else: line_offset = line_idx_0_based * num_versions * 10; version_seeds = [(base_seed + line_offset + i) % (2**32) for i in range(num_versions)]; seed_info = ', '.join([f'V{i+1}={s}' for i, s in enumerate(version_seeds)])
    status += f"\nUsing seeds: {seed_info}"; print(f"Regenerating Line {line_idx_0_based + 1} - Seeds: {seed_info}"); yield status, all_options_copy, selections_copy
    new_line_options: List[Optional[str]] = [None] * num_versions; generation_successful = True; line_status = f"\nRegenerating {num_versions} versions for Line {line_idx_0_based + 1} ({speaker_filename})..."; status += line_status; print(line_status); yield status, all_options_copy, selections_copy
    for j in progress.tqdm(range(num_versions), desc=f"Regen Line {line_idx_0_based+1}"):
        version_idx_1_based = j + 1; current_seed = version_seeds[j]; segment_filename = f"line{line_idx_0_based:03d}_spk-{Path(speaker_filename).stem}_v{j+1:02d}_s{current_seed}.wav"; output_path = str(TEMP_CONVO_MULTI_DIR / segment_filename)
        print(f"  Regenerating V{version_idx_1_based} (Seed: {current_seed})... Text: '{editable_text[:30]}...'")
        try:
            if Path(output_path).exists():
                 try: Path(output_path).unlink()
                 except OSError as del_e: print(f"    Warn: Could not delete existing file {output_path}: {del_e}")
            tts.infer(speaker_path_str, editable_text, output_path, temperature=temperature_val, top_p=top_p_val, top_k=int(top_k_val), seed=current_seed)
            if Path(output_path).is_file() and Path(output_path).stat().st_size > 0: new_line_options[j] = output_path; print(f"    V{version_idx_1_based} OK: {segment_filename}")
            else: print(f"    V{version_idx_1_based} FAILED (file not found or empty after generation)."); status += f"\n  Warning: Line {line_idx_0_based + 1}, V{version_idx_1_based} failed (file missing/empty)."
        except Exception as e: err_msg = f"Regen L{line_idx_0_based+1} V{version_idx_1_based}: Failed during TTS inference: {type(e).__name__}: {e}"; print(f"    {err_msg}"); status += f"\n❌ {err_msg}"; traceback.print_exc(); generation_successful = False # Track if *any* version fails
    # Check if *at least one* version was successfully generated
    if any(p is not None for p in new_line_options):
         all_options_copy[line_idx_0_based] = new_line_options
         # Find the index of the first successfully generated version
         first_valid_index = -1
         for idx, opt_path in enumerate(new_line_options):
             if opt_path and Path(opt_path).is_file() and Path(opt_path).stat().st_size > 0:
                 first_valid_index = idx
                 break
         # Only update selection if a valid version was created
         if first_valid_index != -1:
             selections_copy[line_idx_0_based] = first_valid_index
         else: # No valid version created, maybe remove selection? Or keep old? Let's remove.
            if line_idx_0_based in selections_copy:
                del selections_copy[line_idx_0_based]
         status += f"\n✅ Regeneration finished for Line {line_idx_0_based + 1}."
         # Check if *all* requested versions were successful
         if not all(p is not None and Path(p).is_file() and Path(p).stat().st_size > 0 for p in new_line_options):
             status += " (with errors)"
             generation_successful = False # Explicitly mark as not fully successful
         print(f"Regeneration finished for Line {line_idx_0_based + 1}.")
         yield status, all_options_copy, selections_copy # Yield updated state
    else:
        status += f"\n❌ Regeneration completely failed for Line {line_idx_0_based + 1}. No files generated.";
        print(f"Regeneration completely failed for Line {line_idx_0_based + 1}.")
        # Yield the *original* state if regeneration totally failed
        yield status, all_options_state, selections_state

# Main Concatenation Function
def concatenate_conversation_versions(
    parsed_script: ParsedScript, all_options_state: AllOptionsState, selections_state: SelectionsState,
    output_format: str, mp3_bitrate: str,
    normalize_segments_flag: bool, # NEW: Flag for per-segment normalization
    trim_silence_flag: bool, trim_thresh_dbfs_val: float, trim_min_silence_len_ms_val: int,
    apply_compression_flag: bool, compress_thresh_db: float, compress_ratio_val: float, compress_attack_ms: float, compress_release_ms: float,
    apply_nr_flag: bool, nr_strength_val: float,
    eq_low_gain: float, eq_mid_gain: float, eq_high_gain: float,
    apply_peak_norm_flag: bool, peak_norm_target_dbfs: float,
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
        _final_norm_target = float(peak_norm_target_dbfs) # Final peak target
        _segment_norm_target = DEFAULT_SEGMENT_NORM_TARGET_DBFS # Per-segment target
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

    # --- MODIFIED LOADING LOOP ---
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

            # --- NEW: Per-Segment Normalization ---
            if normalize_segments_flag:
                current_peak = segment.max_dBFS
                # Check if peak is valid (not silent)
                if not math.isinf(current_peak):
                    gain_needed = _segment_norm_target - current_peak
                    segment = segment.apply_gain(gain_needed)
                    print(f"    Normalized Line {i+1} (Peak: {current_peak:.2f}dBFS -> {_segment_norm_target:.2f}dBFS, Gain: {gain_needed:.2f}dB)")
                else:
                    print(f"    Skipped normalization for Line {i+1} (appears silent)")
            # --- END: Per-Segment Normalization ---

            # Silence Trimming (applied after potential normalization)
            if trim_silence_flag:
                print(f"    Trimming Line {i+1} (Thresh:{_trim_thresh}dB, MinLen:{_trim_len}ms, Keep:{_trim_keep}ms)...");
                nonsilent_ranges = _silence_Imported.detect_nonsilent(segment, min_silence_len=_trim_len, silence_thresh=_trim_thresh);
                if nonsilent_ranges:
                    start_trim = max(0, nonsilent_ranges[0][0] - _trim_keep);
                    end_trim = min(len(segment), nonsilent_ranges[-1][1] + _trim_keep)
                else:
                    start_trim, end_trim = 0, 0
                if end_trim > start_trim:
                    original_len = len(segment);
                    segment = segment[start_trim:end_trim];
                    print(f"      Trimmed segment {i+1} from {original_len}ms to {len(segment)}ms")
                elif nonsilent_ranges:
                    print(f"    Warn: Trim resulted in empty segment for Line {i+1}, using original (normalized).")
                else:
                    print(f"    Warn: Segment for Line {i+1} detected as entirely silent, using original (normalized).")

            processed_segments.append(segment) # Append original or processed segment
        except Exception as e:
            warn = f"Warning L{i+1}: Error loading/processing '{os.path.basename(path_str)}' ({type(e).__name__}: {e}). Skipping segment.";
            status += f"\n  {warn}"; print(f"    {warn}"); traceback.print_exc()
    # --- END MODIFIED LOADING LOOP ---

    if not processed_segments: status += "\n\nError: Failed to load any valid segments after processing."; yield status, None; return
    status += f"\nConcatenating {len(processed_segments)} segments..."; print(status.split('\n')[-1]); yield status, None; combined = AudioSegment.empty(); silence_between = AudioSegment.silent(duration=300)
    try:
        # Use crossfade during concatenation to smooth transitions
        crossfade_duration = 10 # milliseconds
        print(f"  Using {crossfade_duration}ms crossfade between segments.")
        combined = processed_segments[0] # Start with the first segment
        for i in range(1, len(processed_segments)):
            # Add silence with crossfade
            combined = combined.append(silence_between, crossfade=crossfade_duration)
            # Add next segment with crossfade
            combined = combined.append(processed_segments[i], crossfade=crossfade_duration)
    except Exception as e: status += f"\n\nError during concatenation ({type(e).__name__}: {e})"; yield status, None; return

    if len(combined) == 0: status += "\n\nError: Concatenation resulted in empty audio."; yield status, None; return
    status += "\nApplying post-processing effects to combined audio..."; print(status.split('\n')[-1]); yield status, None; processed_audio = combined; processing_errors = []
    if apply_nr_flag:
        if NOISEREDUCE_AVAILABLE:
            try: processed_audio = apply_noise_reduction(processed_audio, _nr_strength); status += f"\n  Applied Noise Reduction (Strength: {_nr_strength:.2f})"
            except Exception as e: err_msg = f"Noise Reduction failed ({type(e).__name__}: {e})"; processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status += f"\n  ❌ Error during Noise Reduction: {err_msg}"; traceback.print_exc()
        else: status += "\n  Skipped Noise Reduction (Library not available)"
    if _eq_low != 0 or _eq_mid != 0 or _eq_high != 0:
        if SCIPY_AVAILABLE:
            try: processed_audio = apply_eq(processed_audio, _eq_low, _eq_mid, _eq_high); status += f"\n  Applied EQ (L:{_eq_low}, M:{_eq_mid}, H:{_eq_high} dB)"
            except Exception as e: err_msg = f"EQ failed ({type(e).__name__}: {e})"; processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status += f"\n  ❌ Error during EQ: {err_msg}"; traceback.print_exc()
        else: status += "\n  Skipped EQ (Library not available)"
    if apply_compression_flag:
        if PYDUB_COMPRESS_AVAILABLE and _compress_dynamic_range_Imported:
            try:
                print(f"  Applying Compression (Thresh:{_compress_thresh}dB, Ratio:{_compress_ratio:.1f}:1, Att:{_compress_attack}ms, Rel:{_compress_release}ms)...");
                processed_audio = _compress_dynamic_range_Imported( processed_audio, threshold=_compress_thresh, ratio=_compress_ratio, attack=_compress_attack, release=_compress_release );
                status += f"\n  Applied Compression (Thresh:{_compress_thresh}, Ratio:{_compress_ratio:.1f})"
            except Exception as e: err_msg = f"Compression failed ({type(e).__name__}: {e})"; processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status += f"\n  ❌ Error during Compression: {err_msg}"; traceback.print_exc()
        else: status += "\n  Skipped Compression (pydub effects component not available)"
    if apply_peak_norm_flag:
        if PYDUB_AVAILABLE:
            try:
                print(f"  Applying Final Peak Normalization (Target: {_final_norm_target}dBFS)..."); current_peak_dbfs = processed_audio.max_dBFS
                if not math.isinf(current_peak_dbfs): # Check if not silent
                    gain_to_apply = _final_norm_target - current_peak_dbfs;
                    # Only apply gain reduction in the final step
                    if gain_to_apply < -0.01: # Small tolerance
                        processed_audio = processed_audio.apply_gain(gain_to_apply);
                        status += f"\n  Applied Final Peak Normalization (Target: {_final_norm_target}dBFS, Applied Gain: {gain_to_apply:.2f}dB)"
                    else:
                        status += f"\n  Skipped Final Peak Normalization (Audio already at or below target: {current_peak_dbfs:.2f}dBFS)";
                        print(f"    Skipping Final Peak Normalization (Current Peak: {current_peak_dbfs:.2f}dBFS, Target: {_final_norm_target}dBFS)")
                else:
                    status += "\n  Skipped Final Peak Normalization (Audio appears silent)";
                    print("    Skipping Final Peak Normalization - audio appears silent.")
            except Exception as e: err_msg = f"Peak Normalization failed ({type(e).__name__}: {e})"; processing_errors.append(err_msg); print(f"    Error: {err_msg}"); status += f"\n  ❌ Error during Peak Normalization: {err_msg}"; traceback.print_exc()
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
    parsed_script_state = gr.State([]); all_options_state = gr.State([]); selections_state = gr.State({}); current_line_index_state = gr.State(0)
    with gr.Accordion("Generation Parameters (Used by All Generators)", open=True):
        with gr.Row(): temperature_slider = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=1.0, label="Temperature"); top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=0.8, label="Top-P"); top_k_slider = gr.Slider(minimum=1, maximum=100, step=1, value=30, label="Top-K")
    with gr.Tabs() as tabs:
        with gr.TabItem("1. Generate Conversation Lines", id="tab_convo_gen"):
             gr.Markdown("Enter script using `SpeakerFile.ext: Text` format. Set versions/seed, then Generate.")
             with gr.Row():
                 with gr.Column(scale=2): script_input_convo = gr.Textbox(label="Conversation Script", lines=15, placeholder="speaker1.wav: Line 1\nspeaker2.wav: Line 2..."); list_speakers_btn = gr.Button("List Available Speakers"); available_speakers_display = gr.Textbox(label="Available Speaker Files", interactive=False, lines=5)
                 with gr.Column(scale=1): num_versions_convo_radio = gr.Radio(label="Versions per Line", choices=[str(i) for i in range(1, MAX_VERSIONS_ALLOWED + 1)], value=str(MAX_VERSIONS_ALLOWED), interactive=True);
                 with gr.Accordion("Seed Control", open=False): seed_input_convo = gr.Number(label="Base Seed (Optional, -1 for random)", value=-1); randomize_seed_checkbox = gr.Checkbox(label="Force Random Seeds (Overrides Base Seed)", value=True);
                 generate_convo_button = gr.Button("Generate All Lines & Versions", variant="primary")
             convo_gen_status_output = gr.Textbox(label="Generation Status", lines=8, interactive=False, max_lines=20); gr.Markdown("<small>*(During generation, a 'Cancel' button will appear next to the progress bar)*</small>", visible=True)
        with gr.TabItem("2. Review & Select Lines", id="tab_review", interactive=False) as review_tab:
             gr.Markdown("### Review and Select Best Version for Each Line");
             with gr.Row(): prev_line_button = gr.Button("<< Previous Line", interactive=False); regenerate_current_line_button = gr.Button("🔄 Regenerate Current Line", interactive=True); line_nav_display = gr.Markdown("Line 0 / 0"); next_line_button = gr.Button("Next Line >>", interactive=False)
             current_line_display_review = gr.Textbox(label="Current Line Info (Original)", interactive=False, lines=4); editable_line_text_review = gr.Textbox( label="Editable Text for Regeneration", lines=4, interactive=True, placeholder="Edit the text here before clicking Regenerate Current Line..." ); gr.Markdown(f"Listen to the versions below and select the best one:")
             review_audio_outputs = [];
             with gr.Column():
                 for i in range(MAX_VERSIONS_ALLOWED): audio_player = gr.Audio( label=f"Version {i+1}", type="filepath", interactive=False, visible=True, elem_id=f"review_audio_{i}" ); review_audio_outputs.append(audio_player)
             line_choice_radio = gr.Radio(label="Select Best Version", choices=[], interactive=False, value=None); review_status_output = gr.Textbox(label="Review Status", lines=1, interactive=False); proceed_to_concat_button = gr.Button("Proceed to Concatenate Tab ->", interactive=False)
        with gr.TabItem("3. Concatenate & Export", id="tab_concat", interactive=False, visible=PYDUB_AVAILABLE) as concat_tab:
             gr.Markdown("### Concatenate Selected Lines & Apply Post-Processing");
             with gr.Row():
                 with gr.Column(scale=1):
                     with gr.Accordion("Output Format", open=True): output_format_dropdown = gr.Dropdown(label="Output Format", choices=OUTPUT_FORMAT_CHOICES, value=DEFAULT_OUTPUT_FORMAT, interactive=True); mp3_bitrate_dropdown = gr.Dropdown(label="MP3 Bitrate (kbps)", choices=MP3_BITRATE_CHOICES, value=DEFAULT_MP3_BITRATE, interactive=False, visible=(DEFAULT_OUTPUT_FORMAT=="mp3"))
                     # --- NEW: Per-Segment Normalization Accordion ---
                     with gr.Accordion("Per-Segment Normalization (Applied BEFORE Concat/Trim)", open=True):
                         normalize_segments_checkbox = gr.Checkbox(label=f"Enable (Normalize each line to {DEFAULT_SEGMENT_NORM_TARGET_DBFS}dBFS peak)", value=True, interactive=PYDUB_AVAILABLE)
                         if not PYDUB_AVAILABLE: gr.Markdown("<small>*(Requires pydub)*</small>")
                     # --- END: New Accordion ---
                     with gr.Accordion("Silence Trimming (Applied AFTER Segment Norm, BEFORE Concat)", open=False): trim_silence_checkbox = gr.Checkbox( label=f"Enable", value=False, interactive=PYDUB_SILENCE_AVAILABLE); trim_threshold_input = gr.Number(label="Trim Threshold (dBFS, lower is stricter)", value=DEFAULT_TRIM_SILENCE_THRESH_DBFS, interactive=PYDUB_SILENCE_AVAILABLE); trim_length_input = gr.Number(label="Trim Min Silence (ms)", value=DEFAULT_TRIM_MIN_SILENCE_LEN_MS, minimum=50, step=50, precision=0, interactive=PYDUB_SILENCE_AVAILABLE);
                     if not PYDUB_SILENCE_AVAILABLE: gr.Markdown("<small>*(Requires pydub silence component)*</small>")
                     with gr.Accordion("Noise Reduction (Applied AFTER Concat)", open=False): apply_noise_reduction_checkbox = gr.Checkbox( label="Enable", value=False, interactive=NOISEREDUCE_AVAILABLE); noise_reduction_strength_slider = gr.Slider( label="Strength (0=off, 1=max)", minimum=0.0, maximum=1.0, value=0.85, step=0.05, interactive=NOISEREDUCE_AVAILABLE);
                     if not NOISEREDUCE_AVAILABLE: gr.Markdown("<small>*(Requires noisereduce, scipy, numpy)*</small>")
                     with gr.Accordion("Equalization (EQ) (Applied AFTER Concat)", open=False): eq_low_gain_input = gr.Slider(label="Low Gain (Shelf)", minimum=-12, maximum=12, value=0, step=0.5, interactive=SCIPY_AVAILABLE); eq_mid_gain_input = gr.Slider(label="Mid Gain (Peak)", minimum=-12, maximum=12, value=0, step=0.5, interactive=SCIPY_AVAILABLE); eq_high_gain_input = gr.Slider(label="High Gain (Shelf)", minimum=-12, maximum=12, value=0, step=0.5, interactive=SCIPY_AVAILABLE);
                     if not SCIPY_AVAILABLE: gr.Markdown("<small>*(Requires scipy & numpy)*</small>")
                     with gr.Accordion("Compression (Applied AFTER Concat)", open=False): apply_compression_checkbox = gr.Checkbox( label="Enable", value=False, interactive=PYDUB_COMPRESS_AVAILABLE); compress_threshold_input = gr.Slider(label="Threshold (dBFS)", minimum=-60, maximum=0, value=-20, step=1, interactive=PYDUB_COMPRESS_AVAILABLE); compress_ratio_input = gr.Slider(label="Ratio (N:1)", minimum=1.0, maximum=20.0, value=4.0, step=0.1, interactive=PYDUB_COMPRESS_AVAILABLE); compress_attack_input = gr.Slider(label="Attack (ms)", minimum=1, maximum=200, value=5, step=1, interactive=PYDUB_COMPRESS_AVAILABLE); compress_release_input = gr.Slider(label="Release (ms)", minimum=20, maximum=1000, value=100, step=10, interactive=PYDUB_COMPRESS_AVAILABLE);
                     if not PYDUB_COMPRESS_AVAILABLE: gr.Markdown("<small>*(Requires pydub effects component)*</small>")
                     with gr.Accordion("Final Peak Normalization (Applied LAST)", open=False): apply_peak_norm_checkbox = gr.Checkbox( label="Enable", value=True, interactive=PYDUB_AVAILABLE); peak_norm_target_input = gr.Number( label="Target Peak (dBFS)", value=DEFAULT_FINAL_NORM_TARGET_DBFS, minimum=-12.0, maximum=-0.1, step=0.1, interactive=PYDUB_AVAILABLE);
                     if not PYDUB_AVAILABLE: gr.Markdown("<small>*(Requires pydub)*</small>")
                 with gr.Column(scale=1): concatenate_convo_button = gr.Button("Concatenate & Process Selected Lines", variant="primary", interactive=False); concat_status_output = gr.Textbox(label="Concatenation & Processing Status", lines=15, interactive=False, max_lines=30); final_conversation_audio = gr.Audio(label="Final Output Audio", type="filepath", interactive=False)
        if not PYDUB_AVAILABLE:
             with gr.TabItem("Concatenate & Export (Disabled)"): gr.Markdown("### Feature Disabled\nRequires `pydub` library (`pip install pydub`) and `ffmpeg`.")

    # --- Event Handlers ---
    list_speakers_btn.click(lambda: "\n".join(list_speaker_files()[1]), inputs=None, outputs=available_speakers_display)
    review_display_outputs_base = [ review_status_output, line_nav_display, current_line_display_review, editable_line_text_review, current_line_index_state, prev_line_button, next_line_button, *review_audio_outputs, line_choice_radio ]
    review_display_outputs_nav_full = review_display_outputs_base + [proceed_to_concat_button, concatenate_convo_button]
    convo_gen_outputs = [ convo_gen_status_output, parsed_script_state, all_options_state, current_line_index_state, generate_convo_button, review_tab, concat_tab ]
    generate_convo_button.click( fn=prepare_temp_dir, inputs=gr.State(TEMP_CONVO_MULTI_DIR), outputs=None, queue=False ).then( fn=parse_validate_and_start_convo, inputs=[ script_input_convo, num_versions_convo_radio, temperature_slider, top_p_slider, top_k_slider, seed_input_convo, randomize_seed_checkbox ], outputs=convo_gen_outputs, show_progress="full" ).then( fn=display_line_for_review, inputs=[ current_line_index_state, parsed_script_state, all_options_state, selections_state ], outputs=review_display_outputs_base ).then( fn=enable_concatenation_buttons, inputs=[parsed_script_state, all_options_state, selections_state], outputs=[proceed_to_concat_button, concatenate_convo_button] )
    def navigate_and_display( direction: int, current_index: int, parsed_script: ParsedScript, all_options_state: AllOptionsState, selections_state: SelectionsState ):
        new_index = current_index + direction; review_yield_tuple = display_line_for_review(new_index, parsed_script, all_options_state, selections_state); can_proceed_update, can_concat_btn_update = enable_concatenation_buttons(parsed_script, all_options_state, selections_state); return review_yield_tuple + (can_proceed_update, can_concat_btn_update)
    prev_line_button.click( fn=navigate_and_display, inputs=[gr.State(-1), current_line_index_state, parsed_script_state, all_options_state, selections_state], outputs=review_display_outputs_nav_full, queue=False )
    next_line_button.click( fn=navigate_and_display, inputs=[gr.State(1), current_line_index_state, parsed_script_state, all_options_state, selections_state], outputs=review_display_outputs_nav_full, queue=False )
    line_choice_radio.change( fn=update_convo_selection, inputs=[line_choice_radio, current_line_index_state, selections_state], outputs=[selections_state], queue=False ).then( fn=enable_concatenation_buttons, inputs=[parsed_script_state, all_options_state, selections_state], outputs=[proceed_to_concat_button, concatenate_convo_button] )

    regenerate_outputs = [ review_status_output, all_options_state, selections_state ]
    regenerate_current_line_button.click(
        fn=lambda: gr.update(interactive=False),
        inputs=None, outputs=regenerate_current_line_button, queue=False
    ).then(
        fn=regenerate_single_line,
        inputs=[ current_line_index_state, parsed_script_state, all_options_state, selections_state, editable_line_text_review, num_versions_convo_radio, temperature_slider, top_p_slider, top_k_slider, seed_input_convo, randomize_seed_checkbox ],
        outputs=regenerate_outputs, show_progress="full"
    ).then(
        fn=display_line_for_review,
        inputs=[ current_line_index_state, parsed_script_state, all_options_state, selections_state ], outputs=review_display_outputs_base
    ).then(
        fn=enable_concatenation_buttons,
        inputs=[ parsed_script_state, all_options_state, selections_state ], outputs=[proceed_to_concat_button, concatenate_convo_button]
    ).then(
        fn=lambda: gr.update(interactive=True),
        inputs=None, outputs=regenerate_current_line_button, queue=False
    )

    proceed_to_concat_button.click( fn=lambda: (gr.update(interactive=True), gr.update(selected="tab_concat")), inputs=None, outputs=[concat_tab, tabs], queue=False )

    # Update concat_inputs to include the new checkbox
    concat_inputs = [
        parsed_script_state, all_options_state, selections_state,
        output_format_dropdown, mp3_bitrate_dropdown,
        normalize_segments_checkbox, # NEW
        trim_silence_checkbox, trim_threshold_input, trim_length_input,
        apply_compression_checkbox, compress_threshold_input, compress_ratio_input, compress_attack_input, compress_release_input,
        apply_noise_reduction_checkbox, noise_reduction_strength_slider,
        eq_low_gain_input, eq_mid_gain_input, eq_high_gain_input,
        apply_peak_norm_checkbox, peak_norm_target_input
    ]
    if PYDUB_AVAILABLE:
        concatenate_convo_button.click( fn=lambda: gr.update(interactive=False), inputs=None, outputs=concatenate_convo_button, queue=False ).then( fn=concatenate_conversation_versions, inputs=concat_inputs, outputs=[concat_status_output, final_conversation_audio], show_progress="full" ).then( fn=lambda: gr.update(interactive=True), inputs=None, outputs=concatenate_convo_button, queue=False )
        output_format_dropdown.change( fn=lambda fmt: gr.update(visible=(fmt.lower() == 'mp3') if isinstance(fmt, str) else False), inputs=output_format_dropdown, outputs=mp3_bitrate_dropdown, queue=False )

# --- Launch the App ---
if __name__ == "__main__":
    if tts is not None: print("Launching Gradio UI..."); SAVE_DIR.mkdir(parents=True, exist_ok=True); demo.launch(share=False, inbrowser=True)
    else: print("\n" + "="*50); print("❌ ERROR: Gradio UI cannot launch because the TTS model failed to initialize."); print("   Please check the console output above for specific errors during TTS setup."); print("   Common issues include missing model files or incorrect paths in config.yaml."); print("="*50 + "\n")