# -*- coding: utf-8 -*-
# general_utils.py

# --- Imports ---
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import json # Add JSON import

# Check pydub availability for functions that depend on it
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE_GEN = True
except ImportError:
    PYDUB_AVAILABLE_GEN = False
    AudioSegment = None # type: ignore

# --- Constants (Copied/Derived from webui.py) ---
# Required by list_speaker_files
SPEAKER_DIR = Path("./speakers")
SPEAKER_FILE_EXTENSIONS = [".wav", ".mp3"]
NO_SPEAKER_OPTION = "[No Speaker Selected]"
# --- ADDED SAVE DIRECTORY CONSTANT ---
SAVE_DIR = Path("./project_saves")


# Type definitions from webui.py needed for signatures
ParsedScript = List[Dict[str, str]]
AllOptionsState = List[List[Optional[str]]]
SelectionsState = Dict[int, int]

# --- Utility Functions ---

def list_speaker_files() -> tuple[list[str], list[str]]:
    """Lists speaker files from the SPEAKER_DIR."""
    speaker_files = [NO_SPEAKER_OPTION]; speaker_filenames_only = []
    if SPEAKER_DIR.is_dir():
        try:
            # Sort alphabetically, case-insensitive
            sorted_items = sorted(SPEAKER_DIR.iterdir(), key=lambda p: str(p).lower())
            for item in sorted_items:
                if item.is_file() and item.suffix.lower() in SPEAKER_FILE_EXTENSIONS:
                    speaker_files.append(item.name); speaker_filenames_only.append(item.name)
        except OSError as e: print(f"Warning: Error scanning speaker directory '{SPEAKER_DIR}': {e}")
    else:
        print(f"Warning: Speaker directory '{SPEAKER_DIR}' not found.");
        try: SPEAKER_DIR.mkdir(parents=True, exist_ok=True); print(f"Created speaker directory: {SPEAKER_DIR}")
        except OSError as e: print(f"Error creating speaker directory '{SPEAKER_DIR}': {e}")
    return speaker_files, speaker_filenames_only

# --- ADDED FUNCTION TO LIST SAVE FILES ---
def list_save_files() -> List[str]:
    """Lists .json files in the SAVE_DIR."""
    save_files = []
    if SAVE_DIR.is_dir():
        try:
            for item in sorted(SAVE_DIR.glob('*.json'), key=lambda p: str(p).lower()):
                if item.is_file():
                    save_files.append(item.name) # Return only filename
        except OSError as e:
            print(f"Warning: Error scanning save directory '{SAVE_DIR}': {e}")
    else:
        print(f"Warning: Save directory '{SAVE_DIR}' not found. Creating it.")
        try:
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Error creating save directory '{SAVE_DIR}': {e}")
    return save_files
# --- END ADDED FUNCTION ---

def prepare_temp_dir(dir_path: Path) -> bool:
    """Clears or creates the specified temporary directory."""
    try:
        if dir_path.exists():
            print(f"Clearing old files from {dir_path}...")
            count = 0;
            for item in dir_path.iterdir():
                 if item.is_file():
                     try: item.unlink(); count+=1
                     except OSError as e: print(f"  Warn: Could not delete {item.name}: {e}") # More specific warning
            print(f"Cleared {count} old files.")
        else: dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Using temporary directory: {dir_path}")
        return True
    except OSError as e: print(f"Error preparing temporary directory '{dir_path}': {e}"); return False

def split_text_simple(text: str) -> List[str]:
    """Splits text into non-empty lines."""
    if not text: return []; lines = text.strip().split('\n'); chunks = [line.strip() for line in lines if line.strip()]; print(f"Split text by newline into {len(chunks)} chunks."); return chunks

def check_all_selections_valid(parsed_script: ParsedScript, all_options_state: AllOptionsState, selections_state: SelectionsState) -> bool:
    """Checks if a valid, existing audio file is selected for every line."""
    if not isinstance(parsed_script, list) or not isinstance(all_options_state, list) or not isinstance(selections_state, dict) or len(parsed_script) != len(all_options_state): return False
    if not parsed_script: return False # No lines means not ready
    for i in range(len(parsed_script)):
        sel_idx = selections_state.get(i, -1) # Use -1 to indicate no selection yet
        if sel_idx == -1: return False # If any line has no selection, fail
        if not isinstance(all_options_state[i], list) or not (0 <= sel_idx < len(all_options_state[i])): return False # Selection index out of bounds for options
        path = all_options_state[i][sel_idx]
        try:
            if not path or not isinstance(path, str): return False # Path missing or not string
            p = Path(path)
            # Check if file exists AND is not empty
            if not p.is_file() or p.stat().st_size == 0:
                # Optionally add a print here if debugging missing files
                # print(f"DEBUG check_all_selections_valid: File invalid/missing: {p}")
                return False
        except Exception as e:
            print(f"Debug check_all_selections_valid: Error checking path '{path}': {e}")
            return False # Error checking file path
    return True # All lines have a valid file selected

def enable_concatenation_buttons(parsed_script: ParsedScript, all_options_state: AllOptionsState, selections_state: SelectionsState) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Determines interactability for Proceed and Concatenate buttons."""
    # Requires Gradio >= 3.0 (which you have)
    import gradio as gr # Import locally to avoid circular dependency if utils used elsewhere

    can_concat = check_all_selections_valid(parsed_script, all_options_state, selections_state)
    print(f"Checking concat readiness: {can_concat}")
    interactive_state = can_concat and PYDUB_AVAILABLE_GEN
    # Returns updates for both proceed_to_concat_button and concatenate_convo_button
    return gr.update(interactive=interactive_state), gr.update(interactive=interactive_state)