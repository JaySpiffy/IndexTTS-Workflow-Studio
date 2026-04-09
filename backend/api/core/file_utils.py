"""
File Utilities module for standalone FastAPI implementation.
Handles file operations, directory management, and validation without Gradio dependencies.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from .app_paths import (
    PROJECT_SAVES_DIR,
    SPEAKERS_DIR,
    TEMP_CONVERSATION_SEGMENTS_DIR,
)

# Constants
SPEAKER_DIR = SPEAKERS_DIR
print(f"DEBUG: file_utils.py SPEAKER_DIR initialized to: {SPEAKER_DIR}")
NO_SPEAKER_OPTION = "[No Speaker Selected]"
TEMP_CONVO_MULTI_DIR = TEMP_CONVERSATION_SEGMENTS_DIR
SAVE_DIR = PROJECT_SAVES_DIR

def list_speaker_files() -> Tuple[List[str], List[str]]:
    """
    List all speaker files in the speakers directory.
    
    Returns:
        Tuple: (display_names, file_names)
    """
    display_names = []
    file_names = []
    
    if not SPEAKER_DIR.exists():
        SPEAKER_DIR.mkdir(parents=True, exist_ok=True)
        return [NO_SPEAKER_OPTION], []
    
    for file_path in SPEAKER_DIR.glob("*.*"):
        if file_path.is_file():
            display_names.append(file_path.name)
            file_names.append(file_path.name)
    
    if not display_names:
        display_names = [NO_SPEAKER_OPTION]
    
    return display_names, file_names

def prepare_temp_dir(temp_dir) -> bool:
    """
    Prepare temporary directory for conversation generation.

    Args:
        temp_dir: Path or string path to temporary directory

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert to Path if it's a string
        if isinstance(temp_dir, str):
            temp_dir = Path(temp_dir)

        # Never remove the temp directory itself. In Docker this path can be a
        # bind mount, and deleting the mount root may fail with "device or
        # resource busy". Instead, ensure the directory exists and clear only
        # its contents.
        temp_dir.mkdir(parents=True, exist_ok=True)

        for child in temp_dir.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except Exception as child_error:
                print(f"Error removing temp entry {child}: {child_error}")
                return False

        return True
    except Exception as e:
        print(f"Error preparing temp directory {temp_dir}: {e}")
        return False

def split_text_simple(text: str, max_length: int = 200) -> List[str]:
    """
    Simple text splitting for demonstration.
    
    Args:
        text: Text to split
        max_length: Maximum length per segment
    
    Returns:
        List: Text segments
    """
    # This is a placeholder - in real implementation, use proper text segmentation
    words = text.split()
    segments = []
    current_segment = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_length and current_segment:
            segments.append(" ".join(current_segment))
            current_segment = []
            current_length = 0
        
        current_segment.append(word)
        current_length += len(word) + 1
    
    if current_segment:
        segments.append(" ".join(current_segment))
    
    return segments

def check_all_selections_valid(selections: Dict[int, int], total_lines: int) -> bool:
    """
    Check if all lines have valid selections.
    
    Args:
        selections: Dictionary of line selections
        total_lines: Total number of lines
    
    Returns:
        bool: True if all selections are valid
    """
    if not selections:
        return False
    
    for i in range(total_lines):
        if i not in selections or selections[i] < 0:
            return False
    
    return True

def list_save_files() -> List[str]:
    """
    List all saved project files.
    
    Returns:
        List: Saved file names
    """
    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        return []
    
    return [f.name for f in SAVE_DIR.glob("*.json") if f.is_file()]

def parse_conversation_script(script_text: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Parse conversation script with SpeakerFile.ext: Text format.
    
    Args:
        script_text: Conversation script text
    
    Returns:
        Tuple: (parsed_script, errors)
    """
    lines = script_text.strip().split('\n')
    parsed_script = []
    errors = []
    available_speakers = list_speaker_files()[1]
    
    # Debug logging for speaker validation
    print(f"DEBUG: parse_conversation_script called")
    print(f"DEBUG: Available speakers in backend: {available_speakers}")
    print(f"DEBUG: Script text lines: {lines}")
    
    for i, line in enumerate(lines):
        line_num = i + 1
        line = line.strip()
        
        if not line:
            continue
            
        match = re.match(r'^([^:]+):\s*(.*)$', line)
        if match:
            speaker_filename = match.group(1).strip()
            line_text = match.group(2).strip()
            
            print(f"DEBUG: Processing line {line_num}: speaker='{speaker_filename}', text='{line_text}'")
            
            if not line_text:
                errors.append(f"Line {line_num} skipped (no text)")
                continue
                
            # Enhanced speaker validation logging
            print(f"DEBUG: Checking if speaker '{speaker_filename}' exists in available speakers")
            print(f"DEBUG: Exact match: {speaker_filename in available_speakers}")
            print(f"DEBUG: With .wav extension: {speaker_filename + '.wav' in available_speakers}")
            print(f"DEBUG: Without .wav extension: {speaker_filename.replace('.wav', '') in [s.replace('.wav', '') for s in available_speakers]}")
            
            if speaker_filename not in available_speakers:
                # Try alternative matching
                speaker_with_wav = speaker_filename + '.wav'
                speaker_without_wav = speaker_filename.replace('.wav', '')
                available_without_wav = [s.replace('.wav', '') for s in available_speakers]
                
                if speaker_with_wav in available_speakers:
                    print(f"DEBUG: Found speaker by adding .wav extension: {speaker_with_wav}")
                    speaker_filename = speaker_with_wav
                elif speaker_without_wav in available_without_wav:
                    # Find the actual filename with extension
                    for available_speaker in available_speakers:
                        if available_speaker.replace('.wav', '') == speaker_without_wav:
                            print(f"DEBUG: Found speaker by name matching: {available_speaker}")
                            speaker_filename = available_speaker
                            break
                    else:
                        errors.append(f"Speaker file '{speaker_filename}' on line {line_num} not found")
                        continue
                else:
                    errors.append(f"Speaker file '{speaker_filename}' on line {line_num} not found")
                    continue
                
            print(f"DEBUG: Speaker validated successfully: {speaker_filename}")
                
            parsed_script.append({
                'speaker_filename': speaker_filename,
                'text': line_text,
                'line_number': line_num
            })
        else:
            errors.append(f"Invalid format on line {line_num}. Expected 'SpeakerFile.ext: Text'")
    
    print(f"DEBUG: Final parsed script: {parsed_script}")
    print(f"DEBUG: Errors: {errors}")
    return parsed_script, errors

def validate_speaker_files(parsed_script: List[Dict[str, str]]) -> List[str]:
    """
    Validate that all speaker files in parsed script exist.
    
    Args:
        parsed_script: Parsed conversation script
    
    Returns:
        List: Error messages
    """
    errors = []
    available_speakers = list_speaker_files()[1]
    
    for line_info in parsed_script:
        speaker_filename = line_info['speaker_filename']
        if speaker_filename not in available_speakers:
            errors.append(f"Speaker file '{speaker_filename}' not found for line {line_info['line_number']}")
    
    return errors

def list_available_speakers(speakers_dir: str = str(SPEAKER_DIR)) -> List[str]:
    """
    List all available speaker files in the speakers directory.
    
    Args:
        speakers_dir: Speakers directory path
    
    Returns:
        List: Speaker file names
    """
    if not os.path.exists(speakers_dir):
        return []
    
    import glob
    speaker_files = glob.glob(os.path.join(speakers_dir, "*.wav"))
    speaker_names = [os.path.basename(f).replace('.wav', '') for f in speaker_files]
    return sorted(speaker_names)
