"""
Audio Processing module for standalone FastAPI implementation.
Handles audio manipulation, analysis, and speaker similarity without Gradio dependencies.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Generator
import traceback
import warnings

from .app_paths import PRETRAINED_MODELS_DIR, SOURCE_CLIPS_DIR, SPEAKERS_DIR

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# SpeechBrain availability flag
SPEECHBRAIN_AVAILABLE = False
speaker_similarity_model = None

# Try to import SpeechBrain
try:
    from speechbrain.inference import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
    print("SpeechBrain successfully imported")
except ImportError:
    print("SpeechBrain not available. Speaker similarity analysis will be disabled.")
except Exception as e:
    print(f"Error importing SpeechBrain: {e}")

# Try to import required libraries with fallbacks
try:
    import moviepy.editor as mp
    MOVIEPY_AVAILABLE = True
    # Suppress moviepy warnings
    warnings.filterwarnings("ignore", module="moviepy")
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: moviepy not available. Video extraction will be disabled.")

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("Warning: noisereduce not available. Noise reduction will be disabled.")

# Import SpeechBrain for vocal separation
try:
    from speechbrain.pretrained import SepformerSeparation as separator
    SPEECHBRAIN_SEPARATION_AVAILABLE = True
    print("SpeechBrain separation models available for vocal separation")
except ImportError:
    print("Warning: SpeechBrain not available for vocal separation")
    SPEECHBRAIN_SEPARATION_AVAILABLE = False
    separator = None
except Exception as e:
    print(f"Warning: Could not load SpeechBrain separation: {e}")
    SPEECHBRAIN_SEPARATION_AVAILABLE = False
    separator = None

try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Audio processing will be disabled.")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available. Audio processing features will be disabled.")
    AudioSegment = None

# Constants for speaker similarity
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_MODEL_SAVEDIR = str(PRETRAINED_MODELS_DIR / "spkrec-ecapa-voxceleb")
SIMILARITY_THRESHOLD = 0.60  # Default threshold for auto-regen
AUTO_REGEN_ATTEMPTS = 1     # Number of auto-regeneration attempts
SPEAKER_ANALYSIS_DEVICE = os.getenv("INDTEXTS_SPEAKER_ANALYSIS_DEVICE", "cpu").strip().lower()

def initialize_speaker_model(device: str = "cpu") -> bool:
    """
    Initialize the SpeechBrain speaker similarity model.
    
    Args:
        device: Device to use ("cuda" or "cpu")
    
    Returns:
        bool: True if successful, False otherwise
    """
    global speaker_similarity_model, SPEECHBRAIN_AVAILABLE
    
    if not SPEECHBRAIN_AVAILABLE:
        return False
        
    try:
        print(f"Loading speaker embedding model ({SPEAKER_MODEL_SOURCE})...")
        Path(SPEAKER_MODEL_SAVEDIR).mkdir(parents=True, exist_ok=True)
        
        requested_device = (device or "cpu").strip().lower()
        device_str = "cuda" if requested_device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        run_opts = {"device": device_str}
        
        speaker_similarity_model = EncoderClassifier.from_hparams(
            source=SPEAKER_MODEL_SOURCE,
            savedir=SPEAKER_MODEL_SAVEDIR,
            run_opts=run_opts
        )
        speaker_similarity_model.eval()
        print("Speaker embedding model loaded successfully.")
        return True
        
    except Exception as e:
        print(f"Error loading SpeechBrain speaker model: {e}")
        traceback.print_exc()
        speaker_similarity_model = None
        SPEECHBRAIN_AVAILABLE = False
        return False

def analyze_speaker_similarity(
    model: Any, 
    reference_audio_path: str, 
    generated_audio_path: str,
    device: str = "cpu"
) -> float:
    """
    Analyze speaker similarity between reference and generated audio.
    
    Args:
        model: SpeechBrain model instance
        reference_audio_path: Path to reference audio file
        generated_audio_path: Path to generated audio file
        device: Device to use for computation
    
    Returns:
        float: Similarity score between 0.0 and 1.0, or -1.0 on error
    """
    if model is None or not SPEECHBRAIN_AVAILABLE:
        return -1.0
        
    try:
        # Check if files exist
        if not Path(reference_audio_path).is_file() or not Path(generated_audio_path).is_file():
            return -1.0
            
        # Load audio files using torchaudio and process them properly
        try:
            import torchaudio
        except ImportError:
            print("Error: torchaudio is required for speaker similarity analysis")
            return -1.0
            
        # Target sampling rate expected by SpeechBrain models
        target_sr = 16000
        
        # Load and prepare reference audio
        ref_sig, ref_sr = torchaudio.load(reference_audio_path)
        if ref_sr != target_sr:
            ref_sig = torchaudio.functional.resample(ref_sig, ref_sr, target_sr)
        if ref_sig.shape[0] > 1:  # Ensure mono
            ref_sig = torch.mean(ref_sig, dim=0, keepdim=True)
        ref_sig = ref_sig.to(device)
        
        # Load and prepare generated audio
        gen_sig, gen_sr = torchaudio.load(generated_audio_path)
        if gen_sr != target_sr:
            gen_sig = torchaudio.functional.resample(gen_sig, gen_sr, target_sr)
        if gen_sig.shape[0] > 1:  # Ensure mono
            gen_sig = torch.mean(gen_sig, dim=0, keepdim=True)
        gen_sig = gen_sig.to(device)
        
        # Compute embeddings using the audio tensors (not file paths)
        with torch.no_grad():
            ref_embedding = model.encode_batch(ref_sig)
            gen_embedding = model.encode_batch(gen_sig)
        
        # Calculate cosine similarity between the embeddings
        similarity = torch.nn.functional.cosine_similarity(
            ref_embedding.squeeze(), gen_embedding.squeeze(), dim=0
        )
        
        return similarity.item()
        
    except Exception as e:
        print(f"Error in speaker similarity analysis: {e}")
        return -1.0

def batch_similarity_analysis(
    model: Any,
    reference_audio_path: str,
    generated_audio_paths: List[str],
    device: str = "cpu"
) -> List[Tuple[str, float]]:
    """
    Perform batch similarity analysis on multiple generated audio files.
    
    Args:
        model: SpeechBrain model instance
        reference_audio_path: Path to reference audio file
        generated_audio_paths: List of paths to generated audio files
        device: Device to use for computation
    
    Returns:
        List: (audio_path, similarity_score) tuples
    """
    results = []
    
    if model is None or not SPEECHBRAIN_AVAILABLE:
        return [(-1.0) for _ in generated_audio_paths]
        
    for audio_path in generated_audio_paths:
        score = analyze_speaker_similarity(model, reference_audio_path, audio_path, device)
        results.append((audio_path, score))
        
    return results

def ensure_dirs_exist() -> Tuple[bool, str]:
    """
    Ensure required directories exist.
    
    Returns:
        Tuple: (success, message)
    """
    try:
        SOURCE_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
        return True, "Directories ready"
    except Exception as e:
        return False, f"Error creating directories: {str(e)}"

def extract_audio_from_video(video_file: str, output_name: str) -> str:
    """
    Extract audio from video file and save to source_clips directory.
    
    Args:
        video_file: Path to video file
        output_name: Output filename
    
    Returns:
        str: Success/error message
    """
    if not MOVIEPY_AVAILABLE:
        return "Error: moviepy library not available"
    
    try:
        # Ensure directories exist
        success, msg = ensure_dirs_exist()
        if not success:
            return msg
        
        # Validate output name
        if not output_name.endswith('.wav'):
            output_name += '.wav'
        
        output_path = SOURCE_CLIPS_DIR / output_name
        
        # Extract audio using moviepy
        video = mp.VideoFileClip(video_file)
        audio = video.audio
        
        if audio is None:
            return "Error: No audio track found in video"
        
        # Write audio to file with high quality settings
        audio.write_audiofile(str(output_path), verbose=False, logger=None,
                            fps=44100, bitrate='192k')
        
        # Close the video and audio to free resources
        audio.close()
        video.close()
        
        return f"Success: Audio extracted to {output_name}"
        
    except Exception as e:
        return f"Error extracting audio: {str(e)}"

def trim_audio_segment(original_audio: str, output_name: str) -> str:
    """
    Trim audio segment (placeholder - would need time selection UI).
    
    Args:
        original_audio: Path to original audio file
        output_name: Output filename
    
    Returns:
        str: Success/error message
    """
    if not SCIPY_AVAILABLE:
        return "Error: scipy library not available"
    
    try:
        # Ensure directories exist
        success, msg = ensure_dirs_exist()
        if not success:
            return msg
        
        # Validate output name
        if not output_name.endswith('.wav'):
            output_name += '.wav'
        
        output_path = SOURCE_CLIPS_DIR / output_name
        
        # For now, just copy the file as a placeholder
        # In a real implementation, you'd need time selection UI
        import shutil
        shutil.copy2(original_audio, output_path)
        
        return f"Success: Audio copied to {output_name} (trimming UI not implemented)"
        
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def separate_vocals(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Separate vocals from background using SpeechBrain's Sepformer model.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
    
    Returns:
        np.ndarray: Separated vocal track
    """
    if not SPEECHBRAIN_SEPARATION_AVAILABLE or not torch.cuda.is_available():
        return audio_data
    
    try:
        # Load the separation model
        model = separator.from_hparams(
            source="speechbrain/sepformer-whamr",
                savedir=str(PRETRAINED_MODELS_DIR / "sepformer-whamr")
        )
        
        # Convert to tensor format expected by SpeechBrain
        audio_tensor = torch.tensor(audio_data).float()
        if len(audio_tensor.shape) == 1:
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [batch, channels, samples]
        else:
            audio_tensor = audio_tensor.unsqueeze(0)  # [batch, channels, samples]
        
        # Separate sources
        separated_sources = model.separate_batch(audio_tensor)
        
        # Return the first source (typically vocals)
        vocals = separated_sources[:, 0, :].squeeze().numpy()
        return vocals
        
    except Exception as e:
        print(f"Vocal separation failed: {e}")
        return audio_data

def process_all_source_clips(use_noise_reduction: bool = False,
                           use_vocal_separation: bool = False,
                           normalization_strength: float = 0.5,
                           noise_reduction_strength: float = 0.5) -> Generator[str, None, None]:
    """
    Process all source clips through vocal separation and normalization.
    
    Args:
        use_noise_reduction: Whether to apply noise reduction
        use_vocal_separation: Whether to use SpeechBrain for vocal separation
        normalization_strength: Strength of normalization (0.0 to 1.0)
        noise_reduction_strength: Strength of noise reduction (0.0 to 1.0)
    
    Yields:
        str: Progress updates
    """
    # Ensure directories exist
    success, msg = ensure_dirs_exist()
    if not success:
        yield f"Error: {msg}"
        return
    
    # Get list of files to process
    source_files = list(SOURCE_CLIPS_DIR.glob("*.wav"))
    speaker_files = {f.name for f in SPEAKERS_DIR.glob("*.wav")}
    
    # Filter files that haven't been processed yet
    files_to_process = [f for f in source_files if f.name not in speaker_files]
    
    if not files_to_process:
        yield "No new files to process"
        return
    
    yield f"Found {len(files_to_process)} files to process"
    yield f"Source files: {[f.name for f in source_files]}"
    yield f"Speaker files: {list(speaker_files)}"
    yield f"Files to process: {[f.name for f in files_to_process]}"
    
    for i, source_file in enumerate(files_to_process):
        try:
            yield f"Processing {source_file.name} ({i+1}/{len(files_to_process)})"
            yield f"File size: {source_file.stat().st_size} bytes"
            
            # Read audio file
            if SCIPY_AVAILABLE:
                sample_rate, audio_data = wavfile.read(source_file)
                yield f"Original: {sample_rate}Hz, shape: {audio_data.shape}, dtype: {audio_data.dtype}"
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    yield "Converting stereo to mono..."
                    audio_data = np.mean(audio_data, axis=1)
                    yield f"Mono shape: {audio_data.shape}"
                
                # Apply vocal separation if requested
                if use_vocal_separation:
                    yield f"Applying vocal separation using SpeechBrain..."
                    audio_data = separate_vocals(audio_data, sample_rate)
                    yield "Vocal separation completed"
                
                # Apply noise reduction if requested with controlled strength
                if use_noise_reduction and NOISEREDUCE_AVAILABLE:
                    yield f"Applying noise reduction (strength: {noise_reduction_strength:.2f})..."
                    audio_data = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=noise_reduction_strength)
                    yield "Noise reduction completed"
                
                # Apply normalization with controlled strength
                yield f"Applying normalization (strength: {normalization_strength:.2f})..."
                
                # Convert to float32 for safe processing
                if np.issubdtype(audio_data.dtype, np.integer):
                    # Convert integer data to float32 in [-1, 1] range
                    audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Apply normalization based on strength parameter
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:  # Avoid division by zero
                    # Scale normalization based on strength parameter
                    target_max = 0.1 + (0.9 * normalization_strength)  # Range: 0.1 to 1.0
                    if max_val < target_max:
                        audio_data = audio_data / max_val * target_max
                
                # Convert back to 16-bit PCM
                audio_data = (audio_data * 32767).astype(np.int16)
                
                # Save processed audio with consistent sample rate
                output_path = SPEAKERS_DIR / source_file.name
                wavfile.write(output_path, sample_rate, audio_data)  # Keep original sample rate
                yield f"Saved to: {output_path}"
                
                # Verify the file was created
                if output_path.exists():
                    yield f"✓ Successfully processed {source_file.name} -> {output_path.name}"
                    yield f"Output file size: {output_path.stat().st_size} bytes"
                else:
                    yield f"✗ ERROR: Output file {output_path} was not created!"
                
            else:
                # Fallback: just copy the file if scipy not available
                output_path = SPEAKERS_DIR / source_file.name
                shutil.copy2(source_file, output_path)
                yield f"✓ Copied {source_file.name} (scipy not available)"
                
        except Exception as e:
            yield f"✗ Error processing {source_file.name}: {str(e)}"
            yield f"Error details: {traceback.format_exc()}"
            continue
    
    # Final verification
    processed_files = list(SPEAKERS_DIR.glob("*.wav"))
    yield f"Processing complete! Final speaker files: {[f.name for f in processed_files]}"

def refresh_speaker_lists() -> Tuple[str, str]:
    """
    Refresh and return formatted lists of source clips and speakers.
    
    Returns:
        Tuple: (source_clips_list, speakers_list)
    """
    # Ensure directories exist
    ensure_dirs_exist()
    
    # Get source clips
    source_files = list(SOURCE_CLIPS_DIR.glob("*.wav"))
    source_list = "\n".join([f"• {f.name} ({f.stat().st_size / 1024:.1f} KB)" 
                           for f in sorted(source_files, key=lambda x: x.name)])
    
    if not source_list:
        source_list = "No source clips found"
    
    # Get speakers
    speaker_files = list(SPEAKERS_DIR.glob("*.wav"))
    speaker_list = "\n".join([f"• {f.name} ({f.stat().st_size / 1024:.1f} KB)" 
                            for f in sorted(speaker_files, key=lambda x: x.name)])
    
    if not speaker_list:
        speaker_list = "No speaker files found"
    
    return source_list, speaker_list

def check_audio_quality(audio_path: str) -> str:
    """
    Check basic audio quality metrics for debugging.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        str: Quality information string
    """
    if not SCIPY_AVAILABLE:
        return "scipy not available for quality checking"
    
    try:
        sample_rate, audio_data = wavfile.read(audio_path)
        
        info = f"Sample rate: {sample_rate} Hz\n"
        if len(audio_data.shape) > 1:
            info += f"Channels: {audio_data.shape[1]}\n"
            info += f"Duration: {len(audio_data) / sample_rate:.2f} seconds\n"
            info += f"Data type: {audio_data.dtype}\n"
            info += f"Max amplitude: {np.max(np.abs(audio_data))}\n"
        else:
            info += f"Channels: 1\n"
            info += f"Duration: {len(audio_data) / sample_rate:.2f} seconds\n"
            info += f"Data type: {audio_data.dtype}\n"
            info += f"Max amplitude: {np.max(np.abs(audio_data))}\n"
        
        return info
        
    except Exception as e:
        return f"Error checking audio quality: {str(e)}"

# Additional imports for robotic speech detection
try:
    import librosa
    import scipy.signal as signal
    import scipy.stats as stats
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available. Robotic speech detection will be disabled.")

def detect_robotic_speech(audio_path: str) -> float:
    """
    Detect robotic speech characteristics using spectral and prosodic analysis.
    Returns robotic score (0.0 = natural, 1.0 = highly robotic)
    """
    if not LIBROSA_AVAILABLE or not SCIPY_AVAILABLE:
        return 0.0  # Return neutral score if dependencies not available
    
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) == 0:
            return 0.0
            
        # 1. Pitch variability analysis (robotic speech has flat pitch)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        if len(pitch_values) > 0:
            pitch_variability = np.std(pitch_values) / np.mean(pitch_values)
            pitch_score = 1.0 - min(pitch_variability / 0.3, 1.0)  # Normalize
        else:
            pitch_score = 0.5
            
        # 2. Spectral flux analysis (robotic speech has abrupt transitions)
        hop_length = 512
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        flux_variability = np.std(spectral_flux) / np.mean(spectral_flux) if np.mean(spectral_flux) > 0 else 0
        flux_score = min(flux_variability / 0.5, 1.0)  # Higher flux = more robotic
        
        # 3. Energy dynamics (robotic speech has flat amplitude)
        rms_energy = librosa.feature.rms(y=y)[0]
        energy_variability = np.std(rms_energy) / np.mean(rms_energy) if np.mean(rms_energy) > 0 else 0
        energy_score = 1.0 - min(energy_variability / 0.2, 1.0)  # Lower variability = more robotic
        
        # 4. Zero-crossing rate stability (robotic speech is too stable)
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        zcr_variability = np.std(zcr) / np.mean(zcr) if np.mean(zcr) > 0 else 0
        zcr_score = 1.0 - min(zcr_variability / 0.3, 1.0)  # Lower variability = more robotic
        
        # Combine scores with weights
        robotic_score = (
            pitch_score * 0.3 +
            flux_score * 0.25 +
            energy_score * 0.25 +
            zcr_score * 0.2
        )
        
        return min(max(robotic_score, 0.0), 1.0)
        
    except Exception as e:
        print(f"Error in robotic speech detection: {e}")
        return 0.0

def analyze_speaker_similarity_with_quality(
    model: Any,
    reference_audio_path: str,
    generated_audio_path: str,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Enhanced similarity analysis with robotic speech detection.
    Returns comprehensive quality assessment.
    """
    # Get traditional similarity score
    similarity_score = analyze_speaker_similarity(model, reference_audio_path, generated_audio_path, device)
    
    # Get robotic detection score
    robotic_score = detect_robotic_speech(generated_audio_path)
    
    # Calculate combined quality score (higher is better)
    quality_score = similarity_score * (1.0 - robotic_score * 0.5)
    
    return {
        'similarity': similarity_score,
        'robotic_score': robotic_score,
        'quality_score': quality_score
    }

def concatenate_audio_files(
    audio_files: List[str],
    output_path: str,
    overlap_plan_text: Optional[str] = None,
    output_format: Optional[str] = None,
    output_bitrate_kbps: int = 192,
    normalize_segments: bool = True,
    target_level_dbfs: float = -19.0,
    peak_limit_dbfs: float = -1.0,
    normalize_final_mix: bool = True,
    trim_leading_silence: bool = True,
    trim_trailing_silence: bool = True,
    trim_silence_threshold_dbfs: float = -42.0,
    trim_min_silence_len_ms: int = 120,
    fade_in_ms: int = 0,
    fade_out_ms: int = 60,
    scene_pacing_profile: str = "balanced",
    scene_gap_ms: int = 140,
    respect_punctuation_pauses: bool = True,
    line_texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Concatenate multiple audio files into a single file.
    
    Args:
        audio_files: List of paths to audio files to concatenate
        output_path: Path for the output concatenated audio file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not PYDUB_AVAILABLE:
        print("Error: pydub not available for audio concatenation")
        return False
        
    try:
        print(f"DEBUG: Concatenating {len(audio_files)} audio files")
        if not audio_files:
            print("DEBUG: No audio files to concatenate")
            return False

        from .audio_mixing import mix_audio_files

        result = mix_audio_files(
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
            line_texts=line_texts,
        )
        if result.get("success"):
            print(f"DEBUG: Exported concatenated audio to: {output_path}")
            print(f"DEBUG: Mix plan applied: {result.get('plan_applied', False)}")
            print(f"DEBUG: Segment leveling applied: {result.get('normalization', {}).get('normalize_segments', False)}")
            return result

        print("DEBUG: Failed to create concatenated audio file")
        return {"success": False, "output_path": output_path}
            
    except Exception as e:
        print(f"Error concatenating audio files: {str(e)}")
        traceback.print_exc()
        return {"success": False, "output_path": output_path, "error": str(e)}

# Initialize the model if SpeechBrain is available
if SPEECHBRAIN_AVAILABLE:
    device = SPEAKER_ANALYSIS_DEVICE
    initialize_speaker_model(device)

# Initialize directories on import
ensure_dirs_exist()
