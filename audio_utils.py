# -*- coding: utf-8 -*-
# audio_utils.py

# --- Imports ---
import numpy as np
import scipy.signal
from pydub import AudioSegment
import noisereduce as nr
import traceback
import torch # Added for SpeechBrain
import torchaudio # Added for SpeechBrain audio loading
from torch.nn.functional import cosine_similarity # Added for similarity calculation

# --- Attempt to import Librosa ---
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    LIBROSA_AVAILABLE = False

# --- Attempt to import SpeechBrain ---
try:
    # We only import the specific class needed within the function to avoid
    # loading it unless the function is called.
    # Actual model loading will happen in webui.py and be passed in.
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    EncoderClassifier = None # type: ignore
    SPEECHBRAIN_AVAILABLE = False
    print("Warning: speechbrain not found (pip install speechbrain). Speaker similarity analysis disabled.")


# --- Constants (Copied from webui.py) ---
# Required by apply_eq
EQ_LOW_FREQ: float = 250.0
EQ_MID_FREQ: float = 1000.0
EQ_HIGH_FREQ: float = 4000.0
EQ_MID_Q: float = 0.707

# Check library availability (can be useful if these funcs are called elsewhere)
SCIPY_AVAILABLE = np is not None and scipy is not None and scipy.signal is not None
NOISEREDUCE_AVAILABLE = nr is not None and np is not None
PYDUB_AVAILABLE = AudioSegment is not None


# --- EQ Functions ---

def _calculate_biquad_coeffs(gain_db: float, cutoff_freq: float, sampling_rate: float, filter_type: str = "lowshelf", q: float = 0.707) -> np.ndarray:
    """ Calculates SOS filter coefficients for various filter types """
    if not SCIPY_AVAILABLE:
        raise ImportError("numpy/scipy.signal required for EQ.")

    A = 10**(gain_db / 40.0)  # Amplitude derived from gain
    w0 = 2 * np.pi * cutoff_freq / sampling_rate
    alpha = np.sin(w0) / (2 * q)

    # Coefficients derived from Audio EQ Cookbook by Robert Bristow-Johnson
    if filter_type == "lowshelf":
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    elif filter_type == "highshelf":
        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
    else:
        raise ValueError("Unsupported filter_type. Use 'lowshelf', 'highshelf', or 'peaking'.")

    # Return SOS format: [b0, b1, b2, a0, a1, a2]
    # Normalize by a0 for sosfilt input
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])

def apply_eq(segment: AudioSegment, low_gain_db: float, mid_gain_db: float, high_gain_db: float, low_freq: float = EQ_LOW_FREQ, mid_freq: float = EQ_MID_FREQ, high_freq: float = EQ_HIGH_FREQ, mid_q: float = EQ_MID_Q) -> AudioSegment:
    """ Applies a 3-band EQ (low shelf, peaking, high shelf) to an AudioSegment """
    if not SCIPY_AVAILABLE or not PYDUB_AVAILABLE:
        print("Warning: EQ skipped - scipy/numpy/pydub not available.")
        return segment
    if low_gain_db == 0 and mid_gain_db == 0 and high_gain_db == 0:
        return segment # No EQ needed

    print(f"  Applying EQ (L:{low_gain_db}dB@{int(low_freq)}Hz, M:{mid_gain_db}dB@{int(mid_freq)}Hz, H:{high_gain_db}dB@{int(high_freq)}Hz)...")
    # Ensure samples are float32 for filtering
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    sampling_rate = segment.frame_rate

    # Apply filters sequentially
    if low_gain_db != 0:
        sos_low = _calculate_biquad_coeffs(low_gain_db, low_freq, sampling_rate, "lowshelf")
        samples = scipy.signal.sosfilt(sos_low, samples)
    if mid_gain_db != 0:
        sos_mid = _calculate_biquad_coeffs(mid_gain_db, mid_freq, sampling_rate, "peaking", q=mid_q)
        samples = scipy.signal.sosfilt(sos_mid, samples)
    if high_gain_db != 0:
        sos_high = _calculate_biquad_coeffs(high_gain_db, high_freq, sampling_rate, "highshelf")
        samples = scipy.signal.sosfilt(sos_high, samples)

    # Convert back to original integer type (typically int16 for wav)
    # Clipping is crucial after filtering to prevent wrap-around distortion
    max_val = 2**(segment.sample_width * 8 - 1) - 1
    min_val = -max_val - 1
    processed_samples = np.clip(samples, min_val, max_val).astype(segment.array_type)

    # Create new AudioSegment with processed samples
    return segment._spawn(processed_samples.tobytes())

# --- Noise Reduction Function ---

def apply_noise_reduction(segment: AudioSegment, strength: float = 0.85) -> AudioSegment:
    """ Applies noise reduction using the noisereduce library """
    if not NOISEREDUCE_AVAILABLE or not PYDUB_AVAILABLE:
        print("Warning: Noise Reduction skipped - noisereduce/numpy/pydub not available.")
        return segment
    if strength <= 0:
        return segment

    print(f"  Applying Noise Reduction (Strength: {strength:.2f})...")
    # noisereduce works best with float samples
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    sampling_rate = segment.frame_rate

    # Noise reduction - prop_decrease controls how much noise is reduced (0 to 1)
    reduced_noise_samples = nr.reduce_noise(
        y=samples,
        sr=sampling_rate,
        prop_decrease=strength,
        n_fft=1024, # Common defaults
        hop_length=256, # Common defaults
        n_jobs=1 # Use only 1 core to avoid potential joblib/temp file issues
        )

    # Convert back to original integer type
    max_val = 2**(segment.sample_width * 8 - 1) - 1
    min_val = -max_val - 1
    processed_samples = np.clip(reduced_noise_samples, min_val, max_val).astype(segment.array_type)

    return segment._spawn(processed_samples.tobytes())

# --- Reverb Effect Function ---

def apply_reverb(segment: AudioSegment, reverb_amount: float = 0.3, delay_ms: int = 60, decay: float = 0.5) -> AudioSegment:
    """
    Applies a simple reverb effect by mixing the original audio with a delayed, attenuated copy.
    reverb_amount: 0.0 (no reverb) to 1.0 (max reverb, mostly wet)
    delay_ms: delay in milliseconds for the echo
    decay: how much the echo decays (0.0 to 1.0)
    """
    if not PYDUB_AVAILABLE or reverb_amount <= 0.0:
        return segment

    print(f"  Applying Reverb (Amount: {reverb_amount:.2f}, Delay: {delay_ms}ms, Decay: {decay:.2f})...")
    # Create the delayed, decayed echo
    echo = segment - (1.0 - decay) * 10  # Attenuate the echo
    echo = echo.delay(delay_ms)
    # Mix original and echo
    wet = echo - (1.0 - reverb_amount) * 10
    # Overlay the echo onto the original
    output = segment.overlay(wet)
    # Optionally, blend dry/wet
    if reverb_amount < 1.0:
        output = segment.overlay(wet, gain_during_overlay=-10 * (1.0 - reverb_amount))
    return output

# --- Pitch Correction (Auto-Tune) ---

def apply_pitch_correction(segment: AudioSegment, strength: float = 1.0, snap_mode: str = "chromatic") -> AudioSegment:
    """
    Applies basic pitch correction (auto-tune) to an AudioSegment.
    - strength: 0.0 (off) to 1.0 (full correction)
    - snap_mode: "chromatic" (all semitones) or "major"/"minor" (future)
    Uses librosa for pitch tracking and shifting.
    """
    # TODO: Implement pitch correction using librosa.pyin or similar
    print("apply_pitch_correction: Not yet implemented.")
    return segment

# --- Chorus Effect ---

def apply_chorus(segment: AudioSegment, depth_ms: float = 15.0, rate_hz: float = 1.5, mix: float = 0.5) -> AudioSegment:
    """
    Applies a chorus effect by mixing delayed, modulated copies of the audio.
    - depth_ms: maximum delay in milliseconds
    - rate_hz: modulation rate in Hz
    - mix: 0.0 (dry) to 1.0 (wet)
    """
    if not PYDUB_AVAILABLE or not SCIPY_AVAILABLE or mix <= 0.0:
        return segment

    print(f"  Applying Chorus (Depth: {depth_ms}ms, Rate: {rate_hz}Hz, Mix: {mix:.2f})...")
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    sampling_rate = segment.frame_rate
    n_samples = len(samples)
    t = np.arange(n_samples) / sampling_rate

    # LFO for modulated delay (sinusoidal)
    lfo = (np.sin(2 * np.pi * rate_hz * t) + 1) / 2  # 0..1
    delay_samples = (lfo * depth_ms * sampling_rate / 1000).astype(np.int32)

    # Create output buffer
    out = np.copy(samples)
    for i in range(n_samples):
        d = delay_samples[i]
        idx = i - d
        if idx >= 0:
            out[i] += mix * samples[idx]
    # Normalize to prevent clipping
    max_val = 2**(segment.sample_width * 8 - 1) - 1
    min_val = -max_val - 1
    out = np.clip(out, min_val, max_val).astype(segment.array_type)
    return segment._spawn(out.tobytes())

# --- Flanger Effect ---

def apply_flanger(segment: AudioSegment, depth_ms: float = 3.0, rate_hz: float = 0.5, feedback: float = 0.5, mix: float = 0.5) -> AudioSegment:
    """
    Applies a flanger effect by mixing a short, modulated delay with feedback.
    - depth_ms: maximum delay in milliseconds
    - rate_hz: modulation rate in Hz
    - feedback: 0.0 (none) to 1.0 (max)
    - mix: 0.0 (dry) to 1.0 (wet)
    """
    if not PYDUB_AVAILABLE or not SCIPY_AVAILABLE or mix <= 0.0:
        return segment

    print(f"  Applying Flanger (Depth: {depth_ms}ms, Rate: {rate_hz}Hz, Feedback: {feedback:.2f}, Mix: {mix:.2f})...")
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    sampling_rate = segment.frame_rate
    n_samples = len(samples)
    t = np.arange(n_samples) / sampling_rate

    # LFO for modulated delay (sinusoidal)
    lfo = (np.sin(2 * np.pi * rate_hz * t) + 1) / 2  # 0..1
    delay_samples = (lfo * depth_ms * sampling_rate / 1000).astype(np.int32)

    out = np.copy(samples)
    fb = 0.0
    for i in range(n_samples):
        d = delay_samples[i]
        idx = i - d
        delayed = samples[idx] if idx >= 0 else 0.0
        # Add feedback from previous output
        delayed += feedback * fb
        out[i] = (1 - mix) * samples[i] + mix * delayed
        fb = out[i]
    # Normalize to prevent clipping
    max_val = 2**(segment.sample_width * 8 - 1) - 1
    min_val = -max_val - 1
    out = np.clip(out, min_val, max_val).astype(segment.array_type)
    return segment._spawn(out.tobytes())

# --- Noise Gate ---

def apply_noise_gate(segment: AudioSegment, threshold_db: float = -40.0, attack_ms: float = 10.0, release_ms: float = 100.0) -> AudioSegment:
    """
    Applies a noise gate to suppress audio below a threshold.
    - threshold_db: amplitude threshold in dBFS
    - attack_ms: attack time in milliseconds
    - release_ms: release time in milliseconds
    """
    # TODO: Implement noise gate using amplitude envelope
    print("apply_noise_gate: Not yet implemented.")
    return segment

# --- 10-Band Graphical Equalizer ---

def apply_graphical_eq(segment: AudioSegment, gains_db: list) -> AudioSegment:
    """
    Applies a 10-band graphical equalizer to an AudioSegment.
    - gains_db: list of 10 gain values (in dB) for each band
    Center frequencies: [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000] Hz
    """
    if not SCIPY_AVAILABLE or not PYDUB_AVAILABLE:
        print("Warning: Graphical EQ skipped - scipy/numpy/pydub not available.")
        return segment
    if not isinstance(gains_db, (list, tuple)) or len(gains_db) != 10:
        print("Warning: Graphical EQ requires 10 gain values.")
        return segment

    print(f"  Applying 10-band Graphical EQ: {gains_db}")
    center_freqs = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    sampling_rate = segment.frame_rate

    # Apply each band as a peaking filter
    for gain_db, freq in zip(gains_db, center_freqs):
        if gain_db == 0:
            continue
        sos = _calculate_biquad_coeffs(gain_db, freq, sampling_rate, filter_type="peaking", q=1.0)
        samples = scipy.signal.sosfilt(sos, samples)

    max_val = 2**(segment.sample_width * 8 - 1) - 1
    min_val = -max_val - 1
    processed_samples = np.clip(samples, min_val, max_val).astype(segment.array_type)
    return segment._spawn(processed_samples.tobytes())

# --- Pitch Shifting Function ---

def change_pitch(segment: AudioSegment, n_steps: float) -> AudioSegment:
    """
    Changes the pitch of an AudioSegment by n_steps (semitones).
    Positive n_steps increases pitch, negative decreases.
    Requires librosa.
    """
    if not LIBROSA_AVAILABLE or not PYDUB_AVAILABLE:
        print("Warning: Pitch shifting skipped - librosa/pydub not available.")
        return segment
    if n_steps == 0:
        return segment

    print(f"  Changing pitch by {n_steps} semitones...")
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    sampling_rate = segment.frame_rate

    # librosa expects mono, float32
    if segment.channels > 1:
        samples = samples.reshape((-1, segment.channels)).T
        shifted = []
        for ch in samples:
            # Use new librosa API if available, otherwise fallback
            if hasattr(librosa, 'pitch_shift'):
                shifted_ch = librosa.pitch_shift(y=ch, sr=sampling_rate, n_steps=n_steps)
            else: # Older librosa versions might use this structure
                shifted_ch = librosa.effects.pitch_shift(ch, sampling_rate, n_steps)
            shifted.append(shifted_ch)
        shifted = np.stack(shifted, axis=0).T.flatten()
    else:
        if hasattr(librosa, 'pitch_shift'):
            shifted = librosa.pitch_shift(y=samples, sr=sampling_rate, n_steps=n_steps)
        else:
            shifted = librosa.effects.pitch_shift(samples, sampling_rate, n_steps)

    # Convert back to original integer type
    max_val = 2**(segment.sample_width * 8 - 1) - 1
    min_val = -max_val - 1
    processed_samples = np.clip(shifted, min_val, max_val).astype(segment.array_type)

    return segment._spawn(processed_samples.tobytes())

# --- Time Stretching Function ---

def change_speed(segment: AudioSegment, speed_factor: float) -> AudioSegment:
    """
    Changes the speed (tempo) of an AudioSegment by speed_factor.
    speed_factor > 1.0 speeds up, < 1.0 slows down.
    Pitch is preserved (uses phase vocoder).
    Requires librosa.
    """
    if not LIBROSA_AVAILABLE or not PYDUB_AVAILABLE:
        print("Warning: Time stretching skipped - librosa/pydub not available.")
        return segment
    if speed_factor == 1.0:
        return segment

    print(f"  Changing speed by factor {speed_factor:.2f} (preserving pitch)...")
    samples = np.array(segment.get_array_of_samples()).astype(np.float32)
    sampling_rate = segment.frame_rate

    # librosa expects mono, float32
    if segment.channels > 1:
        samples = samples.reshape((-1, segment.channels)).T
        stretched = []
        for ch in samples:
            # Use new librosa API if available
            if hasattr(librosa, 'time_stretch'):
                 stretched_ch = librosa.time_stretch(y=ch, rate=speed_factor)
            else: # Older librosa versions
                 stretched_ch = librosa.effects.time_stretch(ch, speed_factor)
            stretched.append(stretched_ch)
        stretched = np.stack(stretched, axis=0).T.flatten()
    else:
        if hasattr(librosa, 'time_stretch'):
            stretched = librosa.time_stretch(y=samples, rate=speed_factor)
        else:
            stretched = librosa.effects.time_stretch(samples, speed_factor)

    # Convert back to original integer type
    max_val = 2**(segment.sample_width * 8 - 1) - 1
    min_val = -max_val - 1
    processed_samples = np.clip(stretched, min_val, max_val).astype(segment.array_type)

    return segment._spawn(processed_samples.tobytes())


# --- Speaker Similarity Analysis Function ---

def analyze_speaker_similarity(
    spk_model: 'EncoderClassifier',
    ref_audio_path: str,
    test_audio_path: str,
    device: str = 'cpu' # Allow specifying 'cuda' or 'cpu'
    ) -> float:
    """
    Calculates speaker similarity between a reference audio and a test audio
    using a pre-loaded SpeechBrain EncoderClassifier model.

    Args:
        spk_model: The pre-loaded SpeechBrain EncoderClassifier instance.
        ref_audio_path: Path to the reference speaker audio file (e.g., from ./speakers).
        test_audio_path: Path to the generated TTS audio file to test.
        device: The torch device to run the model on ('cuda' or 'cpu').

    Returns:
        Cosine similarity score (float between -1.0 and 1.0),
        or -1.0 if an error occurs or SpeechBrain is unavailable.
    """
    if not SPEECHBRAIN_AVAILABLE or spk_model is None:
        print("Speaker similarity analysis skipped: SpeechBrain not available or model not loaded.")
        return -1.0

    if not ref_audio_path or not test_audio_path:
         print("Speaker similarity analysis skipped: Missing audio path(s).")
         return -1.0

    # Target sampling rate expected by many SpeechBrain models
    target_sr = 16000

    try:
        # --- Load and prepare reference audio ---
        ref_sig, ref_sr = torchaudio.load(ref_audio_path)
        # Resample if necessary
        if ref_sr != target_sr:
            ref_sig = torchaudio.functional.resample(ref_sig, ref_sr, target_sr)
        # Ensure mono
        if ref_sig.shape[0] > 1:
            ref_sig = torch.mean(ref_sig, dim=0, keepdim=True)
        # Move to device
        ref_sig = ref_sig.to(device)

        # --- Load and prepare test audio ---
        test_sig, test_sr = torchaudio.load(test_audio_path)
        # Resample if necessary
        if test_sr != target_sr:
            test_sig = torchaudio.functional.resample(test_sig, test_sr, target_sr)
        # Ensure mono
        if test_sig.shape[0] > 1:
            test_sig = torch.mean(test_sig, dim=0, keepdim=True)
        # Move to device
        test_sig = test_sig.to(device)

        # --- Compute embeddings ---
        # Use torch.no_grad() for efficiency as we don't need gradients
        with torch.no_grad():
             # The model expects input shape [batch, time], length is relative
             ref_embedding = spk_model.encode_batch(ref_sig)
             test_embedding = spk_model.encode_batch(test_sig)

        # --- Calculate Cosine Similarity ---
        # Embeddings usually have shape [batch, 1, embedding_dim], squeeze them
        similarity = cosine_similarity(ref_embedding.squeeze(), test_embedding.squeeze(), dim=0)

        # Return the similarity score as a Python float
        score = similarity.item()
        print(f"  Speaker similarity between '{ref_audio_path}' and '{test_audio_path}': {score:.4f}")
        return score

    except FileNotFoundError as e:
        print(f"Error in speaker similarity: File not found - {e}")
        return -1.0
    except RuntimeError as e:
        # Catch potential CUDA errors or other Torch runtime issues
        print(f"Error during speaker similarity model inference: {e}")
        traceback.print_exc()
        return -1.0
    except Exception as e:
        # Catch any other unexpected errors during loading or processing
        print(f"Unexpected error during speaker similarity analysis: {type(e).__name__}: {e}")
        traceback.print_exc()
        return -1.0