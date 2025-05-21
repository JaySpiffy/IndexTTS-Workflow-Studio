# ui_layout.py
import gradio as gr
import datetime
from constants import * # Import constants for default values, choices etc.
# Import necessary functions if needed for initial values (e.g., list_save_files)
try:
    from general_utils import list_save_files
except ImportError:
    def list_save_files(*args, **kwargs): return []
# Import flags needed for conditional visibility
try:
    from audio_utils import SPEECHBRAIN_AVAILABLE
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
# Import PYDUB_AVAILABLE flag (will be set by webui.py, but needed for initial layout)
# This is slightly awkward, ideally this dependency would be handled differently
# For now, assume False initially if ui_logic hasn't been updated yet.
try:
    from ui_logic import PYDUB_AVAILABLE, PYDUB_SILENCE_AVAILABLE, PYDUB_COMPRESS_AVAILABLE, SCIPY_AVAILABLE, NOISEREDUCE_AVAILABLE, speaker_similarity_model
except ImportError:
    PYDUB_AVAILABLE = False
    PYDUB_SILENCE_AVAILABLE = False
    PYDUB_COMPRESS_AVAILABLE = False
    SCIPY_AVAILABLE = False
    NOISEREDUCE_AVAILABLE = False
    speaker_similarity_model = None # Assume None if ui_logic not ready

def create_save_load_section():
    """Creates the Gradio UI components for the Save/Load section."""
    with gr.Accordion("Project Save/Load", open=True):
        with gr.Row():
            with gr.Column(scale=2):
                save_filename_input = gr.Textbox(label="Save Filename (.json)", value=f"project_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
                save_project_button = gr.Button("Save Current Project")
            with gr.Column(scale=2):
                load_filename_dropdown = gr.Dropdown(label="Load Project File", choices=list_save_files(), interactive=True)
                load_project_button = gr.Button("Load Selected Project")
            with gr.Column(scale=1):
                 refresh_saves_button = gr.Button("Refresh List")
        save_load_status_output = gr.Textbox(label="Save/Load Status", interactive=False, lines=1)
    return save_filename_input, save_project_button, load_filename_dropdown, load_project_button, refresh_saves_button, save_load_status_output

def create_generation_parameters_section():
    """Creates the Gradio UI components for the Generation Parameters section."""
    with gr.Accordion("Generation Parameters (Used by All Generators)", open=True):
        with gr.Row():
            temperature_slider = gr.Slider(minimum=0.0, maximum=1.5, step=0.05, value=1.0, label="Temperature")
            top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, step=0.05, value=0.8, label="Top-P")
            top_k_slider = gr.Slider(minimum=1, maximum=100, step=1, value=30, label="Top-K")
    return temperature_slider, top_p_slider, top_k_slider

def create_generation_tab():
    """Creates the Gradio UI components for the Generation Tab."""
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
    return (script_input_convo, list_speakers_btn, available_speakers_display,
            num_versions_convo_radio, manual_regen_attempts_dd, seed_strategy_dd,
            fixed_base_seed_input, generate_convo_button, convo_gen_status_output)

def create_review_tab():
    """Creates the Gradio UI components for the Review Tab."""
    with gr.TabItem("2. Review & Select Lines", id="tab_review", interactive=False) as review_tab:
         gr.Markdown("### Review and Select Best Version for Each Line");
         # Use the flags/variables imported or assumed at the top of this file
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
    # Return all components created within this tab
    return (review_tab, prev_line_button, line_nav_display, next_line_button,
            current_line_display_review, editable_line_text_review,
            regenerate_current_line_button, manual_regen_threshold_slider,
            threshold_regen_button, review_audio_outputs, line_choice_radio,
            review_status_output, proceed_to_concat_button)

def create_concat_tab():
    """Creates the Gradio UI components for the Concatenate Tab."""
    # Visibility will be controlled in webui.py after checking PYDUB_AVAILABLE
    with gr.TabItem("3. Concatenate & Export", id="tab_concat", interactive=False) as concat_tab:
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
    # Return all components created within this tab
    return (concat_tab, output_format_dropdown, mp3_bitrate_dropdown,
            normalize_segments_checkbox, trim_silence_checkbox, trim_threshold_input,
            trim_length_input, pitch_shift_slider, speed_slider,
            apply_noise_reduction_checkbox, noise_reduction_strength_slider,
            eq_low_gain_input, eq_mid_gain_input, eq_high_gain_input,
            apply_compression_checkbox, compress_threshold_input, compress_ratio_input,
            compress_attack_input, compress_release_input, reverb_amount_slider,
            apply_peak_norm_checkbox, peak_norm_target_input,
            concatenate_convo_button, concat_status_output, final_conversation_audio)

def create_advanced_audio_tab():
    """Creates the Gradio UI components for the Advanced Audio Tab."""
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
    # Return all components created within this tab
    return (test_audio_upload, enable_pitch_correction, pitch_correction_strength,
            pitch_correction_mode, enable_chorus, chorus_depth, chorus_rate, chorus_mix,
            enable_flanger, flanger_depth, flanger_rate, flanger_feedback, flanger_mix,
            enable_noise_gate, noise_gate_threshold, noise_gate_attack, noise_gate_release,
            gain_slider_advanced, enable_graphical_eq, eq_band_sliders,
            preview_button, preview_audio)

def create_timeline_tab():
    """Creates the Gradio UI components for the Timeline / Final Edit Tab."""
    with gr.TabItem("Timeline / Final Edit", id="tab_timeline_edit") as timeline_tab:
        gr.Markdown("### Timeline / Final Edit")
        timeline_line_selector_dd = gr.Dropdown(label="Select Line to Edit", choices=[], interactive=True) # Populate choices dynamically
        with gr.Row():
            with gr.Column(): # Column for displaying information and editing
                timeline_original_speaker_text = gr.Textbox(label="Original Speaker", interactive=False)
                timeline_original_text_display = gr.Textbox(label="Original Text", interactive=False, lines=2)
                timeline_editable_text_input = gr.Textbox(label="Editable Text", interactive=True, lines=3, placeholder="Edit text here if needed for regeneration...")
                timeline_selected_audio_player = gr.Audio(label="Selected Audio", type="filepath", interactive=True)
                timeline_selected_audio_seed_text = gr.Textbox(label="Seed (if available)", interactive=False)
                timeline_regenerate_button = gr.Button("Regenerate Selected Audio")
        timeline_status_text = gr.Textbox(label="Timeline Status", interactive=False, lines=1)

    return (timeline_tab, timeline_line_selector_dd, timeline_original_speaker_text,
            timeline_original_text_display, timeline_editable_text_input,
            timeline_selected_audio_player, timeline_selected_audio_seed_text,
            timeline_regenerate_button, timeline_status_text)
