# IndexTTS Workflow Studio - Gradio UI Guide

This guide explains how to use the different tabs and features within the Gradio user interface for IndexTTS Workflow Studio.

## Overall Layout

The interface consists of:
1.  **Generation Parameters:** Global settings (Temperature, Top-P, Top-K) used for all TTS generation steps.
2.  **Project Save/Load:** An accordion section to save your current work (script, generated files, selections) or load a previous project.
3.  **Main Workflow Tabs:** Four tabs guide you through the process:
    *   Tab 1: Generate Conversation Lines
    *   Tab 2: Review & Select Lines
    *   Tab 3: Concatenate & Export
    *   Tab 4: Advanced Audio (Preview Only)

## 1. Generate Conversation Lines Tab

This is where you start your project.

*   **Conversation Script:** Paste your multi-line script here. Each line **must** follow the format: `SpeakerFilename.ext: Text to be spoken`.
    *   Example: `speaker1.wav: Hello there.`
    *   Example: `speaker2.wav: General Kenobi!`
*   **List Available Speakers:** Click this button to see the `.wav` or `.mp3` files detected in your `./speakers/` directory. Ensure your desired speaker files are listed here before generating.
*   **Versions per Line:** Select how many different audio versions (1 to 5) you want the TTS engine to generate for *each line* of your script. Generating more versions gives you more options to choose from in the Review tab.
*   **Max Manual Regen Attempts (Tab 2):** Set the maximum number of times the "Regenerate Below Manual Threshold" button on Tab 2 will try to create a new version for a single audio slot if it falls below your chosen similarity score.
*   **Seed Control (Initial Generation):**
    *   **Seed Strategy:** Choose how seeds are determined for the initial generation:
        *   `Fully Random`: Every single generated segment gets a unique random seed. High variability.
        *   `Random Base + Per-Line Sequential Offset`: A random starting seed is chosen, and subsequent seeds are calculated sequentially based on line and version number. Consistent variation *within* a line, different starting points per generation run.
        *   `Fixed Base + Per-Line Sequential Offset`: Like above, but uses the specific "Fixed Base Seed" you enter. Reproducible variation *within* a line across different generation runs (if base seed is the same).
        *   `Fixed Base + Reused Sequential List`: Uses the "Fixed Base Seed" to generate a list of sequential seeds (one per version). This *same list* of seeds is reused for *every line* in the script. (Equivalent to old "Sequential" mode).
        *   `Random Base + Reused Random List`: Generates a list of *random* seeds (one per version). This *same list* of random seeds is reused for *every line* in the script. (Equivalent to old "Random" mode).
    *   **Fixed Base Seed:** Enter an integer if using a "Fixed Base" strategy. Ignored otherwise.
*   **Generate All Lines & Versions Button:** Click this to start the process.
    *   It parses your script, validates speakers, and generates the requested number of versions for each line using the selected parameters and seed strategy.
    *   If speaker similarity analysis is enabled (requires SpeechBrain), it will automatically try to regenerate versions that fall below a predefined threshold (`SIMILARITY_THRESHOLD`).
*   **Generation Status:** Shows progress, warnings, and errors during generation.

## 2. Review & Select Lines Tab

This tab becomes active after generation finishes. Here you review the generated audio and choose the best version for each line.

*   **Navigation:** Use the `<< Previous Line` and `Next Line >>` buttons to move through your script lines. The `Line X / Y` display shows your current position.
*   **Current Line Info (Original):** Displays the original speaker and text for the current line.
*   **Editable Text for Regeneration:** You can modify the text for the current line here *before* clicking one of the regenerate buttons below. Changes are saved per line when regeneration occurs.
*   **Audio Players (Version 1, Version 2, ...):** Listen to the different generated versions for the current line. The label shows the version number and, if available, the calculated Speaker Similarity score (higher is closer to the reference speaker).
*   **Select Best Version:** Click the radio button corresponding to the audio version you like best for the current line. Your selection and the associated seed (if found in filename) are saved automatically to the application state and the `selected_seeds.json` file. The system may auto-select the version with the highest similarity score initially.
*   **Regenerate All Button:** Click this to regenerate **all** versions for the **current line only**, using the **edited text** (if modified) and the seed strategy/parameters set on Tab 1. This is useful if none of the initial versions are good or if you changed the text significantly. The first successfully generated version will be auto-selected.
*   **Manual Regen Similarity Threshold:** (Only visible if Speaker Similarity is enabled) Set a similarity score threshold (e.g., 0.70).
*   **Regenerate Below Manual Threshold Button:** (Only visible if Speaker Similarity is enabled) Click this to attempt regeneration for any *current versions* of *this line* that fall below the threshold set on the slider. It uses the **edited text** and generates *new random seeds* for each attempt, up to the "Max Manual Regen Attempts" limit set on Tab 1. It keeps the highest-scoring version (original or regenerated) for each slot and cleans up other temporary files for that slot.
*   **Review Status:** Shows status messages related to selection or regeneration.
*   **Proceed to Concatenate Tab ->:** This button becomes active only when you have made a valid selection for **every line** in your script. Click it to move to the next step.

## 3. Concatenate & Export Tab

This tab becomes active after you click "Proceed" on the Review tab. Here you combine your selected audio lines and apply optional post-processing. (Requires `pydub` library).

*   **Output Format:** Choose WAV or MP3. If MP3, select the desired bitrate.
*   **Processing Accordions:** Open these sections to enable and configure effects:
    *   **Per-Segment Normalization:** Normalizes the peak volume of each selected line *before* trimming or concatenation. Recommended to balance volume between lines.
    *   **Silence Trimming:** Removes leading/trailing silence from each segment *before* concatenation. Configure threshold (dBFS) and minimum silence length (ms). (Requires `pydub.silence`).
    *   **Pitch & Speed:** Adjust pitch (semitones) and speed (factor) of the *entire concatenated* audio.
    *   **Noise Reduction:** Apply noise reduction to the *entire concatenated* audio. Adjust strength. (Requires `noisereduce`).
    *   **Equalization (EQ):** Apply a 3-band (Low/Mid/High shelf/peak) EQ to the *entire concatenated* audio. Adjust gain (dB) for each band. (Requires `scipy`).
    *   **Compression:** Apply dynamic range compression to the *entire concatenated* audio. Adjust threshold, ratio, attack, and release. (Requires `pydub.effects`).
    *   **Reverb:** Add reverb to the *entire concatenated* audio. Adjust amount (0-1).
    *   **Final Peak Normalization:** Normalize the peak volume of the *final processed* audio to a target dBFS. Recommended to prevent clipping and ensure consistent final volume.
*   **Concatenate & Process Selected Lines Button:** Click this to perform the concatenation and apply all enabled processing steps.
*   **Concatenation & Processing Status:** Shows progress and errors.
*   **Final Output Audio:** Displays the final generated audio file for playback. The file is saved in the `./conversation_outputs/` directory.

## 4. Advanced Audio Tab

This tab is for **previewing** experimental or more complex audio effects on a single audio file. Effects applied here **do not** affect the main concatenation export process in Tab 3.

*   **Upload Test Audio:** Upload a WAV or MP3 file to test effects on.
*   **Effects Accordions:** Enable and configure advanced effects like Chorus, Flanger, Gain, and a 10-Band Graphical EQ. (Note: Some effects like Pitch Correction and Noise Gate are marked as not implemented).
*   **Preview Advanced Effects Button:** Click to apply the selected advanced effects to the uploaded test audio.
*   **Preview Output:** Displays the processed audio for playback.

## Project Save/Load Section

Located above the main tabs, this allows you to save your progress or load a previous session.

*   **Save Filename:** Enter a name for your save file (defaults to a timestamped name). It will be saved as a `.json` file in the `./project_saves/` directory.
*   **Save Current Project Button:** Saves the current script, paths to all generated audio versions, your selections for each line, any edited text, and the selected seeds data.
*   **Load Project File:** Select a previously saved `.json` file from the dropdown.
*   **Load Selected Project Button:** Loads the state from the selected file, restoring your script, selections, etc.
*   **Refresh List:** Updates the dropdown list of available save files.
*   **Save/Load Status:** Shows confirmation or error messages.
