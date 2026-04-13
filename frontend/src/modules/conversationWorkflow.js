// IndexTTS2 Conversation Workflow Module
IndexTTSApp.prototype.detectEmotionsForScript = async function(lines) {
    // Extract text from all lines for batch processing
    const texts = lines.map(line => line.text);
    
    console.log('🔍 DEBUG: detectEmotionsForScript called with', lines.length, 'lines');
    console.log('🔍 DEBUG: Extracted texts for batch processing:', texts);
    
    try {
        console.log('🔄 DEBUG: Attempting batch emotion detection...');
        // Use batch emotion detection for efficiency
        const results = await this.detectEmotionFromText(null, texts);
        console.log('✅ DEBUG: Batch emotion detection successful, results:', results);
        if (Array.isArray(results?.results)) {
            return results.results;
        }

        if (Array.isArray(results)) {
            return results;
        }

        return results ? [results] : [];
    } catch (error) {
        console.error('❌ Batch emotion detection failed, falling back to individual detection:', error);
        console.error('❌ Error details:', error.message);
        
        // Fallback to individual detection
        const results = [];
        console.log('🔄 DEBUG: Starting fallback individual emotion detection for', lines.length, 'lines');
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            try {
                console.log(`🔄 DEBUG: Processing line ${i + 1}/${lines.length}: "${line.text}"`);
                const result = await this.detectEmotionFromText(line.text);
                console.log(`✅ DEBUG: Individual emotion detection successful for line ${i + 1}:`, result);
                results.push(result);
            } catch (lineError) {
                console.error(`❌ Failed to detect emotion for line ${i + 1}: "${line.text}"`, lineError);
                console.error(`❌ Line error details:`, lineError.message);
                // Use default emotion on failure
                const defaultResult = {
                    emotion_vectors: [0, 0, 0, 0, 0, 0, 0, 0.8], // Default to 'calm'
                    emotion_dict: { happy: 0, sad: 0, angry: 0, afraid: 0, disgusted: 0, melancholic: 0, surprised: 0, calm: 0.8 }
                };
                console.log(`🔄 DEBUG: Using default emotion for line ${i + 1}:`, defaultResult);
                results.push(defaultResult);
            }
        }
        console.log('✅ DEBUG: Fallback individual emotion detection completed, results:', results);
        return results;
    }
};

IndexTTSApp.prototype.generateDefaultProjectSaveName = function() {
    const title = document.getElementById('conversation-title')?.value?.trim();
    const baseName = (title || 'project')
        .replace(/[\\/*?:"<>|]+/g, '_')
        .replace(/\s+/g, '_')
        .replace(/^_+|_+$/g, '');
    return `${baseName || 'project'}.json`;
};

IndexTTSApp.prototype.setProjectSaveStatus = function(message, type = 'info') {
    const statusElement = document.getElementById('project-save-status');
    if (!statusElement) return;

    statusElement.textContent = message;
    statusElement.dataset.status = type;
};

IndexTTSApp.prototype.cloneProjectState = function(data) {
    if (data === null || data === undefined) {
        return data;
    }

    if (typeof structuredClone === 'function') {
        return structuredClone(data);
    }

    return JSON.parse(JSON.stringify(data));
};

IndexTTSApp.prototype.getDefaultGenerationSettings = function() {
    return {
        generation_preset: 'balanced',
        versions_per_line: 3,
        similarity_threshold: 0.6,
        auto_regen_attempts: 1,
        seed_strategy: 'fully_random',
        fixed_base_seed: 1234,
        scene_pacing_profile: 'balanced',
        scene_gap_ms: 140,
        respect_punctuation_pauses: true,
        speaker_pacing: [],
        emotion_weight: 1.0,
        use_random_sampling: false,
        max_text_tokens_per_segment: 120,
        do_sample: true,
        top_p: 0.8,
        top_k: 30,
        temperature: 0.8,
        length_penalty: 0.0,
        num_beams: 3,
        repetition_penalty: 10,
        max_mel_tokens: 1500
    };
};

IndexTTSApp.prototype.getSeedStrategyHelpText = function(seedStrategy) {
    const helpText = {
        fully_random: 'Every generated version gets a fresh unique seed. Highest variation, lowest reproducibility.',
        random_base_sequential: 'Picks a random base seed for this run, then offsets it by line and version so the run is reproducible after export.',
        fixed_base_sequential: 'Uses your fixed base seed and offsets it by line and version. Best when you want a run you can recreate later.',
        fixed_base_reused_list: 'Builds one sequential seed list from the fixed base seed and reuses that same list for every line.',
        random_base_reused_list: 'Builds one random-looking seed list from a random base seed and reuses it for every line in this run.'
    };

    return helpText[seedStrategy] || helpText.fully_random;
};

IndexTTSApp.prototype.updateSeedStrategyUi = function(seedStrategy) {
    const normalizedStrategy = seedStrategy || 'fully_random';
    const helpElement = document.getElementById('seed-strategy-help');
    const fixedSeedGroup = document.getElementById('fixed-base-seed-group');

    if (helpElement) {
        helpElement.textContent = this.getSeedStrategyHelpText(normalizedStrategy);
    }

    if (fixedSeedGroup) {
        const needsFixedSeed = normalizedStrategy === 'fixed_base_sequential' || normalizedStrategy === 'fixed_base_reused_list';
        fixedSeedGroup.style.display = needsFixedSeed ? 'block' : 'none';
    }
};

IndexTTSApp.prototype.getGenerationPresets = function() {
    return {
        clone_fidelity: {
            label: 'Clone Fidelity',
            helpText: 'Best when the voice match matters most. Uses steadier decoding, lower emotion weight, and longer segments for less robotic output.',
            settings: {
                generation_preset: 'clone_fidelity',
                emotion_weight: 0.7,
                use_random_sampling: false,
                max_text_tokens_per_segment: 180,
                do_sample: false,
                top_p: 0.75,
                top_k: 20,
                temperature: 0.7,
                length_penalty: 0.0,
                num_beams: 5,
                repetition_penalty: 7,
                max_mel_tokens: 2000
            }
        },
        balanced: {
            label: 'Balanced',
            helpText: 'Keeps the upstream-style defaults and works well as the normal starting point.',
            settings: {
                generation_preset: 'balanced',
                emotion_weight: 1.0,
                use_random_sampling: false,
                max_text_tokens_per_segment: 120,
                do_sample: true,
                top_p: 0.8,
                top_k: 30,
                temperature: 0.8,
                length_penalty: 0.0,
                num_beams: 3,
                repetition_penalty: 10,
                max_mel_tokens: 1500
            }
        },
        expressive: {
            label: 'Expressive',
            helpText: 'Useful when you want more dramatic delivery, but it can drift farther from the reference voice than Clone Fidelity.',
            settings: {
                generation_preset: 'expressive',
                emotion_weight: 1.0,
                use_random_sampling: false,
                max_text_tokens_per_segment: 120,
                do_sample: true,
                top_p: 0.9,
                top_k: 40,
                temperature: 0.9,
                length_penalty: 0.0,
                num_beams: 3,
                repetition_penalty: 9,
                max_mel_tokens: 1700
            }
        }
    };
};

IndexTTSApp.prototype.getScenePacingPresets = function() {
    return {
        relaxed: {
            label: 'Relaxed',
            default_gap_ms: 220,
            helpText: 'Leaves more air between lines. Good for reflective scenes and slower speakers.'
        },
        balanced: {
            label: 'Balanced',
            default_gap_ms: 140,
            helpText: 'A natural middle ground. Good default for most conversations.'
        },
        snappy: {
            label: 'Snappy',
            default_gap_ms: 75,
            helpText: 'Keeps exchanges moving. Good for comedy and fast back-and-forth scenes.'
        },
        tense: {
            label: 'Tense',
            default_gap_ms: 45,
            helpText: 'Compresses pauses for anxious scenes, interruptions, and arguments.'
        }
    };
};

IndexTTSApp.prototype.getDialoguePacingPresets = function() {
    return {
        natural: {
            label: 'Natural',
            helpText: 'Balanced scene rhythm with neutral speaker delivery. Best everyday starting point before fine-tuning any individual speaker.',
            scene_pacing_profile: 'balanced',
            scene_gap_ms: 140,
            respect_punctuation_pauses: true,
            speaker_delivery_rate: 1.0,
        },
        calm: {
            label: 'Calm',
            helpText: 'Adds more breathing room and gently slows delivery. Good for softer scenes, narration, and thoughtful exchanges.',
            scene_pacing_profile: 'relaxed',
            scene_gap_ms: 220,
            respect_punctuation_pauses: true,
            speaker_delivery_rate: 0.96,
        },
        argument: {
            label: 'Argument',
            helpText: 'Tightens pauses and nudges everyone a little faster. Good for heated exchanges where people keep stepping on each other.',
            scene_pacing_profile: 'tense',
            scene_gap_ms: 45,
            respect_punctuation_pauses: true,
            speaker_delivery_rate: 1.04,
        },
        panic: {
            label: 'Panic',
            helpText: 'Pushes pace hardest. Use this for spirals, breathless reactions, and scenes that should feel close to breaking point.',
            scene_pacing_profile: 'tense',
            scene_gap_ms: 25,
            respect_punctuation_pauses: true,
            speaker_delivery_rate: 1.09,
        },
    };
};

IndexTTSApp.prototype.getDialoguePacingPresetDefaultRate = function(presetName = null) {
    const presets = this.getDialoguePacingPresets();
    const resolvedPreset = presets[presetName || document.getElementById('dialogue-pacing-preset')?.value || 'natural'] || presets.natural;
    return Number(resolvedPreset.speaker_delivery_rate || 1.0);
};

IndexTTSApp.prototype.updateDialoguePacingPresetHelp = function(presetName) {
    const helpElement = document.getElementById('dialogue-pacing-preset-help');
    if (!helpElement) return;

    const presets = this.getDialoguePacingPresets();
    helpElement.textContent = (presets[presetName] || presets.natural).helpText;
};

IndexTTSApp.prototype.updateScenePacingUi = function(profileName, applyPresetGap = false) {
    const presets = this.getScenePacingPresets();
    const preset = presets[profileName] || presets.balanced;
    const helpElement = document.getElementById('scene-pacing-help');
    const gapInput = document.getElementById('scene-gap-ms');
    const gapValue = document.getElementById('scene-gap-ms-value');

    if (helpElement) {
        helpElement.textContent = preset.helpText;
    }

    if (gapInput && applyPresetGap) {
        gapInput.value = preset.default_gap_ms;
        gapInput.dispatchEvent(new Event('input', { bubbles: true }));
    }

    if (gapInput && gapValue) {
        gapValue.textContent = `${parseInt(gapInput.value || `${preset.default_gap_ms}`, 10)} ms`;
    }
};

IndexTTSApp.prototype.captureSpeakerPacingSettings = function() {
    const rows = Array.from(document.querySelectorAll('.speaker-pacing-row'));
    return rows.map((row) => {
        const speakerFilename = row.dataset.speakerFilename || '';
        const deliveryRate = parseFloat(row.querySelector('.speaker-pacing-slider')?.value || '1.0');
        return {
            speaker_filename: speakerFilename,
            delivery_rate: Number.isFinite(deliveryRate) ? deliveryRate : 1.0
        };
    });
};

IndexTTSApp.prototype.updateSpeakerPacingRowLabel = function(row) {
    if (!row) return;
    const slider = row.querySelector('.speaker-pacing-slider');
    const valueElement = row.querySelector('.speaker-pacing-value');
    if (!slider || !valueElement) return;

    const deliveryRate = parseFloat(slider.value || '1.0');
    let tone = 'Natural';
    if (deliveryRate < 0.97) tone = 'Slower';
    if (deliveryRate > 1.03) tone = 'Faster';
    valueElement.textContent = `${tone} (${deliveryRate.toFixed(2)}x)`;
};

IndexTTSApp.prototype.applySpeakerPacingDefaults = function(deliveryRate = 1.0) {
    const resolvedRate = Number.isFinite(Number(deliveryRate)) ? Number(deliveryRate) : 1.0;
    const rows = Array.from(document.querySelectorAll('.speaker-pacing-row'));
    rows.forEach((row) => {
        const slider = row.querySelector('.speaker-pacing-slider');
        if (!slider) return;
        slider.value = resolvedRate.toFixed(2);
        this.updateSpeakerPacingRowLabel(row);
    });
    this.pendingSpeakerPacingSettings = this.captureSpeakerPacingSettings();
};

IndexTTSApp.prototype.applySpeakerPacingSettings = function(settingsList = []) {
    const normalizedSettings = Array.isArray(settingsList) ? settingsList : [];
    this.pendingSpeakerPacingSettings = normalizedSettings;
    const fallbackRate = this.getDialoguePacingPresetDefaultRate();

    const rows = Array.from(document.querySelectorAll('.speaker-pacing-row'));
    if (!rows.length) {
        return;
    }

    const settingMap = {};
    normalizedSettings.forEach((entry) => {
        const speakerFilename = String(entry?.speaker_filename || '').trim();
        if (!speakerFilename) return;
        settingMap[speakerFilename.toLowerCase()] = parseFloat(entry.delivery_rate || '1.0');
        settingMap[speakerFilename.replace(/\.(wav|mp3|ogg|flac|m4a)$/i, '').toLowerCase()] = parseFloat(entry.delivery_rate || '1.0');
    });

    rows.forEach((row) => {
        const filename = String(row.dataset.speakerFilename || '').toLowerCase();
        const stem = filename.replace(/\.(wav|mp3|ogg|flac|m4a)$/i, '');
        const resolvedRate = settingMap[filename] ?? settingMap[stem] ?? fallbackRate;
        const slider = row.querySelector('.speaker-pacing-slider');
        if (slider) {
            slider.value = Number.isFinite(resolvedRate) ? resolvedRate : 1.0;
        }
        this.updateSpeakerPacingRowLabel(row);
    });
};

IndexTTSApp.prototype.applyDialoguePacingPreset = function(presetName, notify = false) {
    const presets = this.getDialoguePacingPresets();
    const preset = presets[presetName] || presets.natural;
    const presetSelect = document.getElementById('dialogue-pacing-preset');
    const scenePacingSelect = document.getElementById('scene-pacing-profile');
    const sceneGapInput = document.getElementById('scene-gap-ms');
    const punctuationInput = document.getElementById('respect-punctuation-pauses');

    if (presetSelect) {
        presetSelect.value = presetName in presets ? presetName : 'natural';
    }

    this.updateDialoguePacingPresetHelp(presetSelect?.value || 'natural');

    if (scenePacingSelect) {
        scenePacingSelect.value = preset.scene_pacing_profile;
    }
    if (sceneGapInput) {
        sceneGapInput.value = preset.scene_gap_ms;
    }
    if (punctuationInput) {
        punctuationInput.checked = Boolean(preset.respect_punctuation_pauses);
    }

    this.updateScenePacingUi(preset.scene_pacing_profile, false);

    if (document.querySelectorAll('.speaker-pacing-row').length) {
        this.applySpeakerPacingDefaults(preset.speaker_delivery_rate);
    } else {
        this.pendingSpeakerPacingSettings = [];
    }

    if (typeof this.applyConversationMixPacingSettings === 'function') {
        this.applyConversationMixPacingSettings({
            pacing_preset: presetSelect?.value || 'natural',
            scene_pacing_profile: preset.scene_pacing_profile,
            scene_gap_ms: preset.scene_gap_ms,
            respect_punctuation_pauses: preset.respect_punctuation_pauses,
        });
    }

    if (notify && this.showNotification) {
        this.showNotification('Pacing Preset Applied', `${preset.label} pacing loaded`, 'info');
    }
};

IndexTTSApp.prototype.renderSpeakerPacingControls = function(lines = null) {
    const container = document.getElementById('speaker-pacing-list');
    if (!container) return;

    const sourceLines = Array.isArray(lines) ? lines : this.conversationScript;
    const existingSettings = this.captureSpeakerPacingSettings();
    const presetDefaultRate = this.getDialoguePacingPresetDefaultRate();
    const preferredSettings = (this.pendingSpeakerPacingSettings && this.pendingSpeakerPacingSettings.length)
        ? this.pendingSpeakerPacingSettings
        : existingSettings;

    const speakerMap = new Map();
    (sourceLines || []).forEach((line) => {
        const speakerFilename = line?.speaker_filename || line?.speaker || '';
        if (!speakerFilename) return;
        if (!speakerMap.has(speakerFilename)) {
            speakerMap.set(speakerFilename, {
                speaker_filename: speakerFilename,
                speaker_label: line?.speaker || speakerFilename.replace(/\.(wav|mp3|ogg|flac|m4a)$/i, '')
            });
        }
    });

    container.innerHTML = '';

    if (!speakerMap.size) {
        container.innerHTML = `
            <div class="empty-state compact-empty-state">
                <i class="fas fa-user-clock"></i>
                <p>Parse a script first to set speaker pacing.</p>
            </div>
        `;
        this.pendingSpeakerPacingSettings = preferredSettings;
        return;
    }

    speakerMap.forEach((speakerInfo) => {
        const row = document.createElement('div');
        row.className = 'speaker-pacing-row';
        row.dataset.speakerFilename = speakerInfo.speaker_filename;
        row.innerHTML = `
            <div class="speaker-pacing-name">
                <strong>${speakerInfo.speaker_label}</strong>
                <span>${speakerInfo.speaker_filename}</span>
            </div>
            <div class="speaker-pacing-controls">
                <input
                    type="range"
                    class="speaker-pacing-slider"
                    min="0.85"
                    max="1.15"
                    step="0.01"
                    value="${presetDefaultRate.toFixed(2)}"
                >
                <span class="speaker-pacing-value">Natural (${presetDefaultRate.toFixed(2)}x)</span>
                <button type="button" class="btn btn-secondary btn-small speaker-pacing-reset">Reset</button>
            </div>
        `;

        const slider = row.querySelector('.speaker-pacing-slider');
        const resetButton = row.querySelector('.speaker-pacing-reset');
        if (slider) {
            slider.addEventListener('input', () => {
                this.updateSpeakerPacingRowLabel(row);
            });
        }
        if (resetButton) {
            resetButton.addEventListener('click', () => {
                if (slider) {
                    slider.value = this.getDialoguePacingPresetDefaultRate().toFixed(2);
                    this.updateSpeakerPacingRowLabel(row);
                }
            });
        }

        container.appendChild(row);
    });

    if (preferredSettings.length) {
        this.applySpeakerPacingSettings(preferredSettings);
    } else {
        this.applySpeakerPacingDefaults(presetDefaultRate);
    }
};

IndexTTSApp.prototype.updateGenerationPresetHelp = function(presetName) {
    const helpElement = document.getElementById('generation-preset-help');
    if (!helpElement) return;

    const presets = this.getGenerationPresets();
    helpElement.textContent = presets[presetName]?.helpText || presets.balanced.helpText;
};

IndexTTSApp.prototype.applyGenerationPreset = function(presetName, notify = false) {
    const presets = this.getGenerationPresets();
    const preset = presets[presetName] || presets.balanced;
    const presetSelect = document.getElementById('generation-preset');

    if (presetSelect) {
        presetSelect.value = preset.settings.generation_preset;
    }

    this.applyGenerationSettings(preset.settings);
    this.updateGenerationPresetHelp(preset.settings.generation_preset);

    if (notify && this.showNotification) {
        this.showNotification('Preset Applied', `${preset.label} preset loaded`, 'info');
    }
};

IndexTTSApp.prototype.captureGenerationSettings = function() {
    const fixedBaseSeed = parseInt(document.getElementById('fixed-base-seed')?.value || '1234', 10);

    return {
        generation_preset: document.getElementById('generation-preset')?.value || 'balanced',
        pacing_preset: document.getElementById('dialogue-pacing-preset')?.value || 'natural',
        versions_per_line: parseInt(document.getElementById('versions-per-line')?.value || '3', 10),
        similarity_threshold: parseFloat(document.getElementById('similarity-threshold')?.value || '0.6'),
        auto_regen_attempts: parseInt(document.getElementById('auto-regen-attempts')?.value || '1', 10),
        seed_strategy: document.getElementById('seed-strategy')?.value || 'fully_random',
        fixed_base_seed: Number.isFinite(fixedBaseSeed) ? fixedBaseSeed : 1234,
        scene_pacing_profile: document.getElementById('scene-pacing-profile')?.value || 'balanced',
        scene_gap_ms: parseInt(document.getElementById('scene-gap-ms')?.value || '140', 10),
        respect_punctuation_pauses: document.getElementById('respect-punctuation-pauses')?.checked ?? true,
        speaker_pacing: this.captureSpeakerPacingSettings(),
        emotion_weight: parseFloat(document.getElementById('emotion-weight')?.value || '1.0'),
        use_random_sampling: document.getElementById('use-random-sampling')?.checked || false,
        max_text_tokens_per_segment: parseInt(document.getElementById('max-text-tokens')?.value || '120', 10),
        do_sample: document.getElementById('do-sample')?.checked ?? true,
        top_p: parseFloat(document.getElementById('top-p')?.value || '0.8'),
        top_k: parseInt(document.getElementById('top-k')?.value || '30', 10),
        temperature: parseFloat(document.getElementById('temperature')?.value || '0.8'),
        length_penalty: parseFloat(document.getElementById('length-penalty')?.value || '0.0'),
        num_beams: parseInt(document.getElementById('num-beams')?.value || '3', 10),
        repetition_penalty: parseFloat(document.getElementById('repetition-penalty')?.value || '10'),
        max_mel_tokens: parseInt(document.getElementById('max-mel-tokens')?.value || '1500', 10)
    };
};

IndexTTSApp.prototype.applyGenerationSettings = function(settings = {}) {
    const presetSelect = document.getElementById('generation-preset');
    if (presetSelect) {
        presetSelect.value = settings.generation_preset || 'balanced';
    }
    const pacingPresetSelect = document.getElementById('dialogue-pacing-preset');
    const effectivePacingPreset = settings.pacing_preset || pacingPresetSelect?.value || 'natural';
    if (pacingPresetSelect) {
        pacingPresetSelect.value = effectivePacingPreset;
    }

    const assignments = [
        ['versions-per-line', settings.versions_per_line],
        ['similarity-threshold', settings.similarity_threshold],
        ['auto-regen-attempts', settings.auto_regen_attempts],
        ['fixed-base-seed', settings.fixed_base_seed],
        ['scene-gap-ms', settings.scene_gap_ms],
        ['emotion-weight', settings.emotion_weight],
        ['max-text-tokens', settings.max_text_tokens_per_segment],
        ['top-p', settings.top_p],
        ['top-k', settings.top_k],
        ['temperature', settings.temperature],
        ['length-penalty', settings.length_penalty],
        ['num-beams', settings.num_beams],
        ['repetition-penalty', settings.repetition_penalty],
        ['max-mel-tokens', settings.max_mel_tokens]
    ];

    assignments.forEach(([elementId, value]) => {
        if (value === undefined || value === null) return;
        const element = document.getElementById(elementId);
        if (!element) return;
        element.value = value;
        element.dispatchEvent(new Event('input', { bubbles: true }));
    });

    const seedStrategySelect = document.getElementById('seed-strategy');
    const effectiveSeedStrategy = settings.seed_strategy || seedStrategySelect?.value || 'fully_random';
    if (seedStrategySelect) {
        seedStrategySelect.value = effectiveSeedStrategy;
    }

    const scenePacingSelect = document.getElementById('scene-pacing-profile');
    const effectiveScenePacing = settings.scene_pacing_profile || scenePacingSelect?.value || 'balanced';
    if (scenePacingSelect) {
        scenePacingSelect.value = effectiveScenePacing;
    }

    const doSample = document.getElementById('do-sample');
    if (doSample && settings.do_sample !== undefined) {
        doSample.checked = Boolean(settings.do_sample);
    }

    const useRandomSampling = document.getElementById('use-random-sampling');
    if (useRandomSampling && settings.use_random_sampling !== undefined) {
        useRandomSampling.checked = Boolean(settings.use_random_sampling);
    }

    const respectPunctuationPauses = document.getElementById('respect-punctuation-pauses');
    if (respectPunctuationPauses && settings.respect_punctuation_pauses !== undefined) {
        respectPunctuationPauses.checked = Boolean(settings.respect_punctuation_pauses);
    }

    this.updateGenerationPresetHelp(settings.generation_preset || 'balanced');
    this.updateSeedStrategyUi(effectiveSeedStrategy);
    this.updateDialoguePacingPresetHelp(effectivePacingPreset);
    this.updateScenePacingUi(effectiveScenePacing, false);
    this.pendingSpeakerPacingSettings = Array.isArray(settings.speaker_pacing) ? settings.speaker_pacing : [];
    if (this.pendingSpeakerPacingSettings.length) {
        this.applySpeakerPacingSettings(this.pendingSpeakerPacingSettings);
    } else if (document.querySelectorAll('.speaker-pacing-row').length) {
        this.applySpeakerPacingDefaults(this.getDialoguePacingPresetDefaultRate(effectivePacingPreset));
    }
};

IndexTTSApp.prototype.renderScriptPreview = function(lines = []) {
    const previewContent = document.getElementById('preview-content');
    const scriptPreview = document.getElementById('script-preview');
    if (!previewContent || !scriptPreview) return;

    if (!lines.length) {
        previewContent.innerHTML = '';
        scriptPreview.style.display = 'none';
        return;
    }

    previewContent.innerHTML = '';

    lines.forEach((line, index) => {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'preview-line';

        const emotionIndicator = line.emo_auto_detected
            ? `<span class="emotion-indicator auto-detected" title="Auto-detected emotion: ${JSON.stringify(line.emo_detected_dict || {})}">AI</span>`
            : '';

        lineDiv.innerHTML = `
            <strong>Line ${index + 1}:</strong>
            <span class="speaker-name">${line.speaker || (line.speaker_filename || '').replace('.wav', '')}</span>:
            "${line.text}"
            ${emotionIndicator}
        `;
        previewContent.appendChild(lineDiv);
    });

    scriptPreview.style.display = 'block';
};

IndexTTSApp.prototype.setScriptPackStatus = function(message, type = 'info') {
    const statusElement = document.getElementById('conversation-script-pack-status');
    if (!statusElement) return;

    statusElement.textContent = message;
    statusElement.dataset.status = type;
};

IndexTTSApp.prototype.extractScriptPackSection = function(markdown, headings = []) {
    if (!markdown || !headings.length) {
        return '';
    }

    const normalized = String(markdown).replace(/\r\n/g, '\n');
    const lines = normalized.split('\n');
    const normalizedHeadings = headings.map((heading) => String(heading).trim().toLowerCase());

    let startIndex = -1;
    for (let index = 0; index < lines.length; index += 1) {
        const lineHeadingText = lines[index].trim().replace(/^#+\s*/, '').trim().toLowerCase();
        if (/^##+\s+/.test(lines[index]) && normalizedHeadings.includes(lineHeadingText)) {
            startIndex = index + 1;
            break;
        }
    }

    if (startIndex === -1) {
        return '';
    }

    let endIndex = lines.length;
    for (let index = startIndex; index < lines.length; index += 1) {
        if (/^##+\s+/.test(lines[index])) {
            endIndex = index;
            break;
        }
    }

    return lines.slice(startIndex, endIndex).join('\n').trim();
};

IndexTTSApp.prototype.extractScriptPackCodeFence = function(sectionText, languagePattern = '[a-zA-Z0-9_-]*') {
    if (!sectionText) {
        return '';
    }

    const fenceRegex = new RegExp(`\\\`\\\`\\\`(?:${languagePattern})?\\s*([\\s\\S]*?)\\\`\\\`\\\``, 'i');
    const match = sectionText.match(fenceRegex);
    return match ? String(match[1] || '').trim() : '';
};

IndexTTSApp.prototype.parseScriptPackLinePlans = function(planText) {
    const normalized = String(planText || '').replace(/\r/g, '').trim();
    if (!normalized || !/^lines:\s*$/im.test(normalized)) {
        return [];
    }

    const linesSectionMatch = normalized.match(/^\s*lines:\s*$([\s\S]*)/im);
    if (!linesSectionMatch) {
        return [];
    }

    const sectionLines = String(linesSectionMatch[1] || '').split('\n');
    const entryBlocks = [];
    let currentBlock = [];

    sectionLines.forEach((line) => {
        if (/^\s*-\s+id:\s*/.test(line)) {
            if (currentBlock.length) {
                entryBlocks.push(currentBlock.join('\n'));
            }
            currentBlock = [line];
            return;
        }

        if (currentBlock.length) {
            currentBlock.push(line);
        }
    });

    if (currentBlock.length) {
        entryBlocks.push(currentBlock.join('\n'));
    }

    const linePlans = [];

    entryBlocks.forEach((blockText) => {
        const block = String(blockText || '');
        const idMatch = block.match(/^\s*-\s+id:\s*(.+?)\s*$/m);
        if (!idMatch) {
            return;
        }

        const rawId = String(idMatch[1] || '').trim().replace(/^['"]|['"]$/g, '');
        const indexMatch = rawId.match(/^L(\d+)$/i);
        if (!indexMatch) {
            return;
        }

        const lineIndex = Math.max(0, parseInt(indexMatch[1], 10) - 1);
        const readField = (fieldName) => {
            const fieldMatch = block.match(new RegExp(`^\\s+${fieldName}:\\s*(.+?)\\s*$`, 'im'));
            if (!fieldMatch) {
                return null;
            }
            return String(fieldMatch[1] || '').trim().replace(/^['"]|['"]$/g, '');
        };

        const readNumber = (fieldName) => {
            const rawValue = readField(fieldName);
            if (rawValue === null || rawValue === '') {
                return null;
            }
            const parsedValue = Number(rawValue);
            return Number.isFinite(parsedValue) ? parsedValue : null;
        };

        const readBoolean = (fieldName) => {
            const rawValue = readField(fieldName);
            if (rawValue === null) {
                return null;
            }
            if (/^(true|yes|on)$/i.test(rawValue)) return true;
            if (/^(false|no|off)$/i.test(rawValue)) return false;
            return null;
        };

        linePlans.push({
            id: rawId,
            lineIndex,
            speaker: readField('speaker'),
            emotion_text: readField('emotion_text'),
            emotion_weight: readNumber('emotion_weight'),
            start_mode: readField('start_mode'),
            gap_after_ms: readNumber('gap_after_ms'),
            overlap_prev_ms: readNumber('overlap_prev_ms'),
            duck_prev_db: readNumber('duck_prev_db'),
            allow_overlap: readBoolean('allow_overlap'),
        });
    });

    return linePlans;
};

IndexTTSApp.prototype.mapEmotionTextToVector = function(emotionText, fallbackVector = null) {
    const fallback = Array.isArray(fallbackVector) && fallbackVector.length === 8
        ? [...fallbackVector]
        : [0, 0, 0, 0, 0, 0, 0, 0.8];
    const text = String(emotionText || '').toLowerCase();

    if (!text) {
        return fallback;
    }

    if (/(angry|irritated|offended|loud|shameless|defiant)/.test(text)) {
        return [0, 0, 0.8, 0, 0.2, 0, 0, 0];
    }

    if (/(sad|cry|crying|hurt|melancholic|worried|voice cracking)/.test(text)) {
        return [0, 0.8, 0, 0, 0, 0.3, 0, 0];
    }

    if (/(surprised|baffled|disbelief|shocked|chaotic)/.test(text)) {
        return [0.2, 0, 0, 0, 0, 0, 0.9, 0];
    }

    if (/(happy|amused|funny|boastful|smug|dry humor)/.test(text)) {
        return [0.65, 0, 0, 0, 0, 0, 0.15, 0.2];
    }

    if (/(calm|controlled|practical|deadpan|whisper|quiet|stern|dignity)/.test(text)) {
        return [0, 0, 0, 0, 0, 0, 0, 0.8];
    }

    return fallback;
};

IndexTTSApp.prototype.applyLoadedScriptPackPlan = function(lines) {
    const linePlans = this.loadedScriptPack?.linePlans || [];
    if (!Array.isArray(lines) || !lines.length || !linePlans.length) {
        return 0;
    }

    let appliedCount = 0;

    lines.forEach((line, index) => {
        const linePlan = linePlans.find(plan => plan.lineIndex === index);
        if (!linePlan) {
            return;
        }

        if (linePlan.emotion_text) {
            line.emotion_text = linePlan.emotion_text;
            line.emotion_control_method = 'from_text';
            line.emotion_weight = linePlan.emotion_weight ?? parseFloat(document.getElementById('emotion-weight')?.value || '1.0');
            line.emo_vector = this.mapEmotionTextToVector(linePlan.emotion_text, line.emo_vector);
            line.emo_auto_detected = false;
            appliedCount += 1;
        }
    });

    return appliedCount;
};

IndexTTSApp.prototype.parseConversationScriptPackContent = function(content, sourceName = 'script pack') {
    const markdown = String(content || '').replace(/\r\n/g, '\n');
    const headingMatch = markdown.match(/^#\s+(.+)$/m);
    const packTitle = headingMatch ? String(headingMatch[1] || '').trim() : '';

    const scriptSection = this.extractScriptPackSection(markdown, ['Pasteable Script', 'Script', 'Paste Script']);
    const scriptText = this.extractScriptPackCodeFence(scriptSection, '(?:text|txt)?')
        || scriptSection
        || markdown;

    const planSection = this.extractScriptPackSection(markdown, ['Emotion And Timing Plan', 'Timing Plan', 'Overlap Plan']);
    const overlapPlanText = this.extractScriptPackCodeFence(planSection, '(?:yaml|yml|json|text|txt)?') || '';
    const linePlans = this.parseScriptPackLinePlans(overlapPlanText);

    return {
        sourceName,
        title: packTitle,
        scriptText: String(scriptText || '').trim(),
        overlapPlanText: String(overlapPlanText || '').trim(),
        linePlans,
    };
};

IndexTTSApp.prototype.loadConversationScriptPackFile = function(event) {
    const file = event?.target?.files?.[0];
    if (!file) {
        return;
    }

    const reader = new FileReader();
    reader.onload = async () => {
        try {
            const parsedPack = this.parseConversationScriptPackContent(reader.result || '', file.name);
            if (!parsedPack.scriptText) {
                throw new Error('No script block was found in that file');
            }

            this.loadedScriptPack = parsedPack;

            const titleInput = document.getElementById('conversation-title');
            const scriptTextarea = document.getElementById('conversation-script');
            const autoDetectEmotion = document.getElementById('auto-detect-emotion');
            const overlapPlanTextarea = document.getElementById('overlap-plan-text');
            const saveNameInput = document.getElementById('project-save-name');

            if (titleInput && parsedPack.title) {
                titleInput.value = parsedPack.title;
            }
            if (scriptTextarea) {
                scriptTextarea.value = parsedPack.scriptText;
            }
            if (overlapPlanTextarea && parsedPack.overlapPlanText) {
                overlapPlanTextarea.value = parsedPack.overlapPlanText;
            }
            if (autoDetectEmotion) {
                autoDetectEmotion.checked = parsedPack.linePlans.length === 0;
            }
            if (saveNameInput && (!saveNameInput.value || saveNameInput.value === 'project.json')) {
                saveNameInput.value = this.generateDefaultProjectSaveName();
            }

            this.setScriptPackStatus(
                parsedPack.linePlans.length
                    ? `Loaded ${file.name}. Imported the script and attached ${parsedPack.linePlans.length} line-level emotion/timing hints.`
                    : `Loaded ${file.name}. Imported the script and any companion timing plan text that was present.`,
                'success'
            );

            await this.parseScript();
            this.showNotification('Success', `Loaded script pack ${file.name}`, 'success');
        } catch (error) {
            console.error('Failed to load conversation script pack:', error);
            this.loadedScriptPack = null;
            this.setScriptPackStatus(`Failed to load script pack: ${error.message}`, 'error');
            this.showNotification('Error', `Failed to load script pack: ${error.message}`, 'error');
        }
    };
    reader.onerror = () => {
        console.error('Failed to read script pack file:', reader.error);
        this.loadedScriptPack = null;
        this.setScriptPackStatus('Failed to read script pack file', 'error');
        this.showNotification('Error', 'Failed to read script pack file', 'error');
    };
    reader.readAsText(file);
};

IndexTTSApp.prototype.buildProjectSavePayload = function() {
    return {
        conversationTitle: document.getElementById('conversation-title')?.value?.trim() || '',
        scriptText: document.getElementById('conversation-script')?.value || '',
        autoDetectEmotion: document.getElementById('auto-detect-emotion')?.checked || false,
        overlapPlanText: document.getElementById('overlap-plan-text')?.value || '',
        generationSettings: this.captureGenerationSettings(),
        parsedScript: this.parsedScript ? this.cloneProjectState(this.parsedScript) : null,
        conversationScript: this.conversationScript ? this.cloneProjectState(this.conversationScript) : [],
        currentConversationId: this.currentConversationId,
        currentConversationData: this.currentConversationData ? this.cloneProjectState(this.currentConversationData) : null,
        currentTab: this.currentTab || 'conversation-workflow'
    };
};

IndexTTSApp.prototype.renderSavedProjects = function(selectedSaveName = '') {
    const select = document.getElementById('saved-projects-select');
    if (!select) return;

    const selectedValue = selectedSaveName || select.value;
    select.innerHTML = '<option value="">Select a saved project…</option>';

    this.savedProjects.forEach(project => {
        const option = document.createElement('option');
        option.value = project.save_name;
        const savedAt = project.saved_at ? new Date(project.saved_at).toLocaleString() : 'unknown time';
        option.textContent = `${project.title || project.save_name} (${savedAt})`;
        select.appendChild(option);
    });

    if (selectedValue && this.savedProjects.some(project => project.save_name === selectedValue)) {
        select.value = selectedValue;
    }
};

IndexTTSApp.prototype.refreshSavedProjects = async function() {
    try {
        const response = await this.apiRequest('/conversation/projects');
        this.savedProjects = response.details?.projects || [];
        this.renderSavedProjects();

        const saveNameInput = document.getElementById('project-save-name');
        if (saveNameInput && !saveNameInput.value.trim()) {
            saveNameInput.value = this.generateDefaultProjectSaveName();
        }

        if (typeof this.refreshStudioShell === 'function') {
            this.refreshStudioShell();
        }
    } catch (error) {
        console.error('Failed to refresh saved projects:', error);
        this.setProjectSaveStatus(`Failed to refresh saved projects: ${error.message}`, 'error');
    }
};

IndexTTSApp.prototype.saveCurrentProject = async function() {
    try {
        const saveNameInput = document.getElementById('project-save-name');
        const saveName = saveNameInput?.value?.trim() || this.generateDefaultProjectSaveName();

        const response = await this.apiRequest('/conversation/projects/save', {
            method: 'POST',
            body: JSON.stringify({
                save_name: saveName,
                project_data: this.buildProjectSavePayload()
            })
        });

        if (saveNameInput) {
            saveNameInput.value = response.details?.save_name || saveName;
        }

        await this.refreshSavedProjects();
        this.renderSavedProjects(response.details?.save_name || saveName);
        this.setProjectSaveStatus(`Saved project to ${response.details?.save_name || saveName}`, 'success');
        this.showNotification('Success', 'Project saved successfully', 'success');
        if (typeof this.refreshStudioShell === 'function') {
            this.refreshStudioShell();
        }
    } catch (error) {
        console.error('Failed to save project:', error);
        this.setProjectSaveStatus(`Failed to save project: ${error.message}`, 'error');
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.restoreProjectState = async function(loadDetails) {
    const projectData = loadDetails?.project_data || {};
    const uiState = projectData.ui_state || {};

    const conversationTitleInput = document.getElementById('conversation-title');
    const scriptTextarea = document.getElementById('conversation-script');
    const autoDetectEmotion = document.getElementById('auto-detect-emotion');
    const saveNameInput = document.getElementById('project-save-name');
    const overlapPlanTextarea = document.getElementById('overlap-plan-text');
    const overlapPlanFileInput = document.getElementById('overlap-plan-file');

    if (conversationTitleInput) conversationTitleInput.value = uiState.conversationTitle || '';
    if (scriptTextarea) scriptTextarea.value = uiState.scriptText || '';
    if (autoDetectEmotion) autoDetectEmotion.checked = Boolean(uiState.autoDetectEmotion);
    if (saveNameInput && loadDetails?.save_name) saveNameInput.value = loadDetails.save_name;
    if (overlapPlanTextarea) overlapPlanTextarea.value = uiState.overlapPlanText || '';
    if (overlapPlanFileInput) overlapPlanFileInput.value = '';

    this.applyGenerationSettings(uiState.generationSettings || {});
    if (typeof this.applyConversationMixPacingSettings === 'function') {
        this.applyConversationMixPacingSettings(uiState.generationSettings || {});
    }

    this.parsedScript = uiState.parsedScript || null;
    this.conversationScript = uiState.conversationScript || (this.parsedScript?.lines ? this.cloneProjectState(this.parsedScript.lines) : []);
    this.currentConversationId = loadDetails?.restored_conversation_id || uiState.currentConversationId || null;
    this.currentConversationData = uiState.currentConversationData || null;

    if (this.conversationScript && this.conversationScript.length > 0) {
        this.renderScriptPreview(this.conversationScript);
        this.renderSpeakerPacingControls(this.conversationScript);
        this.generateTimeline();
    } else {
        this.renderScriptPreview([]);
        this.renderSpeakerPacingControls([]);
        const timelineContainer = document.getElementById('emotion-timeline-container');
        if (timelineContainer) timelineContainer.remove();
    }

    await this.loadConversations();

    if (this.currentConversationId) {
        await this.selectConversation(this.currentConversationId);
    } else if (this.currentConversationData?.lines?.length) {
        this.renderLineVersions();
    }

    const targetTab = uiState.currentTab || (this.currentConversationId ? 'conversation-results' : 'conversation-workflow');
    this.switchTab(targetTab);
    if (typeof this.refreshStudioShell === 'function') {
        this.refreshStudioShell();
    }
};

IndexTTSApp.prototype.loadSelectedProject = async function() {
    const select = document.getElementById('saved-projects-select');
    const saveName = select?.value;

    if (!saveName) {
        this.showNotification('Error', 'Please select a saved project', 'error');
        return;
    }

    try {
        const response = await this.apiRequest(`/conversation/projects/${encodeURIComponent(saveName)}`);
        await this.restoreProjectState(response.details);
        this.setProjectSaveStatus(`Loaded project from ${response.details?.save_name || saveName}`, 'success');
        this.showNotification('Success', 'Project loaded successfully', 'success');
        if (typeof this.refreshStudioShell === 'function') {
            this.refreshStudioShell();
        }
    } catch (error) {
        console.error('Failed to load project:', error);
        this.setProjectSaveStatus(`Failed to load project: ${error.message}`, 'error');
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.deleteSelectedProject = async function() {
    const select = document.getElementById('saved-projects-select');
    const saveName = select?.value;

    if (!saveName) {
        this.showNotification('Error', 'Please select a saved project', 'error');
        return;
    }

    if (!window.confirm(`Delete saved project "${saveName}"?`)) {
        return;
    }

    try {
        await this.apiRequest(`/conversation/projects/${encodeURIComponent(saveName)}`, {
            method: 'DELETE'
        });
        await this.refreshSavedProjects();
        this.setProjectSaveStatus(`Deleted project ${saveName}`, 'success');
        this.showNotification('Success', 'Project deleted successfully', 'success');
    } catch (error) {
        console.error('Failed to delete project:', error);
        this.setProjectSaveStatus(`Failed to delete project: ${error.message}`, 'error');
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.resetActiveProjectUi = function() {
    if (this.generationInterval) {
        clearInterval(this.generationInterval);
        this.generationInterval = null;
    }

    if (this.regenerationInterval) {
        clearInterval(this.regenerationInterval);
        this.regenerationInterval = null;
    }

    this.parsedScript = null;
    this.conversationScript = [];
    this.currentConversationId = null;
    this.currentConversationData = null;
    this.loadedScriptPack = null;

    const conversationTitleInput = document.getElementById('conversation-title');
    const scriptTextarea = document.getElementById('conversation-script');
    const autoDetectEmotion = document.getElementById('auto-detect-emotion');
    const saveNameInput = document.getElementById('project-save-name');
    const savedProjectsSelect = document.getElementById('saved-projects-select');
    const scriptPreview = document.getElementById('script-preview');
    const timelineContainer = document.getElementById('emotion-timeline-container');
    const generationProgress = document.getElementById('generation-progress');
    const generationResult = document.getElementById('generation-result');
    const generateConversationBtn = document.getElementById('generate-conversation-btn');
    const lineVersionsSection = document.getElementById('line-versions-section');
    const linesContainer = document.getElementById('lines-container');
    const concatenationSection = document.getElementById('concatenation-section');
    const concatenationProgress = document.getElementById('concatenation-progress');
    const concatenationResult = document.getElementById('concatenation-result');
    const selectedLinesCount = document.getElementById('selected-lines-count');
    const totalLinesCount = document.getElementById('total-lines-count');
    const audioPlayerModal = document.getElementById('audio-player-modal');
    const overlapPlanTextarea = document.getElementById('overlap-plan-text');
    const overlapPlanFileInput = document.getElementById('overlap-plan-file');
    const scriptPackFileInput = document.getElementById('conversation-script-pack-file');

    if (conversationTitleInput) conversationTitleInput.value = '';
    if (scriptTextarea) scriptTextarea.value = '';
    if (autoDetectEmotion) autoDetectEmotion.checked = true;
    if (saveNameInput) saveNameInput.value = 'project.json';
    if (savedProjectsSelect) savedProjectsSelect.value = '';
    if (overlapPlanTextarea) overlapPlanTextarea.value = '';
    if (overlapPlanFileInput) overlapPlanFileInput.value = '';
    if (scriptPackFileInput) scriptPackFileInput.value = '';
    this.setScriptPackStatus('Pick a markdown script pack from the repo to load its title, pasteable script, and timing/emotion plan into the page.', 'info');

    this.applyGenerationSettings(this.getDefaultGenerationSettings());
    if (typeof this.applyConversationMixPacingSettings === 'function') {
        this.applyConversationMixPacingSettings(this.getDefaultGenerationSettings());
    }
    this.renderScriptPreview([]);
    this.renderSpeakerPacingControls([]);
    this.lastConcatenationPlanApplied = false;

    if (scriptPreview) {
        scriptPreview.style.display = 'none';
    }

    if (timelineContainer) {
        timelineContainer.remove();
    }

    if (generationProgress) {
        generationProgress.style.display = 'none';
        const progressFill = generationProgress.querySelector('.progress-fill');
        const progressText = generationProgress.querySelector('.progress-text');
        const progressPercent = generationProgress.querySelector('.progress-percent');
        const generationLogs = document.getElementById('generation-logs');
        if (progressFill) progressFill.style.width = '0%';
        if (progressText) progressText.textContent = 'Initializing...';
        if (progressPercent) progressPercent.textContent = '0%';
        if (generationLogs) generationLogs.innerHTML = '';
    }

    if (generationResult) {
        generationResult.style.display = 'none';
    }

    if (generateConversationBtn) {
        generateConversationBtn.disabled = false;
    }

    if (lineVersionsSection) {
        lineVersionsSection.style.display = 'none';
    }

    if (linesContainer) {
        linesContainer.innerHTML = '';
    }

    if (concatenationSection) {
        concatenationSection.style.display = 'none';
    }

    if (concatenationProgress) {
        concatenationProgress.style.display = 'none';
    }

    if (concatenationResult) {
        concatenationResult.innerHTML = '';
        concatenationResult.className = 'result-container';
        concatenationResult.style.display = 'none';
    }

    if (selectedLinesCount) selectedLinesCount.textContent = '0';
    if (totalLinesCount) totalLinesCount.textContent = '0';

    if (audioPlayerModal) {
        audioPlayerModal.classList.remove('show');
    }

    if (this.customMediaPlayer?.stop) {
        try {
            this.customMediaPlayer.stop();
        } catch (error) {
            console.warn('Failed to stop media player during project reset:', error);
        }
    }

    if (typeof this.refreshListeningFeedbackExport === 'function') {
        this.refreshListeningFeedbackExport();
    }

    this.renderConversations();
    this.switchTab('conversation-workflow');
    this.setProjectSaveStatus('New project ready. Enter a script or load a saved project.', 'info');
    if (typeof this.refreshStudioShell === 'function') {
        this.refreshStudioShell();
    }
};

IndexTTSApp.prototype.startNewProject = function() {
    const hasProjectState = Boolean(
        document.getElementById('conversation-title')?.value?.trim() ||
        document.getElementById('conversation-script')?.value?.trim() ||
        this.parsedScript ||
        this.currentConversationId ||
        this.currentConversationData?.lines?.length
    );

    if (hasProjectState) {
        const shouldReset = window.confirm(
            'Start a new project? This clears the current script, loaded conversation, and selections from the page. Saved projects and older generated conversations will stay available.'
        );

        if (!shouldReset) {
            return;
        }
    }

    this.resetActiveProjectUi();
    this.showNotification('Success', 'Started a new project', 'success');
};

IndexTTSApp.prototype.parseScript = async function() {
    const scriptText = document.getElementById('conversation-script').value.trim();
    
    if (!scriptText) {
        this.showNotification('Error', 'Please enter a conversation script', 'error');
        return;
    }
    
    // Check if auto emotion detection is enabled
    const autoDetectEmotion = document.getElementById('auto-detect-emotion')?.checked || false;
    
    // Parse script lines
    const lines = [];
    const scriptLines = scriptText.split('\n').filter(line => line.trim());
    
    for (let i = 0; i < scriptLines.length; i++) {
        const line = scriptLines[i].trim();
        const match = line.match(/^([^:]+):\s*(.+)$/);
        
        if (match) {
            const speaker = match[1].trim();
            const text = match[2].trim();
            
            // Check if speaker exists using enhanced matching logic
            console.log('DEBUG: Checking speaker:', speaker);
            console.log('DEBUG: Available speakers:', this.speakers);
            console.log('DEBUG: Speaker filenames:', this.speakers.map(s => s.filename));
            console.log('DEBUG: Speaker names:', this.speakers.map(s => s.name));
            console.log('DEBUG: Total speakers loaded:', this.speakers.length);
            
            let matchedSpeaker = null;
            
            // Try exact filename match first
            matchedSpeaker = this.speakers.find(s => s.filename === speaker);
            if (matchedSpeaker) {
                console.log('DEBUG: Found speaker by exact filename match:', matchedSpeaker.filename);
            }
            
            // Try name match if exact filename match failed
            if (!matchedSpeaker) {
                matchedSpeaker = this.speakers.find(s => s.name === speaker);
                if (matchedSpeaker) {
                    console.log('DEBUG: Found speaker by name match:', matchedSpeaker.filename);
                }
            }
            
            // Try filename with .wav extension if both failed
            if (!matchedSpeaker) {
                const speakerWithWav = speaker.endsWith('.wav') ? speaker : `${speaker}.wav`;
                console.log('DEBUG: Trying with .wav extension:', speakerWithWav);
                matchedSpeaker = this.speakers.find(s => s.filename === speakerWithWav);
                if (matchedSpeaker) {
                    console.log('DEBUG: Found speaker by adding .wav extension:', matchedSpeaker.filename);
                }
            }
            
            // Try name without .wav extension if all else failed
            if (!matchedSpeaker) {
                const speakerWithoutWav = speaker.replace('.wav', '');
                console.log('DEBUG: Trying name without .wav extension:', speakerWithoutWav);
                matchedSpeaker = this.speakers.find(s => s.name === speakerWithoutWav);
                if (matchedSpeaker) {
                    console.log('DEBUG: Found speaker by name without .wav extension:', matchedSpeaker.filename);
                }
            }
            
            // Additional debugging: try case-insensitive matching
            if (!matchedSpeaker) {
                console.log('DEBUG: Trying case-insensitive matching...');
                const speakerLower = speaker.toLowerCase();
                matchedSpeaker = this.speakers.find(s =>
                    s.filename.toLowerCase() === speakerLower ||
                    s.name.toLowerCase() === speakerLower ||
                    s.filename.toLowerCase() === `${speakerLower}.wav` ||
                    s.name.toLowerCase() === speakerLower.replace('.wav', '')
                );
                if (matchedSpeaker) {
                    console.log('DEBUG: Found speaker by case-insensitive match:', matchedSpeaker.filename);
                }
            }
            
            console.log('DEBUG: Final matched speaker:', matchedSpeaker);
            console.log('DEBUG: Speaker matching attempts completed for:', speaker);
            
            if (!matchedSpeaker) {
                this.showNotification('Error', `Speaker not found: ${speaker}`, 'error');
                return;
            }
            
            // Create line object with default emotion vector (calm)
            lines.push({
                line: lines.length + 1,
                speaker: matchedSpeaker.filename.replace('.wav', ''),
                speaker_filename: matchedSpeaker.filename,
                text: text,
                line_number: i,
                emo_vector: [0, 0, 0, 0, 0, 0, 0, 0.8], // Default to 'calm' emotion
                emo_auto_detected: false // Flag to track auto-detected emotions
            });
        } else {
            this.showNotification('Error', `Invalid line format: ${line}`, 'error');
            return;
        }
    }
    
    if (lines.length === 0) {
        this.showNotification('Error', 'No valid conversation lines found', 'error');
        return;
    }
    
    // Auto-detect emotions if enabled
    if (autoDetectEmotion) {
        try {
            this.showNotification('Info', 'Detecting emotions for script lines...', 'info');
            
            // Detect emotions for all lines
            const emotionResults = await this.detectEmotionsForScript(lines);
            
            // Apply detected emotions to lines
            emotionResults.forEach((result, index) => {
                if (index < lines.length) {
                    // Handle both batch response format and individual response format
                    if (result.details && result.details.emotion_vectors) {
                        // Batch response format: result.details.emotion_vectors
                        lines[index].emo_vector = result.details.emotion_vectors;
                        lines[index].emo_detected_dict = result.details.emotion_dict;
                    } else if (result.emotion_vectors) {
                        // Individual response format: result.emotion_vectors
                        lines[index].emo_vector = result.emotion_vectors;
                        lines[index].emo_detected_dict = result.emotion_dict;
                    } else {
                        console.error(`DEBUG: No emotion vectors found in result for line ${index + 1}:`, result);
                        return;
                    }
                    lines[index].emo_auto_detected = true;
                    console.log(`DEBUG: Applied auto-detected emotion to line ${index + 1}:`, lines[index].emo_vector);
                }
            });
            
            this.showNotification('Success', `Emotion detection completed for ${lines.length} lines`, 'success');
        } catch (error) {
            console.error('Emotion detection failed:', error);
            this.showNotification('Warning', 'Emotion detection failed, using default emotions', 'warning');
        }
    }
    
    // Store conversation script with emotion vectors
    this.conversationScript = lines;
    
    // Generate timeline
    this.generateTimeline();
    
    // Show preview
    const previewContent = document.getElementById('preview-content');
    previewContent.innerHTML = '';
    
    lines.forEach((line, index) => {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'preview-line';
        
        // Add emotion indicator if auto-detected
        const emotionIndicator = line.emo_auto_detected ?
            `<span class="emotion-indicator auto-detected" title="Auto-detected emotion: ${JSON.stringify(line.emo_detected_dict)}">🤖</span>` : '';
        
        lineDiv.innerHTML = `
            <strong>Line ${index + 1}:</strong>
            <span class="speaker-name">${line.speaker}</span>:
            "${line.text}"
            ${emotionIndicator}
        `;
        previewContent.appendChild(lineDiv);
    });
    
    document.getElementById('script-preview').style.display = 'block';
    
    // Store parsed script for generation
    this.parsedScript = {
        title: document.getElementById('conversation-title').value.trim() || 'Untitled Conversation',
        lines: lines
    };
    
    this.showNotification('Success', `Parsed ${lines.length} conversation lines`, 'success');
};

IndexTTSApp.prototype.parseScript = async function() {
    const scriptText = document.getElementById('conversation-script').value.trim();

    if (!scriptText) {
        this.showNotification('Error', 'Please enter a conversation script', 'error');
        return;
    }

    const autoDetectEmotion = document.getElementById('auto-detect-emotion')?.checked || false;
    const lines = [];
    const scriptLines = scriptText.split('\n').filter(line => line.trim());

    for (let i = 0; i < scriptLines.length; i++) {
        const line = scriptLines[i].trim();
        const match = line.match(/^([^:]+):\s*(.+)$/);

        if (!match) {
            this.showNotification('Error', `Invalid line format: ${line}`, 'error');
            return;
        }

        const speaker = match[1].trim();
        const text = match[2].trim();

        console.log('DEBUG: Checking speaker:', speaker);
        console.log('DEBUG: Available speakers:', this.speakers);
        console.log('DEBUG: Speaker filenames:', this.speakers.map(s => s.filename));
        console.log('DEBUG: Speaker names:', this.speakers.map(s => s.name));
        console.log('DEBUG: Total speakers loaded:', this.speakers.length);

        let matchedSpeaker = this.speakers.find(s => s.filename === speaker);

        if (!matchedSpeaker) {
            matchedSpeaker = this.speakers.find(s => s.name === speaker);
        }

        if (!matchedSpeaker) {
            const speakerWithWav = speaker.endsWith('.wav') ? speaker : `${speaker}.wav`;
            matchedSpeaker = this.speakers.find(s => s.filename === speakerWithWav);
        }

        if (!matchedSpeaker) {
            const speakerWithoutWav = speaker.replace('.wav', '');
            matchedSpeaker = this.speakers.find(s => s.name === speakerWithoutWav);
        }

        if (!matchedSpeaker) {
            const speakerLower = speaker.toLowerCase();
            matchedSpeaker = this.speakers.find(s =>
                s.filename.toLowerCase() === speakerLower ||
                s.name.toLowerCase() === speakerLower ||
                s.filename.toLowerCase() === `${speakerLower}.wav` ||
                s.name.toLowerCase() === speakerLower.replace('.wav', '')
            );
        }

        if (!matchedSpeaker) {
            this.showNotification('Error', `Speaker not found: ${speaker}`, 'error');
            return;
        }

        lines.push({
            line: lines.length + 1,
            speaker: matchedSpeaker.filename.replace('.wav', ''),
            speaker_filename: matchedSpeaker.filename,
            text: text,
            line_number: i,
            emo_vector: [0, 0, 0, 0, 0, 0, 0, 0.8],
            emo_auto_detected: false
        });
    }

    if (lines.length === 0) {
        this.showNotification('Error', 'No valid conversation lines found', 'error');
        return;
    }

    if (autoDetectEmotion) {
        try {
            this.showNotification('Info', 'Detecting emotions for script lines...', 'info');

            const emotionResults = await this.detectEmotionsForScript(lines);
            emotionResults.forEach((result, index) => {
                if (index >= lines.length) {
                    return;
                }

                if (result.details && result.details.emotion_vectors) {
                    lines[index].emo_vector = result.details.emotion_vectors;
                    lines[index].emo_detected_dict = result.details.emotion_dict;
                } else if (result.emotion_vectors) {
                    lines[index].emo_vector = result.emotion_vectors;
                    lines[index].emo_detected_dict = result.emotion_dict;
                } else {
                    console.error(`DEBUG: No emotion vectors found in result for line ${index + 1}:`, result);
                    return;
                }

                lines[index].emo_auto_detected = true;
            });

            this.showNotification('Success', `Emotion detection completed for ${lines.length} lines`, 'success');
        } catch (error) {
            console.error('Emotion detection failed:', error);
            this.showNotification('Warning', 'Emotion detection failed, using default emotions', 'warning');
        }
    }

    const importedEmotionCount = this.applyLoadedScriptPackPlan(lines);

    this.conversationScript = lines;
    this.renderSpeakerPacingControls(lines);
    this.generateTimeline();
    this.renderScriptPreview(lines);

    this.parsedScript = {
        title: document.getElementById('conversation-title').value.trim() || 'Untitled Conversation',
        lines: lines
    };

    const saveNameInput = document.getElementById('project-save-name');
    if (saveNameInput && !saveNameInput.value.trim()) {
        saveNameInput.value = this.generateDefaultProjectSaveName();
    }

    if (importedEmotionCount > 0) {
        this.setScriptPackStatus(`Parsed ${lines.length} lines and applied ${importedEmotionCount} imported line-level emotion hints.`, 'success');
    }

    this.showNotification('Success', `Parsed ${lines.length} conversation lines`, 'success');
};

// Removed updateEmotionControls - emotions are now handled entirely by the timeline

IndexTTSApp.prototype.generateConversation = async function() {
    if (!this.parsedScript) {
        this.showNotification('Error', 'Please parse a script first', 'error');
        return;
    }

    const generationSettings = this.captureGenerationSettings();
    const generationRequest = {
        ...generationSettings,
        script: {
            title: this.parsedScript?.title || document.getElementById('conversation-title')?.value?.trim() || 'Untitled Conversation',
            lines: this.cloneProjectState(this.conversationScript || this.parsedScript?.lines || [])
        },
        emotion_control_method: 'from_speaker'
    };
    console.log('DEBUG: Complete generation request:', generationRequest);
    
    // Timeline emotions are already embedded in the script lines
    if (generationRequest.script.lines && generationRequest.script.lines.length > 0) {
        console.log('DEBUG: Using timeline-based emotion vectors for all lines');
        generationRequest.script.lines.forEach(line => {
            console.log(`DEBUG: Line ${line.line} emotion vector:`, line.emo_vector);
        });
    }
    
    const progressContainer = document.getElementById('generation-progress');
    const resultContainer = document.getElementById('generation-result');
    const button = document.getElementById('generate-conversation-btn');
    const progressFill = progressContainer.querySelector('.progress-fill');
    const progressText = progressContainer.querySelector('.progress-text');
    const progressPercent = progressContainer.querySelector('.progress-percent');
    const logsContainer = document.getElementById('generation-logs');
    
    try {
        // Show progress
        progressContainer.style.display = 'block';
        resultContainer.style.display = 'none';
        button.disabled = true;
        logsContainer.innerHTML = '';
        
        // Start generation
        const response = await this.apiRequest('/conversation/generate', {
            method: 'POST',
            body: JSON.stringify(generationRequest)
        });
        
        this.currentConversationId = response.conversation_id;
        
        // Start polling for progress
        this.generationInterval = setInterval(() => {
            this.checkGenerationProgress();
        }, 2000);
        
    } catch (error) {
        this.showNotification('Error', error.message, 'error');
        progressContainer.style.display = 'none';
        button.disabled = false;
    }
};

IndexTTSApp.prototype.checkGenerationProgress = async function() {
    if (!this.currentConversationId) return;
    
    try {
        const response = await this.apiRequest(`/conversation/status/${this.currentConversationId}`);
        const task = response.task;
        
        const progressContainer = document.getElementById('generation-progress');
        const progressFill = progressContainer.querySelector('.progress-fill');
        const progressText = progressContainer.querySelector('.progress-text');
        const progressPercent = progressContainer.querySelector('.progress-percent');
        const logsContainer = document.getElementById('generation-logs');
        
        const queueDetails = task.status === 'queued' && Number.isFinite(Number(task.queue_position))
            ? `Queued (#${Number(task.queue_position)} of ${Math.max(Number(task.queued_generation_tasks || 0), Number(task.queue_position))})`
            : null;

        const progressLabel = queueDetails
            ? `${task.current_step} ${queueDetails}`
            : task.current_step;

        // Update progress
        progressFill.style.width = `${task.progress_percent}%`;
        progressText.textContent = progressLabel;
        progressPercent.textContent = `${Math.round(task.progress_percent)}%`;
        
        // Add log entry
        const logEntry = document.createElement('div');
        logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${progressLabel}`;
        logsContainer.appendChild(logEntry);
        logsContainer.scrollTop = logsContainer.scrollHeight;
        
        // Check if completed
        if (task.status === 'completed') {
            clearInterval(this.generationInterval);
            this.generationInterval = null;
            
            const resultContainer = document.getElementById('generation-result');
            const button = document.getElementById('generate-conversation-btn');
            resultContainer.style.display = 'block';
            if (button) {
                button.disabled = false;
            }
            
            this.showNotification('Success', 'Conversation generation completed!', 'success');
            
            // Load conversations
            this.loadConversations();
        } else if (task.status === 'failed') {
            clearInterval(this.generationInterval);
            this.generationInterval = null;
            
            throw new Error(task.error || 'Generation failed');
        }
        
    } catch (error) {
        clearInterval(this.generationInterval);
        this.generationInterval = null;
        
        const progressContainer = document.getElementById('generation-progress');
        const button = document.getElementById('generate-conversation-btn');
        progressContainer.style.display = 'none';
        button.disabled = false;
        
        this.showNotification('Error', error.message, 'error');
    }
};

// Timeline Generation System
IndexTTSApp.prototype.generateTimeline = function() {
    if (!this.conversationScript || this.conversationScript.length === 0) {
        console.log('DEBUG: No conversation script to generate timeline');
        return;
    }
    
    console.log('DEBUG: Generating timeline for', this.conversationScript.length, 'lines');
    
    // Create timeline container if it doesn't exist
    let timelineContainer = document.getElementById('emotion-timeline-container');
    if (!timelineContainer) {
        timelineContainer = document.createElement('div');
        timelineContainer.id = 'emotion-timeline-container';
        timelineContainer.className = 'emotion-timeline-container';
        
        // Insert after script preview
        const scriptPreview = document.getElementById('script-preview');
        scriptPreview.parentNode.insertBefore(timelineContainer, scriptPreview.nextSibling);
    }
    
    // Generate timeline HTML
    timelineContainer.innerHTML = `
        <div class="emotion-timeline-header">
            <h3><i class="fas fa-chart-line"></i> Emotion Timeline</h3>
            <p>Click on any line to adjust emotions for that specific part of the conversation</p>
        </div>
        <div class="emotion-timeline-content" id="emotion-timeline-content">
            <!-- Timeline blocks will be generated here -->
        </div>
        <div class="emotion-control-panel" id="emotion-control-panel" style="display: none;">
            <div class="emotion-control-header">
                <h4><i class="fas fa-sliders-h"></i> Emotion Control for Line <span id="selected-line-number"></span></h4>
                <button class="btn btn-secondary btn-small" id="close-emotion-control">
                    <i class="fas fa-times"></i> Close
                </button>
            </div>
            <div class="emotion-control-content">
                <div class="emotion-sliders">
                    <div class="emotion-slider-group">
                        <label class="emotion-slider-label">
                            <span class="emotion-dot emotion-happy"></span>
                            Happy
                        </label>
                        <input type="range" class="emotion-slider" id="emotion-slider-happy" min="-1" max="1" step="0.1" value="0">
                        <span class="emotion-slider-value">0</span>
                    </div>
                    <div class="emotion-slider-group">
                        <label class="emotion-slider-label">
                            <span class="emotion-dot emotion-sad"></span>
                            Sad
                        </label>
                        <input type="range" class="emotion-slider" id="emotion-slider-sad" min="-1" max="1" step="0.1" value="0">
                        <span class="emotion-slider-value">0</span>
                    </div>
                    <div class="emotion-slider-group">
                        <label class="emotion-slider-label">
                            <span class="emotion-dot emotion-angry"></span>
                            Angry
                        </label>
                        <input type="range" class="emotion-slider" id="emotion-slider-angry" min="-1" max="1" step="0.1" value="0">
                        <span class="emotion-slider-value">0</span>
                    </div>
                    <div class="emotion-slider-group">
                        <label class="emotion-slider-label">
                            <span class="emotion-dot emotion-surprised"></span>
                            Surprised
                        </label>
                        <input type="range" class="emotion-slider" id="emotion-slider-surprised" min="-1" max="1" step="0.1" value="0">
                        <span class="emotion-slider-value">0</span>
                    </div>
                    <div class="emotion-slider-group">
                        <label class="emotion-slider-label">
                            <span class="emotion-dot emotion-calm"></span>
                            Calm
                        </label>
                        <input type="range" class="emotion-slider" id="emotion-slider-calm" min="-1" max="1" step="0.1" value="0.8">
                        <span class="emotion-slider-value">0.8</span>
                    </div>
                </div>
                <div class="emotion-presets">
                    <button class="emotion-preset-btn" data-emotion="neutral">
                        <span class="emotion-dot emotion-neutral"></span>
                        Neutral
                    </button>
                    <button class="emotion-preset-btn" data-emotion="happy">
                        <span class="emotion-dot emotion-happy"></span>
                        Happy
                    </button>
                    <button class="emotion-preset-btn" data-emotion="sad">
                        <span class="emotion-dot emotion-sad"></span>
                        Sad
                    </button>
                    <button class="emotion-preset-btn" data-emotion="angry">
                        <span class="emotion-dot emotion-angry"></span>
                        Angry
                    </button>
                    <button class="emotion-preset-btn" data-emotion="surprised">
                        <span class="emotion-dot emotion-surprised"></span>
                        Surprised
                    </button>
                    <button class="emotion-preset-btn" data-emotion="calm">
                        <span class="emotion-dot emotion-calm"></span>
                        Calm
                    </button>
                </div>
                <div class="emotion-actions">
                    <button class="btn btn-secondary" id="reset-emotion-btn">
                        <i class="fas fa-undo"></i> Reset
                    </button>
                    <button class="btn btn-primary" id="apply-emotion-btn">
                        <i class="fas fa-check"></i> Apply Changes
                    </button>
                </div>
            </div>
        </div>
    `;
    
    // Generate timeline blocks
    const timelineContent = document.getElementById('emotion-timeline-content');
    timelineContent.innerHTML = '';
    
    this.conversationScript.forEach((line, index) => {
        const timelineBlock = document.createElement('div');
        timelineBlock.className = 'timeline-block';
        timelineBlock.dataset.lineIndex = index;
        
        // Create emotion visualization
        const emotionColors = this.getEmotionColors(line.emo_vector);
        
        timelineBlock.innerHTML = `
            <div class="timeline-block-header">
                <span class="timeline-line-number">Line ${index + 1}</span>
                <span class="timeline-speaker">${line.speaker}</span>
            </div>
            <div class="timeline-block-content">
                <div class="timeline-text">"${line.text}"</div>
                <div class="timeline-emotion-visualization">
                    ${emotionColors}
                </div>
            </div>
        `;
        
        // Add click event listener
        timelineBlock.addEventListener('click', () => {
            this.selectTimelineLine(index);
        });
        
        timelineContent.appendChild(timelineBlock);
    });
    
    // Setup emotion control event listeners
    this.setupEmotionControlListeners();
    
    console.log('DEBUG: Timeline generated successfully');
};

// Get emotion colors based on emotion vector
IndexTTSApp.prototype.getEmotionColors = function(emoVector) {
    if (!emoVector || emoVector.length !== 8) {
        return '<div class="emotion-bar emotion-neutral" style="width: 100%;"></div>';
    }
    
    // Map emotion vector indices to emotions
    const emotionMap = [
        { name: 'happy', color: '#FFD700' },
        { name: 'sad', color: '#0000FF' },
        { name: 'angry', color: '#FF0000' },
        { name: 'afraid', color: '#800080' },
        { name: 'disgusted', color: '#8B4513' },
        { name: 'melancholic', color: '#808080' },
        { name: 'surprised', color: '#FFC0CB' },
        { name: 'calm', color: '#90EE90' }
    ];
    
    let emotionBars = '';
    emotionMap.forEach((emotion, index) => {
        const intensity = Math.abs(emoVector[index]);
        if (intensity > 0.1) {
            const width = `${intensity * 100}%`;
            emotionBars += `<div class="emotion-bar emotion-${emotion.name}" style="width: ${width}; background-color: ${emotion.color};"></div>`;
        }
    });
    
    return emotionBars || '<div class="emotion-bar emotion-neutral" style="width: 100%;"></div>';
};

// Select a timeline line and show emotion controls
IndexTTSApp.prototype.selectTimelineLine = function(lineIndex) {
    if (!this.conversationScript || lineIndex < 0 || lineIndex >= this.conversationScript.length) {
        return;
    }
    
    console.log('DEBUG: Selecting timeline line', lineIndex);
    
    // Update selected state
    document.querySelectorAll('.timeline-block').forEach(block => {
        block.classList.remove('selected');
    });
    
    const selectedBlock = document.querySelector(`.timeline-block[data-line-index="${lineIndex}"]`);
    if (selectedBlock) {
        selectedBlock.classList.add('selected');
    }
    
    // Show emotion control panel
    const emotionControlPanel = document.getElementById('emotion-control-panel');
    emotionControlPanel.style.display = 'block';
    
    // Update selected line number
    document.getElementById('selected-line-number').textContent = lineIndex + 1;
    
    // Load current emotion vector into sliders
    const line = this.conversationScript[lineIndex];
    const emoVector = line.emo_vector || [0, 0, 0, 0, 0, 0, 0, 0.8];
    
    // Map sliders to emotion vector indices
    const sliderMapping = {
        'emotion-slider-happy': 0,
        'emotion-slider-sad': 1,
        'emotion-slider-angry': 2,
        'emotion-slider-surprised': 6,
        'emotion-slider-calm': 7
    };
    
    Object.entries(sliderMapping).forEach(([sliderId, vectorIndex]) => {
        const slider = document.getElementById(sliderId);
        const valueDisplay = slider.nextElementSibling;
        if (slider && valueDisplay) {
            slider.value = emoVector[vectorIndex];
            valueDisplay.textContent = emoVector[vectorIndex].toFixed(1);
        }
    });
    
    // Store selected line index
    this.selectedTimelineLine = lineIndex;
};

// Setup emotion control event listeners
IndexTTSApp.prototype.setupEmotionControlListeners = function() {
    // Close button
    const closeBtn = document.getElementById('close-emotion-control');
    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            document.getElementById('emotion-control-panel').style.display = 'none';
            document.querySelectorAll('.timeline-block').forEach(block => {
                block.classList.remove('selected');
            });
        });
    }
    
    // Emotion sliders
    document.querySelectorAll('.emotion-slider').forEach(slider => {
        slider.addEventListener('input', (e) => {
            const valueDisplay = e.target.nextElementSibling;
            if (valueDisplay) {
                valueDisplay.textContent = parseFloat(e.target.value).toFixed(1);
            }
        });
    });
    
    // Emotion preset buttons
    document.querySelectorAll('.emotion-preset-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const emotion = e.currentTarget.dataset.emotion;
            this.applyEmotionPreset(emotion);
        });
    });
    
    // Reset button
    const resetBtn = document.getElementById('reset-emotion-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            this.resetEmotionForSelectedLine();
        });
    }
    
    // Apply button
    const applyBtn = document.getElementById('apply-emotion-btn');
    if (applyBtn) {
        applyBtn.addEventListener('click', () => {
            this.applyEmotionToSelectedLine();
        });
    }
};

// Apply emotion preset
IndexTTSApp.prototype.applyEmotionPreset = function(emotion) {
    const presets = {
        neutral: [0, 0, 0, 0, 0, 0, 0, 0],
        happy: [0.8, 0, 0, 0, 0, 0, 0.2, 0],
        sad: [0, 0.8, 0, 0, 0, 0.3, 0, 0],
        angry: [0, 0, 0.8, 0, 0.2, 0, 0, 0],
        surprised: [0.2, 0, 0, 0, 0, 0, 0.9, 0],
        calm: [0, 0, 0, 0, 0, 0, 0, 0.8]
    };
    
    const presetVector = presets[emotion] || presets.neutral;
    
    // Update sliders
    const sliderMapping = {
        'emotion-slider-happy': 0,
        'emotion-slider-sad': 1,
        'emotion-slider-angry': 2,
        'emotion-slider-surprised': 6,
        'emotion-slider-calm': 7
    };
    
    Object.entries(sliderMapping).forEach(([sliderId, vectorIndex]) => {
        const slider = document.getElementById(sliderId);
        const valueDisplay = slider.nextElementSibling;
        if (slider && valueDisplay) {
            slider.value = presetVector[vectorIndex];
            valueDisplay.textContent = presetVector[vectorIndex].toFixed(1);
        }
    });
};

// Reset emotion for selected line
IndexTTSApp.prototype.resetEmotionForSelectedLine = function() {
    this.applyEmotionPreset('calm');
};

// Apply emotion to selected line
IndexTTSApp.prototype.applyEmotionToSelectedLine = function() {
    if (this.selectedTimelineLine === undefined || !this.conversationScript) {
        return;
    }
    
    const lineIndex = this.selectedTimelineLine;
    const line = this.conversationScript[lineIndex];
    
    // Collect values from sliders
    const sliderMapping = {
        'emotion-slider-happy': 0,
        'emotion-slider-sad': 1,
        'emotion-slider-angry': 2,
        'emotion-slider-surprised': 6,
        'emotion-slider-calm': 7
    };
    
    const newEmoVector = [0, 0, 0, 0, 0, 0, 0, 0];
    
    Object.entries(sliderMapping).forEach(([sliderId, vectorIndex]) => {
        const slider = document.getElementById(sliderId);
        if (slider) {
            newEmoVector[vectorIndex] = parseFloat(slider.value);
        }
    });
    
    // Update the line's emotion vector
    line.emo_vector = newEmoVector;
    
    // Update the timeline visualization
    this.updateTimelineVisualization(lineIndex);
    
    // Update the parsed script as well
    if (this.parsedScript && this.parsedScript.lines[lineIndex]) {
        this.parsedScript.lines[lineIndex].emo_vector = newEmoVector;
    }
    
    this.showNotification('Success', `Emotion updated for line ${lineIndex + 1}`, 'success');
    console.log('DEBUG: Applied emotion vector to line', lineIndex, ':', newEmoVector);
};

// Update timeline visualization for a specific line
IndexTTSApp.prototype.updateTimelineVisualization = function(lineIndex) {
    const timelineBlock = document.querySelector(`.timeline-block[data-line-index="${lineIndex}"]`);
    if (!timelineBlock || !this.conversationScript[lineIndex]) {
        return;
    }
    
    const line = this.conversationScript[lineIndex];
    const emotionVisualization = timelineBlock.querySelector('.timeline-emotion-visualization');
    
    if (emotionVisualization) {
        emotionVisualization.innerHTML = this.getEmotionColors(line.emo_vector);
    }
};

IndexTTSApp.prototype.stopGeneration = async function() {
    if (!this.currentConversationId) return;
    
    try {
        await this.apiRequest(`/conversation/stop/${this.currentConversationId}`, {
            method: 'POST'
        });
        
        if (this.generationInterval) {
            clearInterval(this.generationInterval);
            this.generationInterval = null;
        }
        
        const progressContainer = document.getElementById('generation-progress');
        const button = document.getElementById('generate-conversation-btn');
        progressContainer.style.display = 'none';
        button.disabled = false;
        
        this.showNotification('Info', 'Generation stopped', 'info');
        
    } catch (error) {
        this.showNotification('Error', error.message, 'error');
    }
};
