// IndexTTS2 Speaker Prep Module

if (typeof IndexTTSApp === 'undefined') {
    console.error('IndexTTSApp is not defined! Speaker prep module could not attach.');
}

IndexTTSApp.prototype.setupSpeakerPrepEvents = function() {
    const bindings = [
        ['speaker-prep-upload-btn', () => this.uploadSourceClip()],
        ['speaker-prep-refresh-btn', () => this.loadSourceClips(false)],
        ['speaker-prep-diagnose-btn', () => this.runSourceClipDiagnostics()],
        ['speaker-prep-process-btn', () => this.prepareSelectedSourceClip()],
        ['speaker-prep-create-speaker-btn', () => this.prepareSelectedSourceClip('speakers')],
        ['speaker-prep-play-source-btn', () => this.playSelectedSourceClip()],
        ['speaker-prep-delete-source-btn', () => this.deleteSelectedSourceClip()],
        ['speaker-prep-apply-recommended-btn', () => this.applyRecommendedSpeakerPrep()],
        ['speaker-prep-apply-trim-btn', () => this.applyRecommendedSpeakerPrep('trim')],
        ['speaker-prep-reset-prep-btn', () => this.resetSpeakerPrepControls()],
        ['speaker-prep-play-output-btn', () => this.playLastPreparedOutput()],
        ['speaker-prep-load-output-btn', () => this.loadLastPreparedOutputAsSource()],
    ];

    bindings.forEach(([id, handler]) => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('click', handler);
        }
    });

    ['speaker-prep-target-peak', 'speaker-prep-noise-strength'].forEach((id) => {
        const input = document.getElementById(id);
        if (input) {
            input.addEventListener('input', () => {
                this.updateSpeakerPrepLabels();
            });
        }
    });

    this.updateSpeakerPrepLabels();
};

IndexTTSApp.prototype.updateSpeakerPrepLabels = function() {
    const targetPeak = document.getElementById('speaker-prep-target-peak');
    const targetPeakValue = document.getElementById('speaker-prep-target-peak-value');
    if (targetPeak && targetPeakValue) {
        targetPeakValue.textContent = `${Number(targetPeak.value).toFixed(1)} dBFS`;
    }

    const noiseStrength = document.getElementById('speaker-prep-noise-strength');
    const noiseStrengthValue = document.getElementById('speaker-prep-noise-strength-value');
    if (noiseStrength && noiseStrengthValue) {
        noiseStrengthValue.textContent = Number(noiseStrength.value).toFixed(2);
    }
};

IndexTTSApp.prototype.setSpeakerPrepStatus = function(message, type = 'info') {
    const element = document.getElementById('speaker-prep-status');
    if (!element) {
        return;
    }
    element.textContent = message;
    element.dataset.state = type;
};

IndexTTSApp.prototype.buildSpeakerPrepOutputName = function(filename) {
    const stem = String(filename || '')
        .replace(/\.[^.]+$/, '')
        .trim();
    if (!stem) {
        return '';
    }
    return `${stem}_ready`;
};

IndexTTSApp.prototype.maybeAutofillSpeakerPrepOutputName = function(filename) {
    const outputName = document.getElementById('speaker-prep-output-name');
    if (!outputName) {
        return;
    }

    const suggestedName = this.buildSpeakerPrepOutputName(filename);
    const previousStem = outputName.dataset.autofillStem || '';
    const previousSuggestedName = previousStem ? this.buildSpeakerPrepOutputName(previousStem) : '';
    const currentValue = outputName.value.trim();

    if (!currentValue || currentValue === previousSuggestedName) {
        outputName.value = suggestedName;
    }

    outputName.dataset.autofillStem = filename || '';
};

IndexTTSApp.prototype.getCachedSourceClipDiagnostics = function(filename) {
    if (!filename) {
        return null;
    }
    return this.sourceClipDiagnosticsCache?.[filename] || null;
};

IndexTTSApp.prototype.formatSourceClipDuration = function(seconds) {
    const numericSeconds = Number(seconds);
    if (!Number.isFinite(numericSeconds) || numericSeconds <= 0) {
        return null;
    }
    return `${numericSeconds.toFixed(1)}s`;
};

IndexTTSApp.prototype.buildSourceClipBadges = function(diagnostics) {
    if (!diagnostics) {
        return [];
    }

    const badges = [{
        label: `${String(diagnostics.clone_readiness_label || 'unknown').toUpperCase()} ${diagnostics.clone_readiness_score || 0}`,
        readiness: diagnostics.clone_readiness_label || 'unknown'
    }];

    const durationLabel = this.formatSourceClipDuration(diagnostics.duration_seconds);
    if (durationLabel) {
        badges.push({ label: durationLabel });
    }

    if (diagnostics.channels) {
        badges.push({ label: diagnostics.channels === 1 ? 'mono' : 'stereo' });
    }

    return badges;
};

IndexTTSApp.prototype.describeSuggestedPrep = function(suggestion) {
    if (!suggestion) {
        return 'No suggested prep yet.';
    }

    const parts = [];
    const startTime = Number(suggestion.start_time || 0);
    const endTime = suggestion.end_time == null ? null : Number(suggestion.end_time);

    if (startTime > 0 || endTime != null) {
        if (endTime != null) {
            parts.push(`trim to ${startTime.toFixed(1)}s -> ${endTime.toFixed(1)}s`);
        } else if (startTime > 0) {
            parts.push(`trim the start to ${startTime.toFixed(1)}s`);
        }
    }

    if (suggestion.convert_to_mono) {
        parts.push('convert to mono');
    }
    if (suggestion.normalize_audio) {
        parts.push(`normalize toward ${Number(suggestion.target_peak_dbfs || -1).toFixed(1)} dBFS`);
    }
    if (suggestion.use_noise_reduction) {
        parts.push('try light noise cleanup');
    }
    if (suggestion.use_vocal_separation) {
        parts.push('try vocal isolation');
    }

    const reasons = Array.isArray(suggestion.reasons) && suggestion.reasons.length
        ? ` ${suggestion.reasons.join(' ')}`
        : '';

    if (!parts.length) {
        return `This clip already looks close to ready.${reasons}`;
    }

    return `Suggested prep: ${parts.join(', ')}.${reasons}`;
};

IndexTTSApp.prototype.resetSpeakerPrepControls = function() {
    const defaults = {
        'speaker-prep-start-time': '0',
        'speaker-prep-end-time': '',
        'speaker-prep-target-peak': '-1',
        'speaker-prep-noise-strength': '0.35',
    };

    Object.entries(defaults).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            element.value = value;
        }
    });

    const toggles = {
        'speaker-prep-convert-mono': true,
        'speaker-prep-normalize': true,
        'speaker-prep-noise-reduction': false,
        'speaker-prep-vocal-separation': false,
    };

    Object.entries(toggles).forEach(([id, checked]) => {
        const element = document.getElementById(id);
        if (element) {
            element.checked = checked;
        }
    });

    this.maybeAutofillSpeakerPrepOutputName(this.selectedSourceClip);
    this.updateSpeakerPrepLabels();
    this.setSpeakerPrepStatus('Prep controls reset to the safest defaults.', 'info');
};

IndexTTSApp.prototype.applyRecommendedSpeakerPrep = function(mode = 'all') {
    const suggestion = this.currentSourceClipDiagnostics?.suggested_prep;
    if (!suggestion) {
        this.showNotification('Warning', 'Run diagnostics first so the app can suggest a prep recipe', 'warning');
        return;
    }

    const startField = document.getElementById('speaker-prep-start-time');
    const endField = document.getElementById('speaker-prep-end-time');
    if (startField) {
        startField.value = Number(suggestion.start_time || 0).toFixed(2).replace(/\.00$/, '');
    }
    if (endField) {
        endField.value = suggestion.end_time == null ? '' : Number(suggestion.end_time).toFixed(2).replace(/\.00$/, '');
    }

    if (mode !== 'trim') {
        const convertMono = document.getElementById('speaker-prep-convert-mono');
        const normalizeAudio = document.getElementById('speaker-prep-normalize');
        const targetPeak = document.getElementById('speaker-prep-target-peak');
        const noiseReduction = document.getElementById('speaker-prep-noise-reduction');
        const vocalSeparation = document.getElementById('speaker-prep-vocal-separation');

        if (convertMono) convertMono.checked = Boolean(suggestion.convert_to_mono);
        if (normalizeAudio) normalizeAudio.checked = Boolean(suggestion.normalize_audio);
        if (targetPeak) targetPeak.value = String(Number(suggestion.target_peak_dbfs ?? -1));
        if (noiseReduction) noiseReduction.checked = Boolean(suggestion.use_noise_reduction);
        if (vocalSeparation) vocalSeparation.checked = Boolean(suggestion.use_vocal_separation);
    }

    this.maybeAutofillSpeakerPrepOutputName(this.selectedSourceClip);
    this.updateSpeakerPrepLabels();
    this.setSpeakerPrepStatus(
        mode === 'trim'
            ? `Applied the suggested trim window for ${this.selectedSourceClip}.`
            : `Applied the recommended prep recipe for ${this.selectedSourceClip}.`,
        'success'
    );
};

IndexTTSApp.prototype.loadSourceClips = async function(suppressNotifications = true) {
    try {
        const response = await this.apiRequest('/speakers-tools/list-source-clips', {
            suppressErrorNotification: suppressNotifications
        });
        this.sourceClips = response.details?.files || [];

        const availableFilenames = new Set(this.sourceClips.map((clip) => clip.filename));
        Object.keys(this.sourceClipDiagnosticsCache || {}).forEach((filename) => {
            if (!availableFilenames.has(filename)) {
                delete this.sourceClipDiagnosticsCache[filename];
            }
        });

        if (this.selectedSourceClip && !availableFilenames.has(this.selectedSourceClip)) {
            this.selectedSourceClip = null;
            this.currentSourceClipDiagnostics = null;
        }

        if (!this.selectedSourceClip && this.sourceClips.length) {
            this.selectedSourceClip = this.sourceClips[0].filename;
        }

        this.renderSourceClipList();

        if (!this.selectedSourceClip) {
            this.currentSourceClipDiagnostics = null;
            this.lastSpeakerPrepResult = null;
            this.resetSpeakerPrepDiagnostics();
            this.updateSelectedSourceClipUi();
            return;
        }

        this.maybeAutofillSpeakerPrepOutputName(this.selectedSourceClip);
        const cachedDiagnostics = this.getCachedSourceClipDiagnostics(this.selectedSourceClip);
        if (cachedDiagnostics) {
            this.currentSourceClipDiagnostics = cachedDiagnostics;
            this.renderSourceClipDiagnostics(cachedDiagnostics);
        } else {
            this.currentSourceClipDiagnostics = null;
            this.resetSpeakerPrepDiagnostics();
        }

        this.updateSelectedSourceClipUi();
        if (!cachedDiagnostics) {
            this.runSourceClipDiagnostics(this.selectedSourceClip, true).catch((error) => {
                console.error('Failed to auto-run source clip diagnostics:', error);
            });
        }
    } catch (error) {
        console.error('Failed to load source clips:', error);
        this.sourceClips = [];
        this.selectedSourceClip = null;
        this.currentSourceClipDiagnostics = null;
        this.renderSourceClipList();
        this.resetSpeakerPrepDiagnostics();
        this.updateSelectedSourceClipUi();
        if (!suppressNotifications) {
            this.showNotification('Error', 'Failed to load source clips', 'error');
        }
    }
};

IndexTTSApp.prototype.renderSourceClipList = function() {
    const container = document.getElementById('speaker-prep-source-list');
    const countBadge = document.getElementById('speaker-prep-count');
    if (!container || !countBadge) {
        return;
    }

    const total = this.sourceClips.length;
    countBadge.textContent = `${total} clip${total === 1 ? '' : 's'}`;
    container.innerHTML = '';

      if (!total) {
          container.innerHTML = `
             <div class="empty-state">
                  <i class="fas fa-file-audio"></i>
                  <p>No source clips yet.</p>
                  <p class="empty-state-detail">Upload your own source audio here. Source clips are local-only and are not meant to be bundled with releases.</p>
              </div>
          `;
          return;
      }

    this.sourceClips.forEach((clip) => {
        const diagnostics = this.getCachedSourceClipDiagnostics(clip.filename);
        const item = document.createElement('div');
        item.className = 'speaker-item voice-card source-clip-card';
        if (this.selectedSourceClip === clip.filename) {
            item.classList.add('selected');
        }

        const badgeMarkup = this.buildSourceClipBadges(diagnostics)
            .map((badge) => `
                <span class="source-clip-badge" ${badge.readiness ? `data-readiness="${badge.readiness}"` : ''}>
                    ${badge.label}
                </span>
            `)
            .join('');

        const details = [this.formatSpeakerSize(clip), clip.content_type || 'audio file']
            .filter(Boolean)
            .join(' | ');

        item.innerHTML = `
            <div class="speaker-info">
                <i class="fas fa-file-waveform"></i>
                <div class="voice-card-text">
                    <div class="speaker-name">${clip.filename}</div>
                    <div class="voice-script-label">${diagnostics ? this.describeSuggestedPrep(diagnostics.suggested_prep) : 'Ready for diagnostics and cleanup'}</div>
                    <div class="speaker-size">${details}</div>
                    ${badgeMarkup ? `<div class="source-clip-summary">${badgeMarkup}</div>` : ''}
                </div>
            </div>
            <div class="source-clip-actions">
                <button class="btn btn-secondary btn-small" data-action="select">Select</button>
                <button class="btn btn-secondary btn-small" data-action="play">Play</button>
                <button class="btn btn-secondary btn-small" data-action="diagnose">Diagnose</button>
            </div>
        `;

        item.addEventListener('click', (event) => {
            if (event.target instanceof HTMLElement && event.target.closest('button')) {
                return;
            }
            this.selectSourceClip(clip.filename);
        });

        item.querySelector('[data-action="select"]').addEventListener('click', () => {
            this.selectSourceClip(clip.filename);
        });
        item.querySelector('[data-action="play"]').addEventListener('click', () => {
            this.playSourceClipByName(clip.filename);
        });
        item.querySelector('[data-action="diagnose"]').addEventListener('click', () => {
            this.selectSourceClip(clip.filename);
            this.runSourceClipDiagnostics(clip.filename);
        });

        container.appendChild(item);
    });
};

IndexTTSApp.prototype.selectSourceClip = function(filename, options = {}) {
    this.selectedSourceClip = filename;
    this.currentSourceClipDiagnostics = this.getCachedSourceClipDiagnostics(filename);
    this.maybeAutofillSpeakerPrepOutputName(filename);

    const result = document.getElementById('speaker-prep-result');
    const resultActions = document.getElementById('speaker-prep-result-actions');
    if (!options.preserveResult) {
        this.lastSpeakerPrepResult = null;
    }
    if (result && !options.preserveResult) {
        result.textContent = 'No prep run yet.';
    }
    if (resultActions && !options.preserveResult) {
        resultActions.style.display = 'none';
    }

    if (this.currentSourceClipDiagnostics) {
        this.renderSourceClipDiagnostics(this.currentSourceClipDiagnostics);
    } else if (!options.preserveResult) {
        this.resetSpeakerPrepDiagnostics();
    }

    this.renderSourceClipList();
    this.updateSelectedSourceClipUi();
    this.runSourceClipDiagnostics(filename, true).catch((error) => {
        console.error('Automatic diagnostics failed:', error);
    });
};

IndexTTSApp.prototype.updateSelectedSourceClipUi = function() {
    const info = document.getElementById('speaker-prep-selected-info');
    if (!info) {
        return;
    }
    if (!this.selectedSourceClip) {
        info.textContent = 'Select a source clip to view cloning readiness, silence, loudness, and cleanup suggestions.';
        return;
    }

    const diagnostics = this.getCachedSourceClipDiagnostics(this.selectedSourceClip);
    if (!diagnostics) {
        info.textContent = `Selected clip: ${this.selectedSourceClip}`;
        return;
    }

    const parts = [
        `<strong>${this.selectedSourceClip}</strong>`,
        `${String(diagnostics.clone_readiness_label || 'unknown').toUpperCase()} ${diagnostics.clone_readiness_score || 0}`,
    ];

    const durationLabel = this.formatSourceClipDuration(diagnostics.duration_seconds);
    if (durationLabel) {
        parts.push(durationLabel);
    }
    if (diagnostics.channels) {
        parts.push(diagnostics.channels === 1 ? 'mono' : 'stereo');
    }
    if (diagnostics.level_dbfs != null) {
        parts.push(`${Number(diagnostics.level_dbfs).toFixed(1)} dBFS`);
    }

    info.innerHTML = parts.join(' &bull; ');
};

IndexTTSApp.prototype.resetSpeakerPrepDiagnostics = function() {
    const diagnostics = document.getElementById('speaker-prep-diagnostics');
    const emptyState = document.getElementById('speaker-prep-diagnostics-empty');
    const recipeSummary = document.getElementById('speaker-prep-recipe-summary');
    const result = document.getElementById('speaker-prep-result');
    const resultActions = document.getElementById('speaker-prep-result-actions');

    if (diagnostics) diagnostics.style.display = 'none';
    if (emptyState) emptyState.style.display = 'block';
    if (recipeSummary) recipeSummary.textContent = '';
    if (result && !result.textContent.trim()) {
        result.textContent = 'No prep run yet.';
    }
    if (resultActions && !this.lastSpeakerPrepResult) {
        resultActions.style.display = 'none';
    }
};

IndexTTSApp.prototype.runSourceClipDiagnostics = async function(filename = null, suppressNotification = false) {
    const targetFilename = filename || this.selectedSourceClip;
    if (!targetFilename) {
        this.showNotification('Error', 'Select a source clip first', 'error');
        return;
    }

    try {
        this.setSpeakerPrepStatus(`Checking ${targetFilename}...`);
        const response = await this.apiRequest(`/speakers-tools/source-clip-diagnostics/${encodeURIComponent(targetFilename)}`, {
            suppressErrorNotification: suppressNotification
        });
        this.currentSourceClipDiagnostics = response.details?.diagnostics || null;
        this.sourceClipDiagnosticsCache[targetFilename] = this.currentSourceClipDiagnostics;
        this.renderSourceClipDiagnostics(this.currentSourceClipDiagnostics);
        this.renderSourceClipList();
        this.updateSelectedSourceClipUi();
        this.setSpeakerPrepStatus(`Diagnostics ready for ${targetFilename}`, 'success');
    } catch (error) {
        console.error('Failed to run source clip diagnostics:', error);
        if (!suppressNotification) {
            this.showNotification('Error', 'Failed to analyze source clip', 'error');
        }
        throw error;
    }
};

IndexTTSApp.prototype.renderSourceClipDiagnostics = function(diagnostics) {
    const diagnosticsEl = document.getElementById('speaker-prep-diagnostics');
    const emptyState = document.getElementById('speaker-prep-diagnostics-empty');
    const badge = document.getElementById('speaker-prep-score-badge');
    const copy = document.getElementById('speaker-prep-score-copy');
    const grid = document.getElementById('speaker-prep-metrics-grid');
    const recipeSummary = document.getElementById('speaker-prep-recipe-summary');
    const recommendations = document.getElementById('speaker-prep-recommendations-list');

    if (!diagnosticsEl || !emptyState || !badge || !copy || !grid || !recipeSummary || !recommendations) {
        return;
    }

    if (!diagnostics) {
        diagnosticsEl.style.display = 'none';
        emptyState.style.display = 'block';
        recipeSummary.textContent = '';
        return;
    }

    diagnosticsEl.style.display = 'block';
    emptyState.style.display = 'none';

    badge.textContent = `${String(diagnostics.clone_readiness_label || 'unknown').toUpperCase()} ${diagnostics.clone_readiness_score || 0}/100`;
    badge.dataset.readiness = diagnostics.clone_readiness_label || 'unknown';
    copy.textContent = diagnostics.ready_for_cloning
        ? 'Good cloning candidate. You can usually keep prep light unless you hear obvious room noise or clipping.'
        : 'This clip needs cleanup before it will clone consistently. Start with the suggested trim and safe defaults below.';

    recipeSummary.innerHTML = `<strong>Recommended next step:</strong> ${this.describeSuggestedPrep(diagnostics.suggested_prep)}`;

    const metricItems = [
        ['Duration', `${diagnostics.duration_seconds}s`],
        ['Channels', String(diagnostics.channels)],
        ['Sample Rate', `${diagnostics.sample_rate_hz} Hz`],
        ['Level', diagnostics.level_dbfs == null ? 'n/a' : `${diagnostics.level_dbfs.toFixed(1)} dBFS`],
        ['Peak', diagnostics.peak_dbfs == null ? 'n/a' : `${diagnostics.peak_dbfs.toFixed(1)} dBFS`],
        ['Silence', `${diagnostics.silence_percent}%`],
        ['Leading Silence', `${diagnostics.leading_silence_ms} ms`],
        ['Trailing Silence', `${diagnostics.trailing_silence_ms} ms`],
    ];

    grid.innerHTML = metricItems.map(([label, value]) => `
        <div class="speaker-prep-metric-card">
            <div class="speaker-prep-metric-label">${label}</div>
            <div class="speaker-prep-metric-value">${value}</div>
        </div>
    `).join('');

    const combinedRecommendations = [...(diagnostics.warnings || []), ...(diagnostics.recommendations || [])];
    recommendations.innerHTML = combinedRecommendations
        .map((item) => `<li>${item}</li>`)
        .join('');
};

IndexTTSApp.prototype.uploadSourceClip = async function() {
    const fileInput = document.getElementById('speaker-prep-upload-file');
    const nameInput = document.getElementById('speaker-prep-upload-name');
    const file = fileInput?.files?.[0];

    if (!file) {
        this.showNotification('Error', 'Pick an audio file first', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    if (nameInput?.value?.trim()) {
        formData.append('output_name', nameInput.value.trim());
    }

    try {
        this.setSpeakerPrepStatus(`Uploading ${file.name}...`);
        const response = await fetch(`${this.apiBaseUrl}/speakers-tools/upload-source-clip`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || data.message || 'Upload failed');
        }

        const uploadedFilename = data.details?.filename || file.name;
        this.showNotification('Success', `Uploaded ${uploadedFilename}`, 'success');
        this.setSpeakerPrepStatus(`Uploaded ${uploadedFilename}`, 'success');
        if (fileInput) fileInput.value = '';
        if (nameInput) nameInput.value = '';
        await this.loadSourceClips();
        this.selectSourceClip(uploadedFilename);
    } catch (error) {
        console.error('Failed to upload source clip:', error);
        this.showNotification('Error', error.message || 'Failed to upload source clip', 'error');
        this.setSpeakerPrepStatus('Upload failed', 'error');
    }
};

IndexTTSApp.prototype.playSourceClipByName = async function(filename) {
    if (!filename) {
        return;
    }
    const audioUrl = `${this.apiBaseUrl}/files/download/source_clips/${encodeURIComponent(filename)}`;
    await this.playVersionAudio(`source_clips/${filename}`, audioUrl);
};

IndexTTSApp.prototype.playSelectedSourceClip = async function() {
    if (!this.selectedSourceClip) {
        this.showNotification('Error', 'Select a source clip first', 'error');
        return;
    }
    await this.playSourceClipByName(this.selectedSourceClip);
};

IndexTTSApp.prototype.deleteSelectedSourceClip = async function() {
    if (!this.selectedSourceClip) {
        this.showNotification('Error', 'Select a source clip first', 'error');
        return;
    }

    const target = this.selectedSourceClip;
    if (!window.confirm(`Delete source clip ${target}?`)) {
        return;
    }

    try {
        await this.apiRequest(`/files/source_clips/${encodeURIComponent(target)}`, {
            method: 'DELETE'
        });
        this.showNotification('Success', `Deleted ${target}`, 'success');
        this.selectedSourceClip = null;
        this.currentSourceClipDiagnostics = null;
        delete this.sourceClipDiagnosticsCache[target];
        this.lastSpeakerPrepResult = null;
        await this.loadSourceClips();
    } catch (error) {
        console.error('Failed to delete source clip:', error);
        this.showNotification('Error', 'Failed to delete source clip', 'error');
    }
};

IndexTTSApp.prototype.getSpeakerPrepRequestBody = function(forcedCategory = null) {
    const targetCategory = forcedCategory || document.getElementById('speaker-prep-target-category')?.value || 'speakers';
    const endTimeRaw = document.getElementById('speaker-prep-end-time')?.value?.trim();

    return {
        source_filename: this.selectedSourceClip,
        output_name: document.getElementById('speaker-prep-output-name')?.value?.trim() || this.buildSpeakerPrepOutputName(this.selectedSourceClip),
        target_category: targetCategory,
        start_time: Number(document.getElementById('speaker-prep-start-time')?.value || 0),
        end_time: endTimeRaw ? Number(endTimeRaw) : null,
        convert_to_mono: Boolean(document.getElementById('speaker-prep-convert-mono')?.checked),
        normalize_audio: Boolean(document.getElementById('speaker-prep-normalize')?.checked),
        target_peak_dbfs: Number(document.getElementById('speaker-prep-target-peak')?.value || -1),
        use_noise_reduction: Boolean(document.getElementById('speaker-prep-noise-reduction')?.checked),
        noise_reduction_strength: Number(document.getElementById('speaker-prep-noise-strength')?.value || 0.35),
        use_vocal_separation: Boolean(document.getElementById('speaker-prep-vocal-separation')?.checked),
    };
};

IndexTTSApp.prototype.prepareSelectedSourceClip = async function(forcedCategory = null) {
    if (!this.selectedSourceClip) {
        this.showNotification('Error', 'Select a source clip first', 'error');
        return;
    }

    const requestBody = this.getSpeakerPrepRequestBody(forcedCategory);
    const targetLabel = requestBody.target_category === 'speakers' ? 'speaker file' : 'prepared source clip';

    try {
        this.setSpeakerPrepStatus(`Preparing ${this.selectedSourceClip}...`);
        const response = await this.apiRequest('/speakers-tools/prepare-source-clip', {
            method: 'POST',
            body: JSON.stringify(requestBody)
        });

        const details = response.details || {};
        this.lastSpeakerPrepResult = details;
        this.renderSpeakerPrepResult(details);
        this.showNotification('Success', `Created ${details.output_filename}`, 'success');
        this.setSpeakerPrepStatus(`Built ${targetLabel}: ${details.output_filename}`, 'success');

        await this.loadSourceClips();
        if (requestBody.target_category === 'speakers') {
            await this.loadSpeakers();
        } else if (details.output_filename) {
            this.selectSourceClip(details.output_filename, { preserveResult: true });
        }
    } catch (error) {
        console.error('Failed to prepare source clip:', error);
        this.showNotification('Error', error.message || 'Failed to prepare source clip', 'error');
        this.setSpeakerPrepStatus('Speaker prep failed', 'error');
    }
};

IndexTTSApp.prototype.renderSpeakerPrepResult = function(details) {
    const result = document.getElementById('speaker-prep-result');
    const resultActions = document.getElementById('speaker-prep-result-actions');
    const loadOutputBtn = document.getElementById('speaker-prep-load-output-btn');
    if (!result) {
        return;
    }

    const before = details.before || {};
    const after = details.after || {};
    const notes = details.processing_notes || [];
    const beforeScore = Number(before.clone_readiness_score || 0);
    const afterScore = Number(after.clone_readiness_score || 0);
    const scoreDelta = afterScore - beforeScore;

    result.innerHTML = `
        <div><strong>Output:</strong> ${details.output_filename || 'unknown'} <span class="speaker-prep-result-improvement">${scoreDelta >= 0 ? '+' : ''}${scoreDelta} score</span></div>
        <div class="speaker-prep-result-grid">
            <div class="speaker-prep-result-card">
                <strong>Saved To</strong>
                <div>${details.target_category || 'unknown'}</div>
            </div>
            <div class="speaker-prep-result-card">
                <strong>Before</strong>
                <div>${String(before.clone_readiness_label || 'n/a').toUpperCase()} ${before.clone_readiness_score || 'n/a'}</div>
            </div>
            <div class="speaker-prep-result-card">
                <strong>After</strong>
                <div>${String(after.clone_readiness_label || 'n/a').toUpperCase()} ${after.clone_readiness_score || 'n/a'}</div>
            </div>
            <div class="speaker-prep-result-card">
                <strong>Trimmed Duration</strong>
                <div>${after.duration_seconds || 'n/a'}s</div>
            </div>
        </div>
        <div><strong>Notes:</strong> ${notes.length ? notes.join(' ') : 'No extra processing notes.'}</div>
    `;

    if (resultActions) {
        resultActions.style.display = 'flex';
    }
    if (loadOutputBtn) {
        loadOutputBtn.style.display = details.target_category === 'source_clips' ? 'inline-flex' : 'none';
    }
};

IndexTTSApp.prototype.playLastPreparedOutput = async function() {
    const details = this.lastSpeakerPrepResult;
    if (!details?.output_filename || !details?.target_category) {
        this.showNotification('Warning', 'Prepare a clip first', 'warning');
        return;
    }

    const category = details.target_category === 'speakers' ? 'speakers' : 'source_clips';
    const audioUrl = `${this.apiBaseUrl}/files/download/${category}/${encodeURIComponent(details.output_filename)}`;
    await this.playVersionAudio(`${category}/${details.output_filename}`, audioUrl);
};

IndexTTSApp.prototype.loadLastPreparedOutputAsSource = async function() {
    const details = this.lastSpeakerPrepResult;
    if (!details?.output_filename || details.target_category !== 'source_clips') {
        this.showNotification('Warning', 'The latest prep output is not in Source Clips', 'warning');
        return;
    }

    await this.loadSourceClips();
    this.selectSourceClip(details.output_filename, { preserveResult: true });
    this.showNotification('Success', `Loaded ${details.output_filename} as the active source clip`, 'success');
};
