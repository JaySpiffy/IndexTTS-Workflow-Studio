function escapeTimelineHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function safeTimelineNumber(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function formatTimelineVolumePercent(volume) {
    return `${Math.round(safeTimelineNumber(volume, 1) * 100)}%`;
}

function formatTimelineDb(value) {
    return `${safeTimelineNumber(value, 0).toFixed(1)} dB`;
}

function buildTimelineWaveformBars(peaks) {
    if (!Array.isArray(peaks) || !peaks.length) {
        return `
            <div class="timeline-waveform-empty">
                <i class="fas fa-wave-square"></i>
                <span>No waveform preview yet.</span>
            </div>
        `;
    }

    return peaks.map((peak) => {
        const clamped = Math.max(0.04, Math.min(1, safeTimelineNumber(peak, 0)));
        return `<span class="timeline-waveform-bar" style="height:${Math.round(clamped * 100)}%"></span>`;
    }).join('');
}

IndexTTSApp.prototype.setupTimelineEditorEvents = function() {
    const bindClick = (id, handler) => {
        const element = document.getElementById(id);
        if (element) {
            element.addEventListener('click', handler);
        }
    };

    bindClick('timeline-import-conversation-btn', () => this.importSelectedConversationToTimeline());
    bindClick('timeline-create-blank-btn', () => this.createBlankTimelineProject());
    bindClick('timeline-refresh-projects-btn', () => this.loadTimelineProjects());
    bindClick('timeline-load-project-btn', () => this.loadSelectedTimelineProject());
    bindClick('timeline-delete-project-btn', () => this.deleteSelectedTimelineProject());
    bindClick('timeline-add-track-btn', () => this.addTimelineTrack());
    bindClick('timeline-inline-add-segment-btn', () => this.createInlineTimelineSegment());
    bindClick('timeline-inline-add-track-segment-btn', () => this.createInlineTimelineSegment({ createTrackFirst: true }));
    bindClick('timeline-add-segment-btn', () => this.openCreateTimelineSegmentModal());
    bindClick('timeline-generate-selected-btn', () => this.generateSelectedTimelineSegmentAudio());
    bindClick('timeline-generate-missing-btn', () => this.generateMissingTimelineAudio());
    bindClick('timeline-export-btn', () => this.exportCurrentTimeline());
    bindClick('timeline-play-export-btn', () => this.playTimelineExport());
    bindClick('timeline-download-export-btn', () => this.downloadTimelineExport());
    bindClick('timeline-popout-btn', () => this.openTimelinePopoutWindow());

    bindClick('close-line-editor', () => this.closeCreateTimelineSegmentModal());
    bindClick('cancel-line-editor', () => this.closeCreateTimelineSegmentModal());
    bindClick('create-line-segment', () => this.createTimelineSegment());

    bindClick('close-edit-segment-modal', () => this.closeEditTimelineSegmentModal());
    bindClick('cancel-edit-segment-btn', () => this.closeEditTimelineSegmentModal());
    bindClick('save-edit-segment-btn', () => this.saveEditedTimelineSegment());

    const duckAmountInput = document.getElementById('timeline-duck-amount');
    const duckFadeInput = document.getElementById('timeline-duck-fade-ms');
    const targetLevelInput = document.getElementById('timeline-target-level-dbfs');
    const peakLimitInput = document.getElementById('timeline-peak-limit-dbfs');
    const fadeInInput = document.getElementById('timeline-fade-in-ms');
    const fadeOutInput = document.getElementById('timeline-fade-out-ms');
    const outputFormatInput = document.getElementById('timeline-output-format');
    const timelineTrackSpeaker = document.getElementById('timeline-track-speaker');
    const timelineInlineSegmentTrack = document.getElementById('timeline-inline-segment-track');
    const timelineInlineAutoplace = document.getElementById('timeline-inline-segment-autoplace');
    const refreshDuckingSummary = () => this.updateTimelineDuckingLabels();
    duckAmountInput?.addEventListener('input', refreshDuckingSummary);
    duckFadeInput?.addEventListener('input', refreshDuckingSummary);
    targetLevelInput?.addEventListener('input', refreshDuckingSummary);
    peakLimitInput?.addEventListener('input', refreshDuckingSummary);
    fadeInInput?.addEventListener('input', refreshDuckingSummary);
    fadeOutInput?.addEventListener('input', refreshDuckingSummary);
    outputFormatInput?.addEventListener('change', refreshDuckingSummary);
    timelineTrackSpeaker?.addEventListener('change', () => this.maybeSuggestTimelineTrackName());
    timelineInlineSegmentTrack?.addEventListener('change', () => this.updateTimelineStandaloneBuilder());
    timelineInlineAutoplace?.addEventListener('change', () => this.updateTimelineStandaloneBuilder());

    this.renderTimelineSpeakerOptions();
    this.renderTimelineProjectOptions();
    this.renderTimelineProject();
    this.updateTimelineDuckingLabels();
};

IndexTTSApp.prototype.getTimelineExportSettings = function() {
    return {
        format: document.getElementById('timeline-output-format')?.value || 'wav',
        output_bitrate_kbps: Math.round(safeTimelineNumber(document.getElementById('timeline-output-bitrate')?.value, 192)),
        duck_overlaps: document.getElementById('timeline-duck-overlaps')?.checked ?? true,
        duck_amount_db: safeTimelineNumber(document.getElementById('timeline-duck-amount')?.value, 6),
        duck_fade_ms: Math.max(0, Math.round(safeTimelineNumber(document.getElementById('timeline-duck-fade-ms')?.value, 120))),
        normalize_segments: document.getElementById('timeline-normalize-segments')?.checked ?? true,
        target_level_dbfs: safeTimelineNumber(document.getElementById('timeline-target-level-dbfs')?.value, -19),
        peak_limit_dbfs: safeTimelineNumber(document.getElementById('timeline-peak-limit-dbfs')?.value, -1),
        normalize_final_mix: document.getElementById('timeline-normalize-final-mix')?.checked ?? true,
        trim_leading_silence: document.getElementById('timeline-trim-leading-silence')?.checked ?? true,
        trim_trailing_silence: document.getElementById('timeline-trim-trailing-silence')?.checked ?? true,
        fade_in_ms: Math.max(0, Math.round(safeTimelineNumber(document.getElementById('timeline-fade-in-ms')?.value, 0))),
        fade_out_ms: Math.max(0, Math.round(safeTimelineNumber(document.getElementById('timeline-fade-out-ms')?.value, 60))),
    };
};

IndexTTSApp.prototype.updateTimelineDuckingLabels = function() {
    const amountInput = document.getElementById('timeline-duck-amount');
    const amountLabel = document.getElementById('timeline-duck-amount-value');
    const fadeInput = document.getElementById('timeline-duck-fade-ms');
    const fadeLabel = document.getElementById('timeline-duck-fade-value');
    const targetInput = document.getElementById('timeline-target-level-dbfs');
    const targetLabel = document.getElementById('timeline-target-level-dbfs-value');
    const peakInput = document.getElementById('timeline-peak-limit-dbfs');
    const peakLabel = document.getElementById('timeline-peak-limit-dbfs-value');
    const fadeInInput = document.getElementById('timeline-fade-in-ms');
    const fadeInLabel = document.getElementById('timeline-fade-in-value');
    const fadeOutInput = document.getElementById('timeline-fade-out-ms');
    const fadeOutLabel = document.getElementById('timeline-fade-out-value');
    const outputFormatInput = document.getElementById('timeline-output-format');
    const outputBitrateInput = document.getElementById('timeline-output-bitrate');
    const isMp3 = (outputFormatInput?.value || 'wav') === 'mp3';

    if (amountInput && amountLabel) {
        amountLabel.textContent = formatTimelineDb(amountInput.value);
    }
    if (fadeInput && fadeLabel) {
        fadeLabel.textContent = `${Math.round(safeTimelineNumber(fadeInput.value, 120))} ms`;
    }
    if (targetInput && targetLabel) {
        targetLabel.textContent = `${safeTimelineNumber(targetInput.value, -19).toFixed(1)} dBFS`;
    }
    if (peakInput && peakLabel) {
        peakLabel.textContent = `${safeTimelineNumber(peakInput.value, -1).toFixed(1)} dBFS`;
    }
    if (fadeInInput && fadeInLabel) {
        fadeInLabel.textContent = `${Math.round(safeTimelineNumber(fadeInInput.value, 0))} ms`;
    }
    if (fadeOutInput && fadeOutLabel) {
        fadeOutLabel.textContent = `${Math.round(safeTimelineNumber(fadeOutInput.value, 60))} ms`;
    }
    if (outputBitrateInput) {
        outputBitrateInput.disabled = !isMp3;
        outputBitrateInput.title = isMp3 ? 'MP3 export bitrate' : 'Bitrate only applies to MP3 exports';
    }
};

IndexTTSApp.prototype.renderTimelineSpeakerOptions = function() {
    const select = document.getElementById('timeline-track-speaker');
    if (!select) {
        return;
    }

    const currentValue = select.value;
    select.innerHTML = '<option value="">Select a speaker...</option>';

    this.speakers.forEach((speaker) => {
        const option = document.createElement('option');
        option.value = speaker.filename;
        option.textContent = speaker.name || speaker.filename;
        select.appendChild(option);
    });

    if (currentValue) {
        select.value = currentValue;
    }

    this.maybeSuggestTimelineTrackName();
};

IndexTTSApp.prototype.getTimelineSpeakerDisplayName = function(speakerFilename) {
    if (!speakerFilename) {
        return '';
    }

    const speaker = (this.speakers || []).find((item) => item.filename === speakerFilename);
    return speaker?.name || String(speakerFilename).replace(/\.(wav|mp3)$/i, '');
};

IndexTTSApp.prototype.buildSuggestedTimelineTrackName = function(speakerFilename) {
    const baseName = this.getTimelineSpeakerDisplayName(speakerFilename);
    if (!baseName) {
        return '';
    }

    const existingNames = new Set(
        (this.currentTimelineProject?.tracks || []).map((track) => String(track.track_name || '').trim().toLowerCase()).filter(Boolean)
    );

    if (!existingNames.has(baseName.trim().toLowerCase())) {
        return baseName;
    }

    let suffix = 2;
    while (existingNames.has(`${baseName} ${suffix}`.trim().toLowerCase())) {
        suffix += 1;
    }

    return `${baseName} ${suffix}`;
};

IndexTTSApp.prototype.maybeSuggestTimelineTrackName = function() {
    const trackNameInput = document.getElementById('timeline-track-name');
    const trackSpeakerSelect = document.getElementById('timeline-track-speaker');
    if (!trackNameInput || !trackSpeakerSelect) {
        return;
    }

    const suggestion = this.buildSuggestedTimelineTrackName(trackSpeakerSelect.value);
    const currentValue = (trackNameInput.value || '').trim();
    const previousSuggestion = trackNameInput.dataset.suggestedName || '';

    if (!currentValue || currentValue === previousSuggestion) {
        trackNameInput.value = suggestion;
        trackNameInput.dataset.suggestedName = suggestion;
    }
};

IndexTTSApp.prototype.renderTimelineProjectOptions = function() {
    const select = document.getElementById('timeline-project-select');
    if (!select) {
        return;
    }

    const selectedId = this.currentTimelineProjectId || select.value;
    select.innerHTML = '<option value="">Select a timeline project...</option>';

    this.timelineProjects.forEach((project) => {
        const option = document.createElement('option');
        option.value = project.project_id;
        option.textContent = `${project.project_name} (${project.track_count} tracks, ${project.segment_count} segments)`;
        select.appendChild(option);
    });

    if (selectedId) {
        select.value = selectedId;
    }

    this.updateTimelineStandaloneBuilder();
};

IndexTTSApp.prototype.syncCurrentTimelineProjectListEntry = function() {
    if (!this.currentTimelineProjectId || !this.currentTimelineProject) {
        return;
    }

    const trackCount = this.currentTimelineProject.tracks?.length || 0;
    const segmentCount = (this.currentTimelineProject.tracks || []).reduce(
        (total, track) => total + (track.segments?.length || 0),
        0
    );

    const currentEntry = {
        project_id: this.currentTimelineProjectId,
        project_name: this.currentTimelineProject.project_name || this.currentTimelineProjectId,
        track_count: trackCount,
        segment_count: segmentCount,
        total_duration: this.currentTimelineProject.total_duration || 0,
        updated_at: this.currentTimelineProject.updated_at || new Date().toISOString(),
    };

    const existingIndex = (this.timelineProjects || []).findIndex(
        (project) => project.project_id === this.currentTimelineProjectId
    );

    if (existingIndex >= 0) {
        this.timelineProjects[existingIndex] = {
            ...this.timelineProjects[existingIndex],
            ...currentEntry,
        };
    } else {
        this.timelineProjects = [currentEntry, ...(this.timelineProjects || [])];
    }
};

IndexTTSApp.prototype.getTimelineRecommendedStartTime = function(trackId) {
    const selectedTrack = this.findTimelineTrack(trackId);
    if (!selectedTrack) {
        return this.snapTimelineValue(this.currentTimelineProject?.total_duration || 0);
    }

    const trackEnd = (selectedTrack.segments || []).reduce((latestEnd, segment) => {
        const segmentEnd = safeTimelineNumber(segment.start_time, 0) + safeTimelineNumber(segment.duration, 0);
        return Math.max(latestEnd, segmentEnd);
    }, 0);

    return this.snapTimelineValue(trackEnd);
};

IndexTTSApp.prototype.getTimelineSpeakerOptionsHtml = function(selectedSpeaker = '') {
    const options = ['<option value="">Select a speaker...</option>'];

    (this.speakers || []).forEach((speaker) => {
        const filename = speaker.filename || '';
        const selected = filename === selectedSpeaker ? ' selected' : '';
        options.push(
            `<option value="${escapeTimelineHtml(filename)}"${selected}>${escapeTimelineHtml(speaker.name || filename)}</option>`
        );
    });

    return options.join('');
};

IndexTTSApp.prototype.maybeSuggestTimelineEditorTrackName = function() {
    const nameInput = document.getElementById('timeline-editor-track-name');
    const speakerSelect = document.getElementById('timeline-editor-track-speaker');
    if (!nameInput || !speakerSelect) {
        return;
    }

    const suggestion = this.buildSuggestedTimelineTrackName(speakerSelect.value);
    const currentValue = (nameInput.value || '').trim();
    const previousSuggestion = nameInput.dataset.suggestedName || '';

    if (!currentValue || currentValue === previousSuggestion) {
        nameInput.value = suggestion;
        nameInput.dataset.suggestedName = suggestion;
    }
};

IndexTTSApp.prototype.focusTimelineEditorTrackCreator = function(options = {}) {
    const speakerSelect = document.getElementById('timeline-editor-track-speaker');
    const nameInput = document.getElementById('timeline-editor-track-name');
    const toolbar = document.getElementById('timeline-editor-toolbar');

    if (options.speakerFilename && speakerSelect) {
        speakerSelect.value = options.speakerFilename;
    }

    this.maybeSuggestTimelineEditorTrackName();

    toolbar?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    if (nameInput) {
        nameInput.focus();
        nameInput.select();
    }
};

IndexTTSApp.prototype.createTimelineTrackFromEditor = async function() {
    try {
        const track = await this.createTimelineTrackInternal({
            trackName: document.getElementById('timeline-editor-track-name')?.value || '',
            speakerFilename: document.getElementById('timeline-editor-track-speaker')?.value || '',
        });
        this.showNotification('Success', `Added track ${track.track_name}`, 'success');
        this.focusTimelineEditorTrackCreator();
    } catch (error) {
        console.error('Failed to add timeline track from editor:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.getTimelineTimeFromLaneEvent = function(event, laneElement) {
    if (!laneElement) {
        return 0;
    }

    const bounds = laneElement.getBoundingClientRect();
    const relativeX = Math.max(0, (event.clientX || bounds.left) - bounds.left);
    const seconds = relativeX / this.timelinePixelsPerSecond;
    return this.snapTimelineValue(seconds);
};

IndexTTSApp.prototype.updateTimelineStandaloneBuilder = function() {
    const status = document.getElementById('timeline-builder-status');
    const trackSelect = document.getElementById('timeline-inline-segment-track');
    const startInput = document.getElementById('timeline-inline-segment-start-time');
    const autoPlaceCheckbox = document.getElementById('timeline-inline-segment-autoplace');
    const addSegmentButton = document.getElementById('timeline-inline-add-segment-btn');
    const addTrackSegmentButton = document.getElementById('timeline-inline-add-track-segment-btn');
    const advancedSegmentButton = document.getElementById('timeline-add-segment-btn');

    if (trackSelect) {
        const preferredTrackId = this.selectedTimelineTrackId || trackSelect.value || this.currentTimelineProject?.tracks?.[0]?.track_id || '';
        this.populateTimelineTrackSelect(trackSelect, preferredTrackId);
    }

    const hasProject = Boolean(this.currentTimelineProjectId && this.currentTimelineProject);
    const trackCount = this.currentTimelineProject?.tracks?.length || 0;
    const selectedTrackId = trackSelect?.value || this.selectedTimelineTrackId || this.currentTimelineProject?.tracks?.[0]?.track_id || '';
    const selectedTrack = this.findTimelineTrack(selectedTrackId);
    const autoPlace = autoPlaceCheckbox?.checked ?? true;

    if (startInput && autoPlace) {
        startInput.value = this.getTimelineRecommendedStartTime(selectedTrackId);
    }

    if (addSegmentButton) {
        addSegmentButton.disabled = !hasProject || !trackCount;
    }
    if (addTrackSegmentButton) {
        addTrackSegmentButton.disabled = !hasProject;
    }
    if (advancedSegmentButton) {
        advancedSegmentButton.disabled = !hasProject || !trackCount;
    }

    if (!status) {
        return;
    }

    if (!hasProject) {
        status.textContent = 'Create a blank timeline first. Once it exists, you can add speaker tracks and write the scene directly in this tab.';
        return;
    }

    if (!trackCount) {
        status.textContent = `Timeline "${this.currentTimelineProject.project_name}" is ready. Add your first speaker track, then write a segment here.`;
        return;
    }

    if (selectedTrack) {
        const recommendedStart = this.getTimelineRecommendedStartTime(selectedTrack.track_id).toFixed(1);
        status.textContent = `Writing into ${selectedTrack.track_name}. The next clean start point on this track is ${recommendedStart}s. Use another track if you want an interruption or overlap.`;
        return;
    }

    status.textContent = `Timeline "${this.currentTimelineProject.project_name}" is loaded. Choose a track, write a line, and add it straight to the canvas.`;
};

IndexTTSApp.prototype.updateTimelineProjectSummary = function() {
    const summary = document.getElementById('timeline-project-summary');
    const status = document.getElementById('timeline-project-status');

    if (!summary || !status) {
        return;
    }

    if (!this.currentTimelineProject) {
        summary.textContent = 'No timeline loaded';
        status.textContent = 'Create a blank timeline to build a scene here from scratch, or import a conversation from results if you want a head start. Same-track overlap is blocked on purpose; use multiple tracks for interruptions.';
        return;
    }

    const project = this.currentTimelineProject;
    const trackCount = project.tracks?.length || 0;
    const segmentCount = (project.tracks || []).reduce((total, track) => total + (track.segments?.length || 0), 0);
    summary.textContent = `${trackCount} track${trackCount === 1 ? '' : 's'} | ${segmentCount} segment${segmentCount === 1 ? '' : 's'} | ${project.total_duration?.toFixed(1) || '0.0'}s`;

    const selectedTrack = this.findTimelineTrack(this.selectedTimelineTrackId);
    if (selectedTrack) {
        status.textContent = `Selected track: ${selectedTrack.track_name}. Build the scene by adding new lines above, then drag segments horizontally to retime them or move them between tracks from the edit panel.`;
    } else {
        status.textContent = `Timeline "${project.project_name}" loaded. Add tracks and lines from the standalone builder, or drag existing segments to retime them.`;
    }
};

IndexTTSApp.prototype.loadTimelineProjects = async function() {
    try {
        const response = await this.apiRequest('/timeline/list', { suppressErrorNotification: true });
        this.timelineProjects = response.details?.projects || [];
        this.renderTimelineProjectOptions();

        if (this.currentTimelineProjectId) {
            const stillExists = this.timelineProjects.some((project) => project.project_id === this.currentTimelineProjectId);
            if (!stillExists) {
                this.currentTimelineProjectId = null;
                this.currentTimelineProject = null;
                this.selectedTimelineTrackId = null;
                this.selectedTimelineSegmentId = null;
                this.currentTimelineExportFilename = null;
                this.timelineWaveformCache = {};
                this.renderTimelineProject();
            }
        }

        if (this.pendingTimelineRouteProjectId && !this.currentTimelineProjectId) {
            const routeProjectExists = this.timelineProjects.some((project) => project.project_id === this.pendingTimelineRouteProjectId);
            if (routeProjectExists) {
                const projectId = this.pendingTimelineRouteProjectId;
                this.pendingTimelineRouteProjectId = null;
                await this.loadTimelineProject(projectId);
            }
        }
    } catch (error) {
        console.error('Failed to load timeline projects:', error);
        this.timelineProjects = [];
        this.renderTimelineProjectOptions();
    }
};

IndexTTSApp.prototype.loadTimelineProject = async function(projectId) {
    if (!projectId) {
        this.currentTimelineProjectId = null;
        this.currentTimelineProject = null;
        this.selectedTimelineTrackId = null;
        this.selectedTimelineSegmentId = null;
        this.currentTimelineExportFilename = null;
        this.timelineWaveformCache = {};
        this.renderTimelineProject();
        return;
    }

    const response = await this.apiRequest(`/timeline/${projectId}`);
    this.currentTimelineProjectId = response.project.project_id;
    this.currentTimelineProject = response.project;
    this.currentTimelineExportFilename = null;
    this.timelineWaveformCache = {};
    this.timelineExportDirty = true;
    this.syncCurrentTimelineProjectListEntry();

    if (!this.findTimelineTrack(this.selectedTimelineTrackId)) {
        this.selectedTimelineTrackId = this.currentTimelineProject.tracks?.[0]?.track_id || null;
    }

    const selectedSegment = this.findTimelineSegment(this.selectedTimelineTrackId, this.selectedTimelineSegmentId);
    if (!selectedSegment) {
        this.selectedTimelineSegmentId = null;
    }

    this.renderTimelineProjectOptions();
    this.renderTimelineProject();
};

IndexTTSApp.prototype.ensureTimelineExportReady = async function() {
    if (!this.currentTimelineProjectId) {
        throw new Error('No timeline project loaded');
    }

    if (!this.currentTimelineExportFilename || this.timelineExportDirty) {
        await this.exportCurrentTimeline({ suppressSuccessNotification: true });
    }
};

IndexTTSApp.prototype.createBlankTimelineProject = async function() {
    const projectNameInput = document.getElementById('timeline-project-name');
    const projectName = (projectNameInput?.value || '').trim() || `timeline-${Date.now()}`;

    try {
        const response = await this.apiRequest('/timeline/create', {
            method: 'POST',
            body: JSON.stringify({
                project_name: projectName,
                description: 'Created from the standalone timeline editor',
            }),
        });

        if (projectNameInput) {
            projectNameInput.value = response.project.project_name;
        }

        await this.loadTimelineProjects();
        await this.loadTimelineProject(response.project.project_id);
        this.switchTab('timeline-editor');
        this.showNotification('Success', `Created timeline ${response.project.project_name}`, 'success');
    } catch (error) {
        console.error('Failed to create blank timeline project:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.importSelectedConversationToTimeline = async function() {
    if (!this.currentConversationId) {
        this.showNotification('Warning', 'Select a conversation in the results tab first', 'warning');
        return;
    }

    const projectNameInput = document.getElementById('timeline-project-name');
    const projectName = (projectNameInput?.value || '').trim() || `conversation-${this.currentConversationId.substring(0, 8)}-timeline`;

    try {
        const response = await this.apiRequest(`/timeline/create/${this.currentConversationId}`, {
            method: 'POST',
            body: JSON.stringify({
                project_name: projectName,
                description: `Imported from conversation ${this.currentConversationId}`,
                conversation_id: this.currentConversationId,
            }),
        });

        if (projectNameInput) {
            projectNameInput.value = response.project.project_name;
        }

        await this.loadTimelineProjects();
        await this.loadTimelineProject(response.project.project_id);
        this.switchTab('timeline-editor');
        this.showNotification('Success', 'Conversation imported into the timeline editor', 'success');
    } catch (error) {
        console.error('Failed to import conversation into timeline:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.openSelectedConversationInTimeline = async function() {
    this.switchTab('timeline-editor');
    await this.importSelectedConversationToTimeline();
};

IndexTTSApp.prototype.loadSelectedTimelineProject = async function() {
    const select = document.getElementById('timeline-project-select');
    const projectId = select?.value || this.currentTimelineProjectId;

    if (!projectId) {
        this.showNotification('Warning', 'Choose a timeline project to load', 'warning');
        return;
    }

    try {
        await this.loadTimelineProject(projectId);
        this.showNotification('Success', 'Timeline project loaded', 'success');
    } catch (error) {
        console.error('Failed to load selected timeline project:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.deleteSelectedTimelineProject = async function() {
    const select = document.getElementById('timeline-project-select');
    const projectId = select?.value || this.currentTimelineProjectId;

    if (!projectId) {
        this.showNotification('Warning', 'Choose a timeline project to delete', 'warning');
        return;
    }

    if (!window.confirm('Delete this timeline project? This removes the saved layout but not the original conversation files.')) {
        return;
    }

    try {
        await this.apiRequest(`/timeline/${projectId}`, { method: 'DELETE' });

        if (projectId === this.currentTimelineProjectId) {
            this.currentTimelineProjectId = null;
            this.currentTimelineProject = null;
            this.selectedTimelineTrackId = null;
            this.selectedTimelineSegmentId = null;
            this.currentTimelineExportFilename = null;
        }

        await this.loadTimelineProjects();
        this.renderTimelineProject();
        this.showNotification('Success', 'Timeline project deleted', 'success');
    } catch (error) {
        console.error('Failed to delete timeline project:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.findTimelineTrack = function(trackId) {
    if (!this.currentTimelineProject || !trackId) {
        return null;
    }

    return (this.currentTimelineProject.tracks || []).find((track) => track.track_id === trackId) || null;
};

IndexTTSApp.prototype.findTimelineSegment = function(trackId, segmentId) {
    const track = this.findTimelineTrack(trackId);
    if (!track || !segmentId) {
        return null;
    }

    return (track.segments || []).find((segment) => segment.segment_id === segmentId) || null;
};

IndexTTSApp.prototype.getTimelineVisualDuration = function() {
    const totalDuration = safeTimelineNumber(this.currentTimelineProject?.total_duration, 0);
    return Math.max(totalDuration + 1, 8);
};

IndexTTSApp.prototype.snapTimelineValue = function(value) {
    return Math.round(Math.max(0, safeTimelineNumber(value, 0)) * 10) / 10;
};

IndexTTSApp.prototype.renderTimelineProject = function() {
    const shell = document.getElementById('timeline-editor-shell');
    if (!shell) {
        return;
    }

    if (!this.currentTimelineProject) {
        shell.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-grip-lines"></i>
                <p>No timeline loaded yet.</p>
                <p class="empty-state-detail">Create a blank timeline above, add speaker tracks, then write segments directly in the standalone builder.</p>
            </div>
        `;
        this.renderTimelineSelectedSegmentPanel();
        this.updateTimelineProjectSummary();
        this.updateTimelineStandaloneBuilder();
        return;
    }

    const project = this.currentTimelineProject;
    const totalDuration = this.getTimelineVisualDuration();
    const widthPx = Math.ceil(totalDuration * this.timelinePixelsPerSecond);
    const rulerMarks = [];
    const selectedTrackId = this.selectedTimelineTrackId || project.tracks?.[0]?.track_id || '';
    const selectedTrack = this.findTimelineTrack(selectedTrackId);
    const speakerOptionsHtml = this.getTimelineSpeakerOptionsHtml();

    for (let second = 0; second <= Math.ceil(totalDuration); second += 1) {
        rulerMarks.push(`
            <div class="timeline-ruler-mark" style="left:${second * this.timelinePixelsPerSecond}px">
                <span>${second.toFixed(0)}s</span>
            </div>
        `);
    }

    const tracksHtml = (project.tracks || []).map((track) => {
        const isSelectedTrack = track.track_id === this.selectedTimelineTrackId;
        const nextStartTime = this.getTimelineRecommendedStartTime(track.track_id).toFixed(1);
        const segmentsHtml = (track.segments || []).map((segment) => {
            const isSelectedSegment = track.track_id === this.selectedTimelineTrackId && segment.segment_id === this.selectedTimelineSegmentId;
            const leftPx = Math.max(0, safeTimelineNumber(segment.start_time, 0) * this.timelinePixelsPerSecond);
            const widthForDuration = Math.max(120, safeTimelineNumber(segment.duration, 2) * this.timelinePixelsPerSecond);
            const audioClass = segment.audio_filename ? 'has-audio' : 'missing-audio';
            const emotionLabel = segment.emotion_text || (segment.emotion_control_method === 'from_text' ? 'custom' : 'speaker');

            return `
                <button
                    type="button"
                    class="timeline-track-segment ${audioClass} ${isSelectedSegment ? 'selected' : ''}"
                    data-track-id="${track.track_id}"
                    data-segment-id="${segment.segment_id}"
                    style="left:${leftPx}px;width:${widthForDuration}px"
                    title="${escapeTimelineHtml(segment.text)}"
                >
                    <span class="timeline-segment-title">${escapeTimelineHtml(segment.text)}</span>
                    <span class="timeline-segment-meta">${safeTimelineNumber(segment.start_time, 0).toFixed(1)}s • ${safeTimelineNumber(segment.duration, 2).toFixed(1)}s • ${escapeTimelineHtml(emotionLabel)}</span>
                </button>
            `;
        }).join('');

        return `
            <div class="timeline-track-row">
                <div class="timeline-track-header ${isSelectedTrack ? 'selected' : ''}" data-track-id="${track.track_id}">
                    <div class="timeline-track-header-top">
                        <div>
                            <div class="timeline-track-title">${escapeTimelineHtml(track.track_name)}</div>
                            <div class="timeline-track-subtitle">${escapeTimelineHtml(track.speaker_filename)}</div>
                        </div>
                        <div class="timeline-track-header-actions">
                            <button
                                type="button"
                                class="btn btn-secondary btn-small timeline-track-add-segment"
                                data-track-id="${track.track_id}"
                                data-start-time="${nextStartTime}"
                            >
                                <i class="fas fa-plus"></i> Segment
                            </button>
                        </div>
                    </div>
                    <div class="timeline-track-controls">
                        <label class="timeline-track-volume-control">
                            <span class="timeline-track-volume-label">Level</span>
                            <input
                                type="range"
                                class="timeline-track-volume"
                                data-track-id="${track.track_id}"
                                min="0"
                                max="150"
                                step="5"
                                value="${Math.round(safeTimelineNumber(track.volume, 1) * 100)}"
                                aria-label="Track level for ${escapeTimelineHtml(track.track_name)}"
                            />
                            <span class="timeline-track-volume-value">${formatTimelineVolumePercent(track.volume)}</span>
                        </label>
                        <button type="button" class="btn btn-secondary btn-small timeline-track-toggle" data-track-id="${track.track_id}" data-action="mute">
                            <i class="fas ${track.muted ? 'fa-volume-xmark' : 'fa-volume-high'}"></i> ${track.muted ? 'Muted' : 'Mute'}
                        </button>
                        <button type="button" class="btn btn-secondary btn-small timeline-track-toggle" data-track-id="${track.track_id}" data-action="solo">
                            <i class="fas fa-headphones"></i> ${track.solo ? 'Soloed' : 'Solo'}
                        </button>
                    </div>
                </div>
                <div class="timeline-track-lane-wrapper">
                    <div class="timeline-track-lane ${isSelectedTrack ? 'selected' : ''}" data-track-id="${track.track_id}" style="width:${widthPx}px">
                        ${segmentsHtml || `
                            <div class="timeline-lane-empty">
                                <div class="timeline-lane-empty-copy">
                                    <span>No segments on this track yet.</span>
                                    <small>Double-click anywhere on this lane or use the quick add button to place the first line.</small>
                                </div>
                                <button
                                    type="button"
                                    class="btn btn-secondary btn-small timeline-lane-add-btn"
                                    data-track-id="${track.track_id}"
                                    data-start-time="${nextStartTime}"
                                >
                                    <i class="fas fa-plus"></i> Add segment here
                                </button>
                            </div>
                        `}
                    </div>
                </div>
            </div>
        `;
    }).join('');

    shell.innerHTML = `
        <div class="timeline-scroll-container" style="--timeline-lane-width:${widthPx}px">
            <div class="timeline-editor-toolbar" id="timeline-editor-toolbar">
                <div class="timeline-editor-toolbar-copy">
                    <span class="timeline-editor-toolbar-title">Edit On The Canvas</span>
                    <span class="timeline-editor-toolbar-subtitle">Professional timeline editors keep the add-track and add-segment actions close to the ruler and track headers.</span>
                </div>
                <div class="timeline-editor-toolbar-actions">
                    <button type="button" class="btn btn-secondary" id="timeline-editor-add-track-btn">
                        <i class="fas fa-plus"></i> Speaker Track
                    </button>
                    <button type="button" class="btn btn-secondary" id="timeline-editor-add-selected-segment-btn" ${selectedTrack ? '' : 'disabled'}>
                        <i class="fas fa-plus"></i> Segment On Selected Track
                    </button>
                </div>
            </div>
            <div class="timeline-track-create-strip">
                <label class="timeline-track-create-field">
                    <span>Speaker</span>
                    <select id="timeline-editor-track-speaker">
                        ${speakerOptionsHtml}
                    </select>
                </label>
                <label class="timeline-track-create-field timeline-track-create-field-wide">
                    <span>Track Name</span>
                    <input id="timeline-editor-track-name" type="text" placeholder="Trump Lead" />
                </label>
                <button type="button" class="btn btn-primary" id="timeline-editor-create-track-btn">
                    <i class="fas fa-plus"></i> Add Track To Timeline
                </button>
            </div>
            <div class="timeline-ruler-row">
                <div class="timeline-ruler-spacer"></div>
                <div class="timeline-ruler" style="width:${widthPx}px">
                    ${rulerMarks.join('')}
                </div>
            </div>
            <div class="timeline-track-list">
                ${tracksHtml || `
                    <div class="empty-state">
                        <i class="fas fa-layer-group"></i>
                        <p>No tracks yet. Add a speaker track above, then start placing segments directly on the canvas.</p>
                    </div>
                `}
            </div>
        </div>
    `;

    document.getElementById('timeline-editor-track-speaker')?.addEventListener('change', () => this.maybeSuggestTimelineEditorTrackName());
    document.getElementById('timeline-editor-create-track-btn')?.addEventListener('click', () => this.createTimelineTrackFromEditor());
    document.getElementById('timeline-editor-add-track-btn')?.addEventListener('click', () => this.focusTimelineEditorTrackCreator());
    document.getElementById('timeline-editor-add-selected-segment-btn')?.addEventListener('click', () => {
        const trackId = this.selectedTimelineTrackId || this.currentTimelineProject?.tracks?.[0]?.track_id;
        if (!trackId) {
            this.showNotification('Warning', 'Add a speaker track first', 'warning');
            return;
        }

        this.selectedTimelineTrackId = trackId;
        this.openCreateTimelineSegmentModal(trackId, this.getTimelineRecommendedStartTime(trackId));
    });

    this.maybeSuggestTimelineEditorTrackName();

    shell.querySelectorAll('.timeline-track-header, .timeline-track-lane').forEach((element) => {
        element.addEventListener('click', (event) => {
            if (
                event.target.closest('.timeline-track-toggle') ||
                event.target.closest('.timeline-track-add-segment') ||
                event.target.closest('.timeline-lane-add-btn')
            ) {
                return;
            }

            const trackId = event.currentTarget.dataset.trackId;
            this.selectedTimelineTrackId = trackId;
            if (!this.findTimelineSegment(trackId, this.selectedTimelineSegmentId)) {
                this.selectedTimelineSegmentId = null;
            }
            this.renderTimelineProject();
        });

        if (element.classList.contains('timeline-track-lane')) {
            element.addEventListener('dblclick', (event) => {
                if (event.target.closest('.timeline-track-segment') || event.target.closest('.timeline-lane-add-btn')) {
                    return;
                }

                const trackId = event.currentTarget.dataset.trackId;
                this.selectedTimelineTrackId = trackId;
                this.selectedTimelineSegmentId = null;
                this.openCreateTimelineSegmentModal(trackId, this.getTimelineTimeFromLaneEvent(event, event.currentTarget));
            });
        }
    });

    shell.querySelectorAll('.timeline-track-toggle').forEach((button) => {
        button.addEventListener('click', (event) => {
            event.stopPropagation();
            const trackId = event.currentTarget.dataset.trackId;
            const action = event.currentTarget.dataset.action;
            if (action === 'mute') {
                this.toggleTimelineTrackMute(trackId);
            } else if (action === 'solo') {
                this.toggleTimelineTrackSolo(trackId);
            }
        });
    });

    shell.querySelectorAll('.timeline-track-add-segment, .timeline-lane-add-btn').forEach((button) => {
        button.addEventListener('click', (event) => {
            event.stopPropagation();
            const trackId = event.currentTarget.dataset.trackId;
            const startTime = safeTimelineNumber(event.currentTarget.dataset.startTime, this.getTimelineRecommendedStartTime(trackId));
            this.selectedTimelineTrackId = trackId;
            this.selectedTimelineSegmentId = null;
            this.openCreateTimelineSegmentModal(trackId, startTime);
        });
    });

    shell.querySelectorAll('.timeline-track-volume').forEach((input) => {
        const updateLabel = () => {
            const label = input.closest('.timeline-track-volume-control')?.querySelector('.timeline-track-volume-value');
            if (label) {
                label.textContent = `${input.value}%`;
            }
        };

        input.addEventListener('click', (event) => event.stopPropagation());
        input.addEventListener('pointerdown', (event) => event.stopPropagation());
        input.addEventListener('input', (event) => {
            event.stopPropagation();
            updateLabel();
        });
        input.addEventListener('change', async (event) => {
            event.stopPropagation();
            const trackId = input.dataset.trackId;
            const volume = safeTimelineNumber(input.value, 100) / 100;
            await this.updateTimelineTrackVolume(trackId, volume);
        });
    });

    shell.querySelectorAll('.timeline-track-segment').forEach((segmentButton) => {
        segmentButton.addEventListener('click', (event) => {
            if (this.suppressTimelineSegmentClick) {
                this.suppressTimelineSegmentClick = false;
                return;
            }

            const trackId = event.currentTarget.dataset.trackId;
            const segmentId = event.currentTarget.dataset.segmentId;
            this.selectedTimelineTrackId = trackId;
            this.selectedTimelineSegmentId = segmentId;
            this.renderTimelineProject();
        });

        segmentButton.addEventListener('dblclick', (event) => {
            event.preventDefault();
            const trackId = event.currentTarget.dataset.trackId;
            const segmentId = event.currentTarget.dataset.segmentId;
            this.selectedTimelineTrackId = trackId;
            this.selectedTimelineSegmentId = segmentId;
            this.openEditTimelineSegmentModal();
        });

        segmentButton.addEventListener('pointerdown', (event) => {
            this.beginTimelineSegmentDrag(event);
        });
    });

    this.renderTimelineSelectedSegmentPanel();
    this.updateTimelineProjectSummary();
    this.updateTimelineStandaloneBuilder();
};

IndexTTSApp.prototype.getTimelineWaveformCacheKey = function(trackId, segmentId) {
    return `${this.currentTimelineProjectId || 'timeline'}:${trackId || 'track'}:${segmentId || 'segment'}`;
};

IndexTTSApp.prototype.loadTimelineSegmentWaveform = async function(trackId, segmentId, options = {}) {
    if (!this.currentTimelineProjectId || !trackId || !segmentId) {
        return;
    }

    const cacheKey = this.getTimelineWaveformCacheKey(trackId, segmentId);
    const existing = this.timelineWaveformCache?.[cacheKey];
    if (existing?.loading && !options.force) {
        return;
    }

    this.timelineWaveformCache = this.timelineWaveformCache || {};
    this.timelineWaveformCache[cacheKey] = { loading: true };

    try {
        const waveform = await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${trackId}/segments/${segmentId}/waveform?bars=72`, {
            suppressErrorNotification: true,
        });
        this.timelineWaveformCache[cacheKey] = waveform;
    } catch (error) {
        this.timelineWaveformCache[cacheKey] = {
            error: error.message || 'Waveform preview unavailable',
        };
    }

    if (this.selectedTimelineTrackId === trackId && this.selectedTimelineSegmentId === segmentId) {
        this.renderTimelineSelectedSegmentPanel();
    }
};

IndexTTSApp.prototype.renderTimelineSelectedSegmentPanel = function() {
    const panel = document.getElementById('timeline-selected-panel');
    if (!panel) {
        return;
    }

    const track = this.findTimelineTrack(this.selectedTimelineTrackId);
    const segment = this.findTimelineSegment(this.selectedTimelineTrackId, this.selectedTimelineSegmentId);

    if (!track || !segment) {
        panel.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-arrow-pointer"></i>
                <p>Select a segment to edit, play, generate audio, or delete it.</p>
            </div>
        `;
        return;
    }

    const cacheKey = this.getTimelineWaveformCacheKey(track.track_id, segment.segment_id);
    const waveformState = this.timelineWaveformCache?.[cacheKey];
    let waveformMarkup = `
        <div class="timeline-waveform-empty">
            <i class="fas fa-wave-square"></i>
            <span>Generate audio to see the waveform preview.</span>
        </div>
    `;

    if (segment.audio_filename) {
        if (waveformState?.loading) {
            waveformMarkup = `
                <div class="timeline-waveform-empty">
                    <i class="fas fa-spinner fa-spin"></i>
                    <span>Loading waveform preview...</span>
                </div>
            `;
        } else if (waveformState?.peaks?.length) {
            waveformMarkup = `
                <div class="timeline-waveform-bars" aria-label="Waveform preview" style="grid-template-columns: repeat(${waveformState.peaks.length}, minmax(2px, 1fr));">
                    ${buildTimelineWaveformBars(waveformState.peaks)}
                </div>
            `;
        } else if (waveformState?.error) {
            waveformMarkup = `
                <div class="timeline-waveform-empty">
                    <i class="fas fa-triangle-exclamation"></i>
                    <span>${escapeTimelineHtml(waveformState.error)}</span>
                </div>
            `;
        } else {
            this.loadTimelineSegmentWaveform(track.track_id, segment.segment_id);
        }
    }

    panel.innerHTML = `
        <div class="timeline-selected-card">
            <div class="timeline-selected-copy">
                <div class="timeline-selected-title">${escapeTimelineHtml(segment.text)}</div>
                <div class="timeline-selected-meta">
                    ${escapeTimelineHtml(track.track_name)} • ${safeTimelineNumber(segment.start_time, 0).toFixed(1)}s start • ${safeTimelineNumber(segment.duration, 2).toFixed(1)}s duration
                </div>
                <div class="timeline-selected-meta">
                    ${segment.audio_filename ? `Audio ready: ${escapeTimelineHtml(segment.audio_filename)}` : 'Audio has not been generated for this segment yet.'}
                </div>
                <div class="timeline-waveform-panel">
                    <div class="timeline-waveform-header">
                        <span>Waveform Preview</span>
                        ${segment.audio_filename ? '<button type="button" class="btn btn-secondary btn-small" id="timeline-refresh-waveform-btn"><i class="fas fa-rotate"></i> Refresh Waveform</button>' : ''}
                    </div>
                    ${waveformMarkup}
                </div>
            </div>
            <div class="timeline-selected-actions">
                <button type="button" class="btn btn-secondary" id="timeline-edit-selected-btn">
                    <i class="fas fa-pen"></i> Edit Segment
                </button>
                <button type="button" class="btn btn-secondary" id="timeline-generate-selected-panel-btn">
                    <i class="fas fa-microphone-lines"></i> Generate Audio
                </button>
                <button type="button" class="btn btn-secondary" id="timeline-play-selected-btn">
                    <i class="fas fa-play"></i> Play Segment
                </button>
                <button type="button" class="btn btn-secondary" id="timeline-split-selected-btn">
                    <i class="fas fa-scissors"></i> Split Segment
                </button>
                <button type="button" class="btn btn-secondary" id="timeline-popout-selected-btn">
                    <i class="fas fa-up-right-from-square"></i> Pop Out Editor
                </button>
                <button type="button" class="btn btn-secondary" id="timeline-delete-selected-btn">
                    <i class="fas fa-trash"></i> Delete Segment
                </button>
            </div>
        </div>
    `;

    document.getElementById('timeline-edit-selected-btn')?.addEventListener('click', () => this.openEditTimelineSegmentModal());
    document.getElementById('timeline-generate-selected-panel-btn')?.addEventListener('click', () => this.generateSelectedTimelineSegmentAudio());
    document.getElementById('timeline-play-selected-btn')?.addEventListener('click', () => this.playSelectedTimelineSegment());
    document.getElementById('timeline-split-selected-btn')?.addEventListener('click', () => this.splitSelectedTimelineSegment());
    document.getElementById('timeline-popout-selected-btn')?.addEventListener('click', () => this.openTimelinePopoutWindow());
    document.getElementById('timeline-refresh-waveform-btn')?.addEventListener('click', () => this.loadTimelineSegmentWaveform(track.track_id, segment.segment_id, { force: true }));
    document.getElementById('timeline-delete-selected-btn')?.addEventListener('click', () => this.deleteSelectedTimelineSegment());
};

IndexTTSApp.prototype.toggleTimelineTrackMute = async function(trackId) {
    if (!this.currentTimelineProjectId || !trackId) {
        return;
    }

    try {
        await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${trackId}/mute`, { method: 'PUT' });
        await this.loadTimelineProject(this.currentTimelineProjectId);
    } catch (error) {
        console.error('Failed to toggle track mute:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.toggleTimelineTrackSolo = async function(trackId) {
    if (!this.currentTimelineProjectId || !trackId) {
        return;
    }

    try {
        await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${trackId}/solo`, { method: 'PUT' });
        await this.loadTimelineProject(this.currentTimelineProjectId);
    } catch (error) {
        console.error('Failed to toggle track solo:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.updateTimelineTrackVolume = async function(trackId, volume) {
    if (!this.currentTimelineProjectId || !trackId) {
        return;
    }

    try {
        await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${trackId}/volume`, {
            method: 'PUT',
            body: JSON.stringify({ volume }),
        });
        await this.loadTimelineProject(this.currentTimelineProjectId);
        this.showNotification('Success', `Track level set to ${formatTimelineVolumePercent(volume)}`, 'success');
    } catch (error) {
        console.error('Failed to update timeline track volume:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.populateTimelineTrackSelect = function(selectElement, selectedTrackId) {
    if (!selectElement) {
        return;
    }

    selectElement.innerHTML = '<option value="">Select track...</option>';
    (this.currentTimelineProject?.tracks || []).forEach((track) => {
        const option = document.createElement('option');
        option.value = track.track_id;
        option.textContent = `${track.track_name} (${track.speaker_filename})`;
        selectElement.appendChild(option);
    });

    if (selectedTrackId) {
        selectElement.value = selectedTrackId;
    }
};

IndexTTSApp.prototype.createTimelineTrackInternal = async function(options = {}) {
    if (!this.currentTimelineProjectId) {
        throw new Error('Load or create a timeline project first');
    }

    const trackNameInput = document.getElementById('timeline-track-name');
    const trackSpeakerSelect = document.getElementById('timeline-track-speaker');
    const requestedTrackName = (options.trackName ?? trackNameInput?.value ?? '').trim();
    const speakerFilename = options.speakerFilename ?? trackSpeakerSelect?.value ?? '';
    const trackName = requestedTrackName || this.buildSuggestedTimelineTrackName(speakerFilename);

    if (!trackName) {
        throw new Error('Enter a track name first');
    }

    if (!speakerFilename) {
        throw new Error('Choose a speaker for the new track');
    }

    const response = await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks`, {
        method: 'POST',
        body: JSON.stringify({
            track_name: trackName,
            speaker_filename: speakerFilename,
        }),
    });

    this.selectedTimelineTrackId = response.track.track_id;
    await this.loadTimelineProject(this.currentTimelineProjectId);

    if (!options.preserveTrackInputs) {
        if (trackNameInput) {
            trackNameInput.value = '';
            trackNameInput.dataset.suggestedName = '';
        }
        if (trackSpeakerSelect) {
            trackSpeakerSelect.value = '';
        }
    }

    this.updateTimelineStandaloneBuilder();
    return response.track;
};

IndexTTSApp.prototype.addTimelineTrack = async function() {
    try {
        const track = await this.createTimelineTrackInternal();
        this.showNotification('Success', `Added track ${track.track_name}`, 'success');
    } catch (error) {
        console.error('Failed to add timeline track:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.openCreateTimelineSegmentModal = function(defaultTrackId = null, defaultStartTime = null) {
    if (!this.currentTimelineProjectId || !this.currentTimelineProject) {
        this.showNotification('Warning', 'Load or create a timeline project first', 'warning');
        return;
    }

    const modal = document.getElementById('line-editor-modal');
    const trackSelect = document.getElementById('line-speaker-select');
    const textInput = document.getElementById('line-text-input');
    const emotionSelect = document.getElementById('line-emotion-select');
    const startInput = document.getElementById('line-start-time-input');
    const durationInput = document.getElementById('line-duration-input');

    const preferredTrackId = defaultTrackId || this.selectedTimelineTrackId || this.currentTimelineProject.tracks?.[0]?.track_id || '';
    this.populateTimelineTrackSelect(trackSelect, preferredTrackId);

    if (textInput) {
        textInput.value = '';
    }
    if (emotionSelect) {
        emotionSelect.value = 'neutral';
    }
    if (startInput) {
        startInput.value = defaultStartTime ?? this.getTimelineRecommendedStartTime(preferredTrackId);
    }
    if (durationInput) {
        durationInput.value = '2.0';
    }

    modal?.classList.add('show');
};

IndexTTSApp.prototype.closeCreateTimelineSegmentModal = function() {
    document.getElementById('line-editor-modal')?.classList.remove('show');
};

IndexTTSApp.prototype.createTimelineSegmentRequest = async function(options = {}) {
    if (!this.currentTimelineProjectId) {
        throw new Error('No timeline project loaded');
    }

    const trackId = options.trackId || '';
    const text = String(options.text || '').trim();
    const emotion = options.emotion || 'neutral';
    const startTime = this.snapTimelineValue(options.startTime || 0);
    const duration = Math.max(0.1, safeTimelineNumber(options.duration, 2.0));

    if (!trackId) {
        throw new Error('Choose a timeline track first');
    }

    if (!text) {
        throw new Error('Enter text for the new segment');
    }

    const response = await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${trackId}/segments`, {
        method: 'POST',
        body: JSON.stringify({
            track_id: trackId,
            text,
            start_time: startTime,
            duration,
            emotion_control_method: emotion === 'neutral' ? 'from_speaker' : 'from_text',
            emotion_text: emotion === 'neutral' ? null : emotion,
            use_random_sampling: document.getElementById('use-random-sampling')?.checked || false,
            do_sample: document.getElementById('do-sample')?.checked ?? true,
            top_p: safeTimelineNumber(document.getElementById('top-p')?.value, 0.8),
            top_k: safeTimelineNumber(document.getElementById('top-k')?.value, 30),
            temperature: safeTimelineNumber(document.getElementById('temperature')?.value, 0.8),
            length_penalty: safeTimelineNumber(document.getElementById('length-penalty')?.value, 0),
            num_beams: safeTimelineNumber(document.getElementById('num-beams')?.value, 3),
            repetition_penalty: safeTimelineNumber(document.getElementById('repetition-penalty')?.value, 10),
            max_mel_tokens: safeTimelineNumber(document.getElementById('max-mel-tokens')?.value, 1500),
            max_text_tokens_per_segment: safeTimelineNumber(document.getElementById('max-text-tokens')?.value, 120),
        }),
    });

    this.selectedTimelineTrackId = trackId;
    this.selectedTimelineSegmentId = response.segment.segment_id;
    await this.loadTimelineProject(this.currentTimelineProjectId);
    this.updateTimelineStandaloneBuilder();
    return response.segment;
};

IndexTTSApp.prototype.createInlineTimelineSegment = async function(options = {}) {
    try {
        if (!this.currentTimelineProjectId) {
            throw new Error('Create or load a timeline project first');
        }

        let trackId = document.getElementById('timeline-inline-segment-track')?.value || '';
        const textInput = document.getElementById('timeline-inline-segment-text');
        const text = textInput?.value || '';
        const emotion = document.getElementById('timeline-inline-segment-emotion')?.value || 'neutral';
        const autoPlace = document.getElementById('timeline-inline-segment-autoplace')?.checked ?? true;
        const manualStart = document.getElementById('timeline-inline-segment-start-time')?.value || 0;
        const duration = document.getElementById('timeline-inline-segment-duration')?.value || 2.0;

        if (options.createTrackFirst) {
            const track = await this.createTimelineTrackInternal();
            trackId = track.track_id;
            const trackSelect = document.getElementById('timeline-inline-segment-track');
            if (trackSelect) {
                trackSelect.value = track.track_id;
            }
        }

        const startTime = autoPlace ? this.getTimelineRecommendedStartTime(trackId) : manualStart;
        await this.createTimelineSegmentRequest({
            trackId,
            text,
            emotion,
            startTime,
            duration,
        });

        if (textInput) {
            textInput.value = '';
        }
        this.showNotification('Success', 'Timeline segment created', 'success');
    } catch (error) {
        console.error('Failed to create inline timeline segment:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.createTimelineSegment = async function() {
    try {
        await this.createTimelineSegmentRequest({
            trackId: document.getElementById('line-speaker-select')?.value || '',
            text: document.getElementById('line-text-input')?.value || '',
            emotion: document.getElementById('line-emotion-select')?.value || 'neutral',
            startTime: document.getElementById('line-start-time-input')?.value || 0,
            duration: document.getElementById('line-duration-input')?.value || 2.0,
        });

        this.closeCreateTimelineSegmentModal();
        this.showNotification('Success', 'Timeline segment created', 'success');
    } catch (error) {
        console.error('Failed to create timeline segment:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.openEditTimelineSegmentModal = function() {
    const track = this.findTimelineTrack(this.selectedTimelineTrackId);
    const segment = this.findTimelineSegment(this.selectedTimelineTrackId, this.selectedTimelineSegmentId);
    if (!track || !segment) {
        this.showNotification('Warning', 'Select a segment first', 'warning');
        return;
    }

    this.populateTimelineTrackSelect(document.getElementById('edit-segment-track'), track.track_id);
    document.getElementById('edit-segment-text').value = segment.text || '';
    document.getElementById('edit-segment-start-time').value = this.snapTimelineValue(segment.start_time || 0);
    document.getElementById('edit-segment-duration').value = Math.max(0.1, safeTimelineNumber(segment.duration, 2.0)).toFixed(1);
    document.getElementById('edit-segment-emotion').value = segment.emotion_text || 'neutral';
    document.getElementById('edit-segment-regenerate-audio').checked = false;
    document.getElementById('edit-segment-modal')?.classList.add('show');
};

IndexTTSApp.prototype.closeEditTimelineSegmentModal = function() {
    document.getElementById('edit-segment-modal')?.classList.remove('show');
};

IndexTTSApp.prototype.saveEditedTimelineSegment = async function() {
    const originalTrackId = this.selectedTimelineTrackId;
    const originalSegmentId = this.selectedTimelineSegmentId;
    const originalTrack = this.findTimelineTrack(originalTrackId);
    const originalSegment = this.findTimelineSegment(originalTrackId, originalSegmentId);

    if (!this.currentTimelineProjectId || !originalTrack || !originalSegment) {
        this.showNotification('Warning', 'Select a segment first', 'warning');
        return;
    }

    const targetTrackId = document.getElementById('edit-segment-track')?.value || originalTrackId;
    const text = (document.getElementById('edit-segment-text')?.value || '').trim();
    const startTime = this.snapTimelineValue(document.getElementById('edit-segment-start-time')?.value || originalSegment.start_time || 0);
    const duration = Math.max(0.1, safeTimelineNumber(document.getElementById('edit-segment-duration')?.value, originalSegment.duration || 2.0));
    const emotion = document.getElementById('edit-segment-emotion')?.value || 'neutral';
    const regenerateAudio = document.getElementById('edit-segment-regenerate-audio')?.checked || false;

    if (!text) {
        this.showNotification('Warning', 'Segment text cannot be empty', 'warning');
        return;
    }

    try {
        let activeTrackId = originalTrackId;

        if (targetTrackId !== originalTrackId) {
            await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${originalTrackId}/segments/${originalSegmentId}/move`, {
                method: 'PUT',
                body: JSON.stringify({
                    target_track_id: targetTrackId,
                    new_start_time: startTime,
                }),
            });
            activeTrackId = targetTrackId;
        } else if (
            this.snapTimelineValue(originalSegment.start_time || 0) !== startTime
            || safeTimelineNumber(originalSegment.duration, 2.0) !== duration
        ) {
            await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${originalTrackId}/segments/${originalSegmentId}`, {
                method: 'PUT',
                body: JSON.stringify({
                    start_time: startTime,
                    duration,
                }),
            });
        }

        await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${activeTrackId}/segments/${originalSegmentId}/properties`, {
            method: 'PUT',
            body: JSON.stringify({
                text,
                duration,
                emotion_control_method: emotion === 'neutral' ? 'from_speaker' : 'from_text',
                emotion_text: emotion === 'neutral' ? null : emotion,
                audio_filename: regenerateAudio ? null : originalSegment.audio_filename,
            }),
        });

        if (regenerateAudio) {
            await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${activeTrackId}/segments/${originalSegmentId}/generate`, {
                method: 'POST',
            });
        }

        this.closeEditTimelineSegmentModal();
        await this.loadTimelineProject(this.currentTimelineProjectId);
        this.selectedTimelineTrackId = activeTrackId;
        this.selectedTimelineSegmentId = originalSegmentId;
        this.renderTimelineProject();
        this.showNotification('Success', 'Segment updated', 'success');
    } catch (error) {
        console.error('Failed to save edited timeline segment:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.generateSelectedTimelineSegmentAudio = async function() {
    if (!this.currentTimelineProjectId || !this.selectedTimelineTrackId || !this.selectedTimelineSegmentId) {
        this.showNotification('Warning', 'Select a segment first', 'warning');
        return;
    }

    try {
        await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${this.selectedTimelineTrackId}/segments/${this.selectedTimelineSegmentId}/generate`, {
            method: 'POST',
        });
        await this.loadTimelineProject(this.currentTimelineProjectId);
        this.showNotification('Success', 'Segment audio generated', 'success');
    } catch (error) {
        console.error('Failed to generate selected segment audio:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.generateMissingTimelineAudio = async function() {
    if (!this.currentTimelineProjectId || !this.currentTimelineProject) {
        this.showNotification('Warning', 'Load a timeline project first', 'warning');
        return;
    }

    const missingSegments = [];
    (this.currentTimelineProject.tracks || []).forEach((track) => {
        (track.segments || []).forEach((segment) => {
            if (!segment.audio_filename) {
                missingSegments.push({ trackId: track.track_id, segmentId: segment.segment_id });
            }
        });
    });

    if (!missingSegments.length) {
        this.showNotification('Success', 'All segments already have audio', 'success');
        return;
    }

    try {
        for (const segment of missingSegments) {
            await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${segment.trackId}/segments/${segment.segmentId}/generate`, {
                method: 'POST',
            });
        }

        await this.loadTimelineProject(this.currentTimelineProjectId);
        this.showNotification('Success', `Generated audio for ${missingSegments.length} segment${missingSegments.length === 1 ? '' : 's'}`, 'success');
    } catch (error) {
        console.error('Failed to generate missing timeline audio:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.deleteSelectedTimelineSegment = async function() {
    if (!this.currentTimelineProjectId || !this.selectedTimelineTrackId || !this.selectedTimelineSegmentId) {
        this.showNotification('Warning', 'Select a segment first', 'warning');
        return;
    }

    if (!window.confirm('Delete this timeline segment?')) {
        return;
    }

    try {
        await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${this.selectedTimelineTrackId}/segments/${this.selectedTimelineSegmentId}`, {
            method: 'DELETE',
        });
        this.selectedTimelineSegmentId = null;
        await this.loadTimelineProject(this.currentTimelineProjectId);
        this.showNotification('Success', 'Segment deleted', 'success');
    } catch (error) {
        console.error('Failed to delete timeline segment:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.playSelectedTimelineSegment = async function() {
    const segment = this.findTimelineSegment(this.selectedTimelineTrackId, this.selectedTimelineSegmentId);
    if (!this.currentTimelineProjectId || !segment) {
        this.showNotification('Warning', 'Select a segment first', 'warning');
        return;
    }

    if (!segment.audio_filename) {
        this.showNotification('Warning', 'Generate audio for this segment first', 'warning');
        return;
    }

    try {
        const modal = document.getElementById('audio-player-modal');
        const mediaPlayer = await this.initializeCustomMediaPlayer();
        const audioUrl = `${this.apiBaseUrl}/timeline/${this.currentTimelineProjectId}/assets/audio/${segment.audio_filename}`;
        const lineData = {
            line_number: 0,
            speaker: segment.speaker_filename,
            text: segment.text,
            versions: [
                {
                    version: 0,
                    audio_url: audioUrl,
                    audio_filename: segment.audio_filename,
                    quality_score: 0.8,
                    duration: segment.duration,
                    file_size: 0,
                },
            ],
        };

        await mediaPlayer.loadLine(lineData, { autoplay: true });
        modal.classList.add('show');
    } catch (error) {
        console.error('Failed to play selected timeline segment:', error);
        this.showNotification('Error', 'Failed to play timeline audio', 'error');
    }
};

IndexTTSApp.prototype.splitSelectedTimelineSegment = async function() {
    const track = this.findTimelineTrack(this.selectedTimelineTrackId);
    const segment = this.findTimelineSegment(this.selectedTimelineTrackId, this.selectedTimelineSegmentId);
    if (!this.currentTimelineProjectId || !track || !segment) {
        this.showNotification('Warning', 'Select a segment first', 'warning');
        return;
    }

    if (safeTimelineNumber(segment.duration, 0) < 0.5) {
        this.showNotification('Warning', 'This segment is too short to split cleanly', 'warning');
        return;
    }

    const suggestedOffset = Math.max(0.2, Math.min(safeTimelineNumber(segment.duration, 0) - 0.2, safeTimelineNumber(segment.duration, 0) / 2));
    const splitOffsetRaw = window.prompt(
        'Split this segment at how many seconds from its start? Use a value inside the clip length.',
        suggestedOffset.toFixed(1),
    );
    if (splitOffsetRaw === null) {
        return;
    }

    const splitOffset = safeTimelineNumber(splitOffsetRaw, suggestedOffset);
    try {
        const response = await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${track.track_id}/segments/${segment.segment_id}/split`, {
            method: 'POST',
            body: JSON.stringify({
                split_offset: splitOffset,
            }),
        });

        await this.loadTimelineProject(this.currentTimelineProjectId);
        this.selectedTimelineTrackId = track.track_id;
        this.selectedTimelineSegmentId = response.new_segment.segment_id;
        this.renderTimelineProject();
        this.showNotification('Success', 'Segment split. Regenerate audio for both halves when you are ready.', 'success');
    } catch (error) {
        console.error('Failed to split selected timeline segment:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.openTimelinePopoutWindow = function() {
    const params = new URLSearchParams();
    params.set('tab', 'timeline-editor');
    params.set('timelinePopout', '1');
    if (this.currentTimelineProjectId) {
        params.set('timelineProject', this.currentTimelineProjectId);
    }

    const targetUrl = `${window.location.origin}${window.location.pathname}?${params.toString()}`;
    window.open(targetUrl, 'indextts-timeline-editor', 'width=1560,height=980,resizable=yes,scrollbars=yes');
};

IndexTTSApp.prototype.exportCurrentTimeline = async function(options = {}) {
    if (!this.currentTimelineProjectId) {
        this.showNotification('Warning', 'Load a timeline project first', 'warning');
        return;
    }

    const outputBaseName = `timeline_${this.currentTimelineProjectId}_exported`;
    const exportSettings = this.getTimelineExportSettings();

    try {
        const response = await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/export`, {
            method: 'POST',
            body: JSON.stringify({
                project_id: this.currentTimelineProjectId,
                output_filename: outputBaseName,
                format: exportSettings.format,
                output_bitrate_kbps: exportSettings.output_bitrate_kbps,
                sample_rate: 22050,
                duck_overlaps: exportSettings.duck_overlaps,
                duck_amount_db: exportSettings.duck_amount_db,
                duck_fade_ms: exportSettings.duck_fade_ms,
                normalize_segments: exportSettings.normalize_segments,
                target_level_dbfs: exportSettings.target_level_dbfs,
                peak_limit_dbfs: exportSettings.peak_limit_dbfs,
                normalize_final_mix: exportSettings.normalize_final_mix,
                trim_leading_silence: exportSettings.trim_leading_silence,
                trim_trailing_silence: exportSettings.trim_trailing_silence,
                fade_in_ms: exportSettings.fade_in_ms,
                fade_out_ms: exportSettings.fade_out_ms,
            }),
        });

        this.currentTimelineExportFilename = response.output_filename;
        this.timelineExportDirty = false;

        if (!options.suppressSuccessNotification) {
            this.showNotification('Success', `Timeline exported in ${response.export_time_seconds || 0}s`, 'success');
        }

        return response;
    } catch (error) {
        console.error('Failed to export timeline:', error);
        this.showNotification('Error', error.message, 'error');
        throw error;
    }
};

IndexTTSApp.prototype.playTimelineExport = async function() {
    if (!this.currentTimelineProjectId) {
        this.showNotification('Warning', 'Load a timeline project first', 'warning');
        return;
    }

    try {
        await this.ensureTimelineExportReady();

        const modal = document.getElementById('audio-player-modal');
        const mediaPlayer = await this.initializeCustomMediaPlayer();
        const audioUrl = `${this.apiBaseUrl}/timeline/${this.currentTimelineProjectId}/export/download`;
        const lineData = {
            line_number: 0,
            speaker: 'Timeline Export',
            text: this.currentTimelineProject?.project_name || 'Timeline export',
            versions: [
                {
                    version: 0,
                    audio_url: audioUrl,
                    audio_filename: this.currentTimelineExportFilename || `timeline_${this.currentTimelineProjectId}_exported.wav`,
                    quality_score: 0.8,
                    duration: this.currentTimelineProject?.total_duration || 0,
                    file_size: 0,
                },
            ],
        };

        await mediaPlayer.loadLine(lineData, { autoplay: true });
        modal.classList.add('show');
    } catch (error) {
        console.error('Failed to play timeline export:', error);
        this.showNotification('Error', 'Failed to play timeline export', 'error');
    }
};

IndexTTSApp.prototype.downloadTimelineExport = async function() {
    if (!this.currentTimelineProjectId) {
        this.showNotification('Warning', 'Load a timeline project first', 'warning');
        return;
    }

    try {
        await this.ensureTimelineExportReady();

        const link = document.createElement('a');
        link.href = `${this.apiBaseUrl}/timeline/${this.currentTimelineProjectId}/export/download`;
        link.download = this.currentTimelineExportFilename || `timeline_${this.currentTimelineProjectId}_exported.wav`;
        link.click();
    } catch (error) {
        console.error('Failed to download timeline export:', error);
        this.showNotification('Error', 'Failed to download timeline export', 'error');
    }
};

IndexTTSApp.prototype.beginTimelineSegmentDrag = function(event) {
    if (!this.currentTimelineProjectId) {
        return;
    }

    const element = event.currentTarget;
    const trackId = element.dataset.trackId;
    const segmentId = element.dataset.segmentId;
    const segment = this.findTimelineSegment(trackId, segmentId);

    if (!segment) {
        return;
    }

    event.preventDefault();

    this.timelineDragState = {
        trackId,
        segmentId,
        startClientX: event.clientX,
        originalStartTime: safeTimelineNumber(segment.start_time, 0),
        duration: safeTimelineNumber(segment.duration, 2.0),
        element,
        moved: false,
        previewStartTime: safeTimelineNumber(segment.start_time, 0),
    };

    element.classList.add('dragging');

    if (!this.boundTimelinePointerMove) {
        this.boundTimelinePointerMove = (moveEvent) => this.handleTimelineSegmentDragMove(moveEvent);
    }
    if (!this.boundTimelinePointerUp) {
        this.boundTimelinePointerUp = () => this.handleTimelineSegmentDragEnd();
    }

    document.addEventListener('pointermove', this.boundTimelinePointerMove);
    document.addEventListener('pointerup', this.boundTimelinePointerUp);
};

IndexTTSApp.prototype.handleTimelineSegmentDragMove = function(event) {
    const state = this.timelineDragState;
    if (!state) {
        return;
    }

    const deltaSeconds = (event.clientX - state.startClientX) / this.timelinePixelsPerSecond;
    const newStartTime = this.snapTimelineValue(state.originalStartTime + deltaSeconds);
    state.previewStartTime = newStartTime;
    state.moved = Math.abs(event.clientX - state.startClientX) > 3;

    state.element.style.left = `${newStartTime * this.timelinePixelsPerSecond}px`;
    const meta = state.element.querySelector('.timeline-segment-meta');
    if (meta) {
        meta.textContent = `${newStartTime.toFixed(1)}s • ${state.duration.toFixed(1)}s`;
    }
};

IndexTTSApp.prototype.handleTimelineSegmentDragEnd = async function() {
    const state = this.timelineDragState;
    if (!state) {
        return;
    }

    document.removeEventListener('pointermove', this.boundTimelinePointerMove);
    document.removeEventListener('pointerup', this.boundTimelinePointerUp);
    state.element.classList.remove('dragging');
    this.timelineDragState = null;

    if (!state.moved) {
        return;
    }

    this.suppressTimelineSegmentClick = true;

    try {
        await this.apiRequest(`/timeline/${this.currentTimelineProjectId}/tracks/${state.trackId}/segments/${state.segmentId}`, {
            method: 'PUT',
            body: JSON.stringify({
                start_time: state.previewStartTime,
                duration: state.duration,
            }),
        });

        this.selectedTimelineTrackId = state.trackId;
        this.selectedTimelineSegmentId = state.segmentId;
        await this.loadTimelineProject(this.currentTimelineProjectId);
        this.showNotification('Success', `Moved segment to ${state.previewStartTime.toFixed(1)}s`, 'success');
    } catch (error) {
        console.error('Failed to move timeline segment:', error);
        await this.loadTimelineProject(this.currentTimelineProjectId);
        this.showNotification('Error', error.message, 'error');
    }
};
