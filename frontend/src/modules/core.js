// IndexTTS2 Core Application Module
class IndexTTSApp {
    constructor() {
        // Use port 8000 for the API backend
        this.apiBaseUrl = '/api';
        this.currentConversationId = null;
        this.currentTab = 'conversation-workflow';
        this.speakers = [];
        this.sourceClips = [];
        this.conversations = [];
        this.savedProjects = [];
        this.generationInterval = null;
        this.regenerationInterval = null;
        this.healthStatusInterval = null;
        this.isDarkMode = localStorage.getItem('darkMode') === 'true';
        this.customMediaPlayer = null;
        this.parsedScript = null;
        this.conversationScript = [];
        this.currentConversationData = null;
        this.loadedScriptPack = null;
        this.listeningReviews = {};
        this.lastConcatenationPlanApplied = false;
        this.timelineProjects = [];
        this.currentTimelineProjectId = null;
        this.currentTimelineProject = null;
        this.selectedTimelineTrackId = null;
        this.selectedTimelineSegmentId = null;
        this.timelineDragState = null;
        this.currentTimelineExportFilename = null;
        this.timelineExportDirty = true;
        this.timelineWaveformCache = {};
        this.timelineTrackUiState = {};
        this.selectedSourceClip = null;
        this.currentSourceClipDiagnostics = null;
        this.sourceClipDiagnosticsCache = {};
        this.lastSpeakerPrepResult = null;
        this.pendingSpeakerPacingSettings = [];
        this.currentConversationDialoguePacingPreset = 'natural';
        this.routeParams = new URLSearchParams(window.location.search);
        this.pendingTimelineRouteProjectId = null;
        this.webMcp = null;
        this.webMcpReady = false;
        this.webMcpInitStarted = false;
        this.sectionCollapseState = {};
        this.timelinePixelsPerSecond = 140;

        try {
            const savedSectionCollapseState = JSON.parse(localStorage.getItem('indexttsSectionCollapseState') || '{}');
            if (savedSectionCollapseState && typeof savedSectionCollapseState === 'object') {
                this.sectionCollapseState = savedSectionCollapseState;
            }
        } catch (error) {
            console.warn('Failed to restore section collapse state:', error);
        }

        try {
            const savedTimelineTrackUiState = JSON.parse(localStorage.getItem('indexttsTimelineTrackUiState') || '{}');
            if (savedTimelineTrackUiState && typeof savedTimelineTrackUiState === 'object') {
                this.timelineTrackUiState = savedTimelineTrackUiState;
            }
        } catch (error) {
            console.warn('Failed to restore timeline track UI state:', error);
        }

        try {
            const savedTimelineZoom = Number(localStorage.getItem('indexttsTimelineZoom'));
            if (Number.isFinite(savedTimelineZoom)) {
                this.timelinePixelsPerSecond = Math.min(260, Math.max(60, savedTimelineZoom));
            }
        } catch (error) {
            console.warn('Failed to restore timeline zoom:', error);
        }
        
        this.init();
    }

    init() {
        try {
            this.setupEventListeners();
            this.applyInitialRouteState();
            this.switchTab(this.currentTab);
            this.checkApiStatus();
            this.healthStatusInterval = setInterval(() => this.checkApiStatus(), 15000);
            this.applyGenerationPreset('balanced');
            if (typeof this.updateDialoguePacingPresetHelp === 'function') {
                this.updateDialoguePacingPresetHelp('natural');
            }
            if (typeof this.renderSpeakerPacingControls === 'function') {
                this.renderSpeakerPacingControls([]);
            }
            if (typeof this.loadListeningReviewsState === 'function') {
                this.listeningReviews = this.loadListeningReviewsState();
            }
            this.loadSpeakers();
            if (typeof this.loadSourceClips === 'function') {
                this.loadSourceClips();
            }
            this.loadConversations();
            this.refreshSavedProjects();
            this.setupDarkMode();
            if (typeof this.setupWebMcp === 'function') {
                this.setupWebMcp();
            }
        } catch (error) {
            console.error('IndexTTSApp init() error:', error);
        }
    }
}

IndexTTSApp.prototype.persistSectionCollapseState = function() {
    try {
        localStorage.setItem('indexttsSectionCollapseState', JSON.stringify(this.sectionCollapseState || {}));
    } catch (error) {
        console.warn('Failed to persist section collapse state:', error);
    }
};

IndexTTSApp.prototype.persistTimelineTrackUiState = function() {
    try {
        localStorage.setItem('indexttsTimelineTrackUiState', JSON.stringify(this.timelineTrackUiState || {}));
    } catch (error) {
        console.warn('Failed to persist timeline track UI state:', error);
    }
};

IndexTTSApp.prototype.getCollapsibleSectionKey = function(section) {
    if (!section) {
        return '';
    }

    if (section.dataset.sectionKey) {
        return section.dataset.sectionKey;
    }

    const tabId = section.closest('.tab-pane')?.id || 'workspace';
    const headingText = section.querySelector('.section-header h3')?.textContent?.trim() || 'section';
    const normalizedHeading = headingText.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
    return `${tabId}:${normalizedHeading || 'section'}`;
};

IndexTTSApp.prototype.setSectionCollapsed = function(section, collapsed, options = {}) {
    const header = section?.querySelector(':scope > .section-header, :scope > .studio-panel-header');
    const content = section?.querySelector(':scope > .section-content, :scope > .collapsible-content, :scope > .studio-panel-body');
    const toggleButton = section?.querySelector('.section-collapse-toggle');
    if (!section || !header || !content) {
        return;
    }

    const normalizedCollapsed = Boolean(collapsed);
    section.classList.toggle('section-collapsed', normalizedCollapsed);
    section.classList.toggle('expanded', !normalizedCollapsed);
    content.hidden = normalizedCollapsed;
    content.style.display = normalizedCollapsed ? 'none' : '';
    content.classList.toggle('show', !normalizedCollapsed);
    if (toggleButton) {
        toggleButton.setAttribute('aria-expanded', String(!normalizedCollapsed));
        toggleButton.setAttribute('title', normalizedCollapsed ? 'Expand section' : 'Collapse section');
    }

    if (!options.skipPersist) {
        this.sectionCollapseState = this.sectionCollapseState || {};
        this.sectionCollapseState[this.getCollapsibleSectionKey(section)] = normalizedCollapsed;
        this.persistSectionCollapseState();
    }
};

IndexTTSApp.prototype.toggleSectionCollapsed = function(section) {
    if (!section) {
        return;
    }

    const isCollapsed = section.classList.contains('section-collapsed');
    this.setSectionCollapsed(section, !isCollapsed);
};

IndexTTSApp.prototype.setupCollapsibleSections = function() {
    document.querySelectorAll('.studio-panel, .workflow-section, .results-section').forEach((section) => {
        if (section.dataset.collapsibleReady === 'true') {
            return;
        }

        const header = section.querySelector(':scope > .section-header, :scope > .studio-panel-header');
        const content = section.querySelector(':scope > .section-content, :scope > .collapsible-content, :scope > .studio-panel-body');
        if (!header || !content) {
            return;
        }

        section.dataset.collapsibleReady = 'true';
        section.classList.add('is-collapsible-section');

        let actions = header.querySelector(':scope > .section-actions');
        if (!actions) {
            actions = document.createElement('div');
            actions.className = 'section-actions section-actions-collapse';
            header.appendChild(actions);
        }

        const toggleButton = document.createElement('button');
        toggleButton.type = 'button';
        toggleButton.className = 'btn btn-secondary btn-small section-collapse-toggle';
        toggleButton.setAttribute('aria-label', 'Toggle section');
        toggleButton.innerHTML = '<i class="fas fa-chevron-down"></i>';
        actions.appendChild(toggleButton);

        toggleButton.addEventListener('click', (event) => {
            event.preventDefault();
            event.stopPropagation();
            this.toggleSectionCollapsed(section);
        });

        const collapseKey = this.getCollapsibleSectionKey(section);
        const hasStoredState = Object.prototype.hasOwnProperty.call(this.sectionCollapseState || {}, collapseKey);
        const defaultCollapsed = section.dataset.defaultCollapsed === 'true' || content.style.display === 'none' || content.hidden;
        this.setSectionCollapsed(section, hasStoredState ? Boolean(this.sectionCollapseState[collapseKey]) : defaultCollapsed, { skipPersist: true });
    });
};

IndexTTSApp.prototype.setTimelineZoom = function(nextValue) {
    const clamped = Math.min(260, Math.max(60, Math.round(Number(nextValue) || 140)));
    if (clamped === this.timelinePixelsPerSecond) {
        return;
    }

    this.timelinePixelsPerSecond = clamped;
    try {
        localStorage.setItem('indexttsTimelineZoom', String(clamped));
    } catch (error) {
        console.warn('Failed to persist timeline zoom:', error);
    }

    if (this.currentTimelineProject && typeof this.renderTimelineProject === 'function') {
        this.renderTimelineProject();
    }
};

IndexTTSApp.prototype.adjustTimelineZoom = function(step) {
    this.setTimelineZoom(this.timelinePixelsPerSecond + step);
};

IndexTTSApp.prototype.fitTimelineZoomToProject = function() {
    if (!this.currentTimelineProject) {
        return;
    }

    const shell = document.getElementById('timeline-editor-shell');
    const totalDuration = typeof this.getTimelineVisualDuration === 'function' ? this.getTimelineVisualDuration() : 8;
    const availableWidth = Math.max(420, (shell?.clientWidth || 1080) - 280);
    const suggestedZoom = availableWidth / Math.max(totalDuration, 1);
    this.setTimelineZoom(suggestedZoom);
};

IndexTTSApp.prototype.getTimelineTrackUiKey = function(trackId) {
    return `${this.currentTimelineProjectId || 'timeline'}:${trackId || 'track'}`;
};

IndexTTSApp.prototype.isTimelineTrackCollapsed = function(trackId) {
    const key = this.getTimelineTrackUiKey(trackId);
    return Boolean(this.timelineTrackUiState?.[key]?.collapsed);
};

IndexTTSApp.prototype.toggleTimelineTrackCollapsed = function(trackId) {
    if (!trackId) {
        return;
    }

    const key = this.getTimelineTrackUiKey(trackId);
    const currentState = this.timelineTrackUiState?.[key] || {};
    this.timelineTrackUiState = this.timelineTrackUiState || {};
    this.timelineTrackUiState[key] = {
        ...currentState,
        collapsed: !Boolean(currentState.collapsed),
    };
    this.persistTimelineTrackUiState();
    if (typeof this.renderTimelineProject === 'function') {
        this.renderTimelineProject();
    }
};

IndexTTSApp.prototype.getWorkspaceMeta = function(tabName) {
    const metadata = {
        'speaker-prep': {
            eyebrow: 'Voice Library',
            title: 'Speaker Prep',
            subtitle: 'Prepare source clips and promote clean speaker files.',
        },
        'conversation-workflow': {
            eyebrow: 'Draft',
            title: 'Conversation Workflow',
            subtitle: 'Write, parse, generate, and keep the session in sync.',
        },
        'conversation-results': {
            eyebrow: 'Review',
            title: 'Conversation Results',
            subtitle: 'Compare takes, regenerate lines, and lock the best versions.',
        },
        'timeline-editor': {
            eyebrow: 'Timeline',
            title: 'Timeline Editor',
            subtitle: 'Set timing, overlap control, and export order.',
        },
    };

    return metadata[tabName] || metadata['conversation-workflow'];
};

IndexTTSApp.prototype.getStudioSessionLabel = function() {
    const conversationTitle = document.getElementById('conversation-title')?.value?.trim();
    const saveName = document.getElementById('project-save-name')?.value?.trim();

    if (conversationTitle) {
        return conversationTitle;
    }

    if (this.currentConversationId) {
        return `Conversation ${this.currentConversationId.substring(0, 8)}`;
    }

    if (saveName && saveName !== 'project.json') {
        return saveName;
    }

    if (document.getElementById('conversation-script')?.value?.trim()) {
        return 'Draft In Progress';
    }

    return 'New Session';
};

IndexTTSApp.prototype.refreshStudioShell = function() {
    const workspaceMeta = this.getWorkspaceMeta(this.currentTab);
    const voiceCount = Array.isArray(this.speakers) ? this.speakers.length : 0;
    const projectCount = Array.isArray(this.savedProjects) ? this.savedProjects.length : 0;
    const conversationCount = Array.isArray(this.conversations) ? this.conversations.length : 0;
    const pluralize = (count, singular, plural) => `${count} ${count === 1 ? singular : plural}`;

    const workspaceEyebrow = document.getElementById('workspace-eyebrow');
    const workspaceSessionPill = document.getElementById('workspace-session-pill');

    if (workspaceEyebrow) workspaceEyebrow.textContent = workspaceMeta.eyebrow;
    if (workspaceSessionPill) workspaceSessionPill.textContent = this.getStudioSessionLabel();

    const voicesStat = document.getElementById('workspace-stat-voices');
    const projectsStat = document.getElementById('workspace-stat-projects');
    const conversationsStat = document.getElementById('workspace-stat-conversations');
    const sidebarVoicesBadge = document.getElementById('sidebar-voices-badge');
    const sidebarProjectsBadge = document.getElementById('sidebar-projects-badge');

    if (voicesStat) voicesStat.textContent = String(voiceCount);
    if (projectsStat) projectsStat.textContent = String(projectCount);
    if (conversationsStat) conversationsStat.textContent = String(conversationCount);
    if (sidebarVoicesBadge) sidebarVoicesBadge.textContent = pluralize(voiceCount, 'Voice Ready', 'Voices Ready');
    if (sidebarProjectsBadge) sidebarProjectsBadge.textContent = pluralize(projectCount, 'Saved Project', 'Saved Projects');

    document.querySelectorAll('.studio-flow-item').forEach((item) => {
        item.classList.toggle('active', item.dataset.flowTab === this.currentTab);
    });

    document.body.dataset.currentTab = this.currentTab;
};

IndexTTSApp.prototype.applyInitialRouteState = function() {
    const tabFromUrl = this.routeParams.get('tab');
    if (tabFromUrl && ['conversation-workflow', 'speaker-prep', 'conversation-results', 'timeline-editor'].includes(tabFromUrl)) {
        this.currentTab = tabFromUrl;
    }

    if (this.routeParams.get('timelinePopout') === '1') {
        document.body.classList.add('timeline-popout-mode');
        this.currentTab = 'timeline-editor';
    }

    const timelineProjectId = this.routeParams.get('timelineProject');
    if (timelineProjectId) {
        this.pendingTimelineRouteProjectId = timelineProjectId;
    }
};

// Speaker Management
IndexTTSApp.prototype.loadSpeakers = async function() {
    try {
        const response = await this.apiRequest('/speakers/');
        this.speakers = response.speakers || [];
        this.populateSpeakerSelects();
        this.renderAvailableVoices();
        if (typeof this.renderTimelineSpeakerOptions === 'function') {
            this.renderTimelineSpeakerOptions();
        }
    } catch (error) {
        console.error('Failed to load speakers:', error);
        this.speakers = [];
        this.renderAvailableVoices();
        if (typeof this.renderTimelineSpeakerOptions === 'function') {
            this.renderTimelineSpeakerOptions();
        }
        this.showNotification('Error', 'Failed to load speakers', 'error');
    }
};

IndexTTSApp.prototype.populateSpeakerSelects = function() {
    // Populate speaker select in emotion reference section
    const referenceSelect = document.getElementById('emotion-reference-file');
    if (referenceSelect) {
        referenceSelect.innerHTML = '<option value="">Select reference audio...</option>';
        this.speakers.forEach(speaker => {
            const option = document.createElement('option');
            option.value = speaker.filename;
            option.textContent = speaker.name || speaker.filename.replace('.wav', '');
            referenceSelect.appendChild(option);
        });
    }
};

IndexTTSApp.prototype.formatSpeakerSize = function(speaker) {
    const sizeKb = Number(speaker?.size_kb);
    if (!Number.isFinite(sizeKb) || sizeKb <= 0) {
        return '';
    }

    if (sizeKb >= 1024) {
        return `${(sizeKb / 1024).toFixed(1)} MB`;
    }

    return `${Math.round(sizeKb)} KB`;
};

IndexTTSApp.prototype.renderAvailableVoices = function() {
    const voicesList = document.getElementById('available-voices-list');
    const voicesCount = document.getElementById('available-voices-count');

    if (!voicesList || !voicesCount) {
        return;
    }

    const voiceTotal = this.speakers.length;
    voicesCount.textContent = `${voiceTotal} voice${voiceTotal === 1 ? '' : 's'}`;
    voicesList.innerHTML = '';

      if (!voiceTotal) {
          voicesList.innerHTML = `
              <div class="empty-state">
                  <i class="fas fa-microphone-slash"></i>
                  <p>No voices loaded yet.</p>
                  <p class="empty-state-detail">This app does not ship with bundled voice clones. Add your own prepared WAV files to <code>shared/audio/speakers</code> or create them in <strong>Speaker Prep</strong>.</p>
              </div>
          `;
          this.refreshStudioShell();
          return;
      }

    this.speakers.forEach((speaker) => {
        const speakerItem = document.createElement('div');
        speakerItem.className = 'speaker-item voice-card';

        const speakerInfo = document.createElement('div');
        speakerInfo.className = 'speaker-info';

        const icon = document.createElement('i');
        icon.className = 'fas fa-wave-square';

        const speakerText = document.createElement('div');
        speakerText.className = 'voice-card-text';

        const speakerName = document.createElement('div');
        speakerName.className = 'speaker-name';
        speakerName.textContent = speaker.name || (speaker.filename || '').replace(/\.(wav|mp3)$/i, '');

        const speakerLabel = document.createElement('div');
        speakerLabel.className = 'voice-script-label';
        speakerLabel.textContent = `Use in script: ${speakerName.textContent}:`;

        const speakerMeta = document.createElement('div');
        speakerMeta.className = 'speaker-size';
        const speakerDetails = [speaker.filename].filter(Boolean);
        const formattedSize = this.formatSpeakerSize(speaker);
        if (formattedSize) {
            speakerDetails.push(formattedSize);
        }
        speakerMeta.textContent = speakerDetails.join(' | ');

        speakerText.appendChild(speakerName);
        speakerText.appendChild(speakerLabel);
        speakerText.appendChild(speakerMeta);
        speakerInfo.appendChild(icon);
        speakerInfo.appendChild(speakerText);
        speakerItem.appendChild(speakerInfo);
        voicesList.appendChild(speakerItem);
    });

    this.refreshStudioShell();
};

// Speaker Upload
IndexTTSApp.prototype.uploadSpeaker = async function() {
    const fileInput = document.getElementById('speaker-file-input');
    const file = fileInput.files[0];
    
    if (!file) {
        this.showNotification('Error', 'Please select a file', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch(`${this.apiBaseUrl}/speakers/upload`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || data.message || 'Upload failed');
        }
        
        this.showNotification('Success', 'Speaker uploaded successfully', 'success');
        this.loadSpeakers();
        fileInput.value = '';
        
    } catch (error) {
        console.error('Upload failed:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

// Tab Management
IndexTTSApp.prototype.switchTab = function(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
        button.setAttribute('aria-selected', 'false');
    });
    const activeButton = document.querySelector(`[data-tab="${tabName}"]`);
    activeButton?.classList.add('active');
    activeButton?.setAttribute('aria-selected', 'true');

    // Update tab panes
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });
    document.getElementById(tabName).classList.add('active');

    this.currentTab = tabName;
    this.refreshStudioShell();

    // Load data for specific tabs
    if (tabName === 'conversation-results') {
        this.loadConversations();
    } else if (tabName === 'speaker-prep' && typeof this.loadSourceClips === 'function') {
        this.loadSourceClips();
    } else if (tabName === 'timeline-editor' && typeof this.loadTimelineProjects === 'function') {
        this.loadTimelineProjects();
    }
};

// Missing methods that are called in init()
IndexTTSApp.prototype.loadConversations = async function() {
    try {
        // This will be implemented in conversationResults.js
        // For now, just initialize empty array
        this.conversations = [];
    } catch (error) {
        console.error('Failed to load conversations:', error);
    }
};

IndexTTSApp.prototype.setupDarkMode = function() {
    try {
        // Apply dark mode based on saved preference
        if (this.isDarkMode) {
            document.body.classList.add('dark-mode');
        }
    } catch (error) {
        console.error('Failed to setup dark mode:', error);
    }
};

// Placeholder for methods that should be defined in other modules
// These will be overridden by the actual implementations in the respective modules
IndexTTSApp.prototype.setupEventListeners = function() {
    // This will be implemented in eventListeners.js
};

IndexTTSApp.prototype.checkApiStatus = function() {
    // This will be implemented in api.js
};

IndexTTSApp.prototype.showNotification = function(title, message, type = 'info') {
    // This will be implemented in uiUtils.js
    // For now, just use console.log
    console.log(`[${type.toUpperCase()}] ${title}: ${message}`);
};

IndexTTSApp.prototype.apiRequest = function(endpoint, options = {}) {
    // This will be implemented in api.js
    // For now, return a rejected promise to show it's not implemented
    return Promise.reject(new Error('apiRequest not implemented yet'));
};

// Export the class for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = IndexTTSApp;
} else if (typeof window !== 'undefined') {
    window.IndexTTSApp = IndexTTSApp;
} else {
    console.error('Unable to export IndexTTSApp class - no module or window object available');
}
