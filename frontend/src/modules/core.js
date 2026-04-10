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
        this.timelinePixelsPerSecond = 140;
        this.timelineDragState = null;
        this.currentTimelineExportFilename = null;
        this.timelineExportDirty = true;
        this.timelineWaveformCache = {};
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
    });
    document.querySelector(`[data-tab="${tabName}"]`)?.classList.add('active');

    // Update tab panes
    document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
    });
    document.getElementById(tabName).classList.add('active');

    this.currentTab = tabName;

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
