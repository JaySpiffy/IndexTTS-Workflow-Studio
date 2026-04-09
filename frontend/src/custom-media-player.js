/**
 * Custom Media Player for IndexTTS2 Conversation Workflow
 * Features: Waveform visualization, version comparison, advanced controls
 */

class CustomMediaPlayer {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container with id '${containerId}' not found`);
        }

        this.options = {
            waveColor: options.waveColor || '#4299e1',
            progressColor: options.progressColor || '#2b6cb0',
            cursorColor: options.cursorColor || '#1a365d',
            barWidth: options.barWidth || 2,
            barRadius: options.barRadius || 2,
            responsive: options.responsive !== false,
            height: options.height || 100,
            normalize: options.normalize !== false,
            backend: options.backend || 'WebAudio',
            ...options
        };

        this.currentLine = null;
        this.currentVersion = null;
        this.versions = [];
        this.wavesurfer = null;
        this.comparisonWavesurfer = null;
        this.isComparisonMode = false;
        this.isPlaying = false;
        this.isLooping = false;
        this.playbackSpeed = 1.0;
        this.volume = 1.0;
        this.isTrimMode = false;
        this.trimRegion = null;
        this.isInitialized = false;
        this.pendingAutoplay = false;

        this.ready = this.init();
    }

    async init() {
        try {
            // Load WaveSurfer.js if not already loaded
            await this.loadWaveSurfer();
            
            // Load HTML structure
            this.loadHTMLStructure();
            
            // Initialize UI elements
            this.initializeUI();
            
            // Set up event listeners
            this.setupEventListeners();
            
            // Initialize WaveSurfer
            this.initializeWaveSurfer();
            this.isInitialized = true;
            
            console.log('Custom Media Player initialized successfully');
            return this;
        } catch (error) {
            console.error('Failed to initialize Custom Media Player:', error);
            this.showError('Failed to initialize media player: ' + error.message);
            throw error;
        }
    }

    async ensureReady() {
        if (this.ready) {
            await this.ready;
        }

        if (!this.isInitialized || !this.wavesurfer) {
            throw new Error('Custom media player is not ready yet');
        }

        return this;
    }

    loadHTMLStructure() {
        // Load the HTML structure from the template
        const htmlStructure = `
            <!-- Version Selector -->
            <div class="version-selector">
                <label>Version:</label>
                <select id="versionSelect" class="version-dropdown">
                    <!-- Versions will be populated dynamically -->
                </select>
                <button id="compareBtn" class="compare-btn" disabled>Compare</button>
            </div>

            <!-- Main Player Area -->
            <div class="player-main">
                <!-- Waveform Container -->
                <div class="waveform-container">
                    <div id="waveform" class="waveform"></div>
                    <div id="waveformComparison" class="waveform-comparison hidden">
                        <div class="comparison-track">
                            <div class="track-label">Original</div>
                            <div id="waveformOriginal" class="waveform"></div>
                        </div>
                        <div class="comparison-track">
                            <div class="track-label">Comparison</div>
                            <div id="waveformCompare" class="waveform"></div>
                        </div>
                    </div>
                </div>

                <!-- Audio Info -->
                <div class="audio-info">
                    <div class="audio-title">
                        <span id="audioTitle">Line X - Speaker Name</span>
                        <span id="audioQuality" class="quality-badge">Quality: <span id="qualityScore">--</span></span>
                    </div>
                    <div class="audio-meta">
                        <span id="audioDuration">0:00</span>
                        <span id="audioSize">-- MB</span>
                    </div>
                </div>

                <!-- Playback Controls -->
                <div class="playback-controls">
                    <div class="primary-controls">
                        <button id="playPauseBtn" class="control-btn play-pause-btn">
                            <i class="fas fa-play"></i>
                        </button>
                        <button id="stopBtn" class="control-btn">
                            <i class="fas fa-stop"></i>
                        </button>
                        <button id="rewindBtn" class="control-btn">
                            <i class="fas fa-backward"></i>
                        </button>
                        <button id="forwardBtn" class="control-btn">
                            <i class="fas fa-forward"></i>
                        </button>
                    </div>

                    <div class="progress-container">
                        <span id="currentTime" class="time-display">0:00</span>
                        <div class="progress-bar">
                            <div id="progressFill" class="progress-fill"></div>
                            <div id="progressHandle" class="progress-handle"></div>
                        </div>
                        <span id="totalTime" class="time-display">0:00</span>
                    </div>

                    <div class="secondary-controls">
                        <button id="loopBtn" class="control-btn">
                            <i class="fas fa-redo"></i>
                        </button>
                        <button id="muteBtn" class="control-btn">
                            <i class="fas fa-volume-up"></i>
                        </button>
                        <div class="volume-control">
                            <input type="range" id="volumeSlider" class="volume-slider" min="0" max="100" value="100">
                        </div>
                    </div>
                </div>

                <!-- Advanced Controls -->
                <div class="advanced-controls">
                    <div class="control-group">
                        <label>Speed:</label>
                        <select id="playbackSpeed" class="speed-select">
                            <option value="0.5">0.5x</option>
                            <option value="0.75">0.75x</option>
                            <option value="1" selected>1x</option>
                            <option value="1.25">1.25x</option>
                            <option value="1.5">1.5x</option>
                            <option value="2">2x</option>
                        </select>
                    </div>

                    <div class="control-group">
                        <label>Zoom:</label>
                        <select id="zoomLevel" class="zoom-select">
                            <option value="10">10px/s</option>
                            <option value="25">25px/s</option>
                            <option value="50" selected>50px/s</option>
                            <option value="100">100px/s</option>
                            <option value="200">200px/s</option>
                        </select>
                    </div>

                    <div class="control-group">
                        <button id="trimBtn" class="control-btn">
                            <i class="fas fa-cut"></i> Trim
                        </button>
                        <button id="downloadBtn" class="control-btn">
                            <i class="fas fa-download"></i> Download
                        </button>
                        <button id="shareBtn" class="control-btn">
                            <i class="fas fa-share"></i> Share
                        </button>
                    </div>
                </div>
            </div>

            <!-- Version Comparison Panel -->
            <div id="comparisonPanel" class="comparison-panel hidden">
                <div class="comparison-header">
                    <h3>Version Comparison</h3>
                    <button id="closeComparison" class="close-btn">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="comparison-content">
                    <div class="comparison-stats">
                        <div class="stat-item">
                            <span class="stat-label">Duration Diff:</span>
                            <span id="durationDiff" class="stat-value">--</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Quality Diff:</span>
                            <span id="qualityDiff" class="stat-value">--</span>
                        </div>
                    </div>
                    <div class="comparison-actions">
                        <button id="useOriginalBtn" class="action-btn">Use Original</button>
                        <button id="useComparisonBtn" class="action-btn">Use Comparison</button>
                    </div>
                </div>
            </div>
        `;
        
        this.container.innerHTML = htmlStructure;
    }

    async loadWaveSurfer() {
        if (window.WaveSurfer) {
            return; // Already loaded
        }

        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://unpkg.com/wavesurfer.js@7';
            script.onload = resolve;
            script.onerror = () => reject(new Error('Failed to load WaveSurfer.js'));
            document.head.appendChild(script);
        });
    }

    initializeUI() {
        // Get all UI elements
        this.elements = {
            versionSelect: document.getElementById('versionSelect'),
            compareBtn: document.getElementById('compareBtn'),
            waveform: document.getElementById('waveform'),
            waveformComparison: document.getElementById('waveformComparison'),
            waveformOriginal: document.getElementById('waveformOriginal'),
            waveformCompare: document.getElementById('waveformCompare'),
            audioTitle: document.getElementById('audioTitle'),
            qualityScore: document.getElementById('qualityScore'),
            audioDuration: document.getElementById('audioDuration'),
            audioSize: document.getElementById('audioSize'),
            playPauseBtn: document.getElementById('playPauseBtn'),
            stopBtn: document.getElementById('stopBtn'),
            rewindBtn: document.getElementById('rewindBtn'),
            forwardBtn: document.getElementById('forwardBtn'),
            currentTime: document.getElementById('currentTime'),
            totalTime: document.getElementById('totalTime'),
            progressFill: document.getElementById('progressFill'),
            progressHandle: document.getElementById('progressHandle'),
            progressBar: this.container.querySelector('.progress-bar'),
            loopBtn: document.getElementById('loopBtn'),
            muteBtn: document.getElementById('muteBtn'),
            volumeSlider: document.getElementById('volumeSlider'),
            playbackSpeed: document.getElementById('playbackSpeed'),
            zoomLevel: document.getElementById('zoomLevel'),
            trimBtn: document.getElementById('trimBtn'),
            downloadBtn: document.getElementById('downloadBtn'),
            shareBtn: document.getElementById('shareBtn'),
            comparisonPanel: document.getElementById('comparisonPanel'),
            closeComparison: document.getElementById('closeComparison'),
            durationDiff: document.getElementById('durationDiff'),
            qualityDiff: document.getElementById('qualityDiff'),
            useOriginalBtn: document.getElementById('useOriginalBtn'),
            useComparisonBtn: document.getElementById('useComparisonBtn')
        };

        // Set initial states
        this.updatePlayPauseButton(false);
        this.updateVolumeButton();
        this.updateLoopButton();
    }

    setupEventListeners() {
        // Version selection
        this.elements.versionSelect.addEventListener('change', (e) => {
            this.selectVersion(parseInt(e.target.value));
        });

        // Comparison button
        this.elements.compareBtn.addEventListener('click', () => {
            this.toggleComparisonMode();
        });

        // Playback controls
        this.elements.playPauseBtn.addEventListener('click', () => {
            this.togglePlayPause();
        });

        this.elements.stopBtn.addEventListener('click', () => {
            this.stop();
        });

        this.elements.rewindBtn.addEventListener('click', () => {
            this.skip(-10); // Skip back 10 seconds
        });

        this.elements.forwardBtn.addEventListener('click', () => {
            this.skip(10); // Skip forward 10 seconds
        });

        // Progress bar
        this.elements.progressBar.addEventListener('click', (e) => {
            this.seek(e);
        });

        // Volume control
        this.elements.volumeSlider.addEventListener('input', (e) => {
            this.setVolume(parseInt(e.target.value) / 100);
        });

        this.elements.muteBtn.addEventListener('click', () => {
            this.toggleMute();
        });

        // Loop control
        this.elements.loopBtn.addEventListener('click', () => {
            this.toggleLoop();
        });

        // Playback speed
        this.elements.playbackSpeed.addEventListener('change', (e) => {
            this.setPlaybackSpeed(parseFloat(e.target.value));
        });

        // Zoom level
        this.elements.zoomLevel.addEventListener('change', (e) => {
            this.setZoomLevel(parseInt(e.target.value));
        });

        // Trim button
        this.elements.trimBtn.addEventListener('click', () => {
            this.toggleTrimMode();
        });

        // Download button
        this.elements.downloadBtn.addEventListener('click', () => {
            this.downloadCurrentAudio();
        });

        // Share button
        this.elements.shareBtn.addEventListener('click', () => {
            this.shareCurrentAudio();
        });

        // Comparison panel
        this.elements.closeComparison.addEventListener('click', () => {
            this.closeComparisonPanel();
        });

        this.elements.useOriginalBtn.addEventListener('click', () => {
            this.useOriginalVersion();
        });

        this.elements.useComparisonBtn.addEventListener('click', () => {
            this.useComparisonVersion();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return; // Ignore keyboard events when typing
            }

            switch (e.code) {
                case 'Space':
                    e.preventDefault();
                    this.togglePlayPause();
                    break;
                case 'ArrowLeft':
                    this.skip(-5);
                    break;
                case 'ArrowRight':
                    this.skip(5);
                    break;
                case 'ArrowUp':
                    this.adjustVolume(0.1);
                    break;
                case 'ArrowDown':
                    this.adjustVolume(-0.1);
                    break;
                case 'KeyM':
                    this.toggleMute();
                    break;
                case 'KeyL':
                    this.toggleLoop();
                    break;
            }
        });
    }

    initializeWaveSurfer() {
        this.wavesurfer = WaveSurfer.create({
            container: this.elements.waveform,
            waveColor: this.options.waveColor,
            progressColor: this.options.progressColor,
            cursorColor: this.options.cursorColor,
            barWidth: this.options.barWidth,
            barRadius: this.options.barRadius,
            responsive: this.options.responsive,
            height: this.options.height,
            normalize: this.options.normalize,
            backend: this.options.backend,
            interact: true
        });

        // WaveSurfer event listeners
        this.wavesurfer.on('ready', () => {
            this.onAudioReady();
        });

        this.wavesurfer.on('play', () => {
            this.onPlay();
        });

        this.wavesurfer.on('pause', () => {
            this.onPause();
        });

        this.wavesurfer.on('finish', () => {
            this.onFinish();
        });

        this.wavesurfer.on('audioprocess', () => {
            this.updateProgress();
        });

        this.wavesurfer.on('seeking', () => {
            this.updateProgress();
        });
    }

    // Public methods
    async loadLine(lineData, options = {}) {
        await this.ensureReady();

        const { autoplay = false } = options;
        this.currentLine = lineData;
        this.versions = lineData.versions || [];
        this.currentVersion = this.versions.length > 0 ? this.versions[0] : null;
        this.pendingAutoplay = autoplay;

        // Update UI
        this.updateVersionSelector();
        this.updateAudioInfo();

        if (this.currentVersion) {
            this.loadAudio(this.currentVersion.audio_url);
            this.elements.compareBtn.disabled = this.versions.length < 2;
        } else {
            this.clearWaveform();
            this.elements.compareBtn.disabled = true;
        }
    }

    loadAudio(audioUrl) {
        if (!audioUrl) {
            this.clearWaveform();
            return;
        }

        if (!this.wavesurfer) {
            throw new Error('Audio waveform is not ready yet');
        }

        this.showLoading(true);
        
        // Add cache-busting parameter
        const url = audioUrl.includes('?') ? `${audioUrl}&_t=${Date.now()}` : `${audioUrl}?_t=${Date.now()}`;
        
        this.wavesurfer.load(url)
            .catch(error => {
                console.error('Failed to load audio:', error);
                this.showError('Failed to load audio: ' + error.message);
                this.showLoading(false);
            });
    }

    selectVersion(versionIndex) {
        if (versionIndex < 0 || versionIndex >= this.versions.length) {
            return;
        }

        this.currentVersion = this.versions[versionIndex];
        this.updateAudioInfo();
        this.loadAudio(this.currentVersion.audio_url);

        // Exit comparison mode if active
        if (this.isComparisonMode) {
            this.closeComparisonPanel();
        }
    }

    togglePlayPause() {
        if (!this.wavesurfer) return;

        if (this.isPlaying) {
            this.wavesurfer.pause();
        } else {
            this.wavesurfer.play();
        }
    }

    stop() {
        if (!this.wavesurfer) return;

        this.wavesurfer.stop();
        this.isPlaying = false;
        this.updatePlayPauseButton(false);
    }

    skip(seconds) {
        if (!this.wavesurfer) return;

        const currentTime = this.wavesurfer.getCurrentTime();
        const newTime = Math.max(0, Math.min(currentTime + seconds, this.wavesurfer.getDuration()));
        this.wavesurfer.seekTo(newTime / this.wavesurfer.getDuration());
    }

    seek(event) {
        if (!this.wavesurfer) return;

        const rect = this.elements.progressBar.getBoundingClientRect();
        const percent = (event.clientX - rect.left) / rect.width;
        this.wavesurfer.seekTo(percent);
    }

    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume));
        this.elements.volumeSlider.value = this.volume * 100;
        
        if (this.wavesurfer) {
            this.wavesurfer.setVolume(this.volume);
        }

        this.updateVolumeButton();
    }

    toggleMute() {
        if (this.volume > 0) {
            this.previousVolume = this.volume;
            this.setVolume(0);
        } else {
            this.setVolume(this.previousVolume || 1.0);
        }
    }

    toggleLoop() {
        this.isLooping = !this.isLooping;
        this.updateLoopButton();
    }

    setPlaybackSpeed(speed) {
        this.playbackSpeed = speed;
        this.elements.playbackSpeed.value = speed;
        
        if (this.wavesurfer) {
            this.wavesurfer.setPlaybackRate(speed);
        }
    }

    setZoomLevel(zoomLevel) {
        this.options.minPxPerSec = zoomLevel;
        
        if (this.wavesurfer) {
            this.wavesurfer.zoom(zoomLevel);
        }
    }

    downloadCurrentAudio() {
        if (!this.currentVersion || !this.currentVersion.audio_url) {
            this.showError('No audio available for download');
            return;
        }

        const link = document.createElement('a');
        link.href = this.currentVersion.audio_url;
        link.download = `${this.currentLine.text.substring(0, 20)}_v${this.currentVersion.version}.wav`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    shareCurrentAudio() {
        if (!this.currentVersion || !this.currentVersion.audio_url) {
            this.showError('No audio available for sharing');
            return;
        }

        if (navigator.share) {
            navigator.share({
                title: 'IndexTTS2 Audio',
                text: this.currentLine.text,
                url: this.currentVersion.audio_url
            }).catch(error => {
                console.log('Share cancelled or failed:', error);
            });
        } else {
            // Fallback: copy to clipboard
            navigator.clipboard.writeText(this.currentVersion.audio_url)
                .then(() => {
                    this.showSuccess('Audio URL copied to clipboard');
                })
                .catch(error => {
                    console.error('Failed to copy URL:', error);
                    this.showError('Failed to copy URL to clipboard');
                });
        }
    }

    toggleComparisonMode() {
        if (this.versions.length < 2) {
            this.showError('Need at least 2 versions to compare');
            return;
        }

        if (this.isComparisonMode) {
            this.closeComparisonPanel();
        } else {
            this.openComparisonPanel();
        }
    }

    async openComparisonPanel() {
        this.isComparisonMode = true;
        this.elements.comparisonPanel.classList.remove('hidden');
        this.elements.waveformComparison.classList.remove('hidden');
        
        // Initialize comparison waveform if not already done
        if (!this.comparisonWavesurfer) {
            this.comparisonWavesurfer = WaveSurfer.create({
                container: this.elements.waveformCompare,
                waveColor: '#e53e3e',
                progressColor: '#c53030',
                cursorColor: this.options.cursorColor,
                barWidth: this.options.barWidth,
                barRadius: this.options.barRadius,
                responsive: this.options.responsive,
                height: this.options.height,
                normalize: this.options.normalize,
                backend: this.options.backend,
                interact: false
            });
        }

        // Load comparison audio (next version)
        const currentVersionIndex = this.versions.findIndex(v => v === this.currentVersion);
        const nextVersionIndex = (currentVersionIndex + 1) % this.versions.length;
        const comparisonVersion = this.versions[nextVersionIndex];

        await this.comparisonWavesurfer.load(comparisonVersion.audio_url);
        
        // Update comparison stats
        this.updateComparisonStats(comparisonVersion);
    }

    closeComparisonPanel() {
        this.isComparisonMode = false;
        this.elements.comparisonPanel.classList.add('hidden');
        this.elements.waveformComparison.classList.add('hidden');
    }

    useOriginalVersion() {
        this.closeComparisonPanel();
        this.showSuccess('Using original version');
    }

    useComparisonVersion() {
        const currentVersionIndex = this.versions.findIndex(v => v === this.currentVersion);
        const nextVersionIndex = (currentVersionIndex + 1) % this.versions.length;
        
        this.selectVersion(nextVersionIndex);
        this.closeComparisonPanel();
        this.showSuccess('Switched to comparison version');
    }

    // Private methods
    updateVersionSelector() {
        this.elements.versionSelect.innerHTML = '';
        
        this.versions.forEach((version, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `Version ${version.version + 1}`;
            if (version === this.currentVersion) {
                option.selected = true;
            }
            this.elements.versionSelect.appendChild(option);
        });
    }

    updateAudioInfo() {
        if (!this.currentVersion) {
            this.elements.audioTitle.textContent = 'No audio loaded';
            this.elements.qualityScore.textContent = '--';
            this.elements.audioDuration.textContent = '0:00';
            this.elements.audioSize.textContent = '-- MB';
            return;
        }

        this.elements.audioTitle.textContent = 
            `Line ${this.currentLine.line_number} - ${this.currentLine.speaker}`;
        
        const quality = this.currentVersion.quality_score || 0;
        this.elements.qualityScore.textContent = quality.toFixed(1);
        
        const duration = this.currentVersion.duration || 0;
        this.elements.audioDuration.textContent = this.formatTime(duration);
        
        const size = this.currentVersion.file_size || 0;
        this.elements.audioSize.textContent = this.formatFileSize(size);
    }

    updatePlayPauseButton(isPlaying) {
        this.isPlaying = isPlaying;
        const icon = this.elements.playPauseBtn.querySelector('i');
        
        if (isPlaying) {
            icon.className = 'fas fa-pause';
        } else {
            icon.className = 'fas fa-play';
        }
    }

    updateVolumeButton() {
        const icon = this.elements.muteBtn.querySelector('i');
        
        if (this.volume === 0) {
            icon.className = 'fas fa-volume-mute';
        } else if (this.volume < 0.5) {
            icon.className = 'fas fa-volume-down';
        } else {
            icon.className = 'fas fa-volume-up';
        }
    }

    updateLoopButton() {
        this.elements.loopBtn.style.color = this.isLooping ? 
            'var(--primary-color)' : 'var(--text-color)';
    }

    updateProgress() {
        if (!this.wavesurfer) return;

        const currentTime = this.wavesurfer.getCurrentTime();
        const duration = this.wavesurfer.getDuration();
        const percent = duration > 0 ? (currentTime / duration) * 100 : 0;

        this.elements.currentTime.textContent = this.formatTime(currentTime);
        this.elements.totalTime.textContent = this.formatTime(duration);
        this.elements.progressFill.style.width = `${percent}%`;
        this.elements.progressHandle.style.left = `${percent}%`;
    }

    updateComparisonStats(comparisonVersion) {
        if (!this.currentVersion || !comparisonVersion) return;

        const durationDiff = comparisonVersion.duration - this.currentVersion.duration;
        const qualityDiff = comparisonVersion.quality_score - this.currentVersion.quality_score;

        this.elements.durationDiff.textContent = 
            `${durationDiff >= 0 ? '+' : ''}${durationDiff.toFixed(1)}s`;
        
        this.elements.qualityDiff.textContent = 
            `${qualityDiff >= 0 ? '+' : ''}${qualityDiff.toFixed(1)}`;
        
        // Color code the differences
        this.elements.durationDiff.style.color = 
            Math.abs(durationDiff) < 0.5 ? 'var(--text-secondary)' : 
            durationDiff > 0 ? '#e53e3e' : '#38a169';
        
        this.elements.qualityDiff.style.color = 
            Math.abs(qualityDiff) < 0.5 ? 'var(--text-secondary)' : 
            qualityDiff > 0 ? '#38a169' : '#e53e3e';
    }

    clearWaveform() {
        if (this.wavesurfer) {
            this.wavesurfer.empty();
        }
        
        this.elements.currentTime.textContent = '0:00';
        this.elements.totalTime.textContent = '0:00';
        this.elements.progressFill.style.width = '0%';
        this.elements.progressHandle.style.left = '0%';
    }

    showLoading(show) {
        if (show) {
            this.elements.waveform.classList.add('loading');
        } else {
            this.elements.waveform.classList.remove('loading');
        }
    }

    showError(message) {
        // You can integrate this with your existing notification system
        console.error(message);
        alert(message); // Simple fallback
    }

    showSuccess(message) {
        // You can integrate this with your existing notification system
        console.log(message);
        // Could use a toast notification instead of alert
    }

    // Event handlers
    onAudioReady() {
        this.showLoading(false);
        this.updateProgress();
        
        if (this.wavesurfer) {
            this.wavesurfer.setVolume(this.volume);
            this.wavesurfer.setPlaybackRate(this.playbackSpeed);
        }

        if (this.pendingAutoplay && this.wavesurfer) {
            this.pendingAutoplay = false;
            Promise.resolve(this.wavesurfer.play()).catch(error => {
                console.error('Failed to autoplay audio:', error);
                this.showError('Failed to start audio playback: ' + error.message);
            });
        }
    }

    onPlay() {
        this.updatePlayPauseButton(true);
    }

    onPause() {
        this.updatePlayPauseButton(false);
    }

    onFinish() {
        this.updatePlayPauseButton(false);
        
        if (this.isLooping) {
            this.wavesurfer.play();
        }
    }

    // Utility methods
    formatTime(seconds) {
        if (isNaN(seconds) || seconds < 0) return '0:00';
        
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    formatFileSize(bytes) {
        if (isNaN(bytes) || bytes === 0) return '-- MB';
        
        const mb = bytes / (1024 * 1024);
        return `${mb.toFixed(1)} MB`;
    }

    adjustVolume(delta) {
        this.setVolume(this.volume + delta);
    }

    // Trim functionality
    toggleTrimMode() {
        if (!this.wavesurfer) return;
        
        this.isTrimMode = !this.isTrimMode;
        
        if (this.isTrimMode) {
            this.enableTrimMode();
        } else {
            this.disableTrimMode();
        }
    }

    enableTrimMode() {
        // Update button appearance
        this.elements.trimBtn.style.color = 'var(--primary-color)';
        this.elements.trimBtn.innerHTML = '<i class="fas fa-times"></i> Cancel Trim';
        
        // Create trim region if it doesn't exist
        if (!this.trimRegion) {
            this.trimRegion = {
                start: 0,
                end: this.wavesurfer.getDuration()
            };
        }
        
        // Add visual trim indicators to waveform
        this.addTrimIndicators();
        
        // Show trim controls
        this.showTrimControls();
    }

    disableTrimMode() {
        // Reset button appearance
        this.elements.trimBtn.style.color = 'var(--text-color)';
        this.elements.trimBtn.innerHTML = '<i class="fas fa-cut"></i> Trim';
        
        // Remove trim indicators
        this.removeTrimIndicators();
        
        // Hide trim controls
        this.hideTrimControls();
    }

    addTrimIndicators() {
        if (!this.wavesurfer) return;
        
        const duration = this.wavesurfer.getDuration();
        const container = this.wavesurfer.container;
        
        // Create start indicator
        const startIndicator = document.createElement('div');
        startIndicator.className = 'trim-indicator trim-start';
        startIndicator.style.left = `${(this.trimRegion.start / duration) * 100}%`;
        startIndicator.innerHTML = '<i class="fas fa-grip-lines-vertical"></i>';
        
        // Create end indicator
        const endIndicator = document.createElement('div');
        endIndicator.className = 'trim-indicator trim-end';
        endIndicator.style.left = `${(this.trimRegion.end / duration) * 100}%`;
        endIndicator.innerHTML = '<i class="fas fa-grip-lines-vertical"></i>';
        
        // Create trim region overlay
        const trimOverlay = document.createElement('div');
        trimOverlay.className = 'trim-overlay';
        
        // Add to container
        container.appendChild(startIndicator);
        container.appendChild(endIndicator);
        container.appendChild(trimOverlay);
        
        // Store references
        this.trimStartIndicator = startIndicator;
        this.trimEndIndicator = endIndicator;
        this.trimOverlay = trimOverlay;
        
        // Make indicators draggable
        this.makeTrimIndicatorDraggable(startIndicator, 'start');
        this.makeTrimIndicatorDraggable(endIndicator, 'end');
        
        // Update overlay
        this.updateTrimOverlay();
    }

    removeTrimIndicators() {
        if (this.trimStartIndicator) {
            this.trimStartIndicator.remove();
            this.trimStartIndicator = null;
        }
        
        if (this.trimEndIndicator) {
            this.trimEndIndicator.remove();
            this.trimEndIndicator = null;
        }
        
        if (this.trimOverlay) {
            this.trimOverlay.remove();
            this.trimOverlay = null;
        }
    }

    makeTrimIndicatorDraggable(indicator, type) {
        let isDragging = false;
        
        const handleMouseDown = (e) => {
            isDragging = true;
            e.preventDefault();
        };
        
        const handleMouseMove = (e) => {
            if (!isDragging) return;
            
            const rect = this.wavesurfer.container.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const percentage = x / rect.width;
            const duration = this.wavesurfer.getDuration();
            const time = percentage * duration;
            
            // Update trim region
            if (type === 'start') {
                this.trimRegion.start = Math.max(0, Math.min(time, this.trimRegion.end - 0.1));
                indicator.style.left = `${(this.trimRegion.start / duration) * 100}%`;
            } else {
                this.trimRegion.end = Math.min(duration, Math.max(time, this.trimRegion.start + 0.1));
                indicator.style.left = `${(this.trimRegion.end / duration) * 100}%`;
            }
            
            this.updateTrimOverlay();
            this.updateTrimTimeDisplay();
        };
        
        const handleMouseUp = () => {
            isDragging = false;
        };
        
        indicator.addEventListener('mousedown', handleMouseDown);
        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
    }

    updateTrimOverlay() {
        if (!this.trimOverlay || !this.wavesurfer) return;
        
        const duration = this.wavesurfer.getDuration();
        const startPercent = (this.trimRegion.start / duration) * 100;
        const endPercent = (this.trimRegion.end / duration) * 100;
        const widthPercent = endPercent - startPercent;
        
        this.trimOverlay.style.left = `${startPercent}%`;
        this.trimOverlay.style.width = `${widthPercent}%`;
    }

    showTrimControls() {
        // Create trim controls panel
        const trimControls = document.createElement('div');
        trimControls.className = 'trim-controls';
        trimControls.innerHTML = `
            <div class="trim-time-display">
                <span>Start: <span id="trimStartTime">${this.formatTime(this.trimRegion.start)}</span></span>
                <span>End: <span id="trimEndTime">${this.formatTime(this.trimRegion.end)}</span></span>
                <span>Duration: <span id="trimDuration">${this.formatTime(this.trimRegion.end - this.trimRegion.start)}</span></span>
            </div>
            <div class="trim-actions">
                <button id="resetTrimBtn" class="btn btn-secondary btn-small">
                    <i class="fas fa-undo"></i> Reset
                </button>
                <button id="applyTrimBtn" class="btn btn-primary btn-small">
                    <i class="fas fa-check"></i> Apply Trim
                </button>
            </div>
        `;
        
        // Add to waveform container
        this.elements.waveform.appendChild(trimControls);
        this.trimControlsPanel = trimControls;
        
        // Add event listeners
        document.getElementById('resetTrimBtn').addEventListener('click', () => {
            this.resetTrimRegion();
        });
        
        document.getElementById('applyTrimBtn').addEventListener('click', () => {
            this.applyTrim();
        });
    }

    hideTrimControls() {
        if (this.trimControlsPanel) {
            this.trimControlsPanel.remove();
            this.trimControlsPanel = null;
        }
    }

    updateTrimTimeDisplay() {
        if (!this.trimControlsPanel) return;
        
        document.getElementById('trimStartTime').textContent = this.formatTime(this.trimRegion.start);
        document.getElementById('trimEndTime').textContent = this.formatTime(this.trimRegion.end);
        document.getElementById('trimDuration').textContent = this.formatTime(this.trimRegion.end - this.trimRegion.start);
    }

    resetTrimRegion() {
        if (!this.wavesurfer) return;
        
        this.trimRegion.start = 0;
        this.trimRegion.end = this.wavesurfer.getDuration();
        
        // Update indicators
        if (this.trimStartIndicator) {
            this.trimStartIndicator.style.left = '0%';
        }
        
        if (this.trimEndIndicator) {
            this.trimEndIndicator.style.left = '100%';
        }
        
        this.updateTrimOverlay();
        this.updateTrimTimeDisplay();
    }

    async applyTrim() {
        if (!this.currentVersion || !this.trimRegion) return;
        
        try {
            this.showLoading(true);
            
            // Call API to trim the audio
            const formData = new FormData();
            formData.append('audio_url', this.currentVersion.audio_url);
            formData.append('start_time', this.trimRegion.start);
            formData.append('end_time', this.trimRegion.end);
            formData.append('output_name', `trimmed_${this.currentVersion.audio_filename || 'audio.wav'}`);
            
            const response = await fetch(`${window.location.origin}/api/audio-process/trim`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            // Update current version with trimmed audio
            this.currentVersion.audio_url = result.trimmed_audio_url;
            this.currentVersion.audio_filename = result.output_filename;
            this.currentVersion.duration = this.trimRegion.end - this.trimRegion.start;
            
            // Reload the audio
            this.loadAudio(this.currentVersion.audio_url);
            
            // Exit trim mode
            this.toggleTrimMode();
            
            this.showSuccess('Audio trimmed successfully');
            
        } catch (error) {
            console.error('Failed to trim audio:', error);
            this.showError('Failed to trim audio: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }

    // Cleanup
    destroy() {
        if (this.wavesurfer) {
            this.wavesurfer.destroy();
        }
        
        if (this.comparisonWavesurfer) {
            this.comparisonWavesurfer.destroy();
        }
        
        // Clean up trim elements
        this.disableTrimMode();
        
        // Remove event listeners
        // (Implementation depends on your specific needs)
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CustomMediaPlayer;
} else if (typeof window !== 'undefined') {
    window.CustomMediaPlayer = CustomMediaPlayer;
}
