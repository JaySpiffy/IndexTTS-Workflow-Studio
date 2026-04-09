// IndexTTS2 Event Listeners Module
IndexTTSApp.prototype.setupEventListeners = function() {
    console.log('DEBUG: setupEventListeners called');
    
    try {
        // Tab navigation
        const tabButtons = document.querySelectorAll('.tab-button');
        console.log(`DEBUG: Found ${tabButtons.length} tab buttons`);
        
        tabButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                const tabName = e.target.closest('.tab-button').dataset.tab;
                console.log(`DEBUG: Tab button clicked: ${tabName}`);
                this.switchTab(tabName);
            });
        });

        // Conversation Workflow
        this.setupConversationWorkflowEvents();

        // Speaker Prep
        if (typeof this.setupSpeakerPrepEvents === 'function') {
            this.setupSpeakerPrepEvents();
        }
        
        // Conversation Results
        this.setupConversationResultsEvents();

        // Timeline Editor
        if (typeof this.setupTimelineEditorEvents === 'function') {
            this.setupTimelineEditorEvents();
        }
        
        // Modal
        this.setupModalEvents();
        
        // Range inputs
        this.setupRangeInputs();
        
        // Dark mode toggle
        this.setupDarkModeToggle();
        
        console.log('DEBUG: setupEventListeners completed successfully');
    } catch (error) {
        console.error('DEBUG: setupEventListeners error:', error);
    }
};

IndexTTSApp.prototype.setupConversationWorkflowEvents = function() {
    console.log('DEBUG: setupConversationWorkflowEvents called');
    
    try {
        // Script parsing
        const parseScriptBtn = document.getElementById('parse-script-btn');
        if (parseScriptBtn) {
            parseScriptBtn.addEventListener('click', () => {
                console.log('DEBUG: Parse script button clicked');
                this.parseScript();
            });
            console.log('DEBUG: Parse script button event listener added');
        } else {
            console.warn('DEBUG: parse-script-btn not found');
        }
        
        const clearScriptBtn = document.getElementById('clear-script-btn');
        if (clearScriptBtn) {
            clearScriptBtn.addEventListener('click', () => {
                const scriptTextarea = document.getElementById('conversation-script');
                const scriptPreview = document.getElementById('script-preview');
                const timelineContainer = document.getElementById('emotion-timeline-container');
                const overlapPlanTextarea = document.getElementById('overlap-plan-text');
                const scriptPackFileInput = document.getElementById('conversation-script-pack-file');
                if (scriptTextarea) scriptTextarea.value = '';
                if (scriptPreview) scriptPreview.style.display = 'none';
                if (timelineContainer) timelineContainer.remove();
                if (overlapPlanTextarea) overlapPlanTextarea.value = '';
                if (scriptPackFileInput) scriptPackFileInput.value = '';
                this.parsedScript = null;
                this.conversationScript = [];
                this.loadedScriptPack = null;
                this.pendingSpeakerPacingSettings = [];
                if (typeof this.applyConversationMixPacingSettings === 'function' && typeof this.getDefaultGenerationSettings === 'function') {
                    this.applyConversationMixPacingSettings(this.getDefaultGenerationSettings());
                }
                if (typeof this.renderSpeakerPacingControls === 'function') {
                    this.renderSpeakerPacingControls([]);
                }
                if (typeof this.setScriptPackStatus === 'function') {
                    this.setScriptPackStatus('Pick a markdown script pack from the repo to load its title, pasteable script, and timing/emotion plan into the page.', 'info');
                }
            });
            console.log('DEBUG: Clear script button event listener added');
        } else {
            console.warn('DEBUG: clear-script-btn not found');
        }

        const scriptPackFile = document.getElementById('conversation-script-pack-file');
        if (scriptPackFile) {
            scriptPackFile.addEventListener('change', (event) => {
                this.loadConversationScriptPackFile(event);
            });
        }

        const saveProjectBtn = document.getElementById('save-project-btn');
        if (saveProjectBtn) {
            saveProjectBtn.addEventListener('click', () => {
                this.saveCurrentProject();
            });
        }

        const newProjectBtn = document.getElementById('new-project-btn');
        if (newProjectBtn) {
            newProjectBtn.addEventListener('click', () => {
                this.startNewProject();
            });
        }

        const refreshProjectsBtn = document.getElementById('refresh-projects-btn');
        if (refreshProjectsBtn) {
            refreshProjectsBtn.addEventListener('click', () => {
                this.refreshSavedProjects();
            });
        }

        const refreshVoicesBtn = document.getElementById('refresh-voices-btn');
        if (refreshVoicesBtn) {
            refreshVoicesBtn.addEventListener('click', () => {
                this.loadSpeakers();
            });
        }

        const generationPresetSelect = document.getElementById('generation-preset');
        if (generationPresetSelect) {
            generationPresetSelect.addEventListener('change', (event) => {
                this.applyGenerationPreset(event.target.value, true);
            });
        }

        const dialoguePacingPresetSelect = document.getElementById('dialogue-pacing-preset');
        if (dialoguePacingPresetSelect) {
            dialoguePacingPresetSelect.addEventListener('change', (event) => {
                this.applyDialoguePacingPreset(event.target.value, true);
            });
        }

        const seedStrategySelect = document.getElementById('seed-strategy');
        if (seedStrategySelect) {
            seedStrategySelect.addEventListener('change', (event) => {
                this.updateSeedStrategyUi(event.target.value);
            });
        }

        const scenePacingSelect = document.getElementById('scene-pacing-profile');
        if (scenePacingSelect) {
            scenePacingSelect.addEventListener('change', (event) => {
                this.updateScenePacingUi(event.target.value, true);
            });
        }

        const sceneGapInput = document.getElementById('scene-gap-ms');
        if (sceneGapInput) {
            sceneGapInput.addEventListener('input', () => {
                const currentProfile = document.getElementById('scene-pacing-profile')?.value || 'balanced';
                this.updateScenePacingUi(currentProfile, false);
            });
        }

        const loadProjectBtn = document.getElementById('load-project-btn');
        if (loadProjectBtn) {
            loadProjectBtn.addEventListener('click', () => {
                this.loadSelectedProject();
            });
        }

        const deleteProjectBtn = document.getElementById('delete-project-btn');
        if (deleteProjectBtn) {
            deleteProjectBtn.addEventListener('click', () => {
                this.deleteSelectedProject();
            });
        }

        // Emotion control method removed - emotions are now handled entirely by the timeline
        console.log('DEBUG: Emotion control method removed - using timeline emotions');

        // Advanced settings toggle
        const advancedSettingsToggle = document.getElementById('advanced-settings-toggle');
        if (advancedSettingsToggle) {
            advancedSettingsToggle.addEventListener('click', () => {
                const collapsible = advancedSettingsToggle.closest('.collapsible');
                if (collapsible) {
                    const content = collapsible.querySelector('.collapsible-content');

                    if (content) {
                        const isCollapsed = content.style.display === 'none' || !content.classList.contains('show');

                        if (isCollapsed) {
                            content.style.display = 'block';
                            content.classList.add('show');
                            collapsible.classList.add('expanded');
                        } else {
                            content.classList.remove('show');
                            content.style.display = 'none';
                            collapsible.classList.remove('expanded');
                        }
                    }
                }
            });
            console.log('DEBUG: Advanced settings toggle event listener added');
        } else {
            console.warn('DEBUG: advanced-settings-toggle not found');
        }

        // Generate conversation
        const generateConversationBtn = document.getElementById('generate-conversation-btn');
        if (generateConversationBtn) {
            generateConversationBtn.addEventListener('click', () => {
                console.log('DEBUG: Generate conversation button clicked');
                this.generateConversation();
            });
            console.log('DEBUG: Generate conversation button event listener added');
        } else {
            console.warn('DEBUG: generate-conversation-btn not found');
        }

        // Stop generation
        const stopGenerationBtn = document.getElementById('stop-generation-btn');
        if (stopGenerationBtn) {
            stopGenerationBtn.addEventListener('click', () => {
                this.stopGeneration();
            });
            console.log('DEBUG: Stop generation button event listener added');
        } else {
            console.warn('DEBUG: stop-generation-btn not found');
        }

        // View results
        const viewResultsBtn = document.getElementById('view-results-btn');
        if (viewResultsBtn) {
            viewResultsBtn.addEventListener('click', () => {
                this.switchTab('conversation-results');
            });
            console.log('DEBUG: View results button event listener added');
        } else {
            console.warn('DEBUG: view-results-btn not found');
        }
        
        console.log('DEBUG: setupConversationWorkflowEvents completed successfully');
    } catch (error) {
        console.error('DEBUG: setupConversationWorkflowEvents error:', error);
    }
};

IndexTTSApp.prototype.setupConversationResultsEvents = function() {
    console.log('DEBUG: setupConversationResultsEvents called');
    
    try {
        // Auto-select best versions
        const autoSelectBestBtn = document.getElementById('auto-select-best-btn');
        if (autoSelectBestBtn) {
            autoSelectBestBtn.addEventListener('click', () => {
                this.autoSelectBestVersions();
            });
            console.log('DEBUG: Auto-select best button event listener added');
        } else {
            console.warn('DEBUG: auto-select-best-btn not found');
        }

        // Clear selections
        const clearSelectionsBtn = document.getElementById('clear-selections-btn');
        if (clearSelectionsBtn) {
            clearSelectionsBtn.addEventListener('click', () => {
                this.clearVersionSelections();
            });
            console.log('DEBUG: Clear selections button event listener added');
        } else {
            console.warn('DEBUG: clear-selections-btn not found');
        }

        // Concatenate audio
        const concatenateBtn = document.getElementById('concatenate-btn');
        if (concatenateBtn) {
            concatenateBtn.addEventListener('click', () => {
                this.concatenateConversation();
            });
            console.log('DEBUG: Concatenate button event listener added');
        } else {
            console.warn('DEBUG: concatenate-btn not found');
        }

        // Download selected
        const downloadSelectedBtn = document.getElementById('download-selected-btn');
        if (downloadSelectedBtn) {
            downloadSelectedBtn.addEventListener('click', () => {
                this.downloadSelectedVersions();
            });
            console.log('DEBUG: Download selected button event listener added');
        } else {
            console.warn('DEBUG: download-selected-btn not found');
        }

        const copyListeningFeedbackBtn = document.getElementById('copy-listening-feedback-btn');
        if (copyListeningFeedbackBtn) {
            copyListeningFeedbackBtn.addEventListener('click', () => {
                this.copyListeningFeedbackExport();
            });
        }

        const downloadListeningFeedbackBtn = document.getElementById('download-listening-feedback-btn');
        if (downloadListeningFeedbackBtn) {
            downloadListeningFeedbackBtn.addEventListener('click', () => {
                this.downloadListeningFeedbackExport();
            });
        }

        const copySeedReportBtn = document.getElementById('copy-seed-report-btn');
        if (copySeedReportBtn) {
            copySeedReportBtn.addEventListener('click', () => {
                this.copySeedReportExport();
            });
        }

        const downloadSeedReportBtn = document.getElementById('download-seed-report-btn');
        if (downloadSeedReportBtn) {
            downloadSeedReportBtn.addEventListener('click', () => {
                this.downloadSeedReportExport();
            });
        }

        const clearListeningFeedbackBtn = document.getElementById('clear-listening-feedback-btn');
        if (clearListeningFeedbackBtn) {
            clearListeningFeedbackBtn.addEventListener('click', () => {
                this.clearCurrentConversationReviews();
            });
        }

        const overlapPlanFile = document.getElementById('overlap-plan-file');
        if (overlapPlanFile) {
            overlapPlanFile.addEventListener('change', (event) => {
                this.loadOverlapPlanFile(event);
            });
        }

        ['concat-target-level-dbfs', 'concat-peak-limit-dbfs', 'concat-fade-in-ms', 'concat-fade-out-ms', 'concat-output-format', 'concat-scene-gap-ms', 'concat-scene-pacing-profile'].forEach((id) => {
            const input = document.getElementById(id);
            if (input) {
                const eventName = id === 'concat-output-format' || id === 'concat-scene-pacing-profile' ? 'change' : 'input';
                input.addEventListener(eventName, () => {
                    if (id === 'concat-scene-pacing-profile' && typeof this.getScenePacingPresets === 'function') {
                        const presets = this.getScenePacingPresets();
                        const selectedPreset = presets[input.value];
                        const gapInput = document.getElementById('concat-scene-gap-ms');
                        if (selectedPreset && gapInput) {
                            gapInput.value = selectedPreset.default_gap_ms;
                        }
                    }
                    this.updateConversationMixLabels();
                });
            }
        });
        this.updateConversationMixLabels();

        const openSelectedInTimelineBtn = document.getElementById('open-selected-in-timeline-btn');
        if (openSelectedInTimelineBtn) {
            openSelectedInTimelineBtn.addEventListener('click', () => {
                this.openSelectedConversationInTimeline();
            });
        }
        
        console.log('DEBUG: setupConversationResultsEvents completed successfully');
    } catch (error) {
        console.error('DEBUG: setupConversationResultsEvents error:', error);
    }
};

IndexTTSApp.prototype.setupModalEvents = function() {
    console.log('DEBUG: setupModalEvents called');
    
    try {
        const modal = document.getElementById('audio-player-modal');
        const closeBtn = document.getElementById('close-audio-player');
        
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                if (modal) modal.classList.remove('show');
                if (this.customMediaPlayer) {
                    this.customMediaPlayer.stop();
                }
            });
            console.log('DEBUG: Modal close button event listener added');
        } else {
            console.warn('DEBUG: close-audio-player not found');
        }
        
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.classList.remove('show');
                    if (this.customMediaPlayer) {
                        this.customMediaPlayer.stop();
                    }
                }
            });

            // Initialize custom media player when modal is first opened
            modal.addEventListener('shown', () => {
                this.initializeCustomMediaPlayer().catch(error => {
                    console.error('DEBUG: Modal player initialization failed:', error);
                });
            });
            console.log('DEBUG: Modal event listeners added');
        } else {
            console.warn('DEBUG: audio-player-modal not found');
        }
        
        console.log('DEBUG: setupModalEvents completed successfully');
    } catch (error) {
        console.error('DEBUG: setupModalEvents error:', error);
    }
};

IndexTTSApp.prototype.initializeCustomMediaPlayer = async function() {
    console.log('DEBUG: initializeCustomMediaPlayer called');
    
    try {
        if (!this.customMediaPlayer) {
            if (typeof CustomMediaPlayer !== 'undefined') {
                this.customMediaPlayer = new CustomMediaPlayer('customMediaPlayer', {
                    waveColor: this.isDarkMode ? '#63b3ed' : '#4299e1',
                    progressColor: this.isDarkMode ? '#2c5282' : '#2b6cb0',
                    cursorColor: this.isDarkMode ? '#1a365d' : '#1a365d',
                    barWidth: 2,
                    barRadius: 2,
                    responsive: true,
                    height: 100,
                    normalize: true,
                    backend: 'WebAudio'
                });
                console.log('DEBUG: CustomMediaPlayer initialized successfully');
            } else {
                throw new Error('CustomMediaPlayer class not available');
            }
        }

        if (!this.customMediaPlayer) {
            throw new Error('Custom media player could not be created');
        }

        if (this.customMediaPlayer.ready) {
            await this.customMediaPlayer.ready;
        }

        return this.customMediaPlayer;
    } catch (error) {
        console.error('DEBUG: initializeCustomMediaPlayer error:', error);
        throw error;
    }
};

IndexTTSApp.prototype.setupRangeInputs = function() {
    document.querySelectorAll('input[type="range"]').forEach(input => {
        const updateValue = () => {
            const valueDisplay = input.nextElementSibling;
            if (valueDisplay && valueDisplay.classList.contains('range-value')) {
                valueDisplay.textContent = input.value;
            }
        };
        
        input.addEventListener('input', updateValue);
        updateValue();
    });
};

IndexTTSApp.prototype.setupDarkMode = function() {
    // Apply dark mode based on saved preference
    if (this.isDarkMode) {
        document.body.classList.add('dark-mode');
    }
};

IndexTTSApp.prototype.setupDarkModeToggle = function() {
    // Create dark mode toggle button if it doesn't exist
    let darkModeToggle = document.getElementById('dark-mode-toggle');
    
    if (!darkModeToggle) {
        darkModeToggle = document.createElement('button');
        darkModeToggle.id = 'dark-mode-toggle';
        darkModeToggle.className = 'dark-mode-toggle';
        darkModeToggle.innerHTML = '<i class="fas fa-moon"></i>';
        darkModeToggle.title = 'Toggle dark mode';
        
        // Add to header or appropriate location
        const header = document.querySelector('header') || document.body;
        header.appendChild(darkModeToggle);
    }
    
    // Set initial state
    darkModeToggle.innerHTML = this.isDarkMode ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
    
    // Add click event listener
    darkModeToggle.addEventListener('click', () => {
        this.toggleDarkMode();
    });
};

IndexTTSApp.prototype.toggleDarkMode = function() {
    this.isDarkMode = !this.isDarkMode;
    
    // Update body class
    if (this.isDarkMode) {
        document.body.classList.add('dark-mode');
    } else {
        document.body.classList.remove('dark-mode');
    }
    
    // Update toggle button
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    if (darkModeToggle) {
        darkModeToggle.innerHTML = this.isDarkMode ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
    }
    
    // Save preference
    localStorage.setItem('darkMode', this.isDarkMode.toString());
};
