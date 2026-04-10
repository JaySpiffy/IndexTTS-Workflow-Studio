// IndexTTS2 WebMCP Integration Module

if (typeof IndexTTSApp === 'undefined') {
    console.error('IndexTTSApp is not defined! WebMCP module could not attach.');
}

const WEBMCP_SCRIPT_URL = 'https://webmcp.dev/webmcp.js';

function webMcpJsonText(payload) {
    return JSON.stringify(payload, null, 2);
}

function buildWebMcpTextResult(payload) {
    return {
        content: [
            {
                type: 'text',
                text: typeof payload === 'string' ? payload : webMcpJsonText(payload),
            },
        ],
    };
}

function buildWebMcpPromptText(text) {
    return {
        messages: [
            {
                role: 'user',
                content: {
                    type: 'text',
                    text,
                },
            },
        ],
    };
}

function parseWebMcpScriptText(scriptText, speakers = []) {
    const normalizedScript = String(scriptText || '').trim();
    if (!normalizedScript) {
        throw new Error('Script text cannot be empty');
    }

    const lines = [];
    const availableSpeakers = Array.isArray(speakers) ? speakers : [];
    const scriptLines = normalizedScript.split('\n').filter((line) => line.trim());

    scriptLines.forEach((rawLine, index) => {
        const line = rawLine.trim();
        const match = line.match(/^([^:]+):\s*(.+)$/);
        if (!match) {
            throw new Error(`Invalid line format: ${line}`);
        }

        const speakerName = match[1].trim();
        const text = match[2].trim();
        const lowerName = speakerName.toLowerCase();

        let matchedSpeaker = availableSpeakers.find((speaker) => speaker.filename === speakerName)
            || availableSpeakers.find((speaker) => speaker.name === speakerName)
            || availableSpeakers.find((speaker) => speaker.filename === (speakerName.endsWith('.wav') ? speakerName : `${speakerName}.wav`))
            || availableSpeakers.find((speaker) => String(speaker.name || '').toLowerCase() === lowerName.replace(/\.wav$/i, ''))
            || availableSpeakers.find((speaker) => String(speaker.filename || '').toLowerCase() === lowerName)
            || availableSpeakers.find((speaker) => String(speaker.filename || '').toLowerCase() === `${lowerName}.wav`);

        if (!matchedSpeaker) {
            throw new Error(`Speaker not found: ${speakerName}`);
        }

        lines.push({
            line: lines.length + 1,
            speaker: matchedSpeaker.filename.replace(/\.wav$/i, ''),
            speaker_filename: matchedSpeaker.filename,
            text,
            line_number: index,
            emo_vector: [0, 0, 0, 0, 0, 0, 0, 0.8],
            emo_auto_detected: false,
        });
    });

    return lines;
}

IndexTTSApp.prototype.loadWebMcpScript = function() {
    if (window.WebMCP) {
        return Promise.resolve(window.WebMCP);
    }

    if (this.webMcpScriptPromise) {
        return this.webMcpScriptPromise;
    }

    this.webMcpScriptPromise = new Promise((resolve, reject) => {
        const existingScript = document.querySelector('script[data-webmcp-loader="true"]');
        if (existingScript) {
            existingScript.addEventListener('load', () => resolve(window.WebMCP), { once: true });
            existingScript.addEventListener('error', () => reject(new Error('Failed to load WebMCP script')), { once: true });
            return;
        }

        const script = document.createElement('script');
        script.src = WEBMCP_SCRIPT_URL;
        script.async = true;
        script.defer = true;
        script.dataset.webmcpLoader = 'true';
        script.onload = () => {
            if (window.WebMCP) {
                resolve(window.WebMCP);
                return;
            }
            reject(new Error('WebMCP loaded but did not expose window.WebMCP'));
        };
        script.onerror = () => reject(new Error('Failed to load WebMCP script'));
        document.head.appendChild(script);
    });

    return this.webMcpScriptPromise;
};

IndexTTSApp.prototype.getWebMcpSpeakerList = async function() {
    const response = await this.apiRequest('/speakers/', { suppressErrorNotification: true });
    this.speakers = response.speakers || [];
    return this.speakers;
};

IndexTTSApp.prototype.getWebMcpSourceClipList = async function() {
    const response = await this.apiRequest('/speakers-tools/list-source-clips', { suppressErrorNotification: true });
    this.sourceClips = response.details?.files || [];
    return this.sourceClips;
};

IndexTTSApp.prototype.getWebMcpStateSnapshot = function() {
    return {
        current_tab: this.currentTab,
        current_conversation_id: this.currentConversationId,
        current_timeline_project_id: this.currentTimelineProjectId,
        available_speakers: (this.speakers || []).map((speaker) => ({
            name: speaker.name,
            filename: speaker.filename,
            size_kb: speaker.size_kb,
        })),
        available_source_clips: (this.sourceClips || []).map((clip) => ({
            filename: clip.filename,
            size_kb: clip.size_kb,
            content_type: clip.content_type,
        })),
        loaded_conversation_title: this.parsedScript?.title || null,
        parsed_line_count: this.parsedScript?.lines?.length || 0,
        timeline_project_name: this.currentTimelineProject?.project_name || null,
        timeline_track_count: this.currentTimelineProject?.tracks?.length || 0,
    };
};

IndexTTSApp.prototype.createWebMcpConversationRequest = async function(args = {}) {
    const speakers = await this.getWebMcpSpeakerList();
    const title = String(args.title || 'Untitled Conversation').trim() || 'Untitled Conversation';
    const lines = parseWebMcpScriptText(args.script_text, speakers);
    const autoDetectEmotion = args.auto_detect_emotion !== false;

    if (autoDetectEmotion) {
        try {
            const emotionResults = await this.detectEmotionsForScript(lines);
            emotionResults.forEach((result, index) => {
                if (index >= lines.length) {
                    return;
                }

                if (result?.details?.emotion_vectors) {
                    lines[index].emo_vector = result.details.emotion_vectors;
                    lines[index].emo_detected_dict = result.details.emotion_dict;
                } else if (result?.emotion_vectors) {
                    lines[index].emo_vector = result.emotion_vectors;
                    lines[index].emo_detected_dict = result.emotion_dict;
                }
                lines[index].emo_auto_detected = true;
            });
        } catch (error) {
            console.warn('WebMCP emotion detection failed, continuing with default calm vectors:', error);
        }
    }

    const baseSettings = typeof this.getDefaultGenerationSettings === 'function'
        ? this.getDefaultGenerationSettings()
        : {};
    const request = {
        ...baseSettings,
        generation_preset: args.generation_preset || baseSettings.generation_preset || 'balanced',
        pacing_preset: args.pacing_preset || this.currentConversationDialoguePacingPreset || 'natural',
        versions_per_line: Number.isFinite(Number(args.versions_per_line))
            ? Math.max(1, Math.min(5, Number(args.versions_per_line)))
            : (baseSettings.versions_per_line || 3),
        script: {
            title,
            lines,
        },
        emotion_control_method: 'from_speaker',
    };

    this.conversationScript = lines;
    this.parsedScript = { title, lines };

    if (typeof this.renderSpeakerPacingControls === 'function') {
        this.renderSpeakerPacingControls(lines);
    }
    if (typeof this.renderScriptPreview === 'function') {
        this.renderScriptPreview(lines);
    }

    return request;
};

IndexTTSApp.prototype.registerWebMcpTools = function() {
    this.webMcp.registerTool(
        'studio_list_speakers',
        'List the available speaker prompt files loaded by the app.',
        {},
        async () => {
            const speakers = await this.getWebMcpSpeakerList();
            return buildWebMcpTextResult({
                count: speakers.length,
                speakers,
            });
        }
    );

    this.webMcp.registerTool(
        'studio_list_source_clips',
        'List uploaded source clips available in Speaker Prep.',
        {},
        async () => {
            const clips = await this.getWebMcpSourceClipList();
            return buildWebMcpTextResult({
                count: clips.length,
                files: clips,
            });
        }
    );

    this.webMcp.registerTool(
        'studio_get_source_clip_diagnostics',
        'Get cloning-focused diagnostics for a source clip.',
        {
            filename: { type: 'string', description: 'The source clip filename to analyze.' },
        },
        async (args) => {
            const response = await this.apiRequest(`/speakers-tools/source-clip-diagnostics/${encodeURIComponent(args.filename)}`, {
                suppressErrorNotification: true,
            });
            return buildWebMcpTextResult(response.details || {});
        }
    );

    this.webMcp.registerTool(
        'studio_prepare_source_clip',
        'Prepare a source clip with trim, mono conversion, normalization, and optional noise cleanup.',
        {
            source_filename: { type: 'string', description: 'The source clip filename to prepare.' },
            output_name: { type: 'string', description: 'The output clip name without extension.' },
            target_category: { type: 'string', description: 'Either source_clips or speakers.' },
            start_time: { type: 'number', description: 'Optional trim start time in seconds.' },
            end_time: { type: 'number', description: 'Optional trim end time in seconds.' },
            convert_to_mono: { type: 'boolean', description: 'Whether to convert the clip to mono.' },
            normalize_audio: { type: 'boolean', description: 'Whether to normalize the clip level.' },
            target_peak_dbfs: { type: 'number', description: 'Peak target in dBFS when normalizing.' },
            use_noise_reduction: { type: 'boolean', description: 'Whether to apply noise cleanup.' },
            noise_reduction_strength: { type: 'number', description: 'Noise cleanup strength.' },
            noise_reduction_backend: { type: 'string', description: 'auto, classic, or deepfilter.' },
            use_vocal_separation: { type: 'boolean', description: 'Whether to isolate vocals before saving.' },
        },
        async (args) => {
            const requestBody = {
                source_filename: args.source_filename,
                output_name: args.output_name || String(args.source_filename || '').replace(/\.[^.]+$/, '_ready'),
                target_category: args.target_category || 'source_clips',
                start_time: Number.isFinite(Number(args.start_time)) ? Number(args.start_time) : 0,
                end_time: args.end_time == null || args.end_time === '' ? null : Number(args.end_time),
                convert_to_mono: args.convert_to_mono !== false,
                normalize_audio: args.normalize_audio !== false,
                target_peak_dbfs: Number.isFinite(Number(args.target_peak_dbfs)) ? Number(args.target_peak_dbfs) : -1,
                use_noise_reduction: Boolean(args.use_noise_reduction),
                noise_reduction_strength: Number.isFinite(Number(args.noise_reduction_strength)) ? Number(args.noise_reduction_strength) : 0.35,
                noise_reduction_backend: args.noise_reduction_backend || 'auto',
                use_vocal_separation: Boolean(args.use_vocal_separation),
            };
            const response = await this.apiRequest('/speakers-tools/prepare-source-clip', {
                method: 'POST',
                suppressErrorNotification: true,
                body: JSON.stringify(requestBody),
            });
            return buildWebMcpTextResult(response.details || {});
        }
    );

    this.webMcp.registerTool(
        'studio_parse_script',
        'Parse a multi-speaker script using the same speaker matching rules as the main UI.',
        {
            title: { type: 'string', description: 'Optional conversation title.' },
            script_text: { type: 'string', description: 'Raw multi-speaker script in speaker: text format.' },
            auto_detect_emotion: { type: 'boolean', description: 'Whether to auto-detect emotions for each line.' },
        },
        async (args) => {
            const request = await this.createWebMcpConversationRequest(args);
            const response = await this.apiRequest('/conversation/parse-script', {
                method: 'POST',
                suppressErrorNotification: true,
                body: JSON.stringify(request.script),
            });
            return buildWebMcpTextResult({
                title: request.script.title,
                line_count: request.script.lines.length,
                statistics: response.details || {},
                lines: request.script.lines,
            });
        }
    );

    this.webMcp.registerTool(
        'studio_generate_conversation',
        'Generate a new conversation from raw script text.',
        {
            title: { type: 'string', description: 'Optional conversation title.' },
            script_text: { type: 'string', description: 'Raw multi-speaker script in speaker: text format.' },
            auto_detect_emotion: { type: 'boolean', description: 'Whether to auto-detect emotions for each line.' },
            versions_per_line: { type: 'number', description: 'How many candidate versions to generate per line.' },
            generation_preset: { type: 'string', description: 'balanced, clone_fidelity, or expressive.' },
            pacing_preset: { type: 'string', description: 'natural, calm, argument, or panic.' },
        },
        async (args) => {
            const request = await this.createWebMcpConversationRequest(args);
            const response = await this.apiRequest('/conversation/generate', {
                method: 'POST',
                suppressErrorNotification: true,
                body: JSON.stringify(request),
            });
            this.currentConversationId = response.conversation_id;
            this.currentConversationDialoguePacingPreset = request.pacing_preset || 'natural';
            return buildWebMcpTextResult({
                conversation_id: response.conversation_id,
                title: request.script.title,
                line_count: request.script.lines.length,
                versions_per_line: request.versions_per_line,
                status: 'started',
            });
        }
    );

    this.webMcp.registerTool(
        'studio_get_conversation_status',
        'Check generation status for a conversation.',
        {
            conversation_id: { type: 'string', description: 'The conversation ID to inspect.' },
        },
        async (args) => {
            const response = await this.apiRequest(`/conversation/status/${encodeURIComponent(args.conversation_id)}`, {
                suppressErrorNotification: true,
            });
            if (response.task?.status === 'completed' && typeof this.loadConversations === 'function') {
                this.loadConversations();
            }
            return buildWebMcpTextResult(response.task || response);
        }
    );

    this.webMcp.registerTool(
        'studio_list_timeline_projects',
        'List saved timeline projects.',
        {},
        async () => {
            const response = await this.apiRequest('/timeline/list', {
                suppressErrorNotification: true,
            });
            return buildWebMcpTextResult({
                count: response.details?.projects?.length || 0,
                projects: response.details?.projects || [],
            });
        }
    );

    this.webMcp.registerTool(
        'studio_create_timeline',
        'Create a blank timeline project.',
        {
            project_name: { type: 'string', description: 'The timeline project name.' },
            description: { type: 'string', description: 'Optional project description.' },
            conversation_id: { type: 'string', description: 'Optional conversation ID to link to the timeline.' },
        },
        async (args) => {
            const response = await this.apiRequest('/timeline/create', {
                method: 'POST',
                suppressErrorNotification: true,
                body: JSON.stringify({
                    project_name: args.project_name || `timeline-${Date.now()}`,
                    description: args.description || 'Created via WebMCP',
                    conversation_id: args.conversation_id || null,
                }),
            });
            this.currentTimelineProjectId = response.project?.project_id || null;
            this.currentTimelineProject = response.project || null;
            if (typeof this.loadTimelineProjects === 'function') {
                this.loadTimelineProjects();
            }
            return buildWebMcpTextResult(response.project || {});
        }
    );

    this.webMcp.registerTool(
        'studio_add_timeline_track',
        'Add a speaker track to a timeline project.',
        {
            project_id: { type: 'string', description: 'The timeline project ID.' },
            track_name: { type: 'string', description: 'The track display name.' },
            speaker_filename: { type: 'string', description: 'The speaker filename to assign to the track.' },
        },
        async (args) => {
            const response = await this.apiRequest(`/timeline/${encodeURIComponent(args.project_id)}/tracks`, {
                method: 'POST',
                suppressErrorNotification: true,
                body: JSON.stringify({
                    track_name: args.track_name,
                    speaker_filename: args.speaker_filename,
                }),
            });
            return buildWebMcpTextResult(response.track || {});
        }
    );

    this.webMcp.registerTool(
        'studio_add_timeline_segment',
        'Add a segment to a timeline track.',
        {
            project_id: { type: 'string', description: 'The timeline project ID.' },
            track_id: { type: 'string', description: 'The track ID.' },
            text: { type: 'string', description: 'Segment dialogue text.' },
            start_time: { type: 'number', description: 'Segment start time in seconds.' },
            duration: { type: 'number', description: 'Planned segment duration in seconds.' },
            emotion_text: { type: 'string', description: 'Optional emotion steering text.' },
        },
        async (args) => {
            const response = await this.apiRequest(`/timeline/${encodeURIComponent(args.project_id)}/tracks/${encodeURIComponent(args.track_id)}/segments`, {
                method: 'POST',
                suppressErrorNotification: true,
                body: JSON.stringify({
                    text: args.text,
                    start_time: Number.isFinite(Number(args.start_time)) ? Number(args.start_time) : 0,
                    duration: Number.isFinite(Number(args.duration)) ? Number(args.duration) : 2.5,
                    emotion_control_method: 'from_speaker',
                    emotion_text: args.emotion_text || null,
                }),
            });
            return buildWebMcpTextResult(response.segment || {});
        }
    );
};

IndexTTSApp.prototype.registerWebMcpResources = function() {
    this.webMcp.registerResource(
        'studio-health',
        'Current app health and connection summary.',
        {
            uri: 'studio://health',
            mimeType: 'application/json',
        },
        () => ({
            contents: [
                {
                    uri: 'studio://health',
                    mimeType: 'application/json',
                    text: webMcpJsonText({
                        api_status: document.querySelector('#api-status .status-text')?.textContent || 'Unknown',
                        current_tab: this.currentTab,
                        webmcp_ready: this.webMcpReady,
                    }),
                },
            ],
        })
    );

    this.webMcp.registerResource(
        'studio-state',
        'Current high-level app state snapshot.',
        {
            uri: 'studio://state',
            mimeType: 'application/json',
        },
        () => ({
            contents: [
                {
                    uri: 'studio://state',
                    mimeType: 'application/json',
                    text: webMcpJsonText(this.getWebMcpStateSnapshot()),
                },
            ],
        })
    );

    this.webMcp.registerResource(
        'studio-current-script',
        'The current script text from the Conversation Workflow tab.',
        {
            uri: 'studio://conversation/script',
            mimeType: 'text/plain',
        },
        () => ({
            contents: [
                {
                    uri: 'studio://conversation/script',
                    mimeType: 'text/plain',
                    text: document.getElementById('conversation-script')?.value || '',
                },
            ],
        })
    );

    this.webMcp.registerResource(
        'studio-current-timeline',
        'The currently loaded timeline project summary.',
        {
            uri: 'studio://timeline/current',
            mimeType: 'application/json',
        },
        () => ({
            contents: [
                {
                    uri: 'studio://timeline/current',
                    mimeType: 'application/json',
                    text: webMcpJsonText(this.currentTimelineProject || { message: 'No timeline project loaded' }),
                },
            ],
        })
    );
};

IndexTTSApp.prototype.registerWebMcpPrompts = function() {
    this.webMcp.registerPrompt(
        'studio-generate-dialogue',
        'Generate a multi-speaker dialogue using the current studio workflow.',
        [
            { name: 'goal', description: 'What the dialogue should achieve.', required: true },
            { name: 'speakers', description: 'Which speakers should be used.', required: true },
            { name: 'tone', description: 'Desired tone or pacing.', required: false },
        ],
        (args) => buildWebMcpPromptText(
            `Using the IndexTTS2 Workflow Studio website, create a multi-speaker script for these speakers: ${args.speakers}.\n\nGoal: ${args.goal}\nTone/pacing: ${args.tone || 'natural'}\n\nThen use the available studio tools to parse the script, generate the conversation, and report back the conversation ID.`
        )
    );

    this.webMcp.registerPrompt(
        'studio-prepare-source-clip',
        'Prepare a source clip for voice cloning using Speaker Prep.',
        [
            { name: 'filename', description: 'The source clip filename to prepare.', required: true },
            { name: 'target', description: 'Target category: source_clips or speakers.', required: false },
        ],
        (args) => buildWebMcpPromptText(
            `Use the Speaker Prep tools on this site to inspect and prepare the source clip "${args.filename}".\n\nDefault target category: ${args.target || 'source_clips'}.\n\nRun diagnostics first, then choose a safe prep recipe unless the diagnostics clearly suggest something stronger.`
        )
    );

    this.webMcp.registerPrompt(
        'studio-build-timeline-scene',
        'Create a simple timeline scene with tracks and timed segments.',
        [
            { name: 'scene_goal', description: 'What kind of scene to build.', required: true },
            { name: 'speakers', description: 'Which speakers or speaker filenames should be used.', required: true },
        ],
        (args) => buildWebMcpPromptText(
            `Use the timeline tools on this site to build a new scene.\n\nScene goal: ${args.scene_goal}\nSpeakers: ${args.speakers}\n\nCreate a blank timeline, add speaker tracks, add timed dialogue segments, and summarize the resulting project ID and track layout.`
        )
    );
};

IndexTTSApp.prototype.setupWebMcp = async function() {
    if (this.webMcpInitStarted || this.webMcpReady) {
        return;
    }

    this.webMcpInitStarted = true;

    try {
        await this.loadWebMcpScript();
        const WebMCPConstructor = window.WebMCP;
        if (!WebMCPConstructor) {
            throw new Error('window.WebMCP is unavailable after script load');
        }

        this.webMcp = window.webmcp || new WebMCPConstructor({
            color: '#5b4cf6',
            position: 'bottom-right',
            size: '42px',
            padding: '18px',
        });

        window.webmcp = this.webMcp;

        this.registerWebMcpTools();
        this.registerWebMcpResources();
        this.registerWebMcpPrompts();
        this.webMcpReady = true;
        console.info('WebMCP connected: studio tools, prompts, and resources registered.');
    } catch (error) {
        console.info('WebMCP setup skipped:', error);
    }
};
