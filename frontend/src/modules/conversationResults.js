// IndexTTS2 Conversation Results Module
function escapeConversationResultsHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function getConversationReviewScore(version) {
    const reviewScore = Number(version?.review_score);
    if (Number.isFinite(reviewScore)) {
        return reviewScore;
    }

    const qualityScore = Number(version?.quality_score);
    if (Number.isFinite(qualityScore)) {
        return qualityScore;
    }

    const similarityScore = Number(version?.similarity_score);
    return Number.isFinite(similarityScore) ? similarityScore : 0;
}

function formatPacingLabel(label) {
    return String(label || 'unknown')
        .replace(/_/g, ' ')
        .replace(/\b\w/g, character => character.toUpperCase());
}

function getConversationQualityGateSettings(generationParams = {}) {
    const similarityThreshold = Number.isFinite(Number(generationParams?.similarity_threshold))
        ? Number(generationParams.similarity_threshold)
        : 0.6;
    const roboticThreshold = Number.isFinite(Number(generationParams?.robotic_threshold))
        ? Number(generationParams.robotic_threshold)
        : 0.7;
    const minQualityScore = Number.isFinite(Number(generationParams?.quality_gate_min_quality_score))
        ? Number(generationParams.quality_gate_min_quality_score)
        : Math.max(0.48, similarityThreshold * 0.8);
    const minPacingScore = Number.isFinite(Number(generationParams?.quality_gate_min_pacing_score))
        ? Number(generationParams.quality_gate_min_pacing_score)
        : 0.45;

    return {
        similarityThreshold,
        roboticThreshold,
        minQualityScore,
        minPacingScore,
    };
}

function conversationVersionMeetsQualityGate(version, generationParams = {}) {
    if (!version || typeof version !== 'object') {
        return false;
    }

    if (typeof version.meets_quality_gate === 'boolean') {
        return version.meets_quality_gate;
    }

    const gate = getConversationQualityGateSettings(generationParams);
    const similarityScore = Number(version.similarity_score);
    const roboticScore = Number(version.robotic_score);
    const qualityScore = Number.isFinite(Number(version.quality_score))
        ? Number(version.quality_score)
        : similarityScore;
    const pacingScore = Number.isFinite(Number(version.pacing_score))
        ? Number(version.pacing_score)
        : null;

    if (!Number.isFinite(similarityScore) || similarityScore < gate.similarityThreshold) {
        return false;
    }
    if (!Number.isFinite(roboticScore) || roboticScore > gate.roboticThreshold) {
        return false;
    }
    if (!Number.isFinite(qualityScore) || qualityScore < gate.minQualityScore) {
        return false;
    }
    if (pacingScore !== null && pacingScore < gate.minPacingScore) {
        return false;
    }
    return true;
}

function getBestConversationVersionIndex(line, generationParams = {}, { requireQualityGate = false } = {}) {
    const versions = line?.versions || [];
    let bestGatePassingVersionIndex = -1;
    let bestGatePassingScore = Number.NEGATIVE_INFINITY;
    let bestVersionIndex = -1;
    let bestScore = Number.NEGATIVE_INFINITY;

    versions.forEach((version, index) => {
        const reviewScore = getConversationReviewScore(version);
        const meetsQualityGate = conversationVersionMeetsQualityGate(version, generationParams);

        if (meetsQualityGate && reviewScore > bestGatePassingScore) {
            bestGatePassingScore = reviewScore;
            bestGatePassingVersionIndex = index;
        }

        if (reviewScore > bestScore) {
            bestScore = reviewScore;
            bestVersionIndex = index;
        }
    });

    if (requireQualityGate && bestGatePassingVersionIndex >= 0) {
        return bestGatePassingVersionIndex;
    }

    return bestVersionIndex;
}

IndexTTSApp.prototype.loadListeningReviewsState = function() {
    try {
        const saved = localStorage.getItem('indexttsListeningReviews');
        return saved ? JSON.parse(saved) : {};
    } catch (error) {
        console.error('Failed to load listening reviews:', error);
        return {};
    }
};

IndexTTSApp.prototype.saveListeningReviewsState = function() {
    try {
        localStorage.setItem('indexttsListeningReviews', JSON.stringify(this.listeningReviews || {}));
    } catch (error) {
        console.error('Failed to save listening reviews:', error);
    }
};

IndexTTSApp.prototype.getCurrentConversationReviewBucket = function(createIfMissing = false) {
    if (!this.currentConversationId) {
        return null;
    }

    if (!this.listeningReviews) {
        this.listeningReviews = {};
    }

    if (createIfMissing && !this.listeningReviews[this.currentConversationId]) {
        this.listeningReviews[this.currentConversationId] = {};
    }

    return this.listeningReviews[this.currentConversationId] || null;
};

IndexTTSApp.prototype.getListeningReviewStorageKey = function(lineIndex, versionIndex) {
    return `${lineIndex}:${versionIndex}`;
};

IndexTTSApp.prototype.getClipReviewCode = function(lineIndex, versionIndex) {
    const line = this.currentConversationData?.lines?.[lineIndex];
    const lineNumber = Number.isFinite(Number(line?.line_number)) ? Number(line.line_number) + 1 : lineIndex + 1;
    return `${(this.currentConversationId || 'unknown').substring(0, 8)}/L${lineNumber}/V${versionIndex + 1}`;
};

IndexTTSApp.prototype.getListeningReview = function(lineIndex, versionIndex) {
    const bucket = this.getCurrentConversationReviewBucket(false);
    if (!bucket) {
        return null;
    }
    return bucket[this.getListeningReviewStorageKey(lineIndex, versionIndex)] || null;
};

IndexTTSApp.prototype.getListeningScoreOptionsHtml = function(selectedValue) {
    const options = [
        { value: 1, label: '1' },
        { value: 2, label: '2' },
        { value: 3, label: '3' },
        { value: 4, label: '4' },
        { value: 5, label: '5' },
    ];

    return options.map(({ value, label }) => (
        `<option value="${value}" ${Number(selectedValue) === value ? 'selected' : ''}>${label}</option>`
    )).join('');
};

IndexTTSApp.prototype.buildListeningReviewHtml = function(lineIndex, versionIndex) {
    const review = this.getListeningReview(lineIndex, versionIndex) || {};
    const verdict = escapeConversationResultsHtml(review.verdict || 'ok');
    const issues = escapeConversationResultsHtml((review.issues || []).join(','));
    const actions = escapeConversationResultsHtml((review.action || []).join(','));
    const notes = escapeConversationResultsHtml(review.notes || '');
    const clipCode = escapeConversationResultsHtml(this.getClipReviewCode(lineIndex, versionIndex));

    return `
        <details class="listening-review-panel" onclick="event.stopPropagation()">
            <summary><i class="fas fa-headphones"></i> Review This Version</summary>
            <div class="listening-review-content">
                <div class="listening-review-code">Clip code: <code>${clipCode}</code></div>
                <div class="listening-review-grid">
                    <div class="form-group">
                        <label for="review-verdict-${lineIndex}-${versionIndex}">Verdict</label>
                        <select id="review-verdict-${lineIndex}-${versionIndex}">
                            <option value="bad" ${verdict === 'bad' ? 'selected' : ''}>bad</option>
                            <option value="ok" ${verdict === 'ok' ? 'selected' : ''}>ok</option>
                            <option value="good" ${verdict === 'good' ? 'selected' : ''}>good</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="review-similarity-${lineIndex}-${versionIndex}">Similarity</label>
                        <select id="review-similarity-${lineIndex}-${versionIndex}">
                            ${this.getListeningScoreOptionsHtml(review.similarity || 3)}
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="review-naturalness-${lineIndex}-${versionIndex}">Naturalness</label>
                        <select id="review-naturalness-${lineIndex}-${versionIndex}">
                            ${this.getListeningScoreOptionsHtml(review.naturalness || 3)}
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="review-pace-${lineIndex}-${versionIndex}">Pace</label>
                        <select id="review-pace-${lineIndex}-${versionIndex}">
                            ${this.getListeningScoreOptionsHtml(review.pace || 3)}
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="review-robotic-${lineIndex}-${versionIndex}">Robotic</label>
                        <select id="review-robotic-${lineIndex}-${versionIndex}">
                            ${this.getListeningScoreOptionsHtml(review.robotic || 3)}
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="review-clarity-${lineIndex}-${versionIndex}">Clarity</label>
                        <select id="review-clarity-${lineIndex}-${versionIndex}">
                            ${this.getListeningScoreOptionsHtml(review.clarity || 3)}
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="review-emotion-${lineIndex}-${versionIndex}">Emotion</label>
                        <select id="review-emotion-${lineIndex}-${versionIndex}">
                            ${this.getListeningScoreOptionsHtml(review.emotion || 3)}
                        </select>
                    </div>
                </div>
                <div class="form-group">
                    <label for="review-issues-${lineIndex}-${versionIndex}">Issues (comma separated)</label>
                    <input type="text" id="review-issues-${lineIndex}-${versionIndex}" value="${issues}" placeholder="too_fast,robotic,weak_similarity">
                </div>
                <div class="form-group">
                    <label for="review-actions-${lineIndex}-${versionIndex}">Requested actions (comma separated)</label>
                    <input type="text" id="review-actions-${lineIndex}-${versionIndex}" value="${actions}" placeholder="more_faithful,slower,cleaner_ref">
                </div>
                <div class="form-group">
                    <label for="review-notes-${lineIndex}-${versionIndex}">Notes</label>
                    <textarea id="review-notes-${lineIndex}-${versionIndex}" rows="3" placeholder="What did you hear?">${notes}</textarea>
                </div>
                <div class="listening-review-actions">
                    <button class="btn btn-secondary btn-small" onclick="event.stopPropagation(); app.saveListeningReview(${lineIndex}, ${versionIndex})">
                        <i class="fas fa-save"></i> Save Review
                    </button>
                    <button class="btn btn-secondary btn-small" onclick="event.stopPropagation(); app.copySingleListeningReview(${lineIndex}, ${versionIndex})">
                        <i class="fas fa-copy"></i> Copy Block
                    </button>
                    <button class="btn btn-secondary btn-small" onclick="event.stopPropagation(); app.clearListeningReview(${lineIndex}, ${versionIndex})">
                        <i class="fas fa-trash"></i> Clear
                    </button>
                </div>
            </div>
        </details>
    `;
};

IndexTTSApp.prototype.readListeningReviewForm = function(lineIndex, versionIndex) {
    const readScore = (field) => parseInt(document.getElementById(`review-${field}-${lineIndex}-${versionIndex}`)?.value || '3', 10);
    const splitCsv = (value) => String(value || '')
        .split(',')
        .map(item => item.trim())
        .filter(Boolean);

    return {
        clip: this.getClipReviewCode(lineIndex, versionIndex),
        verdict: (document.getElementById(`review-verdict-${lineIndex}-${versionIndex}`)?.value || 'ok').trim().toLowerCase(),
        similarity: readScore('similarity'),
        naturalness: readScore('naturalness'),
        pace: readScore('pace'),
        robotic: readScore('robotic'),
        clarity: readScore('clarity'),
        emotion: readScore('emotion'),
        issues: splitCsv(document.getElementById(`review-issues-${lineIndex}-${versionIndex}`)?.value),
        action: splitCsv(document.getElementById(`review-actions-${lineIndex}-${versionIndex}`)?.value),
        notes: (document.getElementById(`review-notes-${lineIndex}-${versionIndex}`)?.value || '').trim(),
        line_index: lineIndex,
        version_index: versionIndex,
        updated_at: new Date().toISOString(),
    };
};

IndexTTSApp.prototype.buildListeningFeedbackBlock = function(review) {
    const lines = [
        `CLIP=${review.clip}`,
        `VERDICT=${review.verdict}`,
        `SIMILARITY=${review.similarity}`,
        `NATURALNESS=${review.naturalness}`,
        `PACE=${review.pace}`,
        `ROBOTIC=${review.robotic}`,
        `CLARITY=${review.clarity}`,
        `EMOTION=${review.emotion}`,
    ];

    if (review.issues?.length) {
        lines.push(`ISSUES=${review.issues.join(',')}`);
    }
    if (review.action?.length) {
        lines.push(`ACTION=${review.action.join(',')}`);
    }
    if (review.notes) {
        lines.push(`NOTES=${review.notes}`);
    }

    return lines.join('\n');
};

IndexTTSApp.prototype.getCurrentConversationReviewEntries = function() {
    const bucket = this.getCurrentConversationReviewBucket(false);
    if (!bucket) {
        return [];
    }

    return Object.values(bucket).sort((a, b) => {
        if ((a.line_index ?? 0) !== (b.line_index ?? 0)) {
            return (a.line_index ?? 0) - (b.line_index ?? 0);
        }
        return (a.version_index ?? 0) - (b.version_index ?? 0);
    });
};

IndexTTSApp.prototype.refreshListeningFeedbackExport = function() {
    const section = document.getElementById('listening-feedback-section');
    const exportField = document.getElementById('listening-feedback-export');
    const meta = document.getElementById('listening-feedback-meta');

    if (!section || !exportField || !meta) {
        return;
    }

    if (!this.currentConversationId || !this.currentConversationData) {
        section.style.display = 'none';
        exportField.value = '';
        meta.textContent = 'No conversation selected.';
        return;
    }

    section.style.display = 'block';

    const entries = this.getCurrentConversationReviewEntries();
    if (!entries.length) {
        exportField.value = '';
        meta.textContent = 'No listening reviews saved for this conversation yet.';
        return;
    }

    exportField.value = entries.map(entry => this.buildListeningFeedbackBlock(entry)).join('\n\n');
    meta.textContent = `${entries.length} listening review${entries.length === 1 ? '' : 's'} saved for conversation ${this.currentConversationId.substring(0, 8)}.`;
};

IndexTTSApp.prototype.buildSeedReport = function() {
    if (!this.currentConversationId || !this.currentConversationData) {
        return null;
    }

    const generationParams = this.currentConversationData.generation_params || {};
    const seedRuntime = this.currentConversationData.seed_runtime_metadata || {};

    return {
        conversation_id: this.currentConversationId,
        seed_strategy: generationParams.seed_strategy || seedRuntime.seed_strategy || 'fully_random',
        fixed_base_seed: generationParams.fixed_base_seed ?? null,
        resolved_base_seed: seedRuntime.resolved_base_seed ?? null,
        reused_seed_list: Array.isArray(seedRuntime.reused_seed_list) ? seedRuntime.reused_seed_list : [],
        lines: (this.currentConversationData.lines || []).map((line, lineIndex) => {
            const selectedVersionIndex = (line.versions || []).findIndex(version => Boolean(version.is_selected));
            return {
                line_number: Number.isFinite(Number(line.line_number)) ? Number(line.line_number) : lineIndex,
                speaker_filename: line.speaker_filename,
                text: line.text,
                selected_version_index: selectedVersionIndex >= 0 ? selectedVersionIndex : null,
                versions: (line.versions || []).map((version, versionIndex) => ({
                    version_index: versionIndex,
                    audio_filename: version.audio_filename || '',
                    seed: version.seed ?? null,
                    seed_origin: version.seed_origin || null,
                    seed_strategy: version.seed_strategy || null,
                    similarity_score: version.similarity_score,
                    quality_score: version.quality_score,
                    is_selected: Boolean(version.is_selected),
                })),
            };
        }),
    };
};

IndexTTSApp.prototype.refreshSeedReportExport = function() {
    const section = document.getElementById('seed-report-section');
    const exportField = document.getElementById('seed-report-export');
    const meta = document.getElementById('seed-report-meta');

    if (!section || !exportField || !meta) {
        return;
    }

    const report = this.buildSeedReport();
    if (!report) {
        section.style.display = 'none';
        exportField.value = '';
        meta.textContent = 'No conversation selected.';
        return;
    }

    section.style.display = 'block';
    exportField.value = JSON.stringify(report, null, 2);

    const totalSeededVersions = report.lines.reduce((count, line) => {
        return count + line.versions.filter(version => version.seed !== null && version.seed !== undefined).length;
    }, 0);
    meta.textContent = `Strategy: ${report.seed_strategy} | Seeded versions: ${totalSeededVersions}`;
};

IndexTTSApp.prototype.copySeedReportExport = async function() {
    const exportField = document.getElementById('seed-report-export');
    const content = exportField?.value || '';

    if (!content.trim()) {
        this.showNotification('Warning', 'No seed report available yet', 'warning');
        return;
    }

    try {
        await navigator.clipboard.writeText(content);
        this.showNotification('Success', 'Copied seed report', 'success');
    } catch (error) {
        console.error('Failed to copy seed report:', error);
        this.showNotification('Error', 'Failed to copy seed report', 'error');
    }
};

IndexTTSApp.prototype.downloadSeedReportExport = function() {
    const content = document.getElementById('seed-report-export')?.value || '';
    if (!content.trim()) {
        this.showNotification('Warning', 'No seed report available yet', 'warning');
        return;
    }

    const conversationPrefix = (this.currentConversationId || 'conversation').substring(0, 8);
    const blob = new Blob([content], { type: 'application/json;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `seed-report-${conversationPrefix}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
};

IndexTTSApp.prototype.saveListeningReview = function(lineIndex, versionIndex) {
    const bucket = this.getCurrentConversationReviewBucket(true);
    if (!bucket) {
        this.showNotification('Error', 'No conversation selected', 'error');
        return;
    }

    const review = this.readListeningReviewForm(lineIndex, versionIndex);
    bucket[this.getListeningReviewStorageKey(lineIndex, versionIndex)] = review;
    this.saveListeningReviewsState();
    this.refreshListeningFeedbackExport();
    this.showNotification('Success', `Saved listening review for ${review.clip}`, 'success');
};

IndexTTSApp.prototype.clearListeningReview = function(lineIndex, versionIndex) {
    const bucket = this.getCurrentConversationReviewBucket(false);
    if (!bucket) {
        return;
    }

    delete bucket[this.getListeningReviewStorageKey(lineIndex, versionIndex)];
    if (Object.keys(bucket).length === 0) {
        delete this.listeningReviews[this.currentConversationId];
    }

    this.saveListeningReviewsState();
    this.renderLineVersions();
    this.refreshListeningFeedbackExport();
    this.showNotification('Success', 'Listening review cleared', 'success');
};

IndexTTSApp.prototype.copySingleListeningReview = async function(lineIndex, versionIndex) {
    const review = this.readListeningReviewForm(lineIndex, versionIndex);
    const block = this.buildListeningFeedbackBlock(review);

    try {
        await navigator.clipboard.writeText(block);
        this.showNotification('Success', `Copied ${review.clip}`, 'success');
    } catch (error) {
        console.error('Failed to copy listening review:', error);
        this.showNotification('Error', 'Failed to copy review block', 'error');
    }
};

IndexTTSApp.prototype.copyListeningFeedbackExport = async function() {
    const exportField = document.getElementById('listening-feedback-export');
    const content = exportField?.value || '';

    if (!content.trim()) {
        this.showNotification('Warning', 'No listening reviews saved yet', 'warning');
        return;
    }

    try {
        await navigator.clipboard.writeText(content);
        this.showNotification('Success', 'Copied listening review export', 'success');
    } catch (error) {
        console.error('Failed to copy listening review export:', error);
        this.showNotification('Error', 'Failed to copy listening review export', 'error');
    }
};

IndexTTSApp.prototype.downloadListeningFeedbackExport = function() {
    const content = document.getElementById('listening-feedback-export')?.value || '';
    if (!content.trim()) {
        this.showNotification('Warning', 'No listening reviews saved yet', 'warning');
        return;
    }

    const conversationPrefix = (this.currentConversationId || 'conversation').substring(0, 8);
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `listening-feedback-${conversationPrefix}.txt`;
    link.click();
    URL.revokeObjectURL(link.href);
};

IndexTTSApp.prototype.clearCurrentConversationReviews = function() {
    if (!this.currentConversationId || !this.listeningReviews?.[this.currentConversationId]) {
        this.showNotification('Warning', 'No listening reviews saved for this conversation', 'warning');
        return;
    }

    delete this.listeningReviews[this.currentConversationId];
    this.saveListeningReviewsState();
    this.renderLineVersions();
    this.refreshListeningFeedbackExport();
    this.showNotification('Success', 'Cleared listening reviews for this conversation', 'success');
};

IndexTTSApp.prototype.loadConversations = async function() {
    try {
        console.log('DEBUG: loadConversations called');
        const response = await this.apiRequest('/conversation/list');
        console.log('DEBUG: loadConversations response:', response);
        this.conversations = response.details.conversations;
        console.log('DEBUG: conversations loaded:', this.conversations);
        this.renderConversations();
        if (typeof this.refreshStudioShell === 'function') {
            this.refreshStudioShell();
        }
    } catch (error) {
        console.error('Failed to load conversations:', error);
    }
};

IndexTTSApp.prototype.renderConversations = function() {
    const container = document.getElementById('conversations-list');
    
    if (this.conversations.length === 0) {
        container.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-comments"></i>
                <p>No conversations found. Generate a conversation first.</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = '';
    
    this.conversations.forEach(conversation => {
        const conversationItem = document.createElement('div');
        conversationItem.className = 'conversation-item';
        conversationItem.dataset.conversationId = conversation.conversation_id;
        if (conversation.conversation_id === this.currentConversationId) {
            conversationItem.classList.add('selected');
        }
        
        const statusClass = conversation.status;
        const statusText = conversation.status.charAt(0).toUpperCase() + conversation.status.slice(1);
        
        conversationItem.innerHTML = `
            <div class="conversation-info">
                <h4>Conversation ${conversation.conversation_id.substring(0, 8)}</h4>
                <p>Status: ${statusText} | Progress: ${Math.round(conversation.progress)}%</p>
            </div>
            <div class="conversation-status">
                <span class="status-badge ${statusClass}">${statusText}</span>
            </div>
        `;

        conversationItem.addEventListener('click', () => {
            this.selectConversation(conversation.conversation_id);
        });
        
        container.appendChild(conversationItem);
    });
};

IndexTTSApp.prototype.selectConversation = async function(conversationId) {
    this.currentConversationId = conversationId;
    
    // Update UI selection
    document.querySelectorAll('.conversation-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    const selectedItem = document.querySelector(`.conversation-item[data-conversation-id="${conversationId}"]`);
    if (selectedItem) {
        selectedItem.classList.add('selected');
    }
    
    // Load conversation results
    await this.loadConversationResults(conversationId);
    if (typeof this.refreshStudioShell === 'function') {
        this.refreshStudioShell();
    }
};

IndexTTSApp.prototype.loadConversationResults = async function(conversationId) {
    try {
        console.log('DEBUG: loadConversationResults called with conversationId:', conversationId);
        const response = await this.apiRequest(`/conversation/results/${conversationId}`);
        console.log('DEBUG: loadConversationResults response:', response);
        const conversationData = response.details;
        console.log('DEBUG: conversationData:', conversationData);

        if (conversationData?.lines?.length) {
            conversationData.lines.forEach(line => {
                if (line.edited_text === undefined || line.edited_text === null) {
                    line.edited_text = line.text || '';
                }
                if (line.manual_similarity_threshold === undefined || line.manual_similarity_threshold === null) {
                    line.manual_similarity_threshold = parseFloat(document.getElementById('similarity-threshold')?.value || '0.60');
                }
                if (line.max_manual_attempts === undefined || line.max_manual_attempts === null) {
                    line.max_manual_attempts = parseInt(document.getElementById('auto-regen-attempts')?.value || '1', 10);
                }
            });
        }
        
        this.currentConversationData = conversationData;
        this.currentConversationExportFilename = conversationData?.concatenation_output_filename || this.currentConversationExportFilename;
        this.currentConversationExportPath = conversationData?.concatenation_output_path || this.currentConversationExportPath;
        this.applyConversationMixPacingSettings(conversationData?.generation_params || {});
        this.renderLineVersions();
        this.refreshListeningFeedbackExport();
        this.refreshSeedReportExport();
        if (typeof this.refreshStudioShell === 'function') {
            this.refreshStudioShell();
        }
        
    } catch (error) {
        console.error('DEBUG: loadConversationResults error:', error);
        this.showNotification('Error', error.message, 'error');
    }
};

IndexTTSApp.prototype.updateReviewLineDraft = function(lineIndex, updates = {}) {
    if (!this.currentConversationData?.lines?.[lineIndex]) return;
    Object.assign(this.currentConversationData.lines[lineIndex], updates);
};

IndexTTSApp.prototype.getConversationSelectionStatus = function() {
    if (!this.currentConversationData?.lines?.length) {
        return {
            totalLines: 0,
            selectedLines: 0,
            missingLines: [],
            multiSelectedLines: [],
            canExport: false,
        };
    }

    const summary = {
        totalLines: this.currentConversationData.lines.length,
        selectedLines: 0,
        missingLines: [],
        qualityWarningLines: [],
        multiSelectedLines: [],
        canExport: false,
    };

    const generationParams = this.currentConversationData?.generation_params || {};

    this.currentConversationData.lines.forEach((line, lineIndex) => {
        const selectedVersions = (line.versions || []).filter(version => Boolean(version.is_selected));
        const lineNumber = Number.isFinite(Number(line.line_number))
            ? Number(line.line_number) + 1
            : lineIndex + 1;

        if (selectedVersions.length === 1) {
            summary.selectedLines += 1;
            if (!conversationVersionMeetsQualityGate(selectedVersions[0], generationParams)) {
                summary.qualityWarningLines.push(lineNumber);
            }
        } else if (selectedVersions.length === 0) {
            summary.missingLines.push(lineNumber);
        } else {
            summary.multiSelectedLines.push(lineNumber);
        }
    });

    summary.canExport = (
        summary.totalLines > 0 &&
        summary.missingLines.length === 0 &&
        summary.multiSelectedLines.length === 0
    );
    return summary;
};

IndexTTSApp.prototype.buildSelectionReadinessMessage = function(selectionStatus) {
    if (!selectionStatus.totalLines) {
        return 'No lines are available for export yet.';
    }

    const issues = [];
    if (selectionStatus.missingLines.length) {
        issues.push(`choose a version for lines ${selectionStatus.missingLines.join(', ')}`);
    }
    if (selectionStatus.multiSelectedLines.length) {
        issues.push(`fix multiple selections on lines ${selectionStatus.multiSelectedLines.join(', ')}`);
    }

    if (!issues.length && selectionStatus.qualityWarningLines.length) {
        return `Ready to export, but review low-quality selections on lines ${selectionStatus.qualityWarningLines.join(', ')}.`;
    }

    if (!issues.length) {
        return `Ready to export. ${selectionStatus.selectedLines} of ${selectionStatus.totalLines} lines have a final selection.`;
    }

    return `Pick exactly one version for every line before export: ${issues.join('; ')}.`;
};

IndexTTSApp.prototype.syncSelectedVersionsForExport = async function() {
    if (!this.currentConversationId || !this.currentConversationData?.lines?.length) {
        return;
    }

    for (let lineIndex = 0; lineIndex < this.currentConversationData.lines.length; lineIndex += 1) {
        const line = this.currentConversationData.lines[lineIndex];
        const selectedVersionIndex = (line.versions || []).findIndex(version => Boolean(version.is_selected));

        if (selectedVersionIndex < 0) {
            continue;
        }

        await this.apiRequest(
            `/conversation/results/${this.currentConversationId}/line/${lineIndex}/select-version?version_index=${selectedVersionIndex}`,
            {
                method: 'POST',
            }
        );
    }
};

IndexTTSApp.prototype.renderLineVersions = function() {
    console.log('DEBUG: renderLineVersions called');
    console.log('DEBUG: currentConversationData:', this.currentConversationData);
    
    if (!this.currentConversationData) {
        console.log('DEBUG: No currentConversationData, returning');
        this.refreshSeedReportExport();
        return;
    }
    
    const container = document.getElementById('lines-container');
    const section = document.getElementById('line-versions-section');
    const concatenationSection = document.getElementById('concatenation-section');
    
    console.log('DEBUG: container element:', container);
    console.log('DEBUG: section element:', section);
    console.log('DEBUG: concatenationSection element:', concatenationSection);
    
    if (!container) {
        console.error('DEBUG: lines-container element not found');
        return;
    }
    
    section.style.display = 'block';
    concatenationSection.style.display = 'block';
    
    container.innerHTML = '';
    
    const lines = this.currentConversationData.lines;
    console.log('DEBUG: lines to render:', lines);
    console.log('DEBUG: number of lines:', lines.length);
    
    if (!lines || lines.length === 0) {
        console.log('DEBUG: No lines to render');
        container.innerHTML = '<p>No lines found in conversation</p>';
        this.refreshSeedReportExport();
        return;
    }
    
    lines.forEach((line, lineIndex) => {
        console.log(`DEBUG: Rendering line ${lineIndex}:`, line);
        
        const lineItem = document.createElement('div');
        lineItem.className = 'line-item';
        const generationParams = this.currentConversationData?.generation_params || {};
        const speakerLabel = escapeConversationResultsHtml((line.speaker_filename || 'Unknown').replace(/\.(wav|mp3)$/i, ''));
        const displayText = escapeConversationResultsHtml(line.text || '');
        const reviewText = escapeConversationResultsHtml(line.edited_text ?? line.text ?? '');
        const manualThreshold = Number.isFinite(Number(line.manual_similarity_threshold))
            ? Number(line.manual_similarity_threshold)
            : parseFloat(document.getElementById('similarity-threshold')?.value || '0.60');
        const maxManualAttempts = Number.isFinite(Number(line.max_manual_attempts))
            ? Number(line.max_manual_attempts)
            : parseInt(document.getElementById('auto-regen-attempts')?.value || '1', 10);
        const selectedVersionCount = (line.versions || []).filter(version => Boolean(version.is_selected)).length;
        const hasGatePassingVersion = (line.versions || []).some(version => (
            conversationVersionMeetsQualityGate(version, generationParams)
        ));
        const selectedVersion = (line.versions || []).find(version => Boolean(version.is_selected));
        const selectedVersionNeedsReview = selectedVersionCount === 1 && !conversationVersionMeetsQualityGate(selectedVersion, generationParams);
        const selectionState = selectedVersionCount === 1
            ? selectedVersionNeedsReview
                ? {
                    className: 'line-selection-quality-review',
                    label: 'Selected, review recommended',
                }
                : {
                    className: 'line-selection-ready',
                    label: 'Final version selected',
                }
            : selectedVersionCount === 0
                ? hasGatePassingVersion
                    ? {
                        className: 'line-selection-missing',
                        label: 'Selection required',
                    }
                    : {
                        className: 'line-selection-quality-review',
                        label: 'Needs quality review',
                    }
                : {
                    className: 'line-selection-invalid',
                    label: 'Only one version can be selected',
                };
        
        const versionsHtml = line.versions.map((version, versionIndex) => {
            const isSelected = version.is_selected;
            const qualityScore = Math.round(version.quality_score * 100);
            const similarityScore = Math.round(version.similarity_score * 100);
            const reviewScore = Math.round(getConversationReviewScore(version) * 100);
            const seedText = Number.isFinite(Number(version.seed)) ? `Seed ${Number(version.seed)}` : 'Seed n/a';
            const seedOriginText = version.seed_origin ? escapeConversationResultsHtml(String(version.seed_origin).replace(/_/g, ' ')) : '';
            const pacingScore = Number.isFinite(Number(version.pacing_score))
                ? Math.round(Number(version.pacing_score) * 100)
                : null;
            const pacingLabel = escapeConversationResultsHtml(formatPacingLabel(version.pacing_label || 'unknown'));
            const pacingNote = Array.isArray(version.pacing_notes) && version.pacing_notes.length
                ? escapeConversationResultsHtml(version.pacing_notes[0])
                : '';
            const durationText = Number.isFinite(Number(version.duration_seconds))
                ? `${Number(version.duration_seconds).toFixed(2)}s`
                : 'n/a';
            const meetsQualityGate = conversationVersionMeetsQualityGate(version, generationParams);
            const qualityGateReason = Array.isArray(version.quality_gate_failures) && version.quality_gate_failures.length
                ? escapeConversationResultsHtml(version.quality_gate_failures[0])
                : 'Below the automatic quality gate.';
            
            console.log(`DEBUG: Version ${versionIndex}:`, version);
            
            return `
                <div class="version-item ${isSelected ? 'selected' : ''} ${meetsQualityGate ? '' : 'quality-gate-failed'}"
                     onclick="app.selectVersion(${lineIndex}, ${versionIndex})">
                    <div class="version-header">
                        <span class="version-title">Version ${versionIndex + 1}</span>
                        <div class="version-badges">
                            <span class="score-badge quality">Review: ${reviewScore}%</span>
                            <span class="score-badge similarity">Similarity: ${similarityScore}%</span>
                            <span class="score-badge quality">Quality: ${qualityScore}%</span>
                            ${pacingScore !== null ? `<span class="score-badge similarity">Pacing: ${pacingScore}%</span>` : ''}
                            ${meetsQualityGate ? '' : `<span class="score-badge warning" title="${qualityGateReason}">Needs review</span>`}
                        </div>
                    </div>
                    <div class="version-seed">${escapeConversationResultsHtml(seedText)}${seedOriginText ? ` | ${seedOriginText}` : ''}</div>
                    <div class="version-seed">Duration ${escapeConversationResultsHtml(durationText)} | ${pacingLabel}${pacingNote ? ` | ${pacingNote}` : ''}${meetsQualityGate ? '' : ` | ${qualityGateReason}`}</div>
                    <div class="version-actions">
                        <button class="btn btn-secondary btn-small"
                                onclick="event.stopPropagation(); app.playVersionAudio('${version.audio_path}', '${version.audio_url || ''}')">
                            <i class="fas fa-play"></i> Play
                        </button>
                        <button class="btn btn-primary btn-small"
                                onclick="event.stopPropagation(); app.openLineInMediaPlayer(${lineIndex})">
                            <i class="fas fa-waveform"></i> Compare
                        </button>
                        <button class="btn btn-secondary btn-small"
                                onclick="event.stopPropagation(); app.downloadVersionAudio('${lineIndex}', '${versionIndex}', '${version.audio_filename}')">
                            <i class="fas fa-download"></i> Download
                        </button>
                    </div>
                    ${this.buildListeningReviewHtml(lineIndex, versionIndex)}
                </div>
            `;
        }).join('');
        
        lineItem.innerHTML = `
            <div class="line-header">
                <div class="line-info">
                    <h4>Line ${(line.line_number ?? lineIndex) + 1}: ${speakerLabel}</h4>
                    <p>"${displayText}"</p>
                </div>
                <div class="line-actions">
                    <span class="line-selection-status ${selectionState.className}">${selectionState.label}</span>
                    <button class="btn btn-secondary btn-small" onclick="app.autoSelectBestForLine(${lineIndex})">
                        <i class="fas fa-magic"></i> Auto-Select Best
                    </button>
                </div>
            </div>
            <div class="form-group">
                <label for="review-edit-${lineIndex}">Editable text for regeneration:</label>
                <textarea
                    id="review-edit-${lineIndex}"
                    class="review-edit-text"
                    data-line-index="${lineIndex}"
                    rows="3"
                >${reviewText}</textarea>
            </div>
            <div class="versions-grid">
                ${versionsHtml}
            </div>
        `;
        
        // Add regeneration controls
        const regenerationDiv = document.createElement('div');
        regenerationDiv.className = 'line-regeneration';
        regenerationDiv.innerHTML = `
            <div class="regeneration-controls">
                <div class="form-group">
                    <label for="regen-count-${lineIndex}">New versions to generate:</label>
                    <input type="number" class="regen-count-input" min="1" max="5" value="1" id="regen-count-${lineIndex}">
                </div>
                <div class="form-group">
                    <label for="manual-threshold-${lineIndex}">Manual similarity threshold:</label>
                    <input
                        type="number"
                        class="manual-threshold-input"
                        min="0"
                        max="1"
                        step="0.05"
                        value="${manualThreshold.toFixed(2)}"
                        id="manual-threshold-${lineIndex}"
                        data-line-index="${lineIndex}"
                    >
                </div>
                <div class="form-group">
                    <label for="manual-attempts-${lineIndex}">Max manual attempts:</label>
                    <input
                        type="number"
                        class="manual-attempts-input"
                        min="0"
                        max="10"
                        value="${Math.max(0, Math.round(maxManualAttempts))}"
                        id="manual-attempts-${lineIndex}"
                        data-line-index="${lineIndex}"
                    >
                </div>
                <button class="btn btn-secondary regen-action-btn regen-all-btn" data-line-index="${lineIndex}">
                    <i class="fas fa-redo"></i> Regenerate All
                </button>
                <button class="btn btn-secondary regen-action-btn regen-threshold-btn" data-line-index="${lineIndex}">
                    <i class="fas fa-filter"></i> Regenerate Below Threshold
                </button>
            </div>
            <div class="regeneration-progress" style="display: none;">
                <div class="progress-bar">
                    <div class="progress-fill"></div>
                </div>
                <div class="progress-text">Regenerating...</div>
            </div>
        `;
        lineItem.appendChild(regenerationDiv);
        
        container.appendChild(lineItem);
        console.log(`DEBUG: Added line ${lineIndex} to container`);
    });
    
    console.log('DEBUG: Finished rendering lines, updating selection counts');
    this.updateSelectionCounts();
    this.refreshListeningFeedbackExport();
    this.refreshSeedReportExport();
    
    // Add event listeners for regeneration buttons
    this.setupRegenerationEventListeners();
};

IndexTTSApp.prototype.selectVersion = function(lineIndex, versionIndex) {
    if (!this.currentConversationData) return;
    
    console.log(`DEBUG: selectVersion called with lineIndex: ${lineIndex}, versionIndex: ${versionIndex}`);
    
    const line = this.currentConversationData.lines[lineIndex];
    const versions = line.versions;
    
    // Toggle selection
    versions.forEach((version, index) => {
        const was_selected = version.is_selected;
        version.is_selected = (index === versionIndex);
        console.log(`DEBUG: Version ${index} is_selected changed from ${was_selected} to ${version.is_selected}`);
    });
    
    // Update UI
    this.renderLineVersions();
    
    // Debug: Log selection state after update
    console.log('DEBUG: Selection state after update:');
    this.currentConversationData.lines.forEach((line, i) => {
        line.versions.forEach((version, j) => {
            console.log(`DEBUG: Line ${i}, Version ${j} is_selected: ${version.is_selected}`);
        });
    });
};

IndexTTSApp.prototype.autoSelectBestVersions = function() {
    if (!this.currentConversationData) return;

    const warningLines = [];
    const generationParams = this.currentConversationData?.generation_params || {};

    this.currentConversationData.lines.forEach((line, lineIndex) => {
        const bestVersionIndex = getBestConversationVersionIndex(
            line,
            generationParams,
            { requireQualityGate: true },
        );

        line.versions.forEach((version, index) => {
            version.is_selected = (bestVersionIndex >= 0) && (index === bestVersionIndex);
        });

        if (bestVersionIndex >= 0 && !conversationVersionMeetsQualityGate(line.versions[bestVersionIndex], generationParams)) {
            const lineNumber = Number.isFinite(Number(line.line_number))
                ? Number(line.line_number) + 1
                : lineIndex + 1;
            warningLines.push(lineNumber);
        }
    });
    
    this.renderLineVersions();

    if (warningLines.length) {
        this.showNotification(
            'Warning',
            `Auto-selected the best available versions, but lines ${warningLines.join(', ')} still need review.`,
            'warning',
        );
        return;
    }

    this.showNotification('Success', 'Auto-selected the best gate-passing versions', 'success');
};

IndexTTSApp.prototype.autoSelectBestForLine = function(lineIndex) {
    if (!this.currentConversationData) return;
    
    const line = this.currentConversationData.lines[lineIndex];
    const generationParams = this.currentConversationData?.generation_params || {};
    const bestVersionIndex = getBestConversationVersionIndex(
        line,
        generationParams,
        { requireQualityGate: true },
    );

    line.versions.forEach((version, index) => {
        version.is_selected = (bestVersionIndex >= 0) && (index === bestVersionIndex);
    });
    
    this.renderLineVersions();

    if (bestVersionIndex >= 0 && !conversationVersionMeetsQualityGate(line.versions[bestVersionIndex], generationParams)) {
        const lineNumber = Number.isFinite(Number(line.line_number))
            ? Number(line.line_number) + 1
            : lineIndex + 1;
        this.showNotification(
            'Warning',
            `Line ${lineNumber} picked the best available version, but it still needs review.`,
            'warning',
        );
    }
};

IndexTTSApp.prototype.clearVersionSelections = function() {
    if (!this.currentConversationData) return;
    
    this.currentConversationData.lines.forEach(line => {
        line.versions.forEach(version => {
            version.is_selected = false;
        });
    });
    
    this.renderLineVersions();
};

IndexTTSApp.prototype.updateSelectionCounts = function() {
    if (!this.currentConversationData) return;

    const selectionStatus = this.getConversationSelectionStatus();
    const summaryElement = document.getElementById('selection-readiness-summary');
    const concatenateBtn = document.getElementById('concatenate-btn');
    const downloadSelectedBtn = document.getElementById('download-selected-btn');

    document.getElementById('selected-lines-count').textContent = selectionStatus.selectedLines;
    document.getElementById('total-lines-count').textContent = selectionStatus.totalLines;

    if (summaryElement) {
        summaryElement.textContent = this.buildSelectionReadinessMessage(selectionStatus);
        summaryElement.className = `selection-readiness-summary ${selectionStatus.canExport ? 'is-ready' : 'is-blocked'}`;
    }

    [concatenateBtn, downloadSelectedBtn].forEach((button) => {
        if (!button) return;
        button.disabled = !selectionStatus.canExport;
        button.title = selectionStatus.canExport
            ? 'Ready to export with one selected version per line'
            : this.buildSelectionReadinessMessage(selectionStatus);
    });
};

IndexTTSApp.prototype.loadOverlapPlanFile = function(event) {
    const file = event?.target?.files?.[0];
    const textarea = document.getElementById('overlap-plan-text');

    if (!file || !textarea) {
        return;
    }

    const reader = new FileReader();
    reader.onload = () => {
        textarea.value = String(reader.result || '');
        this.showNotification('Success', `Loaded overlap plan from ${file.name}`, 'success');
    };
    reader.onerror = () => {
        console.error('Failed to read overlap plan file:', reader.error);
        this.showNotification('Error', 'Failed to read overlap plan file', 'error');
    };
    reader.readAsText(file);
};

IndexTTSApp.prototype.getConversationMixSettings = function() {
    return {
        output_format: document.getElementById('concat-output-format')?.value || 'wav',
        output_bitrate_kbps: parseInt(document.getElementById('concat-output-bitrate')?.value || '192', 10),
        scene_pacing_profile: document.getElementById('concat-scene-pacing-profile')?.value || 'balanced',
        scene_gap_ms: parseInt(document.getElementById('concat-scene-gap-ms')?.value || '140', 10),
        respect_punctuation_pauses: document.getElementById('concat-respect-punctuation-pauses')?.checked ?? true,
        normalize_segments: document.getElementById('concat-normalize-segments')?.checked ?? true,
        target_level_dbfs: parseFloat(document.getElementById('concat-target-level-dbfs')?.value || '-19'),
        peak_limit_dbfs: parseFloat(document.getElementById('concat-peak-limit-dbfs')?.value || '-1'),
        normalize_final_mix: document.getElementById('concat-normalize-final-mix')?.checked ?? true,
        trim_leading_silence: document.getElementById('concat-trim-leading-silence')?.checked ?? true,
        trim_trailing_silence: document.getElementById('concat-trim-trailing-silence')?.checked ?? true,
        fade_in_ms: parseInt(document.getElementById('concat-fade-in-ms')?.value || '0', 10),
        fade_out_ms: parseInt(document.getElementById('concat-fade-out-ms')?.value || '60', 10),
    };
};

IndexTTSApp.prototype.applyConversationMixPacingSettings = function(settings = {}) {
    const scenePacingSelect = document.getElementById('concat-scene-pacing-profile');
    const sceneGapInput = document.getElementById('concat-scene-gap-ms');
    const respectPunctuationInput = document.getElementById('concat-respect-punctuation-pauses');
    this.currentConversationDialoguePacingPreset = settings.pacing_preset || document.getElementById('dialogue-pacing-preset')?.value || 'natural';

    const resolvedScenePacing = settings.scene_pacing_profile || document.getElementById('scene-pacing-profile')?.value || 'balanced';
    const resolvedSceneGap = settings.scene_gap_ms ?? document.getElementById('scene-gap-ms')?.value ?? 140;
    const resolvedRespectPunctuation = settings.respect_punctuation_pauses ?? (document.getElementById('respect-punctuation-pauses')?.checked ?? true);

    if (scenePacingSelect) {
        scenePacingSelect.value = resolvedScenePacing;
    }
    if (sceneGapInput) {
        sceneGapInput.value = resolvedSceneGap;
    }
    if (respectPunctuationInput) {
        respectPunctuationInput.checked = Boolean(resolvedRespectPunctuation);
    }

    this.updateConversationMixLabels();
};

IndexTTSApp.prototype.updateConversationMixFormatControls = function() {
    const formatSelect = document.getElementById('concat-output-format');
    const bitrateSelect = document.getElementById('concat-output-bitrate');
    const isMp3 = (formatSelect?.value || 'wav') === 'mp3';

    if (bitrateSelect) {
        bitrateSelect.disabled = !isMp3;
        bitrateSelect.title = isMp3 ? 'MP3 export bitrate' : 'Bitrate only applies to MP3 exports';
    }
};

IndexTTSApp.prototype.updateConversationMixLabels = function() {
    const scenePacingSelect = document.getElementById('concat-scene-pacing-profile');
    const sceneGapInput = document.getElementById('concat-scene-gap-ms');
    const sceneGapLabel = document.getElementById('concat-scene-gap-ms-value');
    const sceneHelp = document.getElementById('concat-scene-pacing-help');
    const targetInput = document.getElementById('concat-target-level-dbfs');
    const targetLabel = document.getElementById('concat-target-level-dbfs-value');
    const peakInput = document.getElementById('concat-peak-limit-dbfs');
    const peakLabel = document.getElementById('concat-peak-limit-dbfs-value');
    const fadeInInput = document.getElementById('concat-fade-in-ms');
    const fadeInLabel = document.getElementById('concat-fade-in-value');
    const fadeOutInput = document.getElementById('concat-fade-out-ms');
    const fadeOutLabel = document.getElementById('concat-fade-out-value');

    const scenePacingPresets = typeof this.getScenePacingPresets === 'function'
        ? this.getScenePacingPresets()
        : {};
    const selectedScenePreset = scenePacingPresets[scenePacingSelect?.value || 'balanced'];
    const dialoguePacingPresets = typeof this.getDialoguePacingPresets === 'function'
        ? this.getDialoguePacingPresets()
        : {};
    const currentDialoguePreset = dialoguePacingPresets[this.currentConversationDialoguePacingPreset || 'natural'];

    if (sceneGapInput && sceneGapLabel) {
        sceneGapLabel.textContent = `${parseInt(sceneGapInput.value || '140', 10)} ms`;
    }
    if (sceneHelp) {
        const presetLead = currentDialoguePreset?.label
            ? `${currentDialoguePreset.label} preset from generation. `
            : '';
        sceneHelp.textContent = selectedScenePreset?.helpText
            ? `${presetLead}${selectedScenePreset.helpText} Export defaults to the conversation's saved pacing, but you can tweak it here before building the full mix.`
            : `${presetLead}Conversation export starts from the pacing used during generation, but you can tweak it here before building the full mix.`;
    }

    if (targetInput && targetLabel) {
        targetLabel.textContent = `${parseFloat(targetInput.value || '-19').toFixed(1)} dBFS`;
    }
    if (peakInput && peakLabel) {
        peakLabel.textContent = `${parseFloat(peakInput.value || '-1').toFixed(1)} dBFS`;
    }
    if (fadeInInput && fadeInLabel) {
        fadeInLabel.textContent = `${parseInt(fadeInInput.value || '0', 10)} ms`;
    }
    if (fadeOutInput && fadeOutLabel) {
        fadeOutLabel.textContent = `${parseInt(fadeOutInput.value || '60', 10)} ms`;
    }

    this.updateConversationMixFormatControls();
};

IndexTTSApp.prototype.concatenateConversation = async function() {
    if (!this.currentConversationId) {
        this.showNotification('Error', 'No conversation selected', 'error');
        return;
    }

    const selectionStatus = this.getConversationSelectionStatus();
    if (!selectionStatus.canExport) {
        this.showNotification('Warning', this.buildSelectionReadinessMessage(selectionStatus), 'warning');
        return;
    }
    
    // Debug: Log selection state before concatenation
    console.log('DEBUG: concatenateConversation called');
    console.log('DEBUG: currentConversationData:', this.currentConversationData);
    
    if (this.currentConversationData && this.currentConversationData.lines) {
        console.log('DEBUG: Checking selection state for concatenation');
        this.currentConversationData.lines.forEach((line, lineIndex) => {
            console.log(`DEBUG: Line ${lineIndex}: ${line.speaker_filename}`);
            line.versions.forEach((version, versionIndex) => {
                console.log(`DEBUG: Version ${versionIndex} is_selected: ${version.is_selected}`);
            });
        });
    }
    
    const progressContainer = document.getElementById('concatenation-progress');
    const resultContainer = document.getElementById('concatenation-result');
    const button = document.getElementById('concatenate-btn');
    const overlapPlanText = document.getElementById('overlap-plan-text')?.value?.trim() || '';
    const mixSettings = this.getConversationMixSettings();
    
    try {
        this.lastConcatenationPlanApplied = false;
        this.lastConcatenationLevelingApplied = Boolean(mixSettings.normalize_segments);

        await this.syncSelectedVersionsForExport();

        // Show progress
        progressContainer.style.display = 'block';
        resultContainer.style.display = 'none';
        button.disabled = true;

        const requestOptions = {
            method: 'POST',
            body: JSON.stringify({
                overlap_plan_text: overlapPlanText || null,
                output_format: mixSettings.output_format,
                output_bitrate_kbps: mixSettings.output_bitrate_kbps,
                scene_pacing_profile: mixSettings.scene_pacing_profile,
                scene_gap_ms: mixSettings.scene_gap_ms,
                respect_punctuation_pauses: mixSettings.respect_punctuation_pauses,
                normalize_segments: mixSettings.normalize_segments,
                target_level_dbfs: mixSettings.target_level_dbfs,
                peak_limit_dbfs: mixSettings.peak_limit_dbfs,
                normalize_final_mix: mixSettings.normalize_final_mix,
                trim_leading_silence: mixSettings.trim_leading_silence,
                trim_trailing_silence: mixSettings.trim_trailing_silence,
                fade_in_ms: mixSettings.fade_in_ms,
                fade_out_ms: mixSettings.fade_out_ms,
            }),
        };

        const response = await this.apiRequest(
            `/conversation/results/${this.currentConversationId}/concatenate`,
            requestOptions
        );
        this.lastConcatenationPlanApplied = Boolean(response?.details?.overlap_plan_applied);
        this.lastConcatenationLevelingApplied = Boolean(response?.details?.segment_leveling_applied);
        this.currentConversationExportFilename = response?.details?.output_filename || this.currentConversationExportFilename;
        this.currentConversationExportPath = response?.details?.output_path || this.currentConversationExportPath;

        if (this.lastConcatenationPlanApplied || this.lastConcatenationLevelingApplied) {
            const appliedBits = [];
            if (this.lastConcatenationPlanApplied) {
                appliedBits.push('the overlap plan');
            }
            if (this.lastConcatenationLevelingApplied) {
                appliedBits.push('volume matching');
            }
            if (mixSettings.trim_leading_silence || mixSettings.trim_trailing_silence) {
                appliedBits.push('silence trim');
            }
            if ((mixSettings.fade_in_ms || 0) > 0 || (mixSettings.fade_out_ms || 0) > 0) {
                appliedBits.push('final fades');
            }
            this.showNotification('Success', `Concatenation started with ${appliedBits.join(' and ')}`, 'success');
        }
        
        // Wait a moment for concatenation to complete
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Check if concatenation completed
        await this.checkConcatenationStatus();
        
    } catch (error) {
        resultContainer.innerHTML = `
            <h4>Error</h4>
            <p>${error.message}</p>
        `;
        resultContainer.className = 'result-container error';
        resultContainer.style.display = 'block';
    } finally {
        progressContainer.style.display = 'none';
        button.disabled = false;
    }
};

IndexTTSApp.prototype.checkConcatenationStatus = async function() {
    try {
        console.log('DEBUG: checkConcatenationStatus called');
        const response = await this.apiRequest(`/conversation/status/${this.currentConversationId}`);
        const task = response.task;
        console.log('DEBUG: Task status response:', task);
        
        // Check if concatenation is completed in the result object
        if (task.result && task.result.concatenation_completed) {
            console.log('DEBUG: Concatenation completed, showing buttons');
            const resultContainer = document.getElementById('concatenation-result');
            const appliedBits = [];
            if (this.lastConcatenationPlanApplied) {
                appliedBits.push('the overlap plan');
            }
            if (this.lastConcatenationLevelingApplied) {
                appliedBits.push('volume matching');
            }
            const overlapMessage = appliedBits.length
                ? `<p>Conversation concatenated successfully using ${appliedBits.join(' and ')}.</p>`
                : '<p>Conversation concatenated successfully.</p>';
            resultContainer.innerHTML = `
                <h4>Success!</h4>
                ${overlapMessage}
                <div class="concatenated-audio-player">
                    <button class="btn btn-primary" onclick="app.playConcatenatedAudio()">
                        <i class="fas fa-play"></i> Play Full Conversation
                    </button>
                    <button class="btn btn-secondary" onclick="app.downloadConcatenatedAudio()">
                        <i class="fas fa-download"></i> Download Full Conversation
                    </button>
                </div>
            `;
            resultContainer.style.display = 'block';
            
            this.showNotification(
                'Success',
                appliedBits.length
                    ? `Conversation concatenated with ${appliedBits.join(' and ')}`
                    : 'Conversation concatenated successfully',
                'success'
            );
        } else if (task.result && task.result.concatenation_error) {
            console.log('DEBUG: Concatenation error:', task.result.concatenation_error);
            throw new Error(task.result.concatenation_error);
        } else {
            console.log('DEBUG: Concatenation not yet completed, checking again in 2 seconds');
            // Check again in a moment
            setTimeout(() => this.checkConcatenationStatus(), 2000);
        }
    } catch (error) {
        console.error('DEBUG: checkConcatenationStatus error:', error);
        const resultContainer = document.getElementById('concatenation-result');
        resultContainer.innerHTML = `
            <h4>Error</h4>
            <p>${error.message}</p>
        `;
        resultContainer.className = 'result-container error';
        resultContainer.style.display = 'block';
    }
};

IndexTTSApp.prototype.playConcatenatedAudio = async function() {
    if (!this.currentConversationId) return;
    
    try {
        console.log('DEBUG: playConcatenatedAudio called with conversationId:', this.currentConversationId);
        
        const audioUrl = `${this.apiBaseUrl}/conversation/results/${this.currentConversationId}/download`;
        console.log('DEBUG: Constructed concatenated audio URL:', audioUrl);
        
        const modal = document.getElementById('audio-player-modal');
        
        // Initialize custom media player if not already done
        const mediaPlayer = await this.initializeCustomMediaPlayer();
        
        // Create a simple line data object for the concatenated audio
        const lineData = {
            line_number: 0,
            speaker: 'Full Conversation',
            text: 'Concatenated conversation audio',
            versions: [{
                version: 0,
                audio_url: audioUrl,
                audio_filename: this.currentConversationExportFilename || `conversation_${this.currentConversationId.substring(0, 8)}.wav`,
                quality_score: 0.8, // Default quality score
                duration: 0,
                file_size: 0
            }]
        };
        
        // Load the audio into the custom media player
        await mediaPlayer.loadLine(lineData, { autoplay: true });
        
        modal.classList.add('show');
        
    } catch (error) {
        console.error('DEBUG: playConcatenatedAudio error:', error);
        this.showNotification('Error', 'Failed to play concatenated audio', 'error');
    }
};

IndexTTSApp.prototype.downloadConcatenatedAudio = async function() {
    if (!this.currentConversationId) return;
    
    try {
        console.log('DEBUG: downloadConcatenatedAudio called with conversationId:', this.currentConversationId);
        
        const downloadUrl = `${this.apiBaseUrl}/conversation/results/${this.currentConversationId}/download`;
        console.log('DEBUG: Constructed concatenated download URL:', downloadUrl);
        
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = this.currentConversationExportFilename || `conversation_${this.currentConversationId.substring(0, 8)}.wav`;
        
        // Add error handling for the download
        link.onerror = (e) => {
            console.error('DEBUG: Concatenated download link error:', e);
            this.showNotification('Error', 'Failed to download audio', 'error');
        };
        
        link.click();
        console.log('DEBUG: Concatenated download link clicked');
    } catch (error) {
        console.error('DEBUG: downloadConcatenatedAudio error:', error);
        this.showNotification('Error', 'Failed to download audio', 'error');
    }
};

IndexTTSApp.prototype.downloadSelectedVersions = async function() {
    if (!this.currentConversationData) {
        this.showNotification('Error', 'No conversation selected', 'error');
        return;
    }

    const selectionStatus = this.getConversationSelectionStatus();
    if (!selectionStatus.canExport) {
        this.showNotification('Warning', this.buildSelectionReadinessMessage(selectionStatus), 'warning');
        return;
    }
    
    let downloadCount = 0;
    
    for (let lineIndex = 0; lineIndex < this.currentConversationData.lines.length; lineIndex++) {
        const line = this.currentConversationData.lines[lineIndex];
        
        for (let versionIndex = 0; versionIndex < line.versions.length; versionIndex++) {
            const version = line.versions[versionIndex];
            
            if (version.is_selected) {
                try {
                    const link = document.createElement('a');
                    link.href = `${this.apiBaseUrl}/conversation/results/${this.currentConversationId}/line/${lineIndex}/version/${versionIndex}/download`;
                    link.download = version.audio_filename;
                    link.click();
                    downloadCount++;
                } catch (error) {
                    console.error('Failed to download version:', error);
                }
            }
        }
    }
    
    if (downloadCount > 0) {
        this.showNotification('Success', `Downloaded ${downloadCount} files`, 'success');
    } else {
        this.showNotification('Warning', 'No versions selected for download', 'warning');
    }
};

IndexTTSApp.prototype.playVersionAudio = async function(audioPath, audioUrl) {
    try {
        console.log('DEBUG: playVersionAudio called with:', { audioPath, audioUrl });
        
        const modal = document.getElementById('audio-player-modal');
        
        // Use the provided audio URL if available, otherwise use the path
        const srcUrl = audioUrl || `${this.apiBaseUrl}/assets/audio/${audioPath.split('/').pop()}`;
        
        console.log('DEBUG: Constructed audio URL:', srcUrl);
        
        // Initialize custom media player if not already done
        const mediaPlayer = await this.initializeCustomMediaPlayer();
        
        // Create a simple line data object for the media player
        const filename = audioPath.split('/').pop();
        const lineData = {
            line_number: 0,
            speaker: filename.replace('.wav', ''),
            text: 'Single audio file',
            versions: [{
                version: 0,
                audio_url: srcUrl,
                audio_filename: filename,
                quality_score: 0.8, // Default quality score
                duration: 0,
                file_size: 0
            }]
        };
        
        // Load the audio into the custom media player
        await mediaPlayer.loadLine(lineData, { autoplay: true });
        
        modal.classList.add('show');
        
    } catch (error) {
        console.error('DEBUG: playVersionAudio error:', error);
        this.showNotification('Error', 'Failed to play audio', 'error');
    }
};

IndexTTSApp.prototype.downloadVersionAudio = async function(lineIndex, versionIndex, filename) {
    try {
        console.log('DEBUG: downloadVersionAudio called with:', { lineIndex, versionIndex, filename, conversationId: this.currentConversationId });
        
        const downloadUrl = `${this.apiBaseUrl}/conversation/results/${this.currentConversationId}/line/${lineIndex}/version/${versionIndex}/download`;
        console.log('DEBUG: Constructed download URL:', downloadUrl);
        
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = filename;
        
        // Add error handling for the download
        link.onerror = (e) => {
            console.error('DEBUG: Download link error:', e);
            this.showNotification('Error', 'Failed to download audio', 'error');
        };
        
        link.click();
        console.log('DEBUG: Download link clicked');
    } catch (error) {
        console.error('DEBUG: downloadVersionAudio error:', error);
        this.showNotification('Error', 'Failed to download audio', 'error');
    }
};

IndexTTSApp.prototype.openLineInMediaPlayer = async function(lineIndex) {
    if (!this.currentConversationData || !this.currentConversationData.lines[lineIndex]) {
        this.showNotification('Error', 'Invalid line selected', 'error');
        return;
    }

    try {
        const modal = document.getElementById('audio-player-modal');
        
        // Initialize custom media player if not already done
        const mediaPlayer = await this.initializeCustomMediaPlayer();
        
        // Get the line data with all versions
        const lineData = this.currentConversationData.lines[lineIndex];
        
        // Load the line into the custom media player
        await mediaPlayer.loadLine(lineData);
        
        modal.classList.add('show');
        
    } catch (error) {
        console.error('DEBUG: openLineInMediaPlayer error:', error);
        this.showNotification('Error', 'Failed to open audio player', 'error');
    }
};

// Regeneration Functions
IndexTTSApp.prototype.setupRegenerationEventListeners = function() {
    document.querySelectorAll('.review-edit-text').forEach(input => {
        input.addEventListener('input', (event) => {
            const lineIndex = parseInt(event.target.dataset.lineIndex, 10);
            this.updateReviewLineDraft(lineIndex, { edited_text: event.target.value });
        });
    });

    document.querySelectorAll('.manual-threshold-input').forEach(input => {
        input.addEventListener('input', (event) => {
            const lineIndex = parseInt(event.target.dataset.lineIndex, 10);
            this.updateReviewLineDraft(lineIndex, {
                manual_similarity_threshold: parseFloat(event.target.value || '0.60')
            });
        });
    });

    document.querySelectorAll('.manual-attempts-input').forEach(input => {
        input.addEventListener('input', (event) => {
            const lineIndex = parseInt(event.target.dataset.lineIndex, 10);
            this.updateReviewLineDraft(lineIndex, {
                max_manual_attempts: parseInt(event.target.value || '1', 10)
            });
        });
    });

    document.querySelectorAll('.regen-all-btn').forEach(button => {
        button.addEventListener('click', (event) => {
            const lineIndex = parseInt(event.target.closest('.regen-all-btn').dataset.lineIndex, 10);
            this.regenerateLine(lineIndex, 'replace_all');
        });
    });

    document.querySelectorAll('.regen-threshold-btn').forEach(button => {
        button.addEventListener('click', (event) => {
            const lineIndex = parseInt(event.target.closest('.regen-threshold-btn').dataset.lineIndex, 10);
            this.regenerateLine(lineIndex, 'below_threshold');
        });
    });
};

IndexTTSApp.prototype.regenerateLine = async function(lineIndex, mode = 'replace_all') {
    if (!this.currentConversationId) {
        this.showNotification('Error', 'No conversation selected', 'error');
        return;
    }

    if (this.regenerationInterval) {
        this.showNotification('Warning', 'Wait for the current regeneration to finish first', 'warning');
        return;
    }

    const lineItem = document.querySelectorAll('.line-item')[lineIndex];
    const regenProgress = lineItem.querySelector('.regeneration-progress');
    const progressFill = regenProgress.querySelector('.progress-fill');
    const progressText = regenProgress.querySelector('.progress-text');
    const regenButtons = lineItem.querySelectorAll('.regen-action-btn');
    const editedText = (document.getElementById(`review-edit-${lineIndex}`)?.value || '').trim();
    const regenCount = parseInt(document.getElementById(`regen-count-${lineIndex}`)?.value || '1', 10);
    const manualThreshold = parseFloat(document.getElementById(`manual-threshold-${lineIndex}`)?.value || document.getElementById('similarity-threshold')?.value || '0.60');
    const maxManualAttempts = parseInt(document.getElementById(`manual-attempts-${lineIndex}`)?.value || document.getElementById('auto-regen-attempts')?.value || '1', 10);

    if (!editedText) {
        this.showNotification('Error', 'Editable text for regeneration cannot be empty', 'error');
        return;
    }

    this.updateReviewLineDraft(lineIndex, {
        edited_text: editedText,
        manual_similarity_threshold: manualThreshold,
        max_manual_attempts: maxManualAttempts
    });

    try {
        // Show progress
        regenProgress.style.display = 'block';
        regenButtons.forEach(button => {
            button.disabled = true;
        });
        progressFill.style.width = '0%';
        progressText.textContent = mode === 'below_threshold'
            ? 'Starting threshold regeneration...'
            : 'Starting regeneration...';

        // Start regeneration
        await this.apiRequest(`/conversation/results/${this.currentConversationId}/line/${lineIndex}/regenerate`, {
            method: 'POST',
            body: JSON.stringify({
                regen_count: regenCount,
                mode,
                edited_text: editedText,
                manual_similarity_threshold: mode === 'below_threshold' ? manualThreshold : undefined,
                max_manual_attempts: mode === 'below_threshold' ? maxManualAttempts : undefined
            })
        });

        // Start polling for regeneration progress
        this.activeRegenerationLineIndex = lineIndex;
        this.regenerationInterval = setInterval(() => {
            this.checkRegenerationProgress(lineIndex);
        }, 2000);

    } catch (error) {
        this.showNotification('Error', error.message, 'error');
        regenProgress.style.display = 'none';
        regenButtons.forEach(button => {
            button.disabled = false;
        });
    }
};

IndexTTSApp.prototype.checkRegenerationProgress = async function(lineIndex) {
    if (!this.currentConversationId) return;

    try {
        console.log('DEBUG: Checking regeneration progress for line', lineIndex);
        const response = await this.apiRequest(`/conversation/results/${this.currentConversationId}/line/${lineIndex}/regenerate/status`);
        console.log('DEBUG: Regeneration status response:', response);
        
        // Handle different response structures
        let task;
        if (response.task) {
            task = response.task;
        } else if (response.details) {
            task = response.details;
        } else if (response.status && response.progress_percent !== undefined) {
            task = response;
        } else {
            console.error('DEBUG: No task or details property in response:', response);
            throw new Error('Invalid response structure: missing task or details property');
        }
        
        console.log('DEBUG: Task object:', task);

        const lineItem = document.querySelectorAll('.line-item')[lineIndex];
        const regenProgress = lineItem.querySelector('.regeneration-progress');
        const regenButtons = lineItem.querySelectorAll('.regen-action-btn');
        const progressFill = regenProgress.querySelector('.progress-fill');
        const progressText = regenProgress.querySelector('.progress-text');

        // Update progress with safe property access
        const progressPercent = task.progress_percent !== undefined ? task.progress_percent : 0;
        const statusMessage = task.status_message || task.status || 'Regenerating...';
        
        progressFill.style.width = `${progressPercent}%`;
        progressText.textContent = statusMessage;

        // Check if completed
        if (task.status === 'completed') {
            console.log('DEBUG: Regeneration completed for line', lineIndex);
            clearInterval(this.regenerationInterval);
            this.regenerationInterval = null;
            this.activeRegenerationLineIndex = null;

            regenProgress.style.display = 'none';
            regenButtons.forEach(button => {
                button.disabled = false;
            });

            this.showNotification('Success', 'Line regeneration completed!', 'success');

            // Reload conversation results to show new versions
            await this.loadConversationResults(this.currentConversationId);
        } else if (task.status === 'failed') {
            console.log('DEBUG: Regeneration failed for line', lineIndex);
            clearInterval(this.regenerationInterval);
            this.regenerationInterval = null;
            this.activeRegenerationLineIndex = null;

            regenProgress.style.display = 'none';
            regenButtons.forEach(button => {
                button.disabled = false;
            });

            throw new Error(task.error || 'Regeneration failed');
        }

    } catch (error) {
        console.error('DEBUG: Regeneration progress error:', error);
        clearInterval(this.regenerationInterval);
        this.regenerationInterval = null;
        this.activeRegenerationLineIndex = null;

        const lineItem = document.querySelectorAll('.line-item')[lineIndex];
        const regenProgress = lineItem.querySelector('.regeneration-progress');
        const regenButtons = lineItem.querySelectorAll('.regen-action-btn');

        if (regenProgress) regenProgress.style.display = 'none';
        if (regenButtons.length) {
            regenButtons.forEach(button => {
                button.disabled = false;
            });
        }

        this.showNotification('Error', error.message, 'error');
    }
};
