// API Service for IndexTTS Frontend
class ApiService {
    constructor() {
        this.apiBaseUrl = window.location.origin + '/api';
    }

    // Generic API request method
    async apiRequest(endpoint, options = {}) {
        const url = `${this.apiBaseUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        try {
            const response = await fetch(url, { ...defaultOptions, ...options });
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || data.message || `HTTP error! status: ${response.status}`);
            }

            return data;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    // Emotion Timeline API Methods
    
    // Add emotion keyframe to a segment
    async addEmotionKeyframe(segmentId, keyframeData) {
        return this.apiRequest(`/emotion-timeline/segments/${segmentId}/keyframes`, {
            method: 'POST',
            body: JSON.stringify(keyframeData)
        });
    }

    // Update emotion keyframe
    async updateEmotionKeyframe(keyframeId, keyframeData) {
        return this.apiRequest(`/emotion-timeline/keyframes/${keyframeId}`, {
            method: 'PUT',
            body: JSON.stringify(keyframeData)
        });
    }

    // Delete emotion keyframe
    async deleteEmotionKeyframe(keyframeId) {
        return this.apiRequest(`/emotion-timeline/keyframes/${keyframeId}`, {
            method: 'DELETE'
        });
    }

    // Get keyframes for a segment
    async getSegmentKeyframes(segmentId) {
        return this.apiRequest(`/emotion-timeline/segments/${segmentId}/keyframes`);
    }

    // Generate emotion preview for a segment
    async generateEmotionPreview(segmentId, previewData) {
        return this.apiRequest(`/emotion-timeline/segments/${segmentId}/preview`, {
            method: 'POST',
            body: JSON.stringify(previewData)
        });
    }

    // Get emotion timeline data for a segment
    async getSegmentTimeline(segmentId) {
        return this.apiRequest(`/emotion-timeline/segments/${segmentId}/timeline`);
    }

    // Update segment emotion settings
    async updateSegmentEmotionSettings(segmentId, settings) {
        return this.apiRequest(`/emotion-timeline/segments/${segmentId}/settings`, {
            method: 'PUT',
            body: JSON.stringify(settings)
        });
    }

    // Calculate emotion at specific timestamp
    async calculateEmotionAtTimestamp(segmentId, timestamp) {
        return this.apiRequest(`/emotion-timeline/segments/${segmentId}/emotion-at-time?timestamp=${timestamp}`);
    }

    // Timeline Management API Methods
    
    // Create a new timeline project
    async createTimelineProject(projectData) {
        return this.apiRequest('/timeline/projects', {
            method: 'POST',
            body: JSON.stringify(projectData)
        });
    }

    // Get all timeline projects
    async getTimelineProjects() {
        return this.apiRequest('/timeline/projects');
    }

    // Get a specific timeline project
    async getTimelineProject(projectId) {
        return this.apiRequest(`/timeline/projects/${projectId}`);
    }

    // Update timeline project
    async updateTimelineProject(projectId, projectData) {
        return this.apiRequest(`/timeline/projects/${projectId}`, {
            method: 'PUT',
            body: JSON.stringify(projectData)
        });
    }

    // Delete timeline project
    async deleteTimelineProject(projectId) {
        return this.apiRequest(`/timeline/projects/${projectId}`, {
            method: 'DELETE'
        });
    }

    // Add segment to timeline
    async addTimelineSegment(projectId, segmentData) {
        return this.apiRequest(`/timeline/projects/${projectId}/segments`, {
            method: 'POST',
            body: JSON.stringify(segmentData)
        });
    }

    // Update timeline segment
    async updateTimelineSegment(projectId, segmentId, segmentData) {
        return this.apiRequest(`/timeline/projects/${projectId}/segments/${segmentId}`, {
            method: 'PUT',
            body: JSON.stringify(segmentData)
        });
    }

    // Delete timeline segment
    async deleteTimelineSegment(projectId, segmentId) {
        return this.apiRequest(`/timeline/projects/${projectId}/segments/${segmentId}`, {
            method: 'DELETE'
        });
    }

    // Generate audio for timeline segment
    async generateSegmentAudio(projectId, segmentId, generationOptions) {
        return this.apiRequest(`/timeline/projects/${projectId}/segments/${segmentId}/generate`, {
            method: 'POST',
            body: JSON.stringify(generationOptions)
        });
    }

    // Get segment generation status
    async getSegmentGenerationStatus(projectId, segmentId, taskId) {
        return this.apiRequest(`/timeline/projects/${projectId}/segments/${segmentId}/generate/status?task_id=${taskId}`);
    }

    // Speaker API Methods
    
    // Get all speakers
    async getSpeakers() {
        return this.apiRequest('/speakers/');
    }

    // Upload speaker
    async uploadSpeaker(formData) {
        return this.apiRequest('/speakers/upload', {
            method: 'POST',
            body: formData,
            headers: {} // Let browser set content-type for FormData
        });
    }

    // Delete speaker
    async deleteSpeaker(filename) {
        return this.apiRequest(`/speakers/${filename}`, {
            method: 'DELETE'
        });
    }

    // Get speaker audio URL
    getSpeakerAudioUrl(filename) {
        return `${this.apiBaseUrl}/speakers/${filename}/audio`;
    }

    // Conversation API Methods
    
    // Generate conversation
    async generateConversation(conversationData) {
        return this.apiRequest('/conversation/generate', {
            method: 'POST',
            body: JSON.stringify(conversationData)
        });
    }

    // Get conversation status
    async getConversationStatus(conversationId) {
        return this.apiRequest(`/conversation/status/${conversationId}`);
    }

    // Stop conversation generation
    async stopConversationGeneration(conversationId) {
        return this.apiRequest(`/conversation/stop/${conversationId}`, {
            method: 'POST'
        });
    }

    // Get conversation list
    async getConversationList() {
        return this.apiRequest('/conversation/list');
    }

    // Get conversation results
    async getConversationResults(conversationId) {
        return this.apiRequest(`/conversation/results/${conversationId}`);
    }

    // Regenerate conversation line
    async regenerateConversationLine(conversationId, lineIndex, regenOptions) {
        return this.apiRequest(`/conversation/results/${conversationId}/line/${lineIndex}/regenerate`, {
            method: 'POST',
            body: JSON.stringify(regenOptions)
        });
    }

    // Get line regeneration status
    async getLineRegenerationStatus(conversationId, lineIndex) {
        return this.apiRequest(`/conversation/results/${conversationId}/line/${lineIndex}/regenerate/status`);
    }

    // Concatenate conversation
    async concatenateConversation(conversationId) {
        return this.apiRequest(`/conversation/results/${conversationId}/concatenate`, {
            method: 'POST'
        });
    }

    // Get conversation download URL
    getConversationDownloadUrl(conversationId) {
        return `${this.apiBaseUrl}/conversation/results/${conversationId}/download`;
    }

    // Get line version download URL
    getLineVersionDownloadUrl(conversationId, lineIndex, versionIndex) {
        return `${this.apiBaseUrl}/conversation/results/${conversationId}/line/${lineIndex}/version/${versionIndex}/download`;
    }

    // Utility API Methods
    
    // Check API health
    async checkApiHealth() {
        return this.apiRequest('/health');
    }

    // Get system info
    async getSystemInfo() {
        return this.apiRequest('/system/info');
    }
}

// Create and export a singleton instance
const apiService = new ApiService();
export default apiService;