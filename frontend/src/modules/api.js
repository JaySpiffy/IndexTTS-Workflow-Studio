// IndexTTS2 API Communication Module

// Check if IndexTTSApp is available before trying to extend it
if (typeof IndexTTSApp === 'undefined') {
    console.error('IndexTTSApp is not defined! This is the source of the error.');
}

IndexTTSApp.prototype.apiRequest = async function(endpoint, options = {}) {
    const url = `${this.apiBaseUrl}${endpoint}`;
    const { suppressErrorNotification = false, ...fetchOptions } = options;
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    try {
        console.log('🔍 DEBUG: Making API request to:', url);
        console.log('🔍 DEBUG: Request options:', options);
        
        const response = await fetch(url, { ...defaultOptions, ...fetchOptions });
        const contentType = response.headers.get('content-type') || '';
        const rawBody = await response.text();
        let data = null;

        if (rawBody) {
            if (contentType.includes('application/json')) {
                try {
                    data = JSON.parse(rawBody);
                } catch (parseError) {
                    console.error('❌ Failed to parse JSON response:', parseError, rawBody);
                    throw new Error(`Backend returned invalid JSON (${response.status} ${response.statusText})`);
                }
            } else {
                data = { raw: rawBody };
            }
        }
        
        console.log('🔍 DEBUG: API response status:', response.status);
        console.log('🔍 DEBUG: API response data:', data);

        if (!response.ok) {
            // Enhanced error message parsing
            let errorMessage = `HTTP error! status: ${response.status}`;
            
            if (data && data.detail) {
                if (typeof data.detail === 'string') {
                    errorMessage = data.detail;
                } else if (Array.isArray(data.detail)) {
                    errorMessage = data.detail.map(detail => detail.msg || JSON.stringify(detail)).join('; ');
                } else {
                    errorMessage = JSON.stringify(data.detail);
                }
            } else if (data && data.message) {
                errorMessage = data.message;
            } else if (data && data.error) {
                errorMessage = `Error: ${data.error}`;
            } else if (data && data.details && data.details.error_message) {
                errorMessage = data.details.error_message;
            } else if (response.status === 502 || response.status === 503 || response.status === 504) {
                errorMessage = 'Backend is starting or temporarily unavailable. Please wait a moment and try again.';
            } else if (data && typeof data === 'object' && !data.raw) {
                // Handle cases where error details are nested
                try {
                    errorMessage = JSON.stringify(data, null, 2);
                } catch (stringifyError) {
                    errorMessage = 'Unknown error occurred';
                }
            } else if (rawBody) {
                errorMessage = `${response.status} ${response.statusText}`;
            }
            
            console.error('❌ API Error Details:', {
                status: response.status,
                data: data,
                message: errorMessage
            });
            
            throw new Error(errorMessage);
        }

        if (contentType.includes('application/json')) {
            return data ?? {};
        }

        if (!rawBody) {
            return {};
        }

        throw new Error(`Expected JSON from API but received ${contentType || 'non-JSON content'}`);
    } catch (error) {
        console.error('❌ API request failed:', error);
        console.error('❌ Error details:', {
            message: error.message,
            stack: error.stack
        });
        
        if (!suppressErrorNotification && this.showNotification) {
            this.showNotification('API Error', error.message, 'error');
        }
        throw error;
    }
};

IndexTTSApp.prototype.checkApiStatus = async function() {
    try {
        const response = await this.apiRequest('/health', {
            suppressErrorNotification: true
        });
        const runtimeDevice = response.runtime_device;
        const requestedDevice = response.requested_device || 'auto';
        const usingDeepSpeed = Boolean(response.using_deepspeed);
        let statusMessage = 'API Connected';

        if (runtimeDevice) {
            let runtimeLabel = response.using_gpu ? `GPU: ${runtimeDevice}` : `CPU fallback: ${runtimeDevice}`;
            if (response.using_gpu && usingDeepSpeed) {
                runtimeLabel += ' + DeepSpeed';
            }
            statusMessage = `API Connected - ${runtimeLabel}`;
        } else if (response.model_loaded === false) {
            statusMessage = `API Starting - Requested: ${requestedDevice}`;
        }

        this.updateApiStatus(true, statusMessage);
    } catch (error) {
        this.updateApiStatus(false, 'API Disconnected');
    }
};

IndexTTSApp.prototype.updateApiStatus = function(connected, message) {
    const statusIndicator = document.getElementById('api-status');
    const statusDot = statusIndicator.querySelector('.status-dot');
    const statusText = statusIndicator.querySelector('.status-text');

    if (connected) {
        statusDot.classList.add('connected');
        statusDot.classList.remove('error');
        statusText.textContent = message;
    } else {
        statusDot.classList.remove('connected');
        statusDot.classList.add('error');
        statusText.textContent = message;
    }
};

// Emotion Detection API Methods
IndexTTSApp.prototype.detectEmotionFromText = async function(text, batchTexts = null) {
    try {
        console.log('🔍 DEBUG: detectEmotionFromText called with:');
        console.log('  - text:', text);
        console.log('  - batchTexts:', batchTexts);
        
        let requestBody;
        
        if (batchTexts && batchTexts.length > 0) {
            // Pure batch processing - only send batch_texts
            requestBody = {
                batch_texts: batchTexts
            };
            console.log('🔍 DEBUG: Pure batch request body:', requestBody);
        } else {
            // Single text processing - only send text
            requestBody = {
                text: text
            };
            console.log('🔍 DEBUG: Single text request body:', requestBody);
        }
        
        console.log('🔍 DEBUG: Final request body:', requestBody);
        
        const response = await this.apiRequest('/emotion-detection/detect', {
            method: 'POST',
            body: JSON.stringify(requestBody),
            suppressErrorNotification: true
        });
        
        console.log('🔍 DEBUG: Emotion detection response:', response);
        return response.details;
    } catch (error) {
        console.error('❌ Emotion detection failed:', error);
        console.error('❌ Error details:', error.message);
        throw error;
    }
};

IndexTTSApp.prototype.getSupportedEmotions = async function() {
    try {
        const response = await this.apiRequest('/emotion-detection/emotions');
        return response.details;
    } catch (error) {
        console.error('Failed to get supported emotions:', error);
        throw error;
    }
};

IndexTTSApp.prototype.detectEmotionsForScript = async function(scriptLines) {
    try {
        if (!scriptLines || scriptLines.length === 0) {
            return [];
        }
        
        // Extract texts from script lines for batch processing
        const texts = scriptLines.map(line => line.text);
        
        // Detect emotions for all texts in batch
        const response = await this.detectEmotionFromText(texts[0], texts.slice(1));
        
        // Process results and match with original lines
        const emotionResults = [];
        
        if (response.results && response.results.length > 0) {
            // Batch processing result
            response.results.forEach((result, index) => {
                if (index < scriptLines.length) {
                    emotionResults.push({
                        details: result,  // Keep the full result structure
                        text: result.text,
                        emotion_vectors: result.emotion_vectors,
                        emotion_dict: result.emotion_dict
                    });
                }
            });
        } else {
            // Single processing result
            emotionResults.push({
                details: response,  // Keep the full result structure
                text: response.text,
                emotion_vectors: response.emotion_vectors,
                emotion_dict: response.emotion_dict
            });
        }
        
        return emotionResults;
    } catch (error) {
        console.error('Script emotion detection failed:', error);
        throw error;
    }
};
