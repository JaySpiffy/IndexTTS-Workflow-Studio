// IndexTTS2 Frontend Application - Main Entry Point (Fixed Version)
// This file includes proper error handling and debugging

// Debug logging function
function debugLog(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    console.log(`[${type.toUpperCase()}] ${message}`);
    
    // Also log to a debug panel if it exists
    const debugPanel = document.getElementById('debug-log');
    if (debugPanel) {
        const logEntry = document.createElement('div');
        logEntry.className = type;
        logEntry.textContent = `[${timestamp}] ${message}`;
        debugPanel.appendChild(logEntry);
        debugPanel.scrollTop = debugPanel.scrollHeight;
    }
}

// Initialize the application with proper error handling
async function initializeApp() {
    try {
        debugLog('Starting app initialization...', 'info');
        
        // Import all modules with error handling
        debugLog('Importing modules...', 'info');
        
        const modules = await Promise.all([
            import('./modules/core.js').catch(error => {
                debugLog(`Failed to import core.js: ${error.message}`, 'error');
                throw error;
            }),
            import('./modules/eventListeners.js').catch(error => {
                debugLog(`Failed to import eventListeners.js: ${error.message}`, 'error');
                throw error;
            }),
            import('./modules/api.js').catch(error => {
                debugLog(`Failed to import api.js: ${error.message}`, 'error');
                throw error;
            }),
            import('./modules/conversationWorkflow.js').catch(error => {
                debugLog(`Failed to import conversationWorkflow.js: ${error.message}`, 'error');
                throw error;
            }),
            import('./modules/conversationResults.js').catch(error => {
                debugLog(`Failed to import conversationResults.js: ${error.message}`, 'error');
                throw error;
            }),
            import('./modules/uiUtils.js').catch(error => {
                debugLog(`Failed to import uiUtils.js: ${error.message}`, 'error');
                throw error;
            })
        ]);
        
        debugLog('All modules imported successfully', 'success');
        
        // Check if IndexTTSApp class is available
        if (typeof IndexTTSApp === 'undefined') {
            throw new Error('IndexTTSApp class is not defined after importing modules');
        }
        
        debugLog('IndexTTSApp class is available', 'success');
        
        // Wait for DOM to be ready
        if (document.readyState !== 'complete') {
            debugLog('Waiting for DOM to be fully loaded...', 'info');
            await new Promise(resolve => {
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', resolve, { once: true });
                } else {
                    resolve();
                }
            });
        }
        
        debugLog('DOM is ready', 'success');
        
        // Check if required DOM elements exist
        const requiredElements = [
            'conversation-workflow',
            'conversation-results',
            'parse-script-btn',
            'generate-conversation-btn'
        ];
        
        for (const elementId of requiredElements) {
            const element = document.getElementById(elementId);
            if (!element) {
                debugLog(`Required element missing: ${elementId}`, 'error');
            } else {
                debugLog(`Required element found: ${elementId}`, 'success');
            }
        }
        
        // Create the app instance
        debugLog('Creating IndexTTSApp instance...', 'info');
        window.app = new IndexTTSApp();
        debugLog('IndexTTSApp instance created successfully', 'success');
        
        // Test basic functionality
        setTimeout(() => {
            if (window.app && typeof window.app.switchTab === 'function') {
                debugLog('switchTab method is available', 'success');
            } else {
                debugLog('switchTab method is not available', 'error');
            }
            
            if (window.app && typeof window.app.parseScript === 'function') {
                debugLog('parseScript method is available', 'success');
            } else {
                debugLog('parseScript method is not available', 'error');
            }
            
            if (window.app && typeof window.app.generateConversation === 'function') {
                debugLog('generateConversation method is available', 'success');
            } else {
                debugLog('generateConversation method is not available', 'error');
            }
        }, 1000);
        
        // Add speaker upload event listener if the element exists
        const speakerFileInput = document.getElementById('speaker-file-input');
        if (speakerFileInput) {
            speakerFileInput.addEventListener('change', () => {
                if (speakerFileInput.files.length > 0) {
                    if (window.app && typeof window.app.uploadSpeaker === 'function') {
                        window.app.uploadSpeaker();
                    } else {
                        debugLog('uploadSpeaker method is not available', 'error');
                    }
                }
            });
            debugLog('Speaker upload event listener added', 'success');
        } else {
            debugLog('Speaker file input not found', 'warning');
        }
        
        debugLog('App initialization completed successfully', 'success');
        
    } catch (error) {
        debugLog(`App initialization failed: ${error.message}`, 'error');
        debugLog(`Stack trace: ${error.stack}`, 'error');
        
        // Show error message to user
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #f8d7da;
            color: #721c24;
            padding: 20px;
            border-radius: 5px;
            border: 1px solid #f5c6cb;
            z-index: 10000;
            max-width: 500px;
            text-align: center;
        `;
        errorDiv.innerHTML = `
            <h3>Application Error</h3>
            <p>Failed to initialize the IndexTTS2 application:</p>
            <p><strong>${error.message}</strong></p>
            <p>Please check the browser console for more details.</p>
            <button onclick="this.parentElement.remove()" style="
                background: #721c24;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                margin-top: 10px;
            ">Close</button>
        `;
        document.body.appendChild(errorDiv);
    }
}

// Initialize the app when the page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// Export for debugging
if (typeof window !== 'undefined') {
    window.IndexTTSAppDebug = {
        initializeApp,
        debugLog
    };
}