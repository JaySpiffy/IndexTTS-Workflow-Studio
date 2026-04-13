// IndexTTS2 Frontend Application - Main Entry Point
// Import all modules with proper error handling
const FRONTEND_BUILD_VERSION = '2026-04-12-studio-shell-4';

// Initialize the app with proper module loading
async function initializeApp() {
    try {
        // Import core module first to ensure IndexTTSApp is defined
        await import(`./modules/core.js?v=${FRONTEND_BUILD_VERSION}`);
        
        // Verify IndexTTSApp is available before importing other modules
        if (typeof IndexTTSApp === 'undefined') {
            throw new Error('IndexTTSApp is not defined after importing core.js');
        }
        
        // Now import the other modules that extend IndexTTSApp
        await Promise.all([
            import(`./modules/eventListeners.js?v=${FRONTEND_BUILD_VERSION}`),
            import(`./modules/api.js?v=${FRONTEND_BUILD_VERSION}`),
            import(`./modules/conversationWorkflow.js?v=${FRONTEND_BUILD_VERSION}`),
            import(`./modules/conversationResults.js?v=${FRONTEND_BUILD_VERSION}`),
            import(`./modules/speakerPrep.js?v=${FRONTEND_BUILD_VERSION}`),
            import(`./modules/timelineEditor.js?v=${FRONTEND_BUILD_VERSION}`),
            import(`./modules/webmcp.js?v=${FRONTEND_BUILD_VERSION}`),
            import(`./modules/uiUtils.js?v=${FRONTEND_BUILD_VERSION}`)
        ]);
        
        // Wait for DOM to be ready
        if (document.readyState !== 'complete') {
            await new Promise(resolve => {
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', resolve, { once: true });
                } else {
                    resolve();
                }
            });
        }
        
        // Initialize the app
        if (typeof IndexTTSApp !== 'undefined') {
            window.app = new IndexTTSApp();

            // Add speaker upload event listener if the element exists
            const speakerFileInput = document.getElementById('speaker-file-input');
            if (speakerFileInput) {
                speakerFileInput.addEventListener('change', () => {
                    if (speakerFileInput.files.length > 0) {
                        app.uploadSpeaker();
                    }
                });
            }
        } else {
            console.error('IndexTTSApp class is not defined');
        }
    } catch (error) {
        console.error('Failed to initialize app:', error);
    }
}

// Initialize the app when the page loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}
