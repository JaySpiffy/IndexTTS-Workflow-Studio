import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const APP_URL = process.env.INDEXTTS_APP_URL || 'http://localhost:3000';
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const OUTPUT_DIR = path.join(REPO_ROOT, 'docs', 'assets', 'manual');
const VIEWPORT = { width: 1600, height: 1800 };
const TAB_SELECTORS = {
    'Speaker Prep': '.tab-button[data-tab="speaker-prep"]',
    'Conversation Workflow': '.tab-button[data-tab="conversation-workflow"]',
    'Conversation Results': '.tab-button[data-tab="conversation-results"]',
    'Timeline Editor': '.tab-button[data-tab="timeline-editor"]',
};
const PUBLIC_VOICE_ALIASES = [
    ['Pr.D.Trump_ready.wav', 'SpeakerSixAlt.wav'],
    ['Pr.D.Trump_ready', 'SpeakerSixAlt'],
    ['Pr.D.Trump.wav', 'SpeakerSix.wav'],
    ['Pr.D.Trump', 'SpeakerSix'],
    ['JeanLucPicard.wav', 'SpeakerThree.wav'],
    ['JeanLucPicard', 'SpeakerThree'],
    ['JohnnyDepp.wav', 'SpeakerFive.wav'],
    ['JohnnyDepp', 'SpeakerFive'],
    ['gordeylaforge.wav', 'SpeakerSeven.wav'],
    ['gordeylaforge', 'SpeakerSeven'],
    ['Asmongold.wav', 'SpeakerOne.wav'],
    ['Asmongold', 'SpeakerOne'],
    ['ElonMusk.wav', 'SpeakerTwo.wav'],
    ['ElonMusk', 'SpeakerTwo'],
    ['JoeRogan.wav', 'SpeakerFour.wav'],
    ['JoeRogan', 'SpeakerFour'],
    ['kajsa.wav', 'SpeakerEight.wav'],
    ['kajsa', 'SpeakerEight'],
];

async function ensureOutputDir() {
    await fs.mkdir(OUTPUT_DIR, { recursive: true });
}

async function launchBrowser() {
    return await chromium.launch({
        headless: true,
        executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    });
}

async function installPublicCaptureSanitizer(page) {
    await page.evaluate((aliases) => {
        const replacements = aliases;

        const replaceText = (value) => {
            if (!value) {
                return value;
            }

            let nextValue = value;
            for (const [actual, publicName] of replacements) {
                nextValue = nextValue.split(actual).join(publicName);
            }

            return nextValue;
        };

        const sanitizeTextNodes = (root = document.body) => {
            const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
            let node;

            while ((node = walker.nextNode())) {
                const original = node.nodeValue;
                const replaced = replaceText(original);
                if (replaced !== original) {
                    node.nodeValue = replaced;
                }
            }
        };

        const hideHelperVoiceCards = () => {
            const voiceList = document.getElementById('available-voices-list');
            if (!voiceList) {
                return;
            }

            const cards = Array.from(voiceList.children);
            let visibleCount = 0;

            for (const card of cards) {
                if ((card.textContent || '').includes('SpeakerSixAlt')) {
                    card.style.display = 'none';
                    continue;
                }

                if (!card.classList.contains('empty-state') && card.style.display !== 'none') {
                    visibleCount += 1;
                }
            }

            const countBadge = document.getElementById('available-voices-count');
            if (countBadge && visibleCount > 0) {
                countBadge.textContent = `${visibleCount} voices`;
            }
        };

        window.__manualSanitizePublicCapture = () => {
            sanitizeTextNodes(document.body);
            hideHelperVoiceCards();
        };

        window.__manualSanitizePublicCapture();
    }, PUBLIC_VOICE_ALIASES);
}

async function runPublicCaptureSanitizer(page) {
    await page.evaluate(() => {
        window.__manualSanitizePublicCapture?.();
    });
}

async function preparePage(page) {
    await page.goto(APP_URL, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.setViewportSize(VIEWPORT);
    await page.addStyleTag({
        content: `
            .notification-container,
            .dark-mode-toggle {
                display: none !important;
            }
        `,
    });
    await installPublicCaptureSanitizer(page);
    await page.waitForSelector('body');
    await page.waitForTimeout(8000);
    await page.waitForFunction(() => {
        const statusText = document.querySelector('#api-status .status-text')?.textContent || '';
        return statusText && !statusText.includes('Checking API');
    }, { timeout: 60000 });
    await runPublicCaptureSanitizer(page);
    await page.waitForTimeout(1200);
}

async function clickTab(page, label) {
    const selector = TAB_SELECTORS[label] || '.tab-button';
    await page.evaluate((tabSelector) => {
        document.querySelector(tabSelector)?.click();
    }, selector);
    await page.waitForTimeout(1200);
    await runPublicCaptureSanitizer(page);
}

async function saveScreenshot(page, filename) {
    const targetPath = path.join(OUTPUT_DIR, filename);
    await page.screenshot({
        path: targetPath,
        fullPage: true,
    });
    return targetPath;
}

async function captureConversationWorkflow(page) {
    await clickTab(page, 'Conversation Workflow');
    await page.evaluate(() => {
        const title = document.getElementById('conversation-title');
        if (title) {
            title.value = '';
        }

        const script = document.getElementById('conversation-script');
        if (script) {
            script.value = '';
        }

        const preview = document.getElementById('script-preview');
        if (preview) {
            preview.style.display = 'none';
        }
    });
    await runPublicCaptureSanitizer(page);
    await page.waitForTimeout(400);
    return await saveScreenshot(page, 'conversation-workflow-tab.png');
}

async function captureSpeakerPrep(page) {
    await clickTab(page, 'Speaker Prep');
    return await saveScreenshot(page, 'speaker-prep-tab.png');
}

async function ensureConversationForResults(page) {
    const conversationCount = await page.evaluate(async () => {
        const response = await fetch('/api/conversation/list');
        const payload = await response.json();
        return payload?.details?.conversations?.length || 0;
    });

    if (conversationCount > 0) {
        return;
    }

    await clickTab(page, 'Conversation Workflow');
    await page.locator('#conversation-title').fill('Manual Demo Conversation');
    await page.locator('#conversation-script').fill('Pr.D.Trump: This is the manual demo line.');

    const autoDetectEmotion = page.locator('#auto-detect-emotion');
    if (await autoDetectEmotion.isChecked()) {
        await autoDetectEmotion.uncheck();
    }

    await page.locator('#parse-script-btn').click();
    await page.waitForTimeout(1200);
    await page.locator('#generate-conversation-btn').click();

    await page.waitForFunction(async () => {
        const response = await fetch('/api/conversation/list');
        const payload = await response.json();
        return (payload?.details?.conversations?.length || 0) > 0;
    }, { timeout: 180000 });

    await page.waitForTimeout(1500);
}

async function captureConversationResults(page) {
    await ensureConversationForResults(page);
    await clickTab(page, 'Conversation Results');
    const conversationItems = page.locator('.conversation-item');
    if ((await conversationItems.count()) > 0) {
        await conversationItems.first().click();
        await page.waitForTimeout(1500);
        await runPublicCaptureSanitizer(page);
    }
    return await saveScreenshot(page, 'conversation-results-tab.png');
}

async function captureTimelineEditor(page) {
    await clickTab(page, 'Timeline Editor');

    const projectSelect = page.locator('#timeline-project-select');
    if (await projectSelect.count()) {
        const optionValues = await projectSelect.locator('option').evaluateAll((options) =>
            options.map((option) => ({ value: option.value, text: option.textContent || '' }))
        );

        const preferred =
            optionValues.find((option) => option.text.includes('standalone-smoke-20260408-b')) ||
            optionValues.find((option) => option.value);

        if (preferred?.value) {
            await projectSelect.selectOption(preferred.value);
            const loadButton = page.locator('#timeline-load-project-btn');
            if (await loadButton.count()) {
                await loadButton.click();
                await page.waitForTimeout(1500);
                await runPublicCaptureSanitizer(page);
            }
        }
    }

    return await saveScreenshot(page, 'timeline-editor-tab.png');
}

async function main() {
    await ensureOutputDir();

    const browser = await launchBrowser();
    const page = await browser.newPage();

    try {
        await preparePage(page);

        const screenshots = {};
        screenshots.workflow = await captureConversationWorkflow(page);
        screenshots.speakerPrep = await captureSpeakerPrep(page);
        screenshots.results = await captureConversationResults(page);
        screenshots.timeline = await captureTimelineEditor(page);

        console.log(JSON.stringify(screenshots, null, 2));
    } finally {
        await browser.close();
    }
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
