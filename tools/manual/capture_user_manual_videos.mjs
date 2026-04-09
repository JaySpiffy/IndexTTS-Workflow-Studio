import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const APP_URL = process.env.INDEXTTS_APP_URL || 'http://localhost:3000';
const CHROME_PATH = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const VIEWPORT = { width: 1440, height: 960 };

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const OUTPUT_DIR = path.join(REPO_ROOT, 'docs', 'assets', 'manual', 'videos');
const TEMP_VIDEO_DIR = path.join(OUTPUT_DIR, '.tmp');
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

async function ensureOutputDirs() {
    await fs.mkdir(OUTPUT_DIR, { recursive: true });
    await fs.mkdir(TEMP_VIDEO_DIR, { recursive: true });
}

async function launchBrowser() {
    return chromium.launch({
        headless: true,
        executablePath: CHROME_PATH,
    });
}

async function installManualVideoStyles(page) {
    await page.addStyleTag({
        content: `
            .notification-container,
            .dark-mode-toggle {
                display: none !important;
            }

            #manual-video-caption {
                position: fixed;
                left: 24px;
                bottom: 24px;
                width: min(520px, calc(100vw - 48px));
                padding: 16px 18px;
                background: rgba(18, 24, 38, 0.88);
                color: #f8fafc;
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-radius: 16px;
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.25);
                backdrop-filter: blur(10px);
                z-index: 2147483647;
                font-family: Inter, system-ui, sans-serif;
                transition: opacity 180ms ease, transform 180ms ease;
                opacity: 0;
                transform: translateY(10px);
                pointer-events: none;
            }

            #manual-video-caption.is-visible {
                opacity: 1;
                transform: translateY(0);
            }

            #manual-video-caption strong {
                display: block;
                font-size: 20px;
                line-height: 1.25;
                margin-bottom: 6px;
                font-weight: 700;
            }

            #manual-video-caption span {
                display: block;
                font-size: 15px;
                line-height: 1.45;
                color: rgba(226, 232, 240, 0.96);
            }

            .manual-video-focus {
                position: relative !important;
                box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.55), 0 0 0 10px rgba(99, 102, 241, 0.16) !important;
                border-radius: 14px !important;
                transition: box-shadow 160ms ease;
                z-index: 50;
            }
        `,
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
    await installManualVideoStyles(page);
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

async function clearCaption(page) {
    await page.evaluate(() => {
        const caption = document.getElementById('manual-video-caption');
        if (caption) {
            caption.classList.remove('is-visible');
        }
    });
}

async function showCaption(page, title, body) {
    await page.evaluate(({ titleText, bodyText }) => {
        let caption = document.getElementById('manual-video-caption');
        if (!caption) {
            caption = document.createElement('div');
            caption.id = 'manual-video-caption';
            document.body.appendChild(caption);
        }

        caption.innerHTML = `<strong>${titleText}</strong><span>${bodyText}</span>`;
        caption.classList.add('is-visible');
    }, { titleText: title, bodyText: body });
    await page.waitForTimeout(1200);
}

async function clearFocus(page) {
    await page.evaluate(() => {
        document.querySelectorAll('.manual-video-focus').forEach((element) => {
            element.classList.remove('manual-video-focus');
        });
    });
}

async function focusSelector(page, selector, pauseMs = 1200) {
    const locator = page.locator(selector).first();
    if ((await locator.count()) === 0) {
        return false;
    }

    const isVisible = await locator.isVisible().catch(() => false);
    if (!isVisible) {
        return false;
    }

    await locator.scrollIntoViewIfNeeded();
    await clearFocus(page);
    await locator.evaluate((element) => {
        element.classList.add('manual-video-focus');
    });
    await page.waitForTimeout(pauseMs);
    return true;
}

async function focusAndClick(page, selector, pauseMs = 900) {
    const locator = page.locator(selector).first();
    if ((await locator.count()) === 0) {
        return false;
    }

    const isVisible = await locator.isVisible().catch(() => false);
    if (!isVisible) {
        return false;
    }

    await locator.scrollIntoViewIfNeeded();
    await clearFocus(page);
    await locator.evaluate((element) => {
        element.classList.add('manual-video-focus');
    });
    await locator.click();
    await page.waitForTimeout(pauseMs);
    return true;
}

async function maybeClickVisible(page, selector, pauseMs = 900) {
    const locator = page.locator(selector).first();
    if ((await locator.count()) === 0) {
        return false;
    }

    const isVisible = await locator.isVisible().catch(() => false);
    if (!isVisible) {
        return false;
    }

    await locator.click();
    await page.waitForTimeout(pauseMs);
    return true;
}

async function clickTab(page, label) {
    const selector = TAB_SELECTORS[label] || '.tab-button';
    await page.evaluate((tabSelector) => {
        document.querySelector(tabSelector)?.click();
    }, selector);
    await page.waitForTimeout(1000);
    await runPublicCaptureSanitizer(page);
}

async function maybeSelectFirstSourceClip(page) {
    const clipButtons = page.locator('#speaker-prep-source-list button');
    const buttonCount = await clipButtons.count();
    if (!buttonCount) {
        return false;
    }

    for (let index = 0; index < buttonCount; index += 1) {
        const button = clipButtons.nth(index);
        const label = (await button.innerText()).trim().toLowerCase();
        if (label.includes('select')) {
            await button.click();
            await page.waitForTimeout(700);
            return true;
        }
    }

    return false;
}

async function ensureConversationForResults(browser) {
    const context = await browser.newContext({ viewport: VIEWPORT });
    const page = await context.newPage();

    try {
        await preparePage(page);

        const conversationCount = await page.evaluate(async () => {
            const response = await fetch('/api/conversation/list');
            const payload = await response.json();
            return payload?.details?.conversations?.length || 0;
        });

        if (conversationCount > 0) {
            return;
        }

        await clickTab(page, 'Conversation Workflow');
        await page.locator('#conversation-title').fill('Manual Video Demo Conversation');
        await page.locator('#conversation-script').fill(
            'Pr.D.Trump: This is the manual video demo line.\nJoeRogan: I am reacting to it right now.'
        );

        const autoDetectEmotion = page.locator('#auto-detect-emotion');
        if ((await autoDetectEmotion.count()) && (await autoDetectEmotion.isChecked())) {
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

        await page.waitForTimeout(1200);
    } finally {
        await context.close();
    }
}

async function withRecordedPage(browser, baseName, captureFn) {
    const context = await browser.newContext({
        viewport: VIEWPORT,
        recordVideo: {
            dir: TEMP_VIDEO_DIR,
            size: VIEWPORT,
        },
        reducedMotion: 'reduce',
    });

    const page = await context.newPage();
    const video = page.video();

    try {
        await preparePage(page);
        await captureFn(page);
        await clearFocus(page);
        await clearCaption(page);
        await page.waitForTimeout(600);
    } finally {
        await context.close();
    }

    const rawVideoPath = await video.path();
    const targetPath = path.join(OUTPUT_DIR, `${baseName}.webm`);
    await fs.rm(targetPath, { force: true });
    await fs.copyFile(rawVideoPath, targetPath);
    await fs.rm(rawVideoPath, { force: true });
    return targetPath;
}

async function recordSpeakerPrepVideo(browser) {
    return withRecordedPage(browser, 'speaker-prep-tab', async (page) => {
        await clickTab(page, 'Speaker Prep');
        await showCaption(page, 'Speaker Prep', 'Upload a source clip, diagnose it, and clean it before using it as a live voice.');
        await focusSelector(page, '#speaker-prep-upload-file');
        await showCaption(page, 'Source Clip Intake', 'Use Audio File and Upload Source Clip to bring in a raw reference.');
        await focusSelector(page, '#speaker-prep-source-list');
        await maybeSelectFirstSourceClip(page);
        await showCaption(page, 'Clip Library', 'Select the clip you want to work on, then diagnose it before prep.');
        await focusAndClick(page, '#speaker-prep-diagnose-btn', 1400);
        await showCaption(page, 'Diagnostics', 'Check the readiness score, read the recommendations, and see the suggested trim window.');
        await focusAndClick(page, '#speaker-prep-apply-recommended-btn', 800);
        await focusAndClick(page, '#speaker-prep-apply-trim-btn', 800);
        await showCaption(page, 'Apply The Recipe', 'Use Apply Recommended Prep and Use Suggested Trim to fill in safer trim and cleanup settings automatically.');
        await focusSelector(page, '#speaker-prep-output-name');
        await focusSelector(page, '#speaker-prep-process-btn', 900);
        await focusSelector(page, '#speaker-prep-create-speaker-btn', 1200);
        await showCaption(page, 'Create The Speaker', 'Prepare Selected Clip keeps it in source clips. Prepare And Create Speaker promotes it into the live speaker library.');
    });
}

async function recordConversationWorkflowVideo(browser) {
    return withRecordedPage(browser, 'conversation-workflow-tab', async (page) => {
        await clickTab(page, 'Conversation Workflow');
        await showCaption(page, 'Conversation Workflow', 'Build a scene from script, choose pacing, and launch generation.');
        await focusSelector(page, '#save-project-btn');
        await focusSelector(page, '#load-project-btn');
        await showCaption(page, 'Project Save And Load', 'Save work in progress here so you can come back later without losing script, settings, or results.');
        await focusSelector(page, '#available-voices-list');
        await showCaption(page, 'Available Voices', 'Use these exact labels at the start of each script line so the parser can match the right speaker.');
        await page.locator('#conversation-title').fill('Manual Video Workflow Demo');
        const actualScript =
            'Pr.D.Trump: These are not flowers. They are consequences.\nJoeRogan: That is one of the weirdest things I have ever agreed with.';
        const publicScript =
            'SpeakerSix: These are not flowers. They are consequences.\nSpeakerFour: That is one of the weirdest things I have ever agreed with.';

        await page.locator('#conversation-script').fill(actualScript);
        await page.locator('#conversation-script').evaluate((element, publicValue) => {
            element.dataset.manualActualValue = element.value;
            element.value = publicValue;
        }, publicScript);
        await runPublicCaptureSanitizer(page);
        await focusSelector(page, '#conversation-script');
        await page.locator('#dialogue-pacing-preset').selectOption('argument');
        await focusSelector(page, '#dialogue-pacing-preset', 900);
        await showCaption(page, 'Script And Pacing', 'Write one speaker turn per line, then choose the pace preset that best fits the scene.');
        await page.locator('#conversation-script').evaluate((element) => {
            if (element.dataset.manualActualValue) {
                element.value = element.dataset.manualActualValue;
                delete element.dataset.manualActualValue;
            }
        });
        await runPublicCaptureSanitizer(page);
        await focusAndClick(page, '#parse-script-btn', 1400);
        await focusSelector(page, '#versions-per-line', 700);
        await focusSelector(page, '#similarity-threshold', 700);
        await focusSelector(page, '#generate-conversation-btn', 1300);
        await showCaption(page, 'Generate The Scene', 'Parse Script checks the lines first. Generate Conversation creates candidate versions for review.');
    });
}

async function recordConversationResultsVideo(browser) {
    await ensureConversationForResults(browser);

    return withRecordedPage(browser, 'conversation-results-tab', async (page) => {
        await clickTab(page, 'Conversation Results');
        await showCaption(page, 'Conversation Results', 'Review versions, pick the final line for each turn, and export the finished mix.');

        const conversationItems = page.locator('.conversation-item');
        if ((await conversationItems.count()) > 0) {
            await conversationItems.first().click();
            await page.waitForTimeout(1200);
            await runPublicCaptureSanitizer(page);
        }

        await focusSelector(page, '.conversation-item');
        await showCaption(page, 'Select A Conversation', 'Pick the generated conversation you want to review or move into the timeline editor.');
        await focusSelector(page, '#auto-select-best-btn', 800);
        await maybeClickVisible(page, '#auto-select-best-btn', 1000);
        await focusSelector(page, '.version-card, .line-version-card', 1000);
        await showCaption(page, 'Review Each Line', 'Play, compare, and regenerate versions until each line has one final selected take.');
        await focusSelector(page, '#selection-readiness-summary');
        await focusSelector(page, '#concat-output-format', 900);
        await focusSelector(page, '#concatenate-btn', 1200);
        await showCaption(page, 'Finish And Export', 'When every line is selected, concatenate the mix and export it with your pacing and finishing settings.');
    });
}

async function chooseTimelineProject(page) {
    const select = page.locator('#timeline-project-select');
    if ((await select.count()) === 0) {
        return false;
    }

    const options = await select.locator('option').evaluateAll((elements) =>
        elements.map((option) => ({ value: option.value, text: option.textContent || '' }))
    );

    const preferred =
        options.find((option) => option.text.includes('standalone-smoke-20260408-b')) ||
        options.find((option) => option.value);

    if (!preferred?.value) {
        return false;
    }

    await select.selectOption(preferred.value);
    await page.waitForTimeout(500);
    await page.locator('#timeline-load-project-btn').click();
    await page.waitForTimeout(1200);
    await runPublicCaptureSanitizer(page);
    return true;
}

async function recordTimelineEditorVideo(browser) {
    return withRecordedPage(browser, 'timeline-editor-tab', async (page) => {
        await clickTab(page, 'Timeline Editor');
        await showCaption(page, 'Timeline Editor', 'Build a scene from scratch, place segments by hand, and export the final arrangement.');
        await focusSelector(page, '#timeline-project-name', 700);
        await focusSelector(page, '#timeline-create-blank-btn', 900);
        await showCaption(page, 'Start A Timeline', 'Create a blank timeline here, or load an existing one from the saved timelines list.');
        await chooseTimelineProject(page);
        await focusSelector(page, '#timeline-track-speaker', 700);
        await focusSelector(page, '#timeline-add-track-btn', 900);
        await showCaption(page, 'Add Speaker Tracks', 'Pick a speaker, name the track, and add it directly from the editor.');
        await focusSelector(page, '#timeline-inline-segment-text', 700);
        await focusSelector(page, '#timeline-inline-add-segment-btn', 900);
        await showCaption(page, 'Add Segments', 'Write the line, choose the track, and create the segment without leaving the timeline.');
        await focusSelector(page, '.timeline-track-add-segment, .timeline-lane-add-btn', 1200);
        await showCaption(page, 'Editor-Native Actions', 'You can also add a segment from the track header or directly from an empty lane in the canvas.');
        await focusSelector(page, '#timeline-export-btn', 900);
        await focusSelector(page, '#timeline-play-export-btn', 900);
        await showCaption(page, 'Render And Preview', 'Generate missing audio, export the mix, and preview the finished scene right from the timeline.');
    });
}

async function main() {
    await ensureOutputDirs();
    const browser = await launchBrowser();

    try {
        const outputs = {};
        outputs.speakerPrep = await recordSpeakerPrepVideo(browser);
        outputs.workflow = await recordConversationWorkflowVideo(browser);
        outputs.results = await recordConversationResultsVideo(browser);
        outputs.timeline = await recordTimelineEditorVideo(browser);

        console.log(JSON.stringify(outputs, null, 2));
    } finally {
        await browser.close();
        await fs.rm(TEMP_VIDEO_DIR, { recursive: true, force: true });
    }
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
