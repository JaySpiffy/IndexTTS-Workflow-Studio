import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const APP_URL = process.env.INDEXTTS_APP_URL || 'http://localhost:3000';
const CHROME_PATH = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const VIEWPORT = { width: 1440, height: 960 };
const TARGET_TITLES = [
    'Podcast Roundtable Demo Pack',
    'Audiobook Night Train Demo Pack',
    'Game Dialogue Checkpoint Breach Pack',
];
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

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');
const OUTPUT_DIR = path.join(REPO_ROOT, 'docs', 'assets', 'social');
const AUDIO_OUTPUT_DIR = path.join(OUTPUT_DIR, 'audio');
const TEMP_VIDEO_DIR = path.join(OUTPUT_DIR, '.tmp');
const VIDEO_PATH = path.join(OUTPUT_DIR, 'timeline-workflow-demo.webm');
const SCREENSHOT_PATH = path.join(OUTPUT_DIR, 'timeline-workflow-demo.png');
const RENDER_SUMMARY_PATH = path.join(AUDIO_OUTPUT_DIR, 'render_summary.json');

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

async function installCaptureStyles(page) {
    await page.addStyleTag({
        content: `
            .notification-container,
            .dark-mode-toggle {
                display: none !important;
            }

            #reddit-followup-caption {
                position: fixed;
                left: 24px;
                bottom: 24px;
                width: min(540px, calc(100vw - 48px));
                padding: 16px 18px;
                background: rgba(15, 23, 42, 0.9);
                color: #f8fafc;
                border: 1px solid rgba(148, 163, 184, 0.35);
                border-radius: 12px;
                box-shadow: 0 18px 48px rgba(15, 23, 42, 0.28);
                font-family: Inter, system-ui, sans-serif;
                z-index: 2147483647;
                opacity: 0;
                transform: translateY(8px);
                transition: opacity 180ms ease, transform 180ms ease;
                pointer-events: none;
            }

            #reddit-followup-caption.is-visible {
                opacity: 1;
                transform: translateY(0);
            }

            #reddit-followup-caption strong {
                display: block;
                margin-bottom: 6px;
                font-size: 20px;
                line-height: 1.25;
            }

            #reddit-followup-caption span {
                display: block;
                font-size: 15px;
                line-height: 1.45;
                color: rgba(226, 232, 240, 0.95);
            }

            .reddit-followup-focus {
                position: relative !important;
                box-shadow: 0 0 0 4px rgba(96, 165, 250, 0.55), 0 0 0 10px rgba(96, 165, 250, 0.16) !important;
                border-radius: 12px !important;
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

        window.__redditSanitizeCapture = () => {
            sanitizeTextNodes(document.body);
        };

        window.__redditSanitizeCapture();
    }, PUBLIC_VOICE_ALIASES);
}

async function runPublicCaptureSanitizer(page) {
    await page.evaluate(() => {
        window.__redditSanitizeCapture?.();
    });
}

async function preparePage(page) {
    await page.goto(APP_URL, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await installCaptureStyles(page);
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

async function clearFocus(page) {
    await page.evaluate(() => {
        document.querySelectorAll('.reddit-followup-focus').forEach((element) => {
            element.classList.remove('reddit-followup-focus');
        });
    });
}

async function showCaption(page, title, body) {
    await page.evaluate(({ titleText, bodyText }) => {
        let caption = document.getElementById('reddit-followup-caption');
        if (!caption) {
            caption = document.createElement('div');
            caption.id = 'reddit-followup-caption';
            document.body.appendChild(caption);
        }

        caption.innerHTML = `<strong>${titleText}</strong><span>${bodyText}</span>`;
        caption.classList.add('is-visible');
    }, { titleText: title, bodyText: body });
    await page.waitForTimeout(1100);
}

async function clearCaption(page) {
    await page.evaluate(() => {
        const caption = document.getElementById('reddit-followup-caption');
        if (caption) {
            caption.classList.remove('is-visible');
        }
    });
}

async function focusSelector(page, selector, pauseMs = 900) {
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
        element.classList.add('reddit-followup-focus');
    });
    await page.waitForTimeout(pauseMs);
    return true;
}

async function clickTab(page, tabId) {
    await page.evaluate((targetTab) => {
        document.querySelector(`.tab-button[data-tab="${targetTab}"]`)?.click();
    }, tabId);
    await page.waitForTimeout(900);
    await runPublicCaptureSanitizer(page);
}

async function loadRenderSummary() {
    const raw = await fs.readFile(RENDER_SUMMARY_PATH, 'utf8');
    return JSON.parse(raw.replace(/^\uFEFF/, ''));
}

async function resolveTargetConversationId(page) {
    const summary = await loadRenderSummary();
    if (!Array.isArray(summary) || !summary.length) {
        throw new Error(`No entries found in ${RENDER_SUMMARY_PATH}`);
    }

    for (const title of TARGET_TITLES) {
        const matchingEntry = summary.find((entry) => entry.title === title);
        if (matchingEntry?.conversation_id) {
            return matchingEntry.conversation_id;
        }
    }

    return summary[0].conversation_id;
}

async function selectConversation(page, conversationId) {
    const selector = `.conversation-item[data-conversation-id="${conversationId}"]`;
    await page.waitForSelector(selector, { timeout: 60000 });
    await page.locator(selector).click();
    await page.waitForTimeout(1400);
    await runPublicCaptureSanitizer(page);
}

async function dragFirstTimelineSegment(page) {
    const segment = page.locator('.timeline-track-segment').first();
    await segment.waitFor({ state: 'visible', timeout: 60000 });
    await segment.click();
    await page.waitForTimeout(600);

    const box = await segment.boundingBox();
    if (!box) {
        throw new Error('Could not resolve the first timeline segment position.');
    }

    await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.5 + 48, box.y + box.height * 0.5, { steps: 10 });
    await page.mouse.up();
    await page.waitForTimeout(1400);
}

async function withRecordedPage(browser, captureFn) {
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
        await page.waitForTimeout(500);
    } finally {
        await context.close();
    }

    const rawVideoPath = await video.path();
    await fs.rm(VIDEO_PATH, { force: true });
    await fs.copyFile(rawVideoPath, VIDEO_PATH);
    await fs.rm(rawVideoPath, { force: true });
}

async function recordTimelineWorkflow(browser) {
    await withRecordedPage(browser, async (page) => {
        const conversationId = await resolveTargetConversationId(page);

        await clickTab(page, 'conversation-results');
        await showCaption(page, 'Review The Demo', 'Start in Conversation Results, select the generated demo, and auto-pick the best take for each line.');
        await selectConversation(page, conversationId);
        await focusSelector(page, `.conversation-item[data-conversation-id="${conversationId}"]`, 900);
        await focusSelector(page, '#auto-select-best-btn', 700);
        await page.locator('#auto-select-best-btn').click();
        await page.waitForTimeout(1300);

        await showCaption(page, 'Send It To Timeline', 'Once the lines are selected, open the conversation directly in the timeline editor.');
        await focusSelector(page, '#open-selected-in-timeline-btn', 700);
        await page.locator('#open-selected-in-timeline-btn').click();
        await page.waitForTimeout(1800);

        await page.waitForSelector('.timeline-track-segment', { timeout: 60000 });
        await showCaption(page, 'Adjust The Scene', 'Nudge a segment, inspect the selected panel, and tighten the handoff before export.');
        await focusSelector(page, '.timeline-track-segment', 900);
        await dragFirstTimelineSegment(page);
        await focusSelector(page, '#timeline-selected-panel', 900);

        await showCaption(page, 'Export The Mix', 'Render the timeline mix, then keep the video frame on the finished arrangement for the post.');
        await focusSelector(page, '#timeline-export-btn', 700);
        await page.locator('#timeline-export-btn').click();
        await page.waitForFunction(() => Boolean(window.app?.currentTimelineExportFilename), { timeout: 120000 });
        await page.waitForTimeout(1200);
        await focusSelector(page, '#timeline-play-export-btn', 600);
        await page.screenshot({ path: SCREENSHOT_PATH, fullPage: false });
    });
}

async function main() {
    await ensureOutputDirs();
    const browser = await launchBrowser();

    try {
        await recordTimelineWorkflow(browser);
        console.log(JSON.stringify({
            video: VIDEO_PATH,
            screenshot: SCREENSHOT_PATH,
            renderSummary: RENDER_SUMMARY_PATH,
        }, null, 2));
    } finally {
        await browser.close();
        await fs.rm(TEMP_VIDEO_DIR, { recursive: true, force: true });
    }
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
