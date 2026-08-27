"use strict";

const API_BASE = window.location.origin;
const WS_BASE  = window.location.origin.replace(/^http/, "ws");

console.log(`[Config] API_BASE="${API_BASE}" WS_BASE="${WS_BASE}"`);

// ─── Global State ─────────────────────────────────────────────────────────────
let activeVoiceName = "Sophia";
let activeVoiceObj  = null;
let voices          = [];
let sessionId       = null;
let websocket       = null;
let wsClosedByUs    = false;
let connectionState = "disconnected";

// Audio Contexts
let captureContext   = null;  // 44100 Hz
let playbackContext  = null;  // 8000 Hz
let mediaStream      = null;
let workletNode      = null;
let activeSources    = [];
let nextPlayTime     = 0;
let isMuted          = false;
let audioContextStarted = false;

// WebSocket keepalive
let pingInterval = null;

// ─── DOM References ───────────────────────────────────────────────────────────
let elIndustry, elLanguage, elStartBtn, elEndBtn,
    elStatus, elMuteBtn, elMicStatus,
    elOrb, elOrbStatus, elVoiceNodes, elActiveAvatarWrap,
    elActiveVoiceName, elActiveVoiceRole, elActiveVoiceLangs, elActiveVoiceDesc,
    elScenarioText, elThankYouModal, elModalRetryBtn, elMicIndicator, elPreviewBtn;

let previewAudio = null;

// ─── CallState Enum ───────────────────────────────────────────────────────────
const CallState = {
    CONNECTED:            "CONNECTED",
    WAITING_FOR_CUSTOMER: "WAITING_FOR_CUSTOMER",
    CUSTOMER_SPEAKING:    "CUSTOMER_SPEAKING",
    TRANSCRIBING:         "TRANSCRIBING",
    THINKING:             "THINKING",
    GENERATING_RESPONSE:  "GENERATING_RESPONSE",
    AI_SPEAKING:          "AI_SPEAKING",
    CALL_COMPLETED:       "CALL_COMPLETED",
    ERROR:                "ERROR"
};

// ─── Init ─────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    elIndustry          = document.getElementById("industry-select");
    elLanguage          = document.getElementById("language-select");
    elStartBtn          = document.getElementById("start-btn");
    elEndBtn            = document.getElementById("end-btn");
    elStatus            = document.getElementById("orb-status-text");
    elMuteBtn           = document.getElementById("mute-btn");
    elMicStatus         = document.getElementById("mic-status");
    elMicIndicator      = document.querySelector(".mic-indicator");
    elOrb               = document.getElementById("ai-orb");
    elOrbStatus         = document.getElementById("orb-status-text");
    elVoiceNodes        = document.getElementById("voice-nodes-container");
    elActiveAvatarWrap  = document.getElementById("active-voice-avatar-wrap");
    elActiveVoiceName   = document.getElementById("active-voice-name");
    elActiveVoiceRole   = document.getElementById("active-voice-role");
    elActiveVoiceLangs  = document.getElementById("active-voice-langs");
    elActiveVoiceDesc   = document.getElementById("active-voice-desc");
    elScenarioText      = document.getElementById("scenario-info-text");
    elThankYouModal     = document.getElementById("thank-you-modal");
    elModalRetryBtn     = document.getElementById("modal-retry-btn");
    elPreviewBtn        = document.getElementById("preview-btn");

    setupListeners();
    loadVoices();
    updateScenarioText();
    initWaveCanvas();
});

// ─── Orb Canvas Wave Animation ──────────────────────────────────────────────
let waveCanvas = null;
let waveCtx = null;
let waveAnimFrame = null;
let currentOrbState = CallState.CALL_COMPLETED;

function initWaveCanvas() {
    waveCanvas = document.getElementById("orb-wave-canvas");
    if (!waveCanvas) return;
    waveCtx = waveCanvas.getContext("2d");
    resizeWaveCanvas();
    window.addEventListener("resize", resizeWaveCanvas);
    startWaveAnimation();
}

function resizeWaveCanvas() {
    if (!waveCanvas) return;
    const rect = waveCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    waveCanvas.width = rect.width * dpr;
    waveCanvas.height = rect.height * dpr;
}

function startWaveAnimation() {
    let phase = 0;
    function render() {
        waveAnimFrame = requestAnimationFrame(render);
        if (!waveCanvas || !waveCtx) return;
        const w = waveCanvas.width;
        const h = waveCanvas.height;
        waveCtx.clearRect(0, 0, w, h);

        if (currentOrbState !== CallState.AI_SPEAKING && currentOrbState !== CallState.CUSTOMER_SPEAKING && currentOrbState !== CallState.WAITING_FOR_CUSTOMER) {
            return;
        }

        phase += 0.035;
        const cx = w / 2;
        const cy = h / 2;
        const baseRadius = Math.min(w, h) * 0.26;
        const isSpeaking = (currentOrbState === CallState.AI_SPEAKING);

        const numWaves = 3;
        for (let i = 0; i < numWaves; i++) {
            waveCtx.beginPath();
            const points = 60;
            const wavePhase = phase + (i * Math.PI / 3);
            const amplitude = isSpeaking ? (8 + i * 3.5) * (window.devicePixelRatio || 1) : (3 + i * 2) * (window.devicePixelRatio || 1);
            const color = isSpeaking ? "rgba(6, 182, 212, " : "rgba(34, 211, 238, ";
            const alpha = 0.4 - i * 0.1;

            for (let j = 0; j <= points; j++) {
                const angle = (j / points) * Math.PI * 2;
                const dist = baseRadius + Math.sin(angle * 4 + wavePhase) * amplitude + Math.cos(angle * 2 - wavePhase) * (amplitude * 0.5);
                const x = cx + Math.cos(angle) * dist;
                const y = cy + Math.sin(angle) * dist;
                if (j === 0) waveCtx.moveTo(x, y);
                else waveCtx.lineTo(x, y);
            }
            waveCtx.closePath();
            waveCtx.strokeStyle = color + alpha + ")";
            waveCtx.lineWidth = 1.8 * (window.devicePixelRatio || 1);
            waveCtx.stroke();
        }
    }
    render();
}

function setupListeners() {
    elStartBtn.addEventListener("click", startConversation);
    elEndBtn.addEventListener("click", stopConversation);
    elMuteBtn.addEventListener("click", toggleMute);
    elPreviewBtn.addEventListener("click", playVoicePreview);
    
    elIndustry.addEventListener("change", () => {
        updateScenarioText();
        autoAdaptVoicesForLangAndIndustry();
    });
    
    elLanguage.addEventListener("change", () => {
        autoAdaptVoicesForLangAndIndustry();
    });

    elModalRetryBtn.addEventListener("click", () => {
        elThankYouModal.classList.add("hidden");
    });
}

function updateScenarioText() {
    if (elIndustry.value === "hospital") {
        elScenarioText.textContent = "Book hospital appointments naturally, reschedule check-up dates, and answer medical FAQs with our automated healthcare receptionist.";
    } else {
        elScenarioText.textContent = "Browse premium listings, understand budget requirements, and schedule visits to Orchard Heights with our real estate AI sales executive.";
    }
}

function playVoicePreview() {
    if (previewAudio) {
        previewAudio.pause();
        previewAudio = null;
        elPreviewBtn.innerHTML = `<i class="fa-solid fa-play"></i>`;
        return;
    }

    const voice = activeVoiceName;
    const lang = elLanguage.value;
    
    elPreviewBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i>`;
    const previewUrl = `${API_BASE}/api/v1/voice-demo/preview?voice=${voice}&lang=${lang}`;
    
    previewAudio = new Audio(previewUrl);
    previewAudio.oncanplaythrough = () => {
        elPreviewBtn.innerHTML = `<i class="fa-solid fa-pause"></i>`;
        previewAudio.play().catch(e => {
            console.error("Failed to play preview audio:", e);
            elPreviewBtn.innerHTML = `<i class="fa-solid fa-play"></i>`;
            previewAudio = null;
        });
    };
    
    previewAudio.onended = () => {
        elPreviewBtn.innerHTML = `<i class="fa-solid fa-play"></i>`;
        previewAudio = null;
    };
    
    previewAudio.onerror = () => {
        console.error("Preview audio failed to load.");
        elPreviewBtn.innerHTML = `<i class="fa-solid fa-play"></i>`;
        previewAudio = null;
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Avatar SVG Helper
// ─────────────────────────────────────────────────────────────────────────────
function makeSvgAvatar(name, gender) {
    const isFemale = (gender === "Female");
    const nameLower = (name || "").toLowerCase();
    
    // Gradient backdrops & skin tones for realistic human portraits
    let c1 = isFemale ? "#2A4032" : "#213945";
    let c2 = isFemale ? "#122017" : "#0E181F";
    let skinBase = "#FCE0D1";
    let skinShadow = "#E5B9A4";
    let hairColor = "#2A1810";
    let hairHighlight = "#4A3024";
    let suitColor = "#1B2E23";
    let accent = "#A7C1A1";
    
    if (nameLower === "sophia") {
        c1 = "#2A4032"; c2 = "#122017"; skinBase = "#FCE0D1"; skinShadow = "#E5B9A4"; hairColor = "#2A1810"; hairHighlight = "#4A3024"; suitColor = "#1B2E23"; accent = "#A7C1A1";
    } else if (nameLower === "maya") {
        c1 = "#344732"; c2 = "#162316"; skinBase = "#F8D7C4"; skinShadow = "#E0AF98"; hairColor = "#422117"; hairHighlight = "#663728"; suitColor = "#263624"; accent = "#C8D6B8";
    } else if (nameLower === "ananya") {
        c1 = "#213D48"; c2 = "#0E1B22"; skinBase = "#F5CEB6"; skinShadow = "#DAA489"; hairColor = "#1A1B22"; hairHighlight = "#323440"; suitColor = "#172A34"; accent = "#7FBFCF";
    } else if (nameLower === "arjun") {
        c1 = "#253429"; c2 = "#101B14"; skinBase = "#E8BC9B"; skinShadow = "#CD9772"; hairColor = "#181513"; hairHighlight = "#302B27"; suitColor = "#17241B"; accent = "#94B48E";
    } else if (nameLower === "david") {
        c1 = "#1B3029"; c2 = "#0B1713"; skinBase = "#FADCB9"; skinShadow = "#DFB087"; hairColor = "#291A14"; hairHighlight = "#452E25"; suitColor = "#111E19"; accent = "#A7C1A1";
    }

    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 100 100");
    svg.setAttribute("width", "100"); 
    svg.setAttribute("height", "100");

    const defs = document.createElementNS(svgNS, "defs");
    const grad = document.createElementNS(svgNS, "linearGradient");
    const gId  = "g_" + name.replace(/\s/g, "_");
    grad.setAttribute("id", gId);
    grad.setAttribute("x1","0%"); grad.setAttribute("y1","0%");
    grad.setAttribute("x2","100%"); grad.setAttribute("y2","100%");
    
    const s1 = document.createElementNS(svgNS, "stop");
    s1.setAttribute("offset","0%"); s1.setAttribute("stop-color", c1);
    const s2 = document.createElementNS(svgNS, "stop");
    s2.setAttribute("offset","100%"); s2.setAttribute("stop-color", c2);
    
    grad.appendChild(s1); grad.appendChild(s2); defs.appendChild(grad);

    // Skin gradient
    const skinGrad = document.createElementNS(svgNS, "linearGradient");
    const skinId = "skin_" + name.replace(/\s/g, "_");
    skinGrad.setAttribute("id", skinId);
    skinGrad.setAttribute("x1","0%"); skinGrad.setAttribute("y1","0%");
    skinGrad.setAttribute("x2","0%"); skinGrad.setAttribute("y2","100%");
    const sk1 = document.createElementNS(svgNS, "stop"); sk1.setAttribute("offset","0%"); sk1.setAttribute("stop-color", skinBase);
    const sk2 = document.createElementNS(svgNS, "stop"); sk2.setAttribute("offset","100%"); sk2.setAttribute("stop-color", skinShadow);
    skinGrad.appendChild(sk1); skinGrad.appendChild(sk2); defs.appendChild(skinGrad);

    svg.appendChild(defs);

    // Background circle
    const bg = document.createElementNS(svgNS, "circle");
    bg.setAttribute("cx","50"); bg.setAttribute("cy","50"); bg.setAttribute("r","50");
    bg.setAttribute("fill", `url(#${gId})`);
    svg.appendChild(bg);

    // Subtle metallic frame ring
    const ring = document.createElementNS(svgNS, "circle");
    ring.setAttribute("cx","50"); ring.setAttribute("cy","50"); ring.setAttribute("r","46");
    ring.setAttribute("fill", "none");
    ring.setAttribute("stroke", accent);
    ring.setAttribute("stroke-opacity", "0.35");
    ring.setAttribute("stroke-width", "1");
    svg.appendChild(ring);

    // Realistic Human Portrait Silhouette
    const gAvatar = document.createElementNS(svgNS, "g");

    // Formal Suit / Blazer Jacket
    const body = document.createElementNS(svgNS, "path");
    body.setAttribute("d", "M 16 88 C 16 66, 28 60, 50 60 C 72 60, 84 66, 84 88 Z");
    body.setAttribute("fill", suitColor);
    gAvatar.appendChild(body);

    // Blazer Lapels
    const lapelL = document.createElementNS(svgNS, "path");
    lapelL.setAttribute("d", "M 28 88 L 42 60 L 50 72 Z");
    lapelL.setAttribute("fill", "#0E1813"); lapelL.setAttribute("fill-opacity", "0.4");
    gAvatar.appendChild(lapelL);

    const lapelR = document.createElementNS(svgNS, "path");
    lapelR.setAttribute("d", "M 72 88 L 58 60 L 50 72 Z");
    lapelR.setAttribute("fill", "#0E1813"); lapelR.setAttribute("fill-opacity", "0.4");
    gAvatar.appendChild(lapelR);

    // White Shirt Inner V
    const shirt = document.createElementNS(svgNS, "polygon");
    shirt.setAttribute("points", "42,60 50,75 58,60");
    shirt.setAttribute("fill", "#F3F0E6");
    gAvatar.appendChild(shirt);

    if (!isFemale) {
        // Formal Silk Tie for Male
        const tie = document.createElementNS(svgNS, "polygon");
        tie.setAttribute("points", "48,64 52,64 53,82 50,87 47,82");
        tie.setAttribute("fill", accent);
        gAvatar.appendChild(tie);
    }

    // Neck with shading
    const neck = document.createElementNS(svgNS, "rect");
    neck.setAttribute("x", "43"); neck.setAttribute("y", "45"); neck.setAttribute("width", "14"); neck.setAttribute("height", "17"); neck.setAttribute("rx", "5");
    neck.setAttribute("fill", `url(#${skinId})`);
    gAvatar.appendChild(neck);

    // Neck shadow under chin
    const neckShadow = document.createElementNS(svgNS, "ellipse");
    neckShadow.setAttribute("cx", "50"); neckShadow.setAttribute("cy", "47"); neckShadow.setAttribute("rx", "7"); neckShadow.setAttribute("ry", "3");
    neckShadow.setAttribute("fill", skinShadow); neckShadow.setAttribute("fill-opacity", "0.6");
    gAvatar.appendChild(neckShadow);

    // Face Shape
    const head = document.createElementNS(svgNS, "ellipse");
    head.setAttribute("cx", "50"); head.setAttribute("cy", "35"); head.setAttribute("rx", "15.5"); head.setAttribute("ry", "17.5");
    head.setAttribute("fill", `url(#${skinId})`);
    gAvatar.appendChild(head);

    // Ears
    const earL = document.createElementNS(svgNS, "circle");
    earL.setAttribute("cx", "34"); earL.setAttribute("cy", "36"); earL.setAttribute("r", "3.2"); earL.setAttribute("fill", skinBase);
    gAvatar.appendChild(earL);

    const earR = document.createElementNS(svgNS, "circle");
    earR.setAttribute("cx", "66"); earR.setAttribute("cy", "36"); earR.setAttribute("r", "3.2"); earR.setAttribute("fill", skinBase);
    gAvatar.appendChild(earR);

    // Cheek soft glow
    const cheekL = document.createElementNS(svgNS, "circle");
    cheekL.setAttribute("cx", "42"); cheekL.setAttribute("cy", "38"); cheekL.setAttribute("r", "4"); cheekL.setAttribute("fill", "#E89B8C"); cheekL.setAttribute("fill-opacity", "0.2");
    gAvatar.appendChild(cheekL);
    const cheekR = document.createElementNS(svgNS, "circle");
    cheekR.setAttribute("cx", "58"); cheekR.setAttribute("cy", "38"); cheekR.setAttribute("r", "4"); cheekR.setAttribute("fill", "#E89B8C"); cheekR.setAttribute("fill-opacity", "0.2");
    gAvatar.appendChild(cheekR);

    // Realistic Eyes
    const eyeWhiteL = document.createElementNS(svgNS, "ellipse");
    eyeWhiteL.setAttribute("cx", "43"); eyeWhiteL.setAttribute("cy", "34"); eyeWhiteL.setAttribute("rx", "3"); eyeWhiteL.setAttribute("ry", "2"); eyeWhiteL.setAttribute("fill", "#FFFFFF");
    gAvatar.appendChild(eyeWhiteL);

    const eyeWhiteR = document.createElementNS(svgNS, "ellipse");
    eyeWhiteR.setAttribute("cx", "57"); eyeWhiteR.setAttribute("cy", "34"); eyeWhiteR.setAttribute("rx", "3"); eyeWhiteR.setAttribute("ry", "2"); eyeWhiteR.setAttribute("fill", "#FFFFFF");
    gAvatar.appendChild(eyeWhiteR);

    // Iris & Pupil
    const irisL = document.createElementNS(svgNS, "circle");
    irisL.setAttribute("cx", "43"); irisL.setAttribute("cy", "34"); irisL.setAttribute("r", "1.8"); irisL.setAttribute("fill", "#2C1D18");
    gAvatar.appendChild(irisL);

    const irisR = document.createElementNS(svgNS, "circle");
    irisR.setAttribute("cx", "57"); irisR.setAttribute("cy", "34"); irisR.setAttribute("r", "1.8"); irisR.setAttribute("fill", "#2C1D18");
    gAvatar.appendChild(irisR);

    // Pupil Glint
    const glintL = document.createElementNS(svgNS, "circle");
    glintL.setAttribute("cx", "42.3"); glintL.setAttribute("cy", "33.3"); glintL.setAttribute("r", "0.6"); glintL.setAttribute("fill", "#FFFFFF");
    gAvatar.appendChild(glintL);

    const glintR = document.createElementNS(svgNS, "circle");
    glintR.setAttribute("cx", "56.3"); glintR.setAttribute("cy", "33.3"); glintR.setAttribute("r", "0.6"); glintR.setAttribute("fill", "#FFFFFF");
    gAvatar.appendChild(glintR);

    // Eyebrows
    const browL = document.createElementNS(svgNS, "path");
    browL.setAttribute("d", "M 39 30 Q 43 28.2 47 30"); browL.setAttribute("stroke", hairColor); browL.setAttribute("stroke-width", "1.4"); browL.setAttribute("fill", "none"); browL.setAttribute("stroke-linecap", "round");
    gAvatar.appendChild(browL);

    const browR = document.createElementNS(svgNS, "path");
    browR.setAttribute("d", "M 53 30 Q 57 28.2 61 30"); browR.setAttribute("stroke", hairColor); browR.setAttribute("stroke-width", "1.4"); browR.setAttribute("fill", "none"); browR.setAttribute("stroke-linecap", "round");
    gAvatar.appendChild(browR);

    // Nose
    const nose = document.createElementNS(svgNS, "path");
    nose.setAttribute("d", "M 50 34 L 49.2 39.5 Q 50 40.5 51 39.8");
    nose.setAttribute("stroke", skinShadow); nose.setAttribute("stroke-width", "1.2"); nose.setAttribute("fill", "none"); nose.setAttribute("stroke-linecap", "round");
    gAvatar.appendChild(nose);

    // Natural Smile Lips
    const lips = document.createElementNS(svgNS, "path");
    lips.setAttribute("d", "M 44 43 Q 50 47 56 43");
    lips.setAttribute("stroke", "#B25E50"); lips.setAttribute("stroke-width", "1.8"); lips.setAttribute("fill", "none"); lips.setAttribute("stroke-linecap", "round");
    gAvatar.appendChild(lips);

    // Realistic Hair Styling (Female vs Male)
    if (isFemale) {
        const hair = document.createElementNS(svgNS, "path");
        if (nameLower === "ananya") {
            // Sleek Bun Updo
            hair.setAttribute("d", "M 33 34 C 33 16, 67 16, 67 34 C 68 22, 60 14, 50 14 C 40 14, 32 22, 33 34 Z M 43 14 C 43 7, 57 7, 57 14 Z");
        } else {
            // Layered Professional Bob Haircut
            hair.setAttribute("d", "M 32 35 C 30 16, 70 16, 68 35 C 70 48, 66 54, 63 56 C 63 43, 64 24, 50 21 C 36 24, 37 43, 37 56 C 34 54, 30 48, 32 35 Z");
        }
        hair.setAttribute("fill", hairColor);
        gAvatar.appendChild(hair);

        // Hair Strand Highlights
        const highlight = document.createElementNS(svgNS, "path");
        highlight.setAttribute("d", "M 37 25 Q 50 19 63 25");
        highlight.setAttribute("stroke", hairHighlight); highlight.setAttribute("stroke-width", "1.5"); highlight.setAttribute("fill", "none"); highlight.setAttribute("stroke-opacity", "0.6");
        gAvatar.appendChild(highlight);
    } else {
        // Executive Male Haircut
        const hair = document.createElementNS(svgNS, "path");
        hair.setAttribute("d", "M 32 33 C 32 17, 68 17, 68 33 C 68 24, 62 19, 50 19 C 38 19, 32 24, 32 33 Z");
        hair.setAttribute("fill", hairColor);
        gAvatar.appendChild(hair);

        // Male Hair Highlight
        const highlight = document.createElementNS(svgNS, "path");
        highlight.setAttribute("d", "M 36 22 Q 50 18 62 23");
        highlight.setAttribute("stroke", hairHighlight); highlight.setAttribute("stroke-width", "1.5"); highlight.setAttribute("fill", "none"); highlight.setAttribute("stroke-opacity", "0.6");
        gAvatar.appendChild(highlight);
    }

    // Modern Metallic AI Headset & Mic Boom
    const headset = document.createElementNS(svgNS, "path");
    headset.setAttribute("d", "M 32 33 A 19.5 19.5 0 0 1 68 33");
    headset.setAttribute("fill", "none");
    headset.setAttribute("stroke", accent);
    headset.setAttribute("stroke-width", "2.2");
    headset.setAttribute("stroke-linecap", "round");
    gAvatar.appendChild(headset);

    const earCap = document.createElementNS(svgNS, "circle");
    earCap.setAttribute("cx", "33"); earCap.setAttribute("cy", "35"); earCap.setAttribute("r", "3.5");
    earCap.setAttribute("fill", accent);
    gAvatar.appendChild(earCap);

    const micBoom = document.createElementNS(svgNS, "path");
    micBoom.setAttribute("d", "M 33 35 L 43 45");
    micBoom.setAttribute("stroke", accent);
    micBoom.setAttribute("stroke-width", "1.8");
    micBoom.setAttribute("stroke-linecap", "round");
    gAvatar.appendChild(micBoom);

    const micTip = document.createElementNS(svgNS, "circle");
    micTip.setAttribute("cx", "44"); micTip.setAttribute("cy", "46"); micTip.setAttribute("r", "2");
    micTip.setAttribute("fill", "#F3F0E6");
    gAvatar.appendChild(micTip);

    svg.appendChild(gAvatar);

    const svgStr = new XMLSerializer().serializeToString(svg);
    return "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgStr)));
}

const REAL_HUMAN_AVATARS = {
    "sophia": "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?auto=format&fit=crop&w=300&q=80",
    "maya":   "https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?auto=format&fit=crop&w=300&q=80",
    "ananya": "https://images.unsplash.com/photo-1589156280159-27698a70f29e?auto=format&fit=crop&w=300&q=80",
    "arjun":  "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=300&q=80",
    "david":  "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?auto=format&fit=crop&w=300&q=80"
};

function createAvatarElement(voice, sizePx) {
    const img = document.createElement("img");
    img.width = sizePx; 
    img.height = sizePx;
    img.alt = voice.name;
    img.style.borderRadius = "50%";
    img.style.objectFit   = "cover";
    img.style.display     = "block";
    const nameKey = (voice.name || "").toLowerCase();
    const photoUrl = REAL_HUMAN_AVATARS[nameKey];
    if (photoUrl) {
        img.src = photoUrl;
        img.onerror = () => {
            img.src = makeSvgAvatar(voice.name, voice.gender);
        };
    } else {
        img.src = makeSvgAvatar(voice.name, voice.gender);
    }
    return img;
}

// ─────────────────────────────────────────────────────────────────────────────
// Load Voice Profiles
// ─────────────────────────────────────────────────────────────────────────────
async function loadVoices() {
    try {
        console.log("[Voices] Loading voice configurations...");
        const res = await fetch(`${API_BASE}/api/v1/voice-demo/voices`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        voices = await res.json();
        renderVoiceCircle();
        if (voices.length > 0) selectVoice(voices[0].name);
    } catch (err) {
        console.error("[Voices] Failed to load voice config:", err);
        elOrbStatus.textContent = "Error loading voices";
    }
}

function renderVoiceCircle() {
    if (!elVoiceNodes) return;
    elVoiceNodes.innerHTML = "";
    const total  = voices.length;
    const isMobile = window.innerWidth <= 768;
    const radius = isMobile ? 104 : 130;

    voices.forEach((voice, idx) => {
        const node = document.createElement("div");
        node.className = "voice-node";
        node.dataset.voiceName = voice.name;
        node.title = `${voice.name} — ${voice.description}`;

        const img = createAvatarElement(voice, isMobile ? 46 : 54);
        node.appendChild(img);

        const angle = (idx / total) * 2 * Math.PI - Math.PI / 2;
        const tx = Math.round(Math.cos(angle) * radius);
        const ty = Math.round(Math.sin(angle) * radius);
        node.style.setProperty("--tx", `${tx}px`);
        node.style.setProperty("--ty", `${ty}px`);
        node.style.transform = `translate(${tx}px, ${ty}px)`;
        node.addEventListener("click", () => selectVoice(voice.name));
        elVoiceNodes.appendChild(node);
    });
}

window.addEventListener("resize", () => {
    if (voices && voices.length > 0) {
        renderVoiceCircle();
    }
});

function selectVoice(voiceName) {
    activeVoiceName = voiceName;
    activeVoiceObj  = voices.find(v => v.name.toLowerCase() === voiceName.toLowerCase());

    elVoiceNodes.querySelectorAll(".voice-node").forEach(n => {
        n.classList.toggle("selected", n.dataset.voiceName.toLowerCase() === voiceName.toLowerCase());
    });

    if (!activeVoiceObj) return;

    elActiveAvatarWrap.innerHTML = "";
    const bigImg = createAvatarElement(activeVoiceObj, 56);
    elActiveAvatarWrap.appendChild(bigImg);

    // Update Center Orb Agent Avatar Face
    const elCenterAvatarWrap = document.getElementById("center-agent-avatar-wrap");
    if (elCenterAvatarWrap) {
        elCenterAvatarWrap.classList.add("avatar-swapping");
        setTimeout(() => {
            elCenterAvatarWrap.innerHTML = "";
            const centerImg = createAvatarElement(activeVoiceObj, 76);
            centerImg.className = "center-agent-face-img";
            elCenterAvatarWrap.appendChild(centerImg);
            elCenterAvatarWrap.classList.remove("avatar-swapping");
        }, 120);
    }

    elActiveVoiceName.textContent = activeVoiceObj.name;
    elActiveVoiceRole.textContent = activeVoiceObj.description;
    elActiveVoiceLangs.textContent = activeVoiceObj.supported_languages.replace(/,/g, " • ");
    
    // Set bio details
    if (activeVoiceObj.name === "Sophia") {
        elActiveVoiceDesc.textContent = "Warm, empathetic and conversational voice suitable for healthcare and patient coordination.";
    } else if (activeVoiceObj.name === "Maya") {
        elActiveVoiceDesc.textContent = "Energetic, bright and engaging female voice, great for sales recommendations and active client conversions.";
    } else if (activeVoiceObj.name === "Ananya") {
        elActiveVoiceDesc.textContent = "Clear, articulate support persona speaking naturally with a pleasant customer-first attitude.";
    } else if (activeVoiceObj.name === "Arjun") {
        elActiveVoiceDesc.textContent = "Steady, authoritative and reassuring male assistant, perfect for real estate catalog assistance.";
    } else if (activeVoiceObj.name === "David") {
        elActiveVoiceDesc.textContent = "Confident, persuasive and highly articulate male sales agent specialized in luxury properties.";
    }

    elOrbStatus.textContent = `Ready`;
    updateStatusBadgeClass("status-ready");
}

function autoAdaptVoicesForLangAndIndustry() {
    const currentLang = elLanguage.value;
    // Sophia and Arjun support all 3 languages. Ananya/Maya support EN/TE, David supports EN only.
    // If the active voice does not support the selected language, auto-adapt to Sophia or Arjun.
    if (activeVoiceObj && !activeVoiceObj.supported_languages.includes(currentLang)) {
        const fallback = currentLang === "Telugu" ? "Ananya" : (currentLang === "Hindi" ? "Sophia" : "Sophia");
        console.log(`[Auto-Adapt] Active voice ${activeVoiceObj.name} does not support ${currentLang}. Swapping to ${fallback}.`);
        selectVoice(fallback);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio Capture & Playout Setup
// ─────────────────────────────────────────────────────────────────────────────
async function ensureAudioContexts() {
    if (!captureContext) {
        captureContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
    }
    if (captureContext.state === "suspended") {
        await captureContext.resume();
    }

    if (!playbackContext) {
        playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    }
    if (playbackContext.state === "suspended") {
        await playbackContext.resume();
    }

    audioContextStarted = true;
    nextPlayTime = 0;
    activeSources = [];
}

let micSeqCounter = 0;

async function setupAudioCapture() {
    if (!captureContext || !mediaStream) return;

    const audioTrack = mediaStream.getAudioTracks()[0];
    const isLive = audioTrack && audioTrack.readyState === "live" && audioTrack.enabled && !audioTrack.muted;
    console.log(`[MIC-HEALTH] permission=granted track_state=${audioTrack ? audioTrack.readyState : 'none'} track_enabled=${audioTrack ? audioTrack.enabled : false} track_muted=${audioTrack ? audioTrack.muted : true} sample_rate=${captureContext.sampleRate} channels=1 websocket_ready=${websocket && websocket.readyState === WebSocket.OPEN}`);

    if (!isLive) {
        console.warn("[MIC-HEALTH] Warning: Audio track is not live/active!");
    }

    try {
        const workletUrl = `${window.location.origin}/static/js/audio-capture-worklet.js`;
        await captureContext.audioWorklet.addModule(workletUrl);

        const sourceNode = captureContext.createMediaStreamSource(mediaStream);
        workletNode = new AudioWorkletNode(captureContext, "audio-capture-processor");

        workletNode.port.onmessage = (evt) => {
            if (!evt.data || evt.data.type !== "frame") return;
            if (!websocket || websocket.readyState !== WebSocket.OPEN || isMuted) return;

            const float32 = evt.data.data;
            const downsampled = downsample(float32, captureContext.sampleRate, 16000);
            const int16 = float32ToInt16PCM(downsampled);
            
            micSeqCounter++;
            const durMs = (int16.byteLength / 32.0); // 16kHz 16-bit mono = 32 bytes/ms
            if (micSeqCounter % 50 === 1) {
                console.log(`[MIC-WS-SEND] session=${sessionId} seq=${micSeqCounter} bytes=${int16.byteLength} format=pcm_s16le sample_rate=16000 dur_ms=${durMs.toFixed(0)}`);
            }
            websocket.send(int16);
        };

        sourceNode.connect(workletNode);
        console.log("[Audio] Capture worklet connected (16kHz PCM_S16LE).");

    } catch (err) {
        console.warn(`[Audio] AudioWorklet failed (${err.message}), falling back to ScriptProcessor.`);
        const sourceNode = captureContext.createMediaStreamSource(mediaStream);
        const scriptProcessor = captureContext.createScriptProcessor(4096, 1, 1);

        scriptProcessor.onaudioprocess = (evt) => {
            if (!websocket || websocket.readyState !== WebSocket.OPEN || isMuted) return;
            const float32 = evt.inputBuffer.getChannelData(0);
            const downsampled = downsample(float32, captureContext.sampleRate, 16000);
            const int16 = float32ToInt16PCM(downsampled);
            
            micSeqCounter++;
            if (micSeqCounter % 50 === 1) {
                console.log(`[MIC-WS-SEND] session=${sessionId} seq=${micSeqCounter} bytes=${int16.byteLength} format=pcm_s16le sample_rate=16000`);
            }
            websocket.send(int16);
        };

        sourceNode.connect(scriptProcessor);
        const silentGain = captureContext.createGain();
        silentGain.gain.value = 0;
        scriptProcessor.connect(silentGain);
        silentGain.connect(captureContext.destination);
    }
}

function teardownAudioCapture() {
    if (workletNode) {
        try { workletNode.port.postMessage({ type: "stop" }); } catch (_) {}
        try { workletNode.disconnect(); } catch (_) {}
        workletNode = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
        mediaStream = null;
    }
}

function downsample(buf, fromRate, toRate) {
    if (fromRate === toRate) return buf;
    const ratio  = fromRate / toRate;
    const outLen = Math.round(buf.length / ratio);
    const out    = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
        const start = Math.round(i * ratio);
        const end   = Math.round((i + 1) * ratio);
        let sum = 0, count = 0;
        for (let j = start; j < end && j < buf.length; j++) { sum += buf[j]; count++; }
        out[i] = count ? sum / count : 0;
    }
    return out;
}

function float32ToInt16PCM(buf) {
    const out = new Int16Array(buf.length);
    for (let i = 0; i < buf.length; i++) {
        const s = Math.max(-1, Math.min(1, buf[i]));
        out[i]  = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return out.buffer;
}

// ─────────────────────────────────────────────────────────────────────────────
// Session Lifecycle Handlers
// ─────────────────────────────────────────────────────────────────────────────
async function startConversation() {
    if (connectionState !== "disconnected") return;

    await ensureAudioContexts();

    connectionState = "connecting";
    elStartBtn.disabled = true;
    elStartBtn.classList.add("hidden");
    elEndBtn.classList.remove("hidden");
    
    elMuteBtn.disabled = false;
    elMicIndicator.classList.add("recording");
    elMicStatus.textContent = "Active";

    setOrbState(CallState.CONNECTED, "Connecting...");

    try {
        const res = await fetch(`${API_BASE}/api/v1/voice-demo/sessions`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                voice_name: activeVoiceName,
                industry: elIndustry.value,
                language: elLanguage.value
            })
        });

        if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Session creation failed: ${errText}`);
        }

        const sessionData = await res.json();
        sessionId = sessionData.session_id;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Microphone access is blocked by your browser because this demo is currently on HTTP (http://147.93.171.146). Modern browsers require HTTPS or enabling the flag chrome://flags/#unsafely-treat-insecure-origin-as-secure for this IP.");
        }

        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl:  true,
                channelCount: 1
            }
        });

        wsClosedByUs = false;
        const wsUrl = `${WS_BASE}/api/v1/voice-demo/stream/${sessionId}`;
        websocket = new WebSocket(wsUrl);
        websocket.binaryType = "arraybuffer";

        websocket.onopen = () => {
            connectionState = "connected";
            setOrbState(CallState.WAITING_FOR_CUSTOMER);
            setupAudioCapture();
            startKeepalive();
        };

        websocket.onmessage = handleWsMessage;

        websocket.onerror = (e) => {
            console.error("[WS] Connection error:", e);
            setOrbState(CallState.ERROR, "Connection Error");
        };

        websocket.onclose = (evt) => {
            stopKeepalive();
            if (!wsClosedByUs) {
                setOrbState(CallState.ERROR, "Disconnected");
                stopAllAudio();
                teardownAudioCapture();
                resetUIAfterCall();
                elThankYouModal.classList.remove("hidden");
            }
        };

    } catch (err) {
        console.error("[Start] Error starting conversation:", err);
        setOrbState(CallState.ERROR, "Start Failed");
        resetUIAfterCall();
    }
}

async function stopConversation() {
    if (connectionState === "disconnected") return;
    connectionState = "disconnected";

    setOrbState(CallState.CALL_COMPLETED, "Conversation Ended");
    stopAllAudio();
    stopKeepalive();

    if (websocket && websocket.readyState < WebSocket.CLOSING) {
        wsClosedByUs = true;
        try { websocket.send(JSON.stringify({ event: "stop" })); } catch (_) {}
        websocket.close(1000, "User ended call");
    }
    websocket = null;
    teardownAudioCapture();
    resetUIAfterCall();

    // Trigger Thank You modal
    elThankYouModal.classList.remove("hidden");
}

function resetUIAfterCall() {
    elStartBtn.disabled = false;
    elStartBtn.classList.remove("hidden");
    elEndBtn.classList.add("hidden");
    
    elMuteBtn.disabled = true;
    elMuteBtn.classList.remove("muted");
    elMuteBtn.innerHTML = `<i class="fa-solid fa-microphone"></i>`;
    elMicIndicator.classList.remove("recording");
    elMicStatus.textContent = "Ready";
    isMuted = false;
}

function toggleMute() {
    isMuted = !isMuted;
    if (isMuted) {
        elMuteBtn.classList.add("muted");
        elMuteBtn.innerHTML = `<i class="fa-solid fa-microphone-slash"></i>`;
        elMicStatus.textContent = "Muted";
    } else {
        elMuteBtn.classList.remove("muted");
        elMuteBtn.innerHTML = `<i class="fa-solid fa-microphone"></i>`;
        elMicStatus.textContent = "Active";
    }
}

function startKeepalive() {
    stopKeepalive();
    pingInterval = setInterval(() => {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({ event: "ping" }));
        }
    }, 15000);
}

function stopKeepalive() {
    if (pingInterval) {
        clearInterval(pingInterval);
        pingInterval = null;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Playout and Audio Output Queue
// ─────────────────────────────────────────────────────────────────────────────
// High-Fidelity 24kHz Linear PCM Audio Playout
// ─────────────────────────────────────────────────────────────────────────────
function playPcmAudio(arrayBuffer) {
    if (!playbackContext || !audioContextStarted) return;
    if (playbackContext.state === "suspended") playbackContext.resume();

    // 24kHz 16-bit Linear PCM (pcm_s16le)
    const int16 = new Int16Array(arrayBuffer);
    const f32   = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
        f32[i] = int16[i] / 32768.0;
    }

    const buf = playbackContext.createBuffer(1, f32.length, 24000);
    buf.getChannelData(0).set(f32);

    const src = playbackContext.createBufferSource();
    src.buffer = buf;
    src.connect(playbackContext.destination);

    const now = playbackContext.currentTime;
    const gapMs = (now - nextPlayTime) * 1000.0;
    if (nextPlayTime > 0 && gapMs > 30) {
        console.warn(`[TTS-FLOW-BROWSER] playback_gap_ms=${gapMs.toFixed(1)}ms > 30ms limit!`);
    }

    // Continuous gapless scheduling: schedule next chunk immediately at nextPlayTime
    // If nextPlayTime is in the past, align with currentTime + 5ms minimal lookahead
    if (nextPlayTime < now + 0.005) {
        nextPlayTime = now + 0.005;
    }
    src.start(nextPlayTime);
    nextPlayTime += buf.duration;

    activeSources.push(src);
    src.onended = () => {
        const i = activeSources.indexOf(src);
        if (i !== -1) activeSources.splice(i, 1);
        if (activeSources.length === 0 && playbackContext && playbackContext.currentTime >= nextPlayTime - 0.05) {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                console.log("[MIC-SYNC] All AI playback completed. Sending playback_ended to backend.");
                try {
                    websocket.send(JSON.stringify({ event: "playback_ended" }));
                } catch (err) {
                    console.warn("[MIC-SYNC] Failed to send playback_ended:", err);
                }
            }
        }
    };
}

function stopAllAudio() {
    activeSources.forEach(src => { try { src.stop(); } catch (_) {} });
    activeSources = [];
    nextPlayTime  = 0;
    console.log("[Audio] All audio playback stopped.");
}

// ─────────────────────────────────────────────────────────────────────────────
// WebSocket Routing
// ─────────────────────────────────────────────────────────────────────────────
function handleWsMessage(evt) {
    if (evt.data instanceof ArrayBuffer) {
        playPcmAudio(evt.data);
        return;
    }
    try {
        const msg = JSON.parse(evt.data);
        if (msg.event === "state_change") {
            setOrbState(msg.state);
        } else if (msg.event === "clear_audio") {
            stopAllAudio();
        }
    } catch (e) {
        console.error("[WS] Fail to parse control packet:", e);
    }
}

// ─── Update UI Orb State ─────────────────────────────────────────────────────
function setOrbState(state, customLabel) {
    currentOrbState = state;
    elOrb.className = "orb-pulsar";
    let label = customLabel;
    let badgeClass = "status-ready";

    switch (state) {
        case CallState.CONNECTED:
            elOrb.classList.add("state-idle");
            label = label || "Connected";
            badgeClass = "status-connected";
            break;
        case CallState.WAITING_FOR_CUSTOMER:
            elOrb.classList.add("state-idle");
            label = "Listening";
            badgeClass = "status-listening";
            break;
        case CallState.CUSTOMER_SPEAKING:
            elOrb.classList.add("state-listening");
            label = "Listening";
            badgeClass = "status-listening";
            break;
        case CallState.TRANSCRIBING:
        case CallState.THINKING:
        case CallState.GENERATING_RESPONSE:
            elOrb.classList.add("state-thinking");
            label = "Thinking";
            badgeClass = "status-thinking";
            break;
        case CallState.AI_SPEAKING:
            elOrb.classList.add("state-speaking");
            label = "Speaking";
            badgeClass = "status-speaking";
            break;
        case CallState.CALL_COMPLETED:
            elOrb.classList.add("state-disconnected");
            label = "Call Ended";
            badgeClass = "status-ready";
            break;
        case CallState.ERROR:
            elOrb.classList.add("state-error");
            label = label || "Error";
            badgeClass = "status-error";
            break;
        default:
            elOrb.classList.add("state-idle");
            label = label || state;
            badgeClass = "status-ready";
    }

    if (elOrbStatus) elOrbStatus.textContent = label;
    updateStatusBadgeClass(badgeClass);
}

function updateStatusBadgeClass(badgeClass) {
    if (!elOrbStatus) return;
    elOrbStatus.className = "status-badge";
    elOrbStatus.classList.add(badgeClass);
}
