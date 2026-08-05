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
});

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
    const initial  = (name || "?").charAt(0).toUpperCase();
    const isFemale = gender === "Female";
    const c1 = isFemale ? "#a855f7" : "#6366f1";
    const c2 = isFemale ? "#ec4899" : "#3b82f6";
    const svgNS = "http://www.w3.org/2000/svg";
    const svg   = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 100 100");
    svg.setAttribute("width", "100"); 
    svg.setAttribute("height", "100");

    const defs = document.createElementNS(svgNS, "defs");
    const grad  = document.createElementNS(svgNS, "linearGradient");
    const gId   = "g_" + name.replace(/\s/g, "_");
    grad.setAttribute("id", gId);
    grad.setAttribute("x1","0%"); grad.setAttribute("y1","0%");
    grad.setAttribute("x2","100%"); grad.setAttribute("y2","100%");
    
    const s1 = document.createElementNS(svgNS, "stop");
    s1.setAttribute("offset","0%"); s1.setAttribute("stop-color", c1);
    
    const s2 = document.createElementNS(svgNS, "stop");
    s2.setAttribute("offset","100%"); s2.setAttribute("stop-color", c2);
    
    grad.appendChild(s1); 
    grad.appendChild(s2); 
    defs.appendChild(grad);
    svg.appendChild(defs);

    const circle = document.createElementNS(svgNS, "circle");
    circle.setAttribute("cx","50"); 
    circle.setAttribute("cy","50");
    circle.setAttribute("r","50"); 
    circle.setAttribute("fill", `url(#${gId})`);
    svg.appendChild(circle);

    const text = document.createElementNS(svgNS, "text");
    text.setAttribute("x","50%"); 
    text.setAttribute("y","54%");
    text.setAttribute("dominant-baseline","middle");
    text.setAttribute("text-anchor","middle");
    text.setAttribute("font-size","42"); 
    text.setAttribute("font-weight","700");
    text.setAttribute("fill","#ffffff");
    text.setAttribute("font-family","Outfit, sans-serif");
    text.textContent = initial;
    svg.appendChild(text);

    const svgStr = new XMLSerializer().serializeToString(svg);
    return "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svgStr)));
}

function createAvatarElement(voice, sizePx) {
    const img = document.createElement("img");
    img.width = sizePx; 
    img.height = sizePx;
    img.alt = voice.name;
    img.style.borderRadius = "50%";
    img.style.objectFit   = "cover";
    img.style.display     = "block";
    img.src = makeSvgAvatar(voice.name, voice.gender);
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
    const radius = isMobile ? 112 : 138;

    voices.forEach((voice, idx) => {
        const node = document.createElement("div");
        node.className = "voice-node";
        node.dataset.voiceName = voice.name;
        node.title = `${voice.name} — ${voice.description}`;

        const img = createAvatarElement(voice, isMobile ? 46 : 54);
        node.appendChild(img);

        const angle = (idx / total) * 2 * Math.PI - Math.PI / 2;
        node.style.transform = `translate(${Math.cos(angle) * radius}px, ${Math.sin(angle) * radius}px)`;
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
        playbackContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 8000 });
    }
    if (playbackContext.state === "suspended") {
        await playbackContext.resume();
    }

    audioContextStarted = true;
    nextPlayTime = 0;
    activeSources = [];
}

async function setupAudioCapture() {
    if (!captureContext || !mediaStream) return;

    try {
        const workletUrl = `${window.location.origin}/static/js/audio-capture-worklet.js`;
        await captureContext.audioWorklet.addModule(workletUrl);

        const sourceNode = captureContext.createMediaStreamSource(mediaStream);
        workletNode = new AudioWorkletNode(captureContext, "audio-capture-processor");

        workletNode.port.onmessage = (evt) => {
            if (!evt.data || evt.data.type !== "frame") return;
            if (!websocket || websocket.readyState !== WebSocket.OPEN || isMuted) return;

            const float32 = evt.data.data;
            const downsampled = downsample(float32, captureContext.sampleRate, 8000);
            const int16 = float32ToInt16PCM(downsampled);
            websocket.send(int16);
        };

        sourceNode.connect(workletNode);
        console.log("[Audio] Capture worklet connected.");

    } catch (err) {
        console.warn(`[Audio] AudioWorklet failed (${err.message}), falling back to ScriptProcessor.`);
        const sourceNode = captureContext.createMediaStreamSource(mediaStream);
        const scriptProcessor = captureContext.createScriptProcessor(4096, 1, 1);

        scriptProcessor.onaudioprocess = (evt) => {
            if (!websocket || websocket.readyState !== WebSocket.OPEN || isMuted) return;
            const float32 = evt.inputBuffer.getChannelData(0);
            const downsampled = downsample(float32, captureContext.sampleRate, 8000);
            const int16 = float32ToInt16PCM(downsampled);
            websocket.send(int16);
        };

        sourceNode.connect(scriptProcessor);
        scriptProcessor.connect(captureContext.destination);
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
function decodeUlaw(u) {
    u = (~u) & 0xFF;
    const sign = u & 0x80;
    const exp  = (u >> 4) & 0x07;
    const mant = u & 0x0F;
    let s = ((mant << 3) + 132) << exp;
    s -= 132;
    return (sign ? -s : s) / 32768.0;
}

function playMulaw(arrayBuffer) {
    if (!playbackContext || !audioContextStarted) return;
    if (playbackContext.state === "suspended") playbackContext.resume();

    const u8  = new Uint8Array(arrayBuffer);
    const f32 = new Float32Array(u8.length);
    for (let i = 0; i < u8.length; i++) f32[i] = decodeUlaw(u8[i]);

    const buf = playbackContext.createBuffer(1, f32.length, 8000);
    buf.getChannelData(0).set(f32);

    const src = playbackContext.createBufferSource();
    src.buffer = buf;
    src.connect(playbackContext.destination);

    const now = playbackContext.currentTime;
    if (nextPlayTime < now + 0.02) nextPlayTime = now + 0.02;
    src.start(nextPlayTime);
    nextPlayTime += buf.duration;

    activeSources.push(src);
    src.onended = () => {
        const i = activeSources.indexOf(src);
        if (i !== -1) activeSources.splice(i, 1);
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
        playMulaw(evt.data);
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
