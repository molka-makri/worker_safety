// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SAFEVISION AI — main.js  (v6 — 10 cameras, Worker Tracking + Posture + Panic)
// CAM 1 : Détection de chute   (worker_falling3.mp4)
// CAM 2 : Détection de fatigue (worker_tired.mp4)
// CAM 3 : Spill detection      (spill.mp4)
// CAM 4 : PPE detection        (ppe_video.mp4)
// CAM 5 : Manhole detection    (hole.mp4)
// CAM 6 : Exit emergency       (exit_emergency.mp4)
// CAM 7 : Sign defect detection(construction_signs2.mp4)
// CAM 8 : Worker tracking      (tracking_workers.mp4)
// CAM 9 : Posture detection    (posture.mp4)
// CAM 10: Panic detection      (panic.mp4)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ── HORLOGE LIVE ─────────────────────────────────────────
function updateClock() {
  const el = document.getElementById("live-clock");
  if (el) {
    const now = new Date();
    el.textContent = now.toLocaleTimeString("fr-FR", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }
}
setInterval(updateClock, 1000);
updateClock();

// ── CONSTANTES VIDÉO ──────────────────────────────────────
const LIVE_VIDEO_SRC = "/media/worker_falling3.mp4";
const LIVE_VIDEO_SRC_CAM2 = "/media/worker_tired.mp4";
const LIVE_VIDEO_SRC_CAM3 = "/media/spill.mp4";
const LIVE_VIDEO_SRC_CAM4 = "/media/ppe_video.mp4";
const LIVE_VIDEO_SRC_CAM6 = "/media/exit_emergency.mp4";
const LIVE_VIDEO_SRC_CAM5 = "/media/hole.mp4";
const LIVE_VIDEO_SRC_CAM7 = "/media/construction_signs2.mp4";
const LIVE_VIDEO_SRC_CAM8 = "/media/tracking_workers.mp4";
const LIVE_VIDEO_SRC_CAM8_PROXIMITY = "/media/Media_Proximity/vid3.mp4";
const LIVE_VIDEO_SRC_CAM9 = "/media/posture.mp4";
const LIVE_VIDEO_SRC_CAM10 = "/media/panic.mp4";
const LIVE_VIDEO_SRC_CAM12 = "/media/fire.mp4";
const DASHBOARD_PPE_SPILL_SRC = "/media/ppe_spill.mp4";
const DASHBOARD_SPILL_FALL_SRC = "/media/spill_fall.mp4";
const VIDEO_CAPTURE_MAX_WIDTH = 960;
const VIDEO_CAPTURE_MAX_HEIGHT = 960;

// ── ÉTAT DES MODULES ──────────────────────────────────────
let liveDetectionInterval = null;
let liveVideoAnalysisActive = false;

let fatigueDetectionInterval = null;
let fatigueVideoAnalysisActive = false;

let cam3DetectionInterval = null;
let cam3VideoAnalysisActive = false;

let cam4DetectionInterval = null;
let cam4VideoAnalysisActive = false;

let cam5DetectionInterval = null;
let cam5VideoAnalysisActive = false;

let cam6DetectionInterval = null;
let cam6VideoAnalysisActive = false;

let cam7DetectionInterval = null; // ADDED SIGN
let cam7VideoAnalysisActive = false; // ADDED SIGN
let cam8FrameLoopToken = 0;
let cam8RafHandle = null;
let cam8VideoFrameCallbackHandle = null;
let cam8LoopVideoRef = null;
let cam8ResetPending = false;
let cam8VideoAnalysisActive = false;
let spillRequestInFlight = {};
let spillLastAlertAt = {};
let cam9DetectionInterval = null; // POSTURE
let cam9VideoAnalysisActive = false; // POSTURE
let cam10DetectionInterval = null; // PANIC
let cam10VideoAnalysisActive = false; // PANIC
let cam12DetectionInterval = null; // FIRE/SMOKE
let cam12VideoAnalysisActive = false; // FIRE/SMOKE
let dashboardHybridInterval = null;
let dashboardHybridActive = false;
let dashboardHybridRequestInFlight = false;
let dashboardHybridLastFrameAt = 0;
let dashboardHybridFrameCount = 0;
let dashboardHybridStartedAt = 0;
let dashboardHybridLastAlertAt = { ppe: 0, spill: 0 };
let dashboardFallSpillInterval = null;
let dashboardFallSpillActive = false;
let dashboardFallSpillRequestInFlight = false;
let dashboardFallSpillFrameCount = 0;
let dashboardFallSpillLastAlertAt = { fall: 0, spill: 0 };

let ppeRequestInFlight = false;
let ppeLastAlertAt = 0;
const PPE_ANALYSIS_INTERVAL_MS = 500;
const PPE_ALERT_COOLDOWN_MS = 10000;

// Sign State Variables
let signRequestInFlight = false;
let signLastAlertAt = 0;
const SIGN_ANALYSIS_INTERVAL_MS = 800; // ResNet can be a bit heavier
const SIGN_ALERT_COOLDOWN_MS = 10000;
const WORKER_TRACKING_ALERT_COOLDOWN_MS = 9000;

// Proximity State Variables
let proximityRequestInFlight = false;
let proximityLastAlertAt = 0;
const PROXIMITY_ANALYSIS_INTERVAL_MS = 1000;
const PROXIMITY_ALERT_COOLDOWN_MS = 10000;
let cam8ProximityDetectionInterval = null;
let cam8ProximityVideoAnalysisActive = false;

// Posture State Variables (CAM 9)
let postureRequestInFlight = false;
let postureLastAlertAt = 0;
const POSTURE_ANALYSIS_INTERVAL_MS = 800;
const POSTURE_ALERT_COOLDOWN_MS = 8000;

// Panic State Variables (CAM 10)
let panicRequestInFlight = false;
let panicLastAlertAt = 0;
const PANIC_ANALYSIS_INTERVAL_MS = 800;
const PANIC_ALERT_COOLDOWN_MS = 6000;

// Fire/Smoke State Variables (CAM 12)
let fireRequestInFlight = false;
let fireLastAlertAt = 0;
const FIRE_ANALYSIS_INTERVAL_MS = 800;
const FIRE_ALERT_COOLDOWN_MS = 10000;

let cam8RequestInFlight = false;
let cam8LastAlertAt = 0;
let cam8LastVideoTime = 0;

let manholeRequestInFlight = false;
let manholeLastAlertAt = 0;
let exitRequestInFlight = false;
let exitLastAlertAt = 0;
let cam6RunToken = 0;
let camerasRunning = false;

const SPILL_ANALYSIS_INTERVAL_MS = 350;
const SPILL_ALERT_COOLDOWN_MS = 10000;
const MANHOLE_ANALYSIS_INTERVAL_MS = 450;
const MANHOLE_ALERT_COOLDOWN_MS = 12000;
const EXIT_ANALYSIS_INTERVAL_MS = 450;
const EXIT_ALERT_COOLDOWN_MS = 10000;

let camerasInitialized = false;
let alertCount = 0;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// UTILITAIRES COMMUNS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function captureVideoFrame(
  video,
  maxWidth = VIDEO_CAPTURE_MAX_WIDTH,
  maxHeight = VIDEO_CAPTURE_MAX_HEIGHT,
) {
  try {
    const videoWidth = video.videoWidth || maxWidth;
    const videoHeight = video.videoHeight || maxHeight;
    const scale = Math.min(maxWidth / videoWidth, maxHeight / videoHeight, 1);
    const width = Math.max(1, Math.round(videoWidth * scale));
    const height = Math.max(1, Math.round(videoHeight * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d").drawImage(video, 0, 0, width, height);

    video.dataset.captureWidth = width;
    video.dataset.captureHeight = height;

    return canvas.toDataURL("image/jpeg", 0.92).split(",")[1];
  } catch (err) {
    console.error("[SafeVision] captureVideoFrame:", err);
    return null;
  }
}

function getCookie(name) {
  const m = document.cookie.match(new RegExp("(^| )" + name + "=([^;]+)"));
  return m ? m[2] : "";
}

function setCanvasSize(canvas, video) {
  if (!canvas || !video) return;
  const w = video.clientWidth || canvas.parentElement?.clientWidth || 320;
  const h = video.clientHeight || canvas.parentElement?.clientHeight || 240;
  canvas.width = w;
  canvas.style.width = w + "px";
  canvas.height = h;
  canvas.style.height = h + "px";
}

function clearOverlayCanvas(canvasId) {
  const c = document.getElementById(canvasId);
  if (c) c.getContext("2d").clearRect(0, 0, c.width, c.height);
}

function clearAllOverlayCanvases() {
  [
    "main",
    "fall-spill",
    "cam1",
    "cam2",
    "cam3",
    "cam4",
    "cam5",
    "cam6",
    "cam7",
    "cam8",
    "cam9",
    "cam10",
    "cam12",
  ].forEach((id) => clearOverlayCanvas(`${id}-overlay-canvas`));
}

function getContainedVideoRect(canvas, video) {
  const captureWidth =
    Number(video.dataset.captureWidth) ||
    video.videoWidth ||
    VIDEO_CAPTURE_MAX_WIDTH;
  const captureHeight =
    Number(video.dataset.captureHeight) ||
    video.videoHeight ||
    VIDEO_CAPTURE_MAX_HEIGHT;
  const scale = Math.min(
    canvas.width / captureWidth,
    canvas.height / captureHeight,
  );
  const width = captureWidth * scale;
  const height = captureHeight * scale;
  return {
    x: (canvas.width - width) / 2,
    y: (canvas.height - height) / 2,
    scaleX: width / captureWidth,
    scaleY: height / captureHeight,
  };
}

function _notifyDetection(result) {
  if (typeof window.onDetectionResult === "function") {
    window.onDetectionResult(result);
  }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TOASTS & NOTIFICATIONS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function createToastContainer() {
  if (document.getElementById("toast-container")) return;
  const c = document.createElement("div");
  c.id = "toast-container";
  c.className = "toast-container";
  document.body.appendChild(c);
}

function saveNotification(message, severity = "info") {
  const n = {
    id: Date.now(),
    message,
    severity,
    timestamp: new Date().toLocaleString("fr-FR"),
    date: new Date().toISOString(),
  };
  const list = JSON.parse(
    localStorage.getItem("safevision_notifications") || "[]",
  );
  list.unshift(n);
  if (list.length > 100) list.pop();
  localStorage.setItem("safevision_notifications", JSON.stringify(list));
  return n;
}

function showPopupNotification(message, severity = "info", persist = true) {
  if (persist) saveNotification(message, severity);
  createToastContainer();
  const container = document.getElementById("toast-container");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `detection-toast detection-toast-${severity}`;
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add("visible"));
  setTimeout(() => {
    toast.classList.remove("visible");
    setTimeout(() => toast.remove(), 300);
  }, 4500);
}

function deleteNotification(id) {
  const list = JSON.parse(
    localStorage.getItem("safevision_notifications") || "[]",
  );
  localStorage.setItem(
    "safevision_notifications",
    JSON.stringify(list.filter((n) => n.id !== id)),
  );
  const el = document.getElementById("notification-" + id);
  if (el) {
    el.style.opacity = "0";
    el.style.transform = "translateX(20px)";
    setTimeout(() => el.remove(), 300);
  }
}

function deleteAllNotifications() {
  if (
    confirm("Êtes-vous sûr de vouloir supprimer toutes les notifications ?")
  ) {
    localStorage.removeItem("safevision_notifications");
    loadNotifications();
  }
}

function loadNotifications() {
  const container = document.getElementById("alerts-container");
  if (!container) return;
  const list = JSON.parse(
    localStorage.getItem("safevision_notifications") || "[]",
  );
  if (list.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" opacity=".3">
          <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/>
        </svg>
        <p>Aucune notification</p>
        <p style="font-size:10px">Les notifications apparaîtront ici</p>
      </div>`;
    return;
  }
  container.innerHTML = list
    .map(
      (n) => `
    <div id="notification-${n.id}" class="notification-item notification-${n.severity}"
         style="transition:all .3s ease">
      <div class="notification-content">
        <div class="notification-header">
          <span class="notification-severity notification-severity-${n.severity}">
            ${n.severity === "critical" ? "🔴" : n.severity === "warning" ? "🟡" : "🔵"} ${n.severity.toUpperCase()}
          </span>
          <span class="notification-date">${n.timestamp}</span>
        </div>
        <p class="notification-message">${n.message}</p>
      </div>
      <button class="btn-delete" onclick="deleteNotification(${n.id})" title="Supprimer">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polyline points="3 6 5 6 21 6"/>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
        </svg>
      </button>
    </div>`,
    )
    .join("");
}

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("alerts-container")) loadNotifications();
  initializeCameras();
  initializeDashboardHybrid();
});

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ALERTES GLOBALES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function addAlert(severity, module, message) {
  alertCount++;
  const badge = document.getElementById("alert-badge");
  if (badge) badge.textContent = alertCount;

  const feed = document.getElementById("alerts-feed");
  if (feed) {
    feed.querySelector(".empty-state")?.remove();
    const item = document.createElement("div");
    item.className = "alert-item fade-in";
    item.innerHTML = `
      <div class="alert-severity sev-${severity}"></div>
      <div class="alert-body">
        <div class="alert-module">${module.toUpperCase()}</div>
        <div class="alert-msg">${message}</div>
        <div class="alert-time">${new Date().toLocaleTimeString("fr-FR")}</div>
      </div>`;
    feed.insertBefore(item, feed.firstChild);
    while (feed.children.length > 50) feed.removeChild(feed.lastChild);
  }

  if (severity === "critical") playAlertSound();
}

function playAlertSound() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.3);
  } catch (_) {}
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 1 — DÉTECTION DE CHUTE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam1Status(text) {
  const el = document.getElementById("cam1-fall-status");
  if (el) el.textContent = "Détection: " + text;
}

function drawBoundingBox(bbox, video, isFall) {
  const canvas = document.getElementById("cam1-overlay-canvas");
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!Array.isArray(bbox) || bbox.length !== 4) return;

  const rect = getContainedVideoRect(canvas, video);
  const [x1, y1, x2, y2] = bbox;
  const rx = rect.x + x1 * rect.scaleX;
  const ry = rect.y + y1 * rect.scaleY;

  ctx.strokeStyle = isFall ? "rgba(255,59,47,0.95)" : "rgba(74,227,181,0.95)";
  ctx.lineWidth = 4;
  ctx.setLineDash([10, 6]);
  ctx.strokeRect(rx, ry, (x2 - x1) * rect.scaleX, (y2 - y1) * rect.scaleY);

  ctx.setLineDash([]);
  ctx.fillStyle = isFall ? "rgba(255,59,47,0.85)" : "rgba(74,227,181,0.85)";
  ctx.font = "bold 11px monospace";
  const label = isFall ? "⚠ CHUTE" : "✓ OK";
  const tw = ctx.measureText(label).width;
  ctx.fillRect(rx, ry - 18, tw + 10, 18);
  ctx.fillStyle = "#fff";
  ctx.fillText(label, rx + 5, ry - 5);
}

async function analyzeVideoFrame(video) {
  if (!video || video.paused || video.ended) return;
  const imageData = captureVideoFrame(video, 720, 720);
  if (!imageData) {
    updateCam1Status("Capture impossible");
    return;
  }
  updateCam1Status("Analyse en cours…");

  try {
    const res = await fetch("/api/fall-detection/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({ image: imageData, camera: "cam1" }),
    });
    const data = await res.json();

    if (data.status === "success") {
      const pct = Math.round(data.confidence * 100);
      updateCam1Status(data.fall_detected ? `CHUTE (${pct}%)` : `OK (${pct}%)`);

      if (Array.isArray(data.details?.bbox)) {
        drawBoundingBox(data.details.bbox, video, data.fall_detected);
      } else {
        clearOverlayCanvas("cam1-overlay-canvas");
      }

      if (data.fall_detected) {
        addAlert("critical", "Fall", "Chute détectée sur caméra 1");
        showPopupNotification(`Chute détectée (${pct}%)`, "critical");
      }

      _notifyDetection({
        cam: "cam1",
        type: "fall",
        detected: data.fall_detected,
        confidence: data.confidence,
        details: data.details || {},
      });
    } else {
      updateCam1Status("Erreur API");
      clearOverlayCanvas("cam1-overlay-canvas");
    }
  } catch (err) {
    updateCam1Status("Erreur détection");
    console.error("[SafeVision] fetch chute:", err);
  }
}

function startLiveDetection(video) {
  if (!video || liveDetectionInterval) return;
  analyzeVideoFrame(video);
  liveDetectionInterval = setInterval(() => analyzeVideoFrame(video), 1200);
  liveVideoAnalysisActive = true;
  updateCam1Status("Analyse active");
}

function stopLiveDetection() {
  if (liveDetectionInterval) {
    clearInterval(liveDetectionInterval);
    liveDetectionInterval = null;
  }
  liveVideoAnalysisActive = false;
  clearOverlayCanvas("cam1-overlay-canvas");
  updateCam1Status("Analyse arrêtée");
}

function loadLiveVideo(path) {
  const v = document.getElementById("cam1-video");
  if (!v) return;
  v.src = path;
  v.load();
  v.play().catch(() => {});
  if (!liveVideoAnalysisActive) startLiveDetection(v);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 2 — DÉTECTION DE FATIGUE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam2Status(text) {
  const el = document.getElementById("cam2-fatigue-status");
  if (el) el.textContent = "Détection: " + text;
}

function drawFatigueOverlay(bbox, video, isFatigued) {
  const canvas = document.getElementById("cam2-overlay-canvas");
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!Array.isArray(bbox) || bbox.length !== 4) return;

  const rect = getContainedVideoRect(canvas, video);
  const [x1, y1, x2, y2] = bbox;
  const rx = rect.x + x1 * rect.scaleX;
  const ry = rect.y + y1 * rect.scaleY;
  const bw = (x2 - x1) * rect.scaleX;
  const bh = (y2 - y1) * rect.scaleY;

  ctx.strokeStyle = isFatigued
    ? "rgba(255,184,0,0.95)"
    : "rgba(74,227,181,0.95)";
  ctx.lineWidth = 4;
  ctx.setLineDash([10, 6]);
  ctx.strokeRect(rx, ry, bw, bh);

  if (isFatigued) {
    ctx.fillStyle = "rgba(255,184,0,0.10)";
    ctx.fillRect(rx, ry, bw, bh);
  }

  ctx.setLineDash([]);
  ctx.fillStyle = isFatigued ? "rgba(255,184,0,0.90)" : "rgba(74,227,181,0.90)";
  ctx.font = "bold 11px monospace";
  const label = isFatigued ? "😴 FATIGUE" : "✓ OK";
  const tw = ctx.measureText(label).width;
  ctx.fillRect(rx, ry - 18, tw + 10, 18);
  ctx.fillStyle = "#000";
  ctx.fillText(label, rx + 5, ry - 5);
}

async function analyzeFatigueFrame(video) {
  if (!video || video.paused || video.ended) return;
  const imageData = captureVideoFrame(video);
  if (!imageData) {
    updateCam2Status("Capture impossible");
    return;
  }
  updateCam2Status("Analyse en cours…");

  try {
    const res = await fetch("/api/fatigue-detection/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({ image: imageData, camera: "cam2" }),
    });
    const data = await res.json();

    if (data.status === "success") {
      const pct = Math.round(data.confidence * 100);
      if (data.fatigue_detected) {
        const level = data.details?.fatigue_level || "modérée";
        updateCam2Status(`FATIGUE ${level.toUpperCase()} (${pct}%)`);
        addAlert("warning", "Fatigue", `Fatigue détectée — niveau ${level}`);
        showPopupNotification(
          `Fatigue (${pct}%) — niveau ${level}`,
          "warning",
          false,
        );
      } else {
        updateCam2Status(`OK (${pct}%)`);
      }

      if (Array.isArray(data.details?.bbox)) {
        drawFatigueOverlay(data.details.bbox, video, data.fatigue_detected);
      } else {
        clearOverlayCanvas("cam2-overlay-canvas");
      }

      _notifyDetection({
        cam: "cam2",
        type: "fatigue",
        detected: data.fatigue_detected,
        confidence: data.confidence,
        details: data.details || {},
      });
    } else {
      updateCam2Status("Erreur API");
      clearOverlayCanvas("cam2-overlay-canvas");
    }
  } catch (err) {
    updateCam2Status("Erreur détection");
    console.error("[SafeVision] fetch fatigue:", err);
  }
}

function startFatigueDetection(video) {
  if (!video || fatigueDetectionInterval) return;
  analyzeFatigueFrame(video);
  fatigueDetectionInterval = setInterval(
    () => analyzeFatigueFrame(video),
    1500,
  );
  fatigueVideoAnalysisActive = true;
  updateCam2Status("Analyse active");
}

function stopFatigueDetection() {
  if (fatigueDetectionInterval) {
    clearInterval(fatigueDetectionInterval);
    fatigueDetectionInterval = null;
  }
  fatigueVideoAnalysisActive = false;
  clearOverlayCanvas("cam2-overlay-canvas");
  updateCam2Status("Analyse arrêtée");
}

function loadCam2Video(path) {
  const v = document.getElementById("cam2-video");
  if (!v) return;
  v.src = path;
  v.load();
  v.play().catch(() => {});
  if (!fatigueVideoAnalysisActive) startFatigueDetection(v);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 3 — SPILL DETECTION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam3Status(text) {
  const el =
    document.getElementById("cam3-spill-status") ||
    document.getElementById("cam3-fall-status");
  if (el) el.textContent = "Detection: " + text;
}

function drawSpillOverlay(canvasId, data, video, isSpill) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect = getContainedVideoRect(canvas, video);
  const polygons = Array.isArray(data?.polygons) ? data.polygons : [];

  ctx.lineWidth = 3;
  ctx.setLineDash([]);
  polygons.forEach((points) => {
    if (!Array.isArray(points) || points.length < 3) return;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = rect.x + point[0] * rect.scaleX;
      const y = rect.y + point[1] * rect.scaleY;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = "rgba(0,194,255,0.28)";
    ctx.strokeStyle = "rgba(0,194,255,0.95)";
    ctx.fill();
    ctx.stroke();
  });

  const bbox = data?.bbox;
  let labelX = 12,
    labelY = 24;
  if (Array.isArray(bbox) && bbox.length === 4) {
    const [x1, y1, x2, y2] = bbox;
    const rx = rect.x + x1 * rect.scaleX;
    const ry = rect.y + y1 * rect.scaleY;
    ctx.strokeStyle = isSpill
      ? "rgba(0,194,255,0.95)"
      : "rgba(74,227,181,0.95)";
    ctx.strokeRect(rx, ry, (x2 - x1) * rect.scaleX, (y2 - y1) * rect.scaleY);
    labelX = rx;
    labelY = Math.max(18, ry);
  }

  if (!isSpill && !polygons.length && !Array.isArray(bbox)) return;
  ctx.fillStyle = isSpill ? "rgba(0,194,255,0.92)" : "rgba(74,227,181,0.90)";
  ctx.font = "bold 11px monospace";
  const label = isSpill ? "SPILL" : "OK";
  const tw = ctx.measureText(label).width;
  ctx.fillRect(labelX, labelY - 18, tw + 10, 18);
  ctx.fillStyle = "#001014";
  ctx.fillText(label, labelX + 5, labelY - 5);
}

async function analyzeSpillFrame(video, cameraId) {
  if (!video || video.paused || video.ended) return;
  if (spillRequestInFlight[cameraId]) return;
  spillRequestInFlight[cameraId] = true;

  const imageData = captureVideoFrame(video);
  const updateStatus =
    cameraId === "cam4" ? updateCam4Status : updateCam3Status; // kept for safety
  const canvasId = `${cameraId}-overlay-canvas`;

  if (!imageData) {
    updateStatus("Capture impossible");
    spillRequestInFlight[cameraId] = false;
    return;
  }
  updateStatus("Analyse en cours...");

  try {
    const res = await fetch("/api/spill-detection/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({ image: imageData, camera: cameraId }),
    });
    const data = await res.json();

    if (data.status === "success") {
      const pct = Math.round(data.confidence * 100);
      const holdText = data.details?.temporal_hold ? " suivi" : "";
      updateStatus(
        data.spill_detected ? `SPILL${holdText} (${pct}%)` : `OK (${pct}%)`,
      );

      if (
        Array.isArray(data.details?.bbox) ||
        Array.isArray(data.details?.polygons)
      ) {
        drawSpillOverlay(
          canvasId,
          data.details || {},
          video,
          data.spill_detected,
        );
      } else {
        clearOverlayCanvas(canvasId);
      }

      const now = Date.now();
      const canAlert =
        now - (spillLastAlertAt[cameraId] || 0) > SPILL_ALERT_COOLDOWN_MS;
      if (data.spill_detected && canAlert) {
        spillLastAlertAt[cameraId] = now;
        addAlert(
          "critical",
          "Spill",
          `Deversement detecte sur ${cameraId.toUpperCase()}`,
        );
        showPopupNotification(
          `[${cameraId.toUpperCase()}] Deversement detecte (${pct}%)`,
          "critical",
        );
      }

      _notifyDetection({
        cam: cameraId,
        type: "spill",
        detected: data.spill_detected,
        confidence: data.confidence,
        details: data.details || {},
      });
    } else {
      updateStatus("Erreur API");
      clearOverlayCanvas(canvasId);
    }
  } catch (err) {
    updateStatus("Erreur detection");
    console.error(`[SafeVision] fetch ${cameraId}:`, err);
  } finally {
    spillRequestInFlight[cameraId] = false;
  }
}

function analyzeCam3Frame(video) {
  return analyzeSpillFrame(video, "cam3");
}

function startCam3Detection(video) {
  if (!video || cam3DetectionInterval) return;
  analyzeCam3Frame(video);
  cam3DetectionInterval = setInterval(
    () => analyzeCam3Frame(video),
    SPILL_ANALYSIS_INTERVAL_MS,
  );
  cam3VideoAnalysisActive = true;
  updateCam3Status("Analyse active");
}

function stopCam3Detection() {
  if (cam3DetectionInterval) {
    clearInterval(cam3DetectionInterval);
    cam3DetectionInterval = null;
  }
  cam3VideoAnalysisActive = false;
  clearOverlayCanvas("cam3-overlay-canvas");
  updateCam3Status("Analyse arretee");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 4 — PPE DETECTION (Motaz)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam4Status(text) {
  const el = document.getElementById("cam4-ppe-status");
  if (el) el.textContent = "Detection: " + text;
}

function drawPPEOverlay(canvasId, data, video, isViolation) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect = getContainedVideoRect(canvas, video);
  const detections = Array.isArray(data?.detections) ? data.detections : [];

  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  detections.forEach((det) => {
    const box = det.bbox;
    if (!box || box.length < 4) return;
    const [x1, y1, x2, y2] = box;
    const rx = rect.x + x1 * rect.scaleX;
    const ry = rect.y + y1 * rect.scaleY;
    const rw = (x2 - x1) * rect.scaleX;
    const rh = (y2 - y1) * rect.scaleY;

    let color = "rgba(74,227,181,0.95)";
    let bgColor = "rgba(74,227,181,0.90)";
    const label = (det.label || "").toLowerCase();

    if (label === "human" || label === "person" || label === "worker") {
      color = "rgba(255,184,0,0.95)";
      bgColor = "rgba(255,184,0,0.90)";
    } else if (isViolation) {
      color = "rgba(255,59,47,0.95)";
      bgColor = "rgba(255,59,47,0.90)";
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(rx, ry, rw, rh);

    const confText = Math.round((det.confidence || 0) * 100) + "%";
    const displayLabel = (det.label || "Object").toUpperCase() + " " + confText;
    ctx.font = "bold 11px monospace";
    const tw = ctx.measureText(displayLabel).width;
    ctx.fillStyle = bgColor;
    ctx.fillRect(rx, ry - 18, tw + 8, 18);
    ctx.fillStyle = "#001014";
    ctx.fillText(displayLabel, rx + 4, ry - 5);
  });

  if (isViolation) {
    ctx.fillStyle = "rgba(255,59,47,0.92)";
    ctx.font = "bold 14px monospace";
    const bannerText = "⚠ PPE VIOLATION";
    const bw = ctx.measureText(bannerText).width + 16;
    ctx.fillRect(8, 8, bw, 26);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillText(bannerText, 16, 26);
  }
}

async function analyzePPEFrame(video, cameraId = "cam4") {
  if (!video || video.paused || video.ended) return;
  if (ppeRequestInFlight) return;
  ppeRequestInFlight = true;

  const imageData = captureVideoFrame(video);
  const updateStatus = updateCam4Status;
  const canvasId = `${cameraId}-overlay-canvas`;

  if (!imageData) {
    updateStatus("Capture impossible");
    ppeRequestInFlight = false;
    return;
  }
  updateStatus("Analyse en cours...");

  try {
    const res = await fetch("/api/ppe-detection/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({ image: imageData, camera: cameraId }),
    });
    const data = await res.json();

    if (data.status === "success") {
      const pct = Math.round(data.confidence * 100);
      const violation = data.ppe_violation;
      updateStatus(violation ? `VIOLATION (${pct}%)` : `OK (${pct}%)`);

      if (
        Array.isArray(data.details?.detections) &&
        data.details.detections.length > 0
      ) {
        drawPPEOverlay(canvasId, data.details || {}, video, violation);
      } else {
        clearOverlayCanvas(canvasId);
      }

      const now = Date.now();
      const canAlert = now - ppeLastAlertAt > PPE_ALERT_COOLDOWN_MS;
      if (violation && canAlert) {
        ppeLastAlertAt = now;
        addAlert(
          "warning",
          "PPE",
          `Violation EPI detectee sur ${cameraId.toUpperCase()}`,
        );
        showPopupNotification(
          `[${cameraId.toUpperCase()}] Violation EPI detectee (${pct}%)`,
          "warning",
        );
      }

      _notifyDetection({
        cam: cameraId,
        type: "ppe",
        detected: violation,
        confidence: data.confidence,
        details: data.details || {},
      });
    } else {
      updateStatus("Erreur API");
      clearOverlayCanvas(canvasId);
    }
  } catch (err) {
    updateStatus("Erreur detection");
    console.error(`[SafeVision] fetch ${cameraId} ppe:`, err);
  } finally {
    ppeRequestInFlight = false;
  }
}

function analyzeCam4Frame(video) {
  return analyzePPEFrame(video, "cam4");
}

function startCam4Detection(video) {
  if (!video || cam4DetectionInterval) return;
  analyzeCam4Frame(video);
  cam4DetectionInterval = setInterval(
    () => analyzeCam4Frame(video),
    PPE_ANALYSIS_INTERVAL_MS,
  );
  cam4VideoAnalysisActive = true;
  updateCam4Status("Analyse active");
}

function stopCam4Detection() {
  if (cam4DetectionInterval) {
    clearInterval(cam4DetectionInterval);
    cam4DetectionInterval = null;
  }
  cam4VideoAnalysisActive = false;
  clearOverlayCanvas("cam4-overlay-canvas");
  updateCam4Status("Analyse arretee");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 5 — MANHOLE DETECTION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam5Status(text) {
  const el = document.getElementById("cam5-manhole-status");
  if (el) el.textContent = "Detection: " + text;
}

function drawManholeOverlay(data, video) {
  const canvas = document.getElementById("cam5-overlay-canvas");
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect = getContainedVideoRect(canvas, video);
  const polygons = Array.isArray(data?.polygons) ? data.polygons : [];
  const isOpen = data?.manhole_state === "open";

  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.lineWidth = 4;
  polygons.forEach((points) => {
    if (!Array.isArray(points) || points.length < 3) return;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = rect.x + point[0] * rect.scaleX;
      const y = rect.y + point[1] * rect.scaleY;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.shadowColor = isOpen ? "rgba(255,107,0,0.35)" : "rgba(74,227,181,0.35)";
    ctx.shadowBlur = 10;
    ctx.fillStyle = isOpen ? "rgba(255,107,0,0.22)" : "rgba(74,227,181,0.18)";
    ctx.strokeStyle = isOpen ? "rgba(255,107,0,0.95)" : "rgba(74,227,181,0.92)";
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;
  });

  const bbox = data?.bbox;
  let labelX = 12,
    labelY = 24;
  if (Array.isArray(bbox) && bbox.length === 4) {
    const [x1, y1, x2, y2] = bbox;
    const rx = rect.x + x1 * rect.scaleX;
    const ry = rect.y + y1 * rect.scaleY;
    ctx.strokeStyle = isOpen ? "rgba(255,107,0,0.95)" : "rgba(74,227,181,0.95)";
    ctx.strokeRect(rx, ry, (x2 - x1) * rect.scaleX, (y2 - y1) * rect.scaleY);
    labelX = rx;
    labelY = Math.max(18, ry);
  }

  if (!Array.isArray(bbox) && !polygons.length) return;
  const label = (data?.manhole_state || "unknown").toUpperCase();
  ctx.fillStyle = isOpen ? "rgba(255,107,0,0.94)" : "rgba(74,227,181,0.94)";
  ctx.font = "bold 11px monospace";
  const tw = ctx.measureText(label).width;
  ctx.fillRect(labelX, labelY - 18, tw + 10, 18);
  ctx.fillStyle = "#081014";
  ctx.fillText(label, labelX + 5, labelY - 5);
}

async function analyzeCam5Frame(video) {
  if (!video || video.paused || video.ended) return;
  if (manholeRequestInFlight) return;
  manholeRequestInFlight = true;
  const imageData = captureVideoFrame(video);
  if (!imageData) {
    updateCam5Status("Capture impossible");
    manholeRequestInFlight = false;
    return;
  }
  updateCam5Status("Analyse en cours...");

  try {
    const res = await fetch("/api/manhole-detection/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({ image: imageData, camera: "cam5" }),
    });
    const data = await res.json();

    if (data.status === "success") {
      const pct = Math.round(data.confidence * 100);
      const holdText = data.details?.temporal_hold ? " suivi" : "";
      const state = (data.details?.manhole_state || "unknown").toUpperCase();
      updateCam5Status(
        data.manhole_detected ? `${state}${holdText} (${pct}%)` : "OK (0%)",
      );

      if (
        Array.isArray(data.details?.bbox) ||
        Array.isArray(data.details?.polygons)
      ) {
        drawManholeOverlay(data.details || {}, video);
      } else {
        clearOverlayCanvas("cam5-overlay-canvas");
      }

      const now = Date.now();
      const canAlert = now - manholeLastAlertAt > MANHOLE_ALERT_COOLDOWN_MS;
      if (
        data.manhole_detected &&
        data.details?.manhole_state === "open" &&
        canAlert
      ) {
        manholeLastAlertAt = now;
        addAlert("critical", "Manhole", "Manhole ouvert detecte sur CAM5");
        showPopupNotification(`[CAM5] Manhole ouvert (${pct}%)`, "critical");
      }

      _notifyDetection({
        cam: "cam5",
        type: "manhole",
        detected: data.manhole_detected,
        confidence: data.confidence,
        details: data.details || {},
      });
    } else {
      updateCam5Status("Erreur API");
      clearOverlayCanvas("cam5-overlay-canvas");
    }
  } catch (err) {
    updateCam5Status("Erreur detection");
    console.error("[SafeVision] fetch cam5:", err);
  } finally {
    manholeRequestInFlight = false;
  }
}

function startCam5Detection(video) {
  if (!video || cam5DetectionInterval) return;
  analyzeCam5Frame(video);
  cam5DetectionInterval = setInterval(
    () => analyzeCam5Frame(video),
    MANHOLE_ANALYSIS_INTERVAL_MS,
  );
  cam5VideoAnalysisActive = true;
  updateCam5Status("Analyse active");
}

function stopCam5Detection() {
  if (cam5DetectionInterval) {
    clearInterval(cam5DetectionInterval);
    cam5DetectionInterval = null;
  }
  cam5VideoAnalysisActive = false;
  clearOverlayCanvas("cam5-overlay-canvas");
  updateCam5Status("Analyse arretee");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 6 — EXIT EMERGENCY DETECTION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam6Status(text) {
  ["cam6-exit-status", "cam6-exit-status-secondary"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.textContent = "Detection: " + text;
  });
}

function drawExitOverlay(
  canvas,
  video,
  detections,
  exitBbox,
  obstacleBbox,
  isBlocked,
) {
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect = getContainedVideoRect(canvas, video);

  (detections || []).forEach((det) => {
    const box = det.bbox || det.bbox_xyxy;
    if (!box || box.length < 4) return;
    const x1 = rect.x + box[0] * rect.scaleX;
    const y1 = rect.y + box[1] * rect.scaleY;
    const x2 = rect.x + box[2] * rect.scaleX;
    const y2 = rect.y + box[3] * rect.scaleY;
    const isExit = det.label === "exit" || det.class_name === "exit";
    const color = isExit ? "#388E3C" : "#F57C00";

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label =
      (det.label || det.class_name || "") +
      "  " +
      Math.round((det.confidence || 0) * 100) +
      "%";
    ctx.font = "bold 11px monospace";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1 - 18, tw + 8, 18);
    ctx.fillStyle = "white";
    ctx.fillText(label, x1 + 4, y1 - 4);
  });

  if (isBlocked && exitBbox && exitBbox.length === 4) {
    const x1 = rect.x + exitBbox[0] * rect.scaleX;
    const y1 = rect.y + exitBbox[1] * rect.scaleY;
    const x2 = rect.x + exitBbox[2] * rect.scaleX;
    const y2 = rect.y + exitBbox[3] * rect.scaleY;
    ctx.strokeStyle = "#D32F2F";
    ctx.lineWidth = 4;
    ctx.setLineDash([]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = "rgba(211,47,47,0.12)";
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    const txt = "EXIT BLOCKED";
    ctx.font = "bold 13px monospace";
    const tw = ctx.measureText(txt).width;
    ctx.fillStyle = "#D32F2F";
    ctx.fillRect(x1, y2 + 2, tw + 10, 20);
    ctx.fillStyle = "white";
    ctx.fillText(txt, x1 + 5, y2 + 16);
  }

  if (isBlocked && obstacleBbox && obstacleBbox.length === 4) {
    const x1 = rect.x + obstacleBbox[0] * rect.scaleX;
    const y1 = rect.y + obstacleBbox[1] * rect.scaleY;
    const x2 = rect.x + obstacleBbox[2] * rect.scaleX;
    const y2 = rect.y + obstacleBbox[3] * rect.scaleY;
    ctx.strokeStyle = "#FF6B00";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }

  const badgeTxt = isBlocked ? "EXIT BLOCKED" : "EXIT ACCESSIBLE";
  const badgeColor = isBlocked ? "#D32F2F" : "#388E3C";
  ctx.font = "bold 12px monospace";
  const bw = ctx.measureText(badgeTxt).width + 16;
  ctx.fillStyle = badgeColor;
  ctx.fillRect(canvas.width - bw - 6, 8, bw, 22);
  ctx.fillStyle = "white";
  ctx.fillText(badgeTxt, canvas.width - bw, 24);
}

async function analyzeCam6Frame(video, runToken = cam6RunToken) {
  if (!video || video.paused || video.ended || !cam6VideoAnalysisActive) return;
  if (exitRequestInFlight || runToken !== cam6RunToken) return;
  exitRequestInFlight = true;

  const imageData = captureVideoFrame(video);
  if (!imageData) {
    if (runToken === cam6RunToken && cam6VideoAnalysisActive)
      updateCam6Status("Capture impossible");
    exitRequestInFlight = false;
    return;
  }
  updateCam6Status("Analyse en cours...");

  try {
    const res = await fetch("/api/blocked-exit-detection/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({ image: imageData, camera: "cam6", conf: 0.25 }),
    });
    const data = await res.json();
    if (runToken !== cam6RunToken || !cam6VideoAnalysisActive || video.paused)
      return;

    if (data.status === "success") {
      const blocked = data.blocked_exit_detected || false;
      const conf = Math.round((data.confidence || 0) * 100);
      const details = data.details || {};
      const detections = details.detections || [];
      const exitBbox = details.exit_bbox;
      const obstacleBbox = details.obstacle_bbox;
      const canvas = document.getElementById("cam6-overlay-canvas");
      const exits = detections.filter(
        (d) => d.label === "exit" || d.class_name === "exit",
      ).length;
      const obs = detections.filter(
        (d) => d.label === "obstacle" || d.class_name === "obstacle",
      ).length;

      if (blocked) {
        updateCam6Status(`BLOQUEE (${conf}%)`);
      } else if (detections.length > 0) {
        updateCam6Status(`exits=${exits}  obstacles=${obs}  LIBRE`);
      } else {
        updateCam6Status("Aucune sortie detectee");
      }

      drawExitOverlay(
        canvas,
        video,
        detections,
        exitBbox,
        obstacleBbox,
        blocked,
      );

      const now = Date.now();
      const canAlert = now - exitLastAlertAt > EXIT_ALERT_COOLDOWN_MS;
      if (blocked && canAlert) {
        exitLastAlertAt = now;
        addAlert(
          "critical",
          "Exit",
          `Sortie de secours bloquee sur CAM6 (${conf}%)`,
        );
        showPopupNotification(`[CAM6] Sortie bloquee (${conf}%)`, "critical");
      }

      _notifyDetection({
        cam: "cam6",
        type: "blocked_exit",
        detected: blocked,
        confidence: data.confidence,
        details,
      });
    } else {
      updateCam6Status("Erreur API");
      clearOverlayCanvas("cam6-overlay-canvas");
    }
  } catch (err) {
    if (runToken === cam6RunToken && cam6VideoAnalysisActive) {
      updateCam6Status("Erreur detection");
      console.error("[SafeVision] fetch cam6:", err);
    }
  } finally {
    exitRequestInFlight = false;
  }
}

function startCam6Detection(video) {
  if (!video || cam6DetectionInterval) return;
  cam6VideoAnalysisActive = true;
  cam6RunToken += 1;
  const runToken = cam6RunToken;
  analyzeCam6Frame(video, runToken);
  cam6DetectionInterval = setInterval(
    () => analyzeCam6Frame(video, runToken),
    EXIT_ANALYSIS_INTERVAL_MS,
  );
  updateCam6Status("Analyse active");
}

function stopCam6Detection() {
  if (cam6DetectionInterval) {
    clearInterval(cam6DetectionInterval);
    cam6DetectionInterval = null;
  }
  cam6VideoAnalysisActive = false;
  cam6RunToken += 1;
  exitRequestInFlight = false;
  clearOverlayCanvas("cam6-overlay-canvas");
  updateCam6Status("Analyse arretee");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 7 — SIGN DEFECT DETECTION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam7Status(text) {
  const el = document.getElementById("cam7-sign-status");
  if (el) el.textContent = "Detection: " + text;
}

function drawSignOverlay(canvasId, data, video, isDefective) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const accentColor = isDefective
    ? "rgba(74, 195, 255, 0.95)"
    : "rgba(181, 227, 74, 0.95)"; // Soft Orange vs Teal

  // 1. Draw Full-Frame Border
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 4;
  ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);

  // 2. Draw HUD Background
  ctx.fillStyle = "rgba(30, 30, 30, 0.75)";
  ctx.fillRect(8, 8, 310, 95);
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 1;
  ctx.strokeRect(8, 8, 310, 95);

  // 3. Draw Text
  ctx.font = "bold 12px monospace";
  ctx.fillStyle = "#F0F0F0";
  ctx.fillText(`CAT: ${data.category} - ${data.category_name}`, 16, 28);

  ctx.fillStyle = "#A0A0A0";
  ctx.font = "11px monospace";
  ctx.fillText(
    `ROUTER CONF: ${Math.round((data.cat_confidence || 0) * 100)}%`,
    16,
    48,
  );
  ctx.fillText(
    `DEFECT: ${data.defect_score?.toFixed(3)} (Th: ${data.threshold?.toFixed(3)})`,
    16,
    64,
  );

  ctx.font = "bold 13px monospace";
  ctx.fillStyle = accentColor;
  ctx.fillText(`VERDICT: ${data.verdict}`, 16, 88);
}

async function analyzeSignFrame(video, cameraId = "cam7") {
  if (!video || video.paused || video.ended) return;
  if (signRequestInFlight) return;
  signRequestInFlight = true;

  const imageData = captureVideoFrame(video);
  const canvasId = `${cameraId}-overlay-canvas`;

  if (!imageData) {
    updateCam7Status("Capture impossible");
    signRequestInFlight = false;
    return;
  }
  updateCam7Status("Analyse en cours...");

  try {
    const res = await fetch("/api/sign-detect/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({ image: imageData, camera: cameraId }),
    });
    const data = await res.json();

    if (data.status === "success") {
      const result = data.data || {};
      const isDefective = result.is_defective || false;
      const verdict = result.verdict || "UNKNOWN";
      const conf = Math.round((result.defect_score || 0) * 100);

      updateCam7Status(
        isDefective ? `${verdict} (${conf}%)` : `${verdict} (${conf}%)`,
      );

      drawSignOverlay(canvasId, result, video, isDefective);

      const now = Date.now();
      const canAlert = now - signLastAlertAt > SIGN_ALERT_COOLDOWN_MS;
      if (isDefective && canAlert) {
        signLastAlertAt = now;
        addAlert(
          "warning",
          "Sign",
          `Panneau défectueux détecté sur ${cameraId.toUpperCase()}`,
        );
        showPopupNotification(
          `[${cameraId.toUpperCase()}] Panneau Défectueux (${conf}%)`,
          "warning",
        );
      }

      _notifyDetection({
        cam: cameraId,
        type: "sign_defect",
        detected: isDefective,
        confidence: result.defect_score,
        details: result,
      });
    } else {
      updateCam7Status("Erreur API");
      clearOverlayCanvas(canvasId);
    }
  } catch (err) {
    updateCam7Status("Erreur detection");
    console.error(`[SafeVision] fetch ${cameraId} sign:`, err);
  } finally {
    signRequestInFlight = false;
  }
}

function analyzeCam7Frame(video) {
  return analyzeSignFrame(video, "cam7");
}

function startCam7Detection(video) {
  if (!video || cam7DetectionInterval) return;
  analyzeCam7Frame(video);
  cam7DetectionInterval = setInterval(
    () => analyzeCam7Frame(video),
    SIGN_ANALYSIS_INTERVAL_MS,
  );
  cam7VideoAnalysisActive = true;
  updateCam7Status("Analyse active");
}

function stopCam7Detection() {
  if (cam7DetectionInterval) {
    clearInterval(cam7DetectionInterval);
    cam7DetectionInterval = null;
  }
  cam7VideoAnalysisActive = false;
  clearOverlayCanvas("cam7-overlay-canvas");
  updateCam7Status("Analyse arretee");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 8 — WORKER TRACKING (PEOPLENET)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam8Status(text) {
  const el = document.getElementById("cam8-worker-status");
  if (el) el.textContent = "Detection: " + text;
}

function drawWorkerTrackingOverlay(canvasId, details, video) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !video) return;

  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect = getContainedVideoRect(canvas, video);

  const lineStart = Array.isArray(details?.line_start)
    ? details.line_start
    : null;
  const lineEnd = Array.isArray(details?.line_end) ? details.line_end : null;

  if (lineStart && lineEnd) {
    const lx1 = rect.x + lineStart[0] * rect.scaleX;
    const ly1 = rect.y + lineStart[1] * rect.scaleY;
    const lx2 = rect.x + lineEnd[0] * rect.scaleX;
    const ly2 = rect.y + lineEnd[1] * rect.scaleY;
    ctx.strokeStyle = "rgba(255,59,47,0.95)";
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 6]);
    ctx.beginPath();
    ctx.moveTo(lx1, ly1);
    ctx.lineTo(lx2, ly2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const tracks = Array.isArray(details?.tracks) ? details.tracks : [];
  tracks.forEach((track) => {
    const box = track.bbox;
    if (!Array.isArray(box) || box.length < 4) return;

    const [x1, y1, x2, y2] = box;
    const rx = rect.x + x1 * rect.scaleX;
    const ry = rect.y + y1 * rect.scaleY;
    const rw = (x2 - x1) * rect.scaleX;
    const rh = (y2 - y1) * rect.scaleY;

    const isCounted = Boolean(track.counted);
    const color = isCounted ? "rgba(0,128,255,0.95)" : "rgba(0,255,128,0.95)";

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(rx, ry, rw, rh);

    const label = `ID ${track.track_id}`;
    ctx.font = "bold 11px monospace";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(rx, Math.max(0, ry - 18), tw + 10, 18);
    ctx.fillStyle = "#001014";
    ctx.fillText(label, rx + 5, Math.max(13, ry - 5));

    const centroid = track.centroid;
    if (Array.isArray(centroid) && centroid.length === 2) {
      const cx = rect.x + centroid[0] * rect.scaleX;
      const cy = rect.y + centroid[1] * rect.scaleY;
      ctx.beginPath();
      ctx.fillStyle = "rgba(255,255,0,0.95)";
      ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  });

  const countIn = Number(details?.count_in || 0);
  const countOut = Number(details?.count_out || 0);
  const total = Number(details?.total_crossings || 0);

  ctx.fillStyle = "rgba(20,20,20,0.72)";
  ctx.fillRect(8, 8, 270, 70);
  ctx.strokeStyle = "rgba(255,255,255,0.24)";
  ctx.lineWidth = 1;
  ctx.strokeRect(8, 8, 270, 70);

  ctx.font = "bold 12px monospace";
  ctx.fillStyle = "rgba(74,227,181,0.95)";
  ctx.fillText(`IN  (left to right): ${countIn}`, 16, 28);
  ctx.fillStyle = "rgba(0,128,255,0.95)";
  ctx.fillText(`OUT (right to left): ${countOut}`, 16, 46);
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.fillText(`Total: ${total}`, 16, 64);
}

async function analyzeCam8Frame(video, options = {}) {
  if (!video || video.paused || video.ended || !cam8VideoAnalysisActive) return;
  if (cam8RequestInFlight) return;
  cam8RequestInFlight = true;

  const imageData = captureVideoFrame(video, 640, 640);
  if (!imageData) {
    updateCam8Status("Capture impossible");
    cam8RequestInFlight = false;
    return;
  }

  const firstCallReset = Boolean(options.reset);
  const loopReset =
    !firstCallReset && video.currentTime + 0.2 < cam8LastVideoTime;
  const shouldReset = firstCallReset || loopReset;
  cam8LastVideoTime = video.currentTime || 0;

  updateCam8Status("Analyse en cours...");

  try {
    const res = await fetch("/api/worker-tracking-detection/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({
        image: imageData,
        camera: "cam8",
        reset: shouldReset,
      }),
    });
    const data = await res.json();

    if (data.status === "success") {
      const details = data.details || {};
      const countIn = Number(details.count_in || 0);
      const countOut = Number(details.count_out || 0);
      const total = Number(details.total_crossings || 0);
      const trackCount = Number(details.track_count || 0);
      const crossedCount = Number(details.crossed_count || 0);
      const conf = Math.round((data.confidence || 0) * 100);

      if (trackCount > 0) {
        updateCam8Status(
          `IN=${countIn} OUT=${countOut} TOTAL=${total} TRACKS=${trackCount}`,
        );
      } else {
        updateCam8Status(`Aucun worker (TOTAL=${total})`);
      }

      drawWorkerTrackingOverlay("cam8-overlay-canvas", details, video);

      const now = Date.now();
      const canAlert =
        now - cam8LastAlertAt > WORKER_TRACKING_ALERT_COOLDOWN_MS;
      if (crossedCount > 0 && canAlert) {
        cam8LastAlertAt = now;
        addAlert(
          "warning",
          "Tracking",
          `Franchissement detecte sur CAM8 (IN=${countIn} OUT=${countOut})`,
        );
        showPopupNotification(
          `[CAM8] Franchissement worker (${conf}%)`,
          "warning",
        );
      }

      _notifyDetection({
        cam: "cam8",
        type: "worker_tracking",
        detected: data.worker_tracking_detected,
        confidence: data.confidence,
        details,
      });
    } else {
      updateCam8Status("Erreur API");
      clearOverlayCanvas("cam8-overlay-canvas");
    }
  } catch (err) {
    updateCam8Status("Erreur detection");
    console.error("[SafeVision] fetch cam8 tracking:", err);
  } finally {
    cam8RequestInFlight = false;
  }
}

function startCam8Detection(video) {
  if (!video || cam8VideoAnalysisActive) return;
  cam8VideoAnalysisActive = true;
  cam8RequestInFlight = false;
  cam8LastVideoTime = 0;
  cam8LoopVideoRef = video;
  cam8ResetPending = true;
  cam8FrameLoopToken += 1;
  _clearCam8FrameSchedule();
  _scheduleNextCam8Frame(video, cam8FrameLoopToken);
  updateCam8Status("Analyse active (chaque frame)");
}

function stopCam8Detection() {
  cam8VideoAnalysisActive = false;
  cam8RequestInFlight = false;
  cam8LastVideoTime = 0;
  cam8ResetPending = false;
  _clearCam8FrameSchedule();
  cam8LoopVideoRef = null;
  clearOverlayCanvas("cam8-overlay-canvas");
  updateCam8Status("Analyse arretee");
}

function _clearCam8FrameSchedule() {
  if (cam8RafHandle !== null) {
    cancelAnimationFrame(cam8RafHandle);
    cam8RafHandle = null;
  }
  if (
    cam8VideoFrameCallbackHandle !== null &&
    cam8LoopVideoRef &&
    typeof cam8LoopVideoRef.cancelVideoFrameCallback === "function"
  ) {
    cam8LoopVideoRef.cancelVideoFrameCallback(cam8VideoFrameCallbackHandle);
  }
  cam8VideoFrameCallbackHandle = null;
}

function _scheduleNextCam8Frame(video, loopToken) {
  if (!cam8VideoAnalysisActive || loopToken !== cam8FrameLoopToken) return;

  const analyzeAndContinue = () => {
    if (!cam8VideoAnalysisActive || loopToken !== cam8FrameLoopToken) return;
    const reset = cam8ResetPending;
    cam8ResetPending = false;
    analyzeCam8Frame(video, { reset }).finally(() => {
      _scheduleNextCam8Frame(video, loopToken);
    });
  };

  if (typeof video.requestVideoFrameCallback === "function") {
    cam8VideoFrameCallbackHandle = video.requestVideoFrameCallback(() => {
      cam8VideoFrameCallbackHandle = null;
      analyzeAndContinue();
    });
    return;
  }

  cam8RafHandle = requestAnimationFrame(() => {
    cam8RafHandle = null;
    analyzeAndContinue();
  });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 8 — PROXIMITY DETECTION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam8ProximityStatus(text) {
  const el = document.getElementById("cam8-proximity-status");
  if (el) el.textContent = "Détection: " + text;
}

function drawProximityOverlay(canvasId, details, video, proximity_detected) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect = getContainedVideoRect(canvas, video);

  // Draw bounding boxes for workers (green) and machines (orange)
  if (details && details.detections) {
    for (const det of details.detections) {
      const [x1, y1, x2, y2] = det.bbox;
      const rx = rect.x + x1 * rect.scaleX;
      const ry = rect.y + y1 * rect.scaleY;
      const rw = (x2 - x1) * rect.scaleX;
      const rh = (y2 - y1) * rect.scaleY;
      let color = "rgba(74,227,181,0.8)"; // green for worker
      if (det.label.toLowerCase().includes("machine")) {
        color = "rgba(255,107,0,0.8)"; // orange for machine
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.fillStyle = color;
      ctx.font = "bold 10px monospace";
      const label = det.label;
      const tw = ctx.measureText(label).width;
      ctx.fillRect(rx, ry - 16, tw + 6, 16);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, rx + 3, ry - 3);
    }
  }

  // Draw proximity zones if proximity_alerts
  if (details && details.proximity_alerts) {
    for (const alert of details.proximity_alerts) {
      const worker_center_x = (alert.worker_bbox[0] + alert.worker_bbox[2]) / 2;
      const worker_center_y = (alert.worker_bbox[1] + alert.worker_bbox[3]) / 2;
      const rx = rect.x + worker_center_x * rect.scaleX;
      const ry = rect.y + worker_center_y * rect.scaleY;
      const radius = alert.distance * 50; // scale for display
      let color = "rgba(0,194,255,0.5)"; // blue for vigilance
      if (alert.severity === "critical")
        color = "rgba(255,59,47,0.5)"; // red
      else if (alert.severity === "warning") color = "rgba(255,184,0,0.5)"; // yellow
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(rx, ry, radius, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }
}
async function analyzeProximityFrame(video, cameraId) {
  if (!video || video.paused || video.ended) return;
  if (proximityRequestInFlight) return;
  proximityRequestInFlight = true;

  const imageData = captureVideoFrame(video);
  const canvasId = `${cameraId}-overlay-canvas`;

  // Status selon la caméra
  const updateStatus =
    cameraId === "cam11" ? updateCam11Status : updateCam8ProximityStatus;

  if (!imageData) {
    updateStatus("Capture impossible");
    proximityRequestInFlight = false;
    return;
  }
  updateStatus("Analyse en cours…");

  try {
    const res = await fetch("/api/proximity-detection/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({ image: imageData, camera: cameraId }),
    });
    const data = await res.json();

    if (data.status === "success") {
      const det = data.details || {};
      const nW = det.workers_count || 0;
      const nM = det.machines_count || 0;
      const sev = det.severity || "safe";
      const inc = det.incident_logs || [];

      // ── Statut texte ──────────────────────────────────────────
      let statusTxt = `OK (${nW}W ${nM}M)`;
      if (inc.length > 0) {
        const minDist = Math.min(...inc.map((i) => i.distance_m || 99));
        const labels = {
          critical: "CRITIQUE",
          alert: "ALERTE",
          vigilance: "VIGILANCE",
        };
        statusTxt = `${labels[sev] || "OK"} ${minDist.toFixed(1)}m (${nW}W ${nM}M)`;
      }
      updateStatus(statusTxt);

      // ── Image annotée côté serveur ─────────────────────────────
      if (data.annotated_image) {
        const canvas = document.getElementById(canvasId);
        if (canvas) {
          canvas.style.position = "absolute";
          canvas.style.top = "0";
          canvas.style.left = "0";
          canvas.style.width = "100%";
          canvas.style.height = "100%";
          canvas.style.zIndex = "10";
          canvas.style.pointerEvents = "none";

          canvas.width = video.videoWidth || video.clientWidth || 640;
          canvas.height = video.videoHeight || video.clientHeight || 360;

          const img = new Image();
          img.onload = () => {
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          };
          img.onerror = () =>
            console.warn(`[${cameraId}] Image annotée non chargée`);
          img.src = data.annotated_image;
        }
      } else {
        clearOverlayCanvas(canvasId);
      }

      // ── Alerte ────────────────────────────────────────────────
      const now = Date.now();
      const canAlert = now - proximityLastAlertAt > PROXIMITY_ALERT_COOLDOWN_MS;
      if (data.proximity_detected && canAlert && inc.length > 0) {
        proximityLastAlertAt = now;
        const minDist = Math.min(...inc.map((i) => i.distance_m || 99));
        const lvl = sev === "critical" ? "critical" : "warning";
        addAlert(
          lvl,
          "Proximite",
          `Proximite dangereuse detectee (${minDist.toFixed(1)}m)`,
        );
        showPopupNotification(
          `[${cameraId.toUpperCase()}] Proximite dangereuse (${minDist.toFixed(1)}m)`,
          lvl,
        );
      }

      _notifyDetection({
        cam: cameraId,
        type: "proximity",
        detected: data.proximity_detected,
        confidence: data.confidence,
        details: data.details,
      });
    } else {
      updateStatus("Erreur API");
      clearOverlayCanvas(canvasId);
    }
  } catch (err) {
    updateStatus("Erreur detection");
    console.error(`[SafeVision] ${cameraId} proximity:`, err);
  } finally {
    proximityRequestInFlight = false;
  }
}
function analyzeCam8ProximityFrame(video) {
  return analyzeProximityFrame(video, "cam8");
}

function startCam8ProximityDetection(video) {
  if (!video || cam8ProximityDetectionInterval) return;
  cam8ProximityVideoAnalysisActive = true;
  analyzeCam8ProximityFrame(video);
  cam8ProximityDetectionInterval = setInterval(
    () => analyzeCam8ProximityFrame(video),
    PROXIMITY_ANALYSIS_INTERVAL_MS,
  );
  updateCam8ProximityStatus("Analyse active");
}

function stopCam8ProximityDetection() {
  if (cam8ProximityDetectionInterval) {
    clearInterval(cam8ProximityDetectionInterval);
    cam8ProximityDetectionInterval = null;
  }
  cam8ProximityVideoAnalysisActive = false;
  clearOverlayCanvas("cam8-overlay-canvas");
  updateCam8ProximityStatus("Analyse arrêtée");
}

// CAM 11 — PROXIMITY DETECTION (copie exacte de CAM 8)
let cam11DetectionInterval = null; // ADDED PROXIMITY
let cam11VideoAnalysisActive = false; // ADDED PROXIMITY

function updateCam11Status(text) {
  const el = document.getElementById("cam11-proximity-status");
  if (el) el.textContent = "Détection: " + text;
}

async function analyzeCam11Frame(video, options = {}) {
  if (!video || video.paused || video.ended || !cam11VideoAnalysisActive)
    return;
  return analyzeProximityFrame(video, "cam11");
}

function analyzeCam11ProximityFrame(video) {
  return analyzeCam11Frame(video, "cam11");
}

function startCam11Detection(video) {
  if (!video || cam11DetectionInterval) return;
  cam11VideoAnalysisActive = true;
  analyzeCam11ProximityFrame(video);
  cam11DetectionInterval = setInterval(
    () => analyzeCam11ProximityFrame(video),
    PROXIMITY_ANALYSIS_INTERVAL_MS,
  );
  updateCam11Status("Analyse active");
}

function stopCam11Detection() {
  if (cam11DetectionInterval) {
    clearInterval(cam11DetectionInterval);
    cam11DetectionInterval = null;
  }
  cam11VideoAnalysisActive = false;
  clearOverlayCanvas("cam11-overlay-canvas");
  updateCam11Status("Analyse arrêtée");
}

function loadCam11Video(file) {
  const v = document.getElementById("cam11-video");
  if (!v || !file) return;
  const url = URL.createObjectURL(file);
  v.src = url;
  v.load();
  v.play().catch(() => {});
  document.getElementById("cam-feed-11").style.display = "none";
  document
    .getElementById("cam11-video-overlay")
    .querySelector("span").textContent = `Fichier: ${file.name}`;
  if (!cam11VideoAnalysisActive) startCam11Detection(v);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 9 — POSTURE DETECTION  (version Hanin — règles géométriques COCO-17)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam9Status(text) {
  const el = document.getElementById("cam9-posture-status");
  if (el) el.textContent = "Detection: " + text;
}

// COCO-17 skeleton connections [from, to]
const POSE_SKELETON = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

function drawPostureOverlay(canvasId, details, video, isUnsafe) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect = getContainedVideoRect(canvas, video);

  const safeColor = "rgba(74,227,181,0.95)";
  const unsafeColor = "rgba(255,107,0,0.95)";
  const color = isUnsafe ? unsafeColor : safeColor;
  const conf = Math.round((details.confidence || 0) * 100);
  const reasons = Array.isArray(details.reasons) ? details.reasons : [];
  const kpts = Array.isArray(details.keypoints)
    ? details.keypoints
    : Array.isArray(details.skeleton_keypoints)
      ? details.skeleton_keypoints
      : [];
  const bbox = Array.isArray(details.bbox) ? details.bbox : null;

  // Bounding box
  if (bbox) {
    const [x1, y1, x2, y2] = bbox;
    const rx = rect.x + x1 * rect.scaleX;
    const ry = rect.y + y1 * rect.scaleY;
    const rw = (x2 - x1) * rect.scaleX;
    const rh = (y2 - y1) * rect.scaleY;
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 4]);
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([]);
    const tag = isUnsafe ? "⚠ UNSAFE" : "✓ SAFE";
    ctx.font = "bold 11px monospace";
    const tw = ctx.measureText(tag).width;
    ctx.fillStyle = color;
    ctx.fillRect(rx, ry - 20, tw + 10, 20);
    ctx.fillStyle = isUnsafe ? "#000" : "#001014";
    ctx.fillText(tag, rx + 5, ry - 5);
  }

  // Skeleton lines
  if (kpts.length === 17) {
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    POSE_SKELETON.forEach(([a, b]) => {
      const pa = kpts[a],
        pb = kpts[b];
      if (!pa || !pb) return;
      if ((pa[0] === 0 && pa[1] === 0) || (pb[0] === 0 && pb[1] === 0)) return;
      const ax = rect.x + pa[0] * rect.scaleX;
      const ay = rect.y + pa[1] * rect.scaleY;
      const bx = rect.x + pb[0] * rect.scaleX;
      const by = rect.y + pb[1] * rect.scaleY;
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    });
    kpts.forEach(([kx, ky], i) => {
      if (kx === 0 && ky === 0) return;
      const px = rect.x + kx * rect.scaleX;
      const py = rect.y + ky * rect.scaleY;
      ctx.beginPath();
      ctx.arc(px, py, i === 0 ? 5 : 3, 0, Math.PI * 2);
      ctx.fillStyle = i === 0 ? "#fff" : color;
      ctx.fill();
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.stroke();
    });
  }

  // HUD (reasons)
  if (reasons.length > 0 || !isUnsafe) {
    const hudH = 22 + Math.min(reasons.length, 4) * 14 + 8;
    ctx.fillStyle = "rgba(15,15,15,0.82)";
    ctx.fillRect(6, 6, 264, hudH);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(6, 6, 264, hudH);
    ctx.font = "bold 11px monospace";
    ctx.fillStyle = color;
    ctx.fillText(isUnsafe ? "⚠ UNSAFE" : "✓ SAFE", 12, 22);
    ctx.font = "10px monospace";
    ctx.fillStyle = "#ddd";
    reasons
      .slice(0, 4)
      .forEach((r, i) => ctx.fillText("• " + r, 12, 36 + i * 14));
  }
}

async function analyzePostureFrame(video) {
  if (!video || video.paused || video.ended) return;
  if (postureRequestInFlight) return;
  postureRequestInFlight = true;

  const imageData = captureVideoFrame(video);
  if (!imageData) {
    updateCam9Status("Capture impossible");
    postureRequestInFlight = false;
    return;
  }
  updateCam9Status("Analyse en cours...");

  try {
    const res = await fetch("/api/posture-detection/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({ image: imageData, camera: "cam9" }),
    });
    const data = await res.json();

    if (data.status === "success") {
      const isUnsafe = data.unsafe_posture_detected;
      const posture = isUnsafe ? "UNSAFE" : "SAFE";
      updateCam9Status(posture);

      drawPostureOverlay(
        "cam9-overlay-canvas",
        data.details || {},
        video,
        isUnsafe,
      );

      if (isUnsafe) {
        const conf = Math.round((data.confidence || 0) * 100);
        const now = Date.now();
        if (now - postureLastAlertAt > POSTURE_ALERT_COOLDOWN_MS) {
          postureLastAlertAt = now;
          const reasons = (data.details?.reasons || []).slice(0, 2).join(", ");
          addAlert(
            "warning",
            "Posture",
            `Posture incorrecte — ${reasons || "mauvaise position"} (${conf}%)`,
          );
          showPopupNotification(
            `[CAM9] Posture UNSAFE (${conf}%) — ${reasons}`,
            "warning",
          );
        }
      }

      _notifyDetection({
        cam: "cam9",
        type: "posture",
        detected: isUnsafe,
        confidence: data.confidence,
        details: data.details || {},
      });
    } else {
      updateCam9Status("Erreur API");
      clearOverlayCanvas("cam9-overlay-canvas");
    }
  } catch (err) {
    updateCam9Status("Erreur detection");
    console.error("[SafeVision] fetch cam9 posture:", err);
  } finally {
    postureRequestInFlight = false;
  }
}

function startCam9Detection(video) {
  if (!video || cam9DetectionInterval) return;
  analyzePostureFrame(video);
  cam9DetectionInterval = setInterval(
    () => analyzePostureFrame(video),
    POSTURE_ANALYSIS_INTERVAL_MS,
  );
  cam9VideoAnalysisActive = true;
  updateCam9Status("Analyse active");
}

function stopCam9Detection() {
  if (cam9DetectionInterval) {
    clearInterval(cam9DetectionInterval);
    cam9DetectionInterval = null;
  }
  cam9VideoAnalysisActive = false;
  clearOverlayCanvas("cam9-overlay-canvas");
  updateCam9Status("Analyse arretee");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 10 — PANIC DETECTION  (version Hanin — BiLSTM 30 frames, SANS squelette)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam10Status(text) {
  const el = document.getElementById("cam10-panic-status");
  if (el) el.textContent = "Detection: " + text;
}

function drawPanicOverlay(canvasId, details, video, isPanic) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const label = details.label || (isPanic ? "PANIC" : "NORMAL");
  const warming = label === "WARMING_UP";
  const possible = label === "POSSIBLE" || details.possible_panic;
  const color = warming
    ? "rgba(0,194,255,0.80)"
    : isPanic
      ? "rgba(255,59,47,0.95)"
      : possible
        ? "rgba(255,184,0,0.95)"
        : "rgba(74,227,181,0.95)";
  const conf = Math.round((details.confidence || 0) * 100);
  const frames = details.frames_collected || 0;
  const needed = details.frames_needed || 30;

  // Frame border
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);

  // Background flash on panic
  if (isPanic) {
    ctx.fillStyle = "rgba(255,59,47,0.08)";
    ctx.fillRect(2, 2, canvas.width - 4, canvas.height - 4);
  }

  // HUD BiLSTM (pas de squelette)
  ctx.fillStyle = "rgba(20,20,20,0.82)";
  ctx.fillRect(8, 8, 270, warming ? 55 : 70);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.strokeRect(8, 8, 270, warming ? 55 : 70);
  ctx.font = "bold 13px monospace";
  const statusText = warming
    ? "WARMING UP..."
    : isPanic
      ? "🚨 PANIC"
      : possible
        ? "⚠ POSSIBLE"
        : "✓ NORMAL";
  ctx.fillStyle = color;
  ctx.fillText(statusText, 14, 28);
  ctx.font = "10px monospace";
  ctx.fillStyle = "#ccc";
  if (warming) {
    ctx.fillText(`Buffer: ${frames} / ${needed} frames`, 14, 46);
  } else {
    const pPanic = Math.round((details.p_panic || 0) * 100);
    const pNormal = Math.round((details.p_normal || 0) * 100);
    ctx.fillText(`P(panic)=${pPanic}%   P(normal)=${pNormal}%`, 14, 44);
    ctx.fillText(`Confiance: ${conf}%   Frames: ${frames}`, 14, 58);
  }

  // Confidence bar (bottom strip)
  if (!warming) {
    const barW = canvas.width - 20;
    ctx.fillStyle = "rgba(50,50,50,0.7)";
    ctx.fillRect(10, canvas.height - 18, barW, 10);
    ctx.fillStyle = isPanic
      ? "rgba(255,59,47,0.90)"
      : possible
        ? "rgba(255,184,0,0.90)"
        : "rgba(74,227,181,0.90)";
    ctx.fillRect(10, canvas.height - 18, barW * (details.confidence || 0), 10);
  }
}

async function analyzePanicFrame(video) {
  if (!video || video.paused || video.ended) return;
  if (panicRequestInFlight) return;
  panicRequestInFlight = true;

  const imageData = captureVideoFrame(video);
  if (!imageData) {
    updateCam10Status("Capture impossible");
    panicRequestInFlight = false;
    return;
  }
  updateCam10Status("Analyse en cours...");

  try {
    const res = await fetch("/api/panic-detection/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({ image: imageData, camera: "cam10" }),
    });
    const data = await res.json();

    if (data.status === "success") {
      const isPanic = data.panic_detected;
      const label = data.label || "NORMAL";
      const conf = Math.round(data.confidence * 100);
      updateCam10Status(`${label} (${conf}%)`);

      drawPanicOverlay(
        "cam10-overlay-canvas",
        data.details || {},
        video,
        isPanic,
      );
      if (!isPanic && label === "POSSIBLE") {
        showPopupNotification(
          `[CAM10] PANIC POSSIBLE ${conf}%`,
          "warning",
          false,
        );
      }

      if (isPanic) {
        showPopupNotification(
          `[CAM10] PANIC detecte ${conf}%`,
          "critical",
          false,
        );
        const now = Date.now();
        if (now - panicLastAlertAt > PANIC_ALERT_COOLDOWN_MS) {
          panicLastAlertAt = now;
          addAlert(
            "critical",
            "Panic",
            `Comportement de panique detecte sur CAM10 (${conf}%)`,
          );
          showPopupNotification(`[CAM10] PANIC detecte (${conf}%)`, "critical");
        }
      }

      _notifyDetection({
        cam: "cam10",
        type: "panic",
        detected: isPanic,
        confidence: data.confidence,
        details: data.details || {},
      });
    } else {
      updateCam10Status("Erreur API");
      clearOverlayCanvas("cam10-overlay-canvas");
    }
  } catch (err) {
    updateCam10Status("Erreur detection");
    console.error("[SafeVision] fetch cam10 panic:", err);
  } finally {
    panicRequestInFlight = false;
  }
}

function startCam10Detection(video) {
  if (!video || cam10DetectionInterval) return;
  analyzePanicFrame(video);
  cam10DetectionInterval = setInterval(
    () => analyzePanicFrame(video),
    PANIC_ANALYSIS_INTERVAL_MS,
  );
  cam10VideoAnalysisActive = true;
  updateCam10Status("Analyse active");
}

function stopCam10Detection() {
  if (cam10DetectionInterval) {
    clearInterval(cam10DetectionInterval);
    cam10DetectionInterval = null;
  }
  cam10VideoAnalysisActive = false;
  clearOverlayCanvas("cam10-overlay-canvas");
  updateCam10Status("Analyse arretee");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 12 — FIRE/SMOKE DETECTION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam12Status(text) {
  const el = document.getElementById("cam12-fire-status");
  if (el) el.textContent = "Detection: " + text;
}

function drawFireOverlay(canvasId, details, video, fireDetected, smokeDetected) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const isDetected = fireDetected || smokeDetected;
  const color = fireDetected
    ? "rgba(255,59,47,0.95)"
    : smokeDetected
      ? "rgba(255,184,0,0.95)"
      : "rgba(74,227,181,0.95)";
  const conf = Math.round((details.confidence || 0) * 100);

  // Frame border
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);

  // Background flash on fire
  if (fireDetected) {
    ctx.fillStyle = "rgba(255,59,47,0.08)";
    ctx.fillRect(2, 2, canvas.width - 4, canvas.height - 4);
  }

  // Draw bounding boxes if available
  const rect = getContainedVideoRect(canvas, video);
  if (Array.isArray(details.detections)) {
    details.detections.forEach((det) => {
      if (!Array.isArray(det.bbox) || det.bbox.length < 4) return;
      const [x1, y1, x2, y2] = det.bbox;
      const rx = rect.x + x1 * rect.scaleX;
      const ry = rect.y + y1 * rect.scaleY;
      const rw = (x2 - x1) * rect.scaleX;
      const rh = (y2 - y1) * rect.scaleY;
      if (rw <= 0 || rh <= 0) return;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.strokeRect(rx, ry, rw, rh);

      const label = det.class || (det.fire_detected ? "FIRE" : "SMOKE");
      const detConf = Math.round((det.confidence || 0) * 100);
      ctx.font = "bold 11px monospace";
      const tw = ctx.measureText(`${label} ${detConf}%`).width;
      ctx.fillStyle = color;
      ctx.fillRect(rx, ry - 20, tw + 12, 20);
      ctx.fillStyle = "#071014";
      ctx.fillText(`${label} ${detConf}%`, rx + 6, ry - 6);
    });
  }

  // HUD
  ctx.fillStyle = "rgba(20,20,20,0.82)";
  ctx.fillRect(8, 8, 260, 60);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.strokeRect(8, 8, 260, 60);
  ctx.font = "bold 13px monospace";
  const statusText = fireDetected
    ? "🔥 FEU DETECTE"
    : smokeDetected
      ? "💨 FUMÉE DETECTÉE"
      : "✓ NORMAL";
  ctx.fillStyle = color;
  ctx.fillText(statusText, 14, 28);
  ctx.font = "10px monospace";
  ctx.fillStyle = "#ccc";
  const fireFlag = details.fire_detected ? "FIRE" : "-";
  const smokeFlag = details.smoke_detected ? "SMOKE" : "-";
  ctx.fillText(`Fire: ${fireFlag}  Smoke: ${smokeFlag}`, 14, 44);
  ctx.fillText(`Confiance: ${conf}%`, 14, 58);

  // Confidence bar (bottom strip)
  const barW = canvas.width - 20;
  ctx.fillStyle = "rgba(50,50,50,0.7)";
  ctx.fillRect(10, canvas.height - 18, barW, 10);
  ctx.fillStyle = color;
  ctx.fillRect(10, canvas.height - 18, barW * (details.confidence || 0), 10);
}

async function analyzeCam12Frame(video) {
  if (!video || video.paused || video.ended) return;
  if (fireRequestInFlight) return;
  fireRequestInFlight = true;

  const imageData = captureVideoFrame(video);
  if (!imageData) {
    updateCam12Status("Capture impossible");
    fireRequestInFlight = false;
    return;
  }
  updateCam12Status("Analyse en cours...");

  try {
    const res = await fetch("/api/fire-detection/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify({ image: imageData, camera: "cam12" }),
    });
    const data = await res.json();

    if (data.status === "success") {
      const fireDetected = data.fire_detected;
      const smokeDetected = data.smoke_detected;
      const isDetected = fireDetected || smokeDetected;
      const conf = Math.round((data.confidence || 0) * 100);

      const statusText = fireDetected
        ? `FEU (${conf}%)`
        : smokeDetected
          ? `FUMÉE (${conf}%)`
          : "OK";
      updateCam12Status(statusText);

      drawFireOverlay(
        "cam12-overlay-canvas",
        data.details || {},
        video,
        fireDetected,
        smokeDetected,
      );

      if (isDetected) {
        const now = Date.now();
        if (now - fireLastAlertAt > FIRE_ALERT_COOLDOWN_MS) {
          fireLastAlertAt = now;
          const type = fireDetected ? "FEU" : "FUMÉE";
          const severity = fireDetected ? "critical" : "warning";
          addAlert(
            severity,
            type,
            `${type} detecte sur CAM12 (${conf}%)`,
          );
          showPopupNotification(
            `[CAM12] ${type} detecte (${conf}%)`,
            severity,
          );
        }
      }

      _notifyDetection({
        cam: "cam12",
        type: "fire",
        detected: isDetected,
        confidence: data.confidence,
        details: data.details || {},
      });
    } else {
      updateCam12Status("Erreur API");
      clearOverlayCanvas("cam12-overlay-canvas");
    }
  } catch (err) {
    updateCam12Status("Erreur detection");
    console.error("[SafeVision] fetch cam12 fire:", err);
  } finally {
    fireRequestInFlight = false;
  }
}

function startCam12Detection(video) {
  if (!video || cam12DetectionInterval) return;
  analyzeCam12Frame(video);
  cam12DetectionInterval = setInterval(
    () => analyzeCam12Frame(video),
    FIRE_ANALYSIS_INTERVAL_MS,
  );
  cam12VideoAnalysisActive = true;
  updateCam12Status("Analyse active");
}

function stopCam12Detection() {
  if (cam12DetectionInterval) {
    clearInterval(cam12DetectionInterval);
    cam12DetectionInterval = null;
  }
  cam12VideoAnalysisActive = false;
  clearOverlayCanvas("cam12-overlay-canvas");
  updateCam12Status("Analyse arretee");
}

function loadCam8Video(file) {
  const v = document.getElementById("cam8-video");
  if (!v || !file) return;
  const url = URL.createObjectURL(file);
  v.src = url;
  v.load();
  v.play().catch(() => {});
  document.getElementById("cam-feed-8").style.display = "none";
  document
    .getElementById("cam8-video-overlay")
    .querySelector("span").textContent = `Fichier: ${file.name}`;
  if (!cam8VideoAnalysisActive) startCam8Detection(v);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// UPLOAD VIDÉO
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const videoUploadEl = document.getElementById("video-upload");
if (videoUploadEl) {
  videoUploadEl.addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    document.title = `[${file.name}] — SafeVision AI`;

    const cam4Feed =
      document.getElementById("cam-feed-4") ||
      document.getElementById("cam4-feed-active");
    if (cam4Feed) {
      stopCam4Detection();
      cam4Feed.outerHTML = `
        <div class="cam-feed" id="cam4-feed-active">
          <video id="cam4-video" class="cam-video" src="${url}" muted loop playsinline preload="auto"></video>
          <canvas id="cam4-overlay-canvas" class="cam-overlay-canvas"></canvas>
          <div class="cam-video-overlay" id="cam4-video-overlay">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;width:100%">
              <span style="font-size:10px;font-family:var(--font-mono);">Fichier: ${file.name}</span>
              <span style="font-size:10px;font-family:var(--font-mono);" id="cam4-ppe-status">Detection: en attente...</span>
            </div>
          </div>
        </div>`;
      const cam4Video = document.getElementById("cam4-video");
      cam4Video.addEventListener(
        "loadeddata",
        () => {
          cam4Video.play().catch(() => {});
          startCam4Detection(cam4Video);
        },
        { once: true },
      );
      cam4Video.load();
      return;
    }

    const liveVideo = document.getElementById("cam1-video");
    if (liveVideo) {
      liveVideo.src = url;
      liveVideo.load();
      liveVideo.play().catch(() => {});
      startLiveDetection(liveVideo);
      return;
    }
    if (document.getElementById("main-video")) {
      loadDashboardVideoSource(url, file.name, true);
    }
  });
}

const camSelect = document.getElementById("camera-select");
if (camSelect) {
  camSelect.addEventListener("change", function () {
    if (this.value === "Fichier vidéo...") {
      document.getElementById("video-upload")?.click();
    }
  });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MODULE TOGGLES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateDashboardHybridStatus(text) {
  const el = document.getElementById("main-hybrid-status");
  if (el) el.textContent = "Detection: " + text;
}

function updateDashboardFallSpillStatus(text) {
  const el = document.getElementById("fall-spill-hybrid-status");
  if (el) el.textContent = "Detection: " + text;
}

function isModuleEnabled(moduleName) {
  const chip = document.querySelector(
    `.module-toggles [data-module="${moduleName}"]`,
  );
  return !chip || chip.classList.contains("on");
}

function drawDashboardBox(ctx, rect, bbox, stroke, fill, label) {
  if (!Array.isArray(bbox) || bbox.length < 4) return;
  const [x1, y1, x2, y2] = bbox.map(Number);
  const rx = rect.x + x1 * rect.scaleX;
  const ry = rect.y + y1 * rect.scaleY;
  const rw = (x2 - x1) * rect.scaleX;
  const rh = (y2 - y1) * rect.scaleY;
  if (rw <= 0 || rh <= 0) return;

  ctx.strokeStyle = stroke;
  ctx.lineWidth = 3;
  ctx.setLineDash([]);
  ctx.strokeRect(rx, ry, rw, rh);

  if (label) {
    ctx.font = "bold 11px monospace";
    const tw = ctx.measureText(label).width;
    const ly = Math.max(20, ry);
    ctx.fillStyle = fill;
    ctx.fillRect(rx, ly - 20, tw + 12, 20);
    ctx.fillStyle = "#071014";
    ctx.fillText(label, rx + 6, ly - 6);
  }
}

function drawDashboardHybridOverlay(video, ppeData, spillData) {
  const canvas = document.getElementById("main-overlay-canvas");
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect = getContainedVideoRect(canvas, video);

  const spillDetails = spillData?.details || {};
  const spillDetected = Boolean(spillData?.spill_detected);
  const spillPolygons = Array.isArray(spillDetails.polygons)
    ? spillDetails.polygons
    : [];
  spillPolygons.forEach((points) => {
    if (!Array.isArray(points) || points.length < 3) return;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = rect.x + point[0] * rect.scaleX;
      const y = rect.y + point[1] * rect.scaleY;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = "rgba(0,194,255,0.24)";
    ctx.strokeStyle = "rgba(0,194,255,0.98)";
    ctx.lineWidth = 3;
    ctx.fill();
    ctx.stroke();
  });
  if (Array.isArray(spillDetails.bbox)) {
    const spillPct = Math.round((spillData?.confidence || 0) * 100);
    drawDashboardBox(
      ctx,
      rect,
      spillDetails.bbox,
      spillDetected ? "rgba(0,194,255,0.98)" : "rgba(74,227,181,0.9)",
      spillDetected ? "rgba(0,194,255,0.94)" : "rgba(74,227,181,0.9)",
      spillDetected ? `SPILL ${spillPct}%` : `SPILL OK ${spillPct}%`,
    );
  }

  const ppeDetails = ppeData?.details || {};
  const violation = Boolean(ppeData?.ppe_violation);
  const ppeDetections = Array.isArray(ppeDetails.detections)
    ? ppeDetails.detections
    : [];
  ppeDetections.forEach((det) => {
    const label = String(det.label || "object").toLowerCase();
    const conf = Math.round((Number(det.confidence) || 0) * 100);
    let stroke = "rgba(74,227,181,0.98)";
    let fill = "rgba(74,227,181,0.92)";
    if (label === "human" || label === "person" || label === "worker") {
      stroke = "rgba(255,184,0,0.98)";
      fill = "rgba(255,184,0,0.92)";
    } else if (violation) {
      stroke = "rgba(255,59,47,0.98)";
      fill = "rgba(255,59,47,0.92)";
    }
    drawDashboardBox(
      ctx,
      rect,
      det.bbox,
      stroke,
      fill,
      `${label.toUpperCase()} ${conf}%`,
    );
  });

  const ppePct = Math.round((ppeData?.confidence || 0) * 100);
  const spillPct = Math.round((spillData?.confidence || 0) * 100);
  const badges = [
    {
      text: violation ? `PPE VIOLATION ${ppePct}%` : `PPE OK ${ppePct}%`,
      color: violation ? "#FF3B2F" : "#4AE3B5",
    },
    {
      text: spillDetected ? `SPILL ${spillPct}%` : `SPILL OK ${spillPct}%`,
      color: spillDetected ? "#00C2FF" : "#4AE3B5",
    },
  ];
  ctx.font = "bold 12px monospace";
  let x = 10;
  badges.forEach((badge) => {
    const w = ctx.measureText(badge.text).width + 18;
    ctx.fillStyle = badge.color;
    ctx.fillRect(x, 10, w, 25);
    ctx.fillStyle =
      badge.color === "#00C2FF" || badge.color === "#4AE3B5"
        ? "#001014"
        : "#fff";
    ctx.fillText(badge.text, x + 9, 27);
    x += w + 8;
  });
}

function drawDashboardFallSpillOverlay(video, fallData, spillData) {
  const canvas = document.getElementById("fall-spill-overlay-canvas");
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect = getContainedVideoRect(canvas, video);

  const spillDetails = spillData?.details || {};
  const spillDetected = Boolean(spillData?.spill_detected);
  const spillPolygons = Array.isArray(spillDetails.polygons)
    ? spillDetails.polygons
    : [];
  spillPolygons.forEach((points) => {
    if (!Array.isArray(points) || points.length < 3) return;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = rect.x + point[0] * rect.scaleX;
      const y = rect.y + point[1] * rect.scaleY;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = "rgba(0,194,255,0.22)";
    ctx.strokeStyle = "rgba(0,194,255,0.98)";
    ctx.lineWidth = 3;
    ctx.fill();
    ctx.stroke();
  });
  if (Array.isArray(spillDetails.bbox)) {
    const spillPct = Math.round((spillData?.confidence || 0) * 100);
    drawDashboardBox(
      ctx,
      rect,
      spillDetails.bbox,
      spillDetected ? "rgba(0,194,255,0.98)" : "rgba(74,227,181,0.9)",
      spillDetected ? "rgba(0,194,255,0.94)" : "rgba(74,227,181,0.9)",
      spillDetected ? `SPILL ${spillPct}%` : `SPILL OK ${spillPct}%`,
    );
  }

  const fallDetails = fallData?.details || {};
  const fallDetected = Boolean(fallData?.fall_detected);
  const fallPct = Math.round((fallData?.confidence || 0) * 100);
  if (Array.isArray(fallDetails.bbox)) {
    drawDashboardBox(
      ctx,
      rect,
      fallDetails.bbox,
      fallDetected ? "rgba(255,59,47,0.98)" : "rgba(74,227,181,0.9)",
      fallDetected ? "rgba(255,59,47,0.94)" : "rgba(74,227,181,0.9)",
      fallDetected ? `FALL ${fallPct}%` : `FALL OK ${fallPct}%`,
    );
  }

  const spillPct = Math.round((spillData?.confidence || 0) * 100);
  const badges = [
    {
      text: fallDetected ? `FALL ${fallPct}%` : `FALL OK ${fallPct}%`,
      color: fallDetected ? "#FF3B2F" : "#4AE3B5",
    },
    {
      text: spillDetected ? `SPILL ${spillPct}%` : `SPILL OK ${spillPct}%`,
      color: spillDetected ? "#00C2FF" : "#4AE3B5",
    },
  ];
  ctx.font = "bold 12px monospace";
  let x = 10;
  badges.forEach((badge) => {
    const w = ctx.measureText(badge.text).width + 18;
    ctx.fillStyle = badge.color;
    ctx.fillRect(x, 10, w, 25);
    ctx.fillStyle =
      badge.color === "#00C2FF" || badge.color === "#4AE3B5"
        ? "#001014"
        : "#fff";
    ctx.fillText(badge.text, x + 9, 27);
    x += w + 8;
  });
}

async function analyzeDashboardHybridFrame(video) {
  if (
    !dashboardHybridActive ||
    dashboardHybridRequestInFlight ||
    !video ||
    video.paused ||
    video.ended
  )
    return;
  const usePPE = isModuleEnabled("ppe");
  const useSpill = isModuleEnabled("hazards");
  if (!usePPE && !useSpill) {
    updateDashboardHybridStatus("PPE + SPILL desactives");
    clearOverlayCanvas("main-overlay-canvas");
    return;
  }

  const imageData = captureVideoFrame(video);
  if (!imageData) {
    updateDashboardHybridStatus("Capture impossible");
    return;
  }

  dashboardHybridRequestInFlight = true;
  updateDashboardHybridStatus("Analyse PPE + SPILL...");
  const startedAt = performance.now();

  try {
    const requests = [];
    if (usePPE) {
      requests.push(
        fetch("/api/ppe-detection/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCookie("csrftoken"),
          },
          body: JSON.stringify({
            image: imageData,
            camera: "dashboard-ppe-spill",
          }),
        })
          .then((res) => res.json())
          .then((data) => ({ key: "ppe", data })),
      );
    }
    if (useSpill) {
      requests.push(
        fetch("/api/spill-detection/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCookie("csrftoken"),
          },
          body: JSON.stringify({
            image: imageData,
            camera: "dashboard-ppe-spill",
          }),
        })
          .then((res) => res.json())
          .then((data) => ({ key: "spill", data })),
      );
    }

    const responses = await Promise.all(requests);
    const ppe = responses.find((item) => item.key === "ppe")?.data;
    const spill = responses.find((item) => item.key === "spill")?.data;
    if (
      (ppe && ppe.status !== "success") ||
      (spill && spill.status !== "success")
    ) {
      updateDashboardHybridStatus("Erreur API");
      return;
    }

    drawDashboardHybridOverlay(video, ppe, spill);
    dashboardHybridFrameCount += 1;
    const fps = Math.max(
      1,
      Math.round(1000 / Math.max(1, performance.now() - startedAt)),
    );
    const ppePct = Math.round((ppe?.confidence || 0) * 100);
    const spillPct = Math.round((spill?.confidence || 0) * 100);
    const statusParts = [];
    if (ppe)
      statusParts.push(
        ppe.ppe_violation ? `PPE danger ${ppePct}%` : `PPE OK ${ppePct}%`,
      );
    if (spill)
      statusParts.push(
        spill.spill_detected ? `SPILL ${spillPct}%` : `SPILL OK ${spillPct}%`,
      );
    updateDashboardHybridStatus(statusParts.join(" | "));

    const overlayFps = document.getElementById("overlay-fps");
    if (overlayFps) overlayFps.textContent = `${fps} FPS`;
    const overlayRes = document.getElementById("overlay-res");
    if (overlayRes)
      overlayRes.textContent = `${video.videoWidth || 0}x${video.videoHeight || 0}`;
    const m1 = document.getElementById("m1-fps");
    if (m1) m1.textContent = String(fps);
    const m5 = document.getElementById("m5-iou");
    if (m5 && spill) m5.textContent = `${spillPct}%`;

    const now = Date.now();
    if (
      ppe?.ppe_violation &&
      now - dashboardHybridLastAlertAt.ppe > PPE_ALERT_COOLDOWN_MS
    ) {
      dashboardHybridLastAlertAt.ppe = now;
      addAlert(
        "warning",
        "PPE",
        `Violation EPI detectee sur ppe_spill.mp4 (${ppePct}%)`,
      );
      showPopupNotification(`Dashboard: Violation EPI (${ppePct}%)`, "warning");
    }
    if (
      spill?.spill_detected &&
      now - dashboardHybridLastAlertAt.spill > SPILL_ALERT_COOLDOWN_MS
    ) {
      dashboardHybridLastAlertAt.spill = now;
      addAlert(
        "critical",
        "Spill",
        `Deversement detecte sur ppe_spill.mp4 (${spillPct}%)`,
      );
      showPopupNotification(
        `Dashboard: Deversement detecte (${spillPct}%)`,
        "critical",
      );
    }

    if (ppe) {
      _notifyDetection({
        cam: "dashboard-ppe-spill",
        type: "ppe",
        detected: ppe.ppe_violation,
        confidence: ppe.confidence,
        details: ppe.details || {},
      });
    }
    if (spill) {
      _notifyDetection({
        cam: "dashboard-ppe-spill",
        type: "spill",
        detected: spill.spill_detected,
        confidence: spill.confidence,
        details: spill.details || {},
      });
    }
  } catch (err) {
    updateDashboardHybridStatus("Erreur detection");
    console.error("[SafeVision] dashboard PPE+SPILL:", err);
  } finally {
    dashboardHybridRequestInFlight = false;
  }
}

async function analyzeDashboardFallSpillFrame(video) {
  if (
    !dashboardFallSpillActive ||
    dashboardFallSpillRequestInFlight ||
    !video ||
    video.paused ||
    video.ended
  )
    return;
  const useFall = isModuleEnabled("fall");
  const useSpill = isModuleEnabled("hazards");
  if (!useFall && !useSpill) {
    updateDashboardFallSpillStatus("FALL + SPILL desactives");
    clearOverlayCanvas("fall-spill-overlay-canvas");
    return;
  }

  const imageData = captureVideoFrame(video);
  if (!imageData) {
    updateDashboardFallSpillStatus("Capture impossible");
    return;
  }

  dashboardFallSpillRequestInFlight = true;
  updateDashboardFallSpillStatus("Analyse FALL + SPILL...");
  const startedAt = performance.now();

  try {
    const requests = [];
    if (useFall) {
      requests.push(
        fetch("/api/fall-detection/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCookie("csrftoken"),
          },
          body: JSON.stringify({
            image: imageData,
            camera: "dashboard-spill-fall",
          }),
        })
          .then((res) => res.json())
          .then((data) => ({ key: "fall", data })),
      );
    }
    if (useSpill) {
      requests.push(
        fetch("/api/spill-detection/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCookie("csrftoken"),
          },
          body: JSON.stringify({
            image: imageData,
            camera: "dashboard-spill-fall",
          }),
        })
          .then((res) => res.json())
          .then((data) => ({ key: "spill", data })),
      );
    }

    const responses = await Promise.all(requests);
    const fall = responses.find((item) => item.key === "fall")?.data;
    const spill = responses.find((item) => item.key === "spill")?.data;
    if (
      (fall && fall.status !== "success") ||
      (spill && spill.status !== "success")
    ) {
      updateDashboardFallSpillStatus("Erreur API");
      return;
    }

    drawDashboardFallSpillOverlay(video, fall, spill);
    dashboardFallSpillFrameCount += 1;
    const fps = Math.max(
      1,
      Math.round(1000 / Math.max(1, performance.now() - startedAt)),
    );
    const fallPct = Math.round((fall?.confidence || 0) * 100);
    const spillPct = Math.round((spill?.confidence || 0) * 100);
    const statusParts = [];
    if (fall)
      statusParts.push(
        fall.fall_detected ? `FALL danger ${fallPct}%` : `FALL OK ${fallPct}%`,
      );
    if (spill)
      statusParts.push(
        spill.spill_detected ? `SPILL ${spillPct}%` : `SPILL OK ${spillPct}%`,
      );
    updateDashboardFallSpillStatus(statusParts.join(" | "));

    const overlayFps = document.getElementById("fall-spill-overlay-fps");
    if (overlayFps) overlayFps.textContent = `${fps} FPS`;
    const overlayRes = document.getElementById("fall-spill-overlay-res");
    if (overlayRes)
      overlayRes.textContent = `${video.videoWidth || 0}x${video.videoHeight || 0}`;
    const m3f1 = document.getElementById("m3-f1");
    if (m3f1 && fall) m3f1.textContent = `${fallPct}%`;
    const m3lat = document.getElementById("m3-lat");
    if (m3lat) m3lat.textContent = `${fps} FPS`;
    const m5dice = document.getElementById("m5-dice");
    if (m5dice && spill) m5dice.textContent = `${spillPct}%`;

    const now = Date.now();
    if (
      fall?.fall_detected &&
      now - dashboardFallSpillLastAlertAt.fall > 9000
    ) {
      dashboardFallSpillLastAlertAt.fall = now;
      addAlert(
        "critical",
        "Fall",
        `Chute detectee sur spill_fall.mp4 (${fallPct}%)`,
      );
      showPopupNotification(
        `Dashboard: Chute detectee (${fallPct}%)`,
        "critical",
      );
    }
    if (
      spill?.spill_detected &&
      now - dashboardFallSpillLastAlertAt.spill > SPILL_ALERT_COOLDOWN_MS
    ) {
      dashboardFallSpillLastAlertAt.spill = now;
      addAlert(
        "critical",
        "Spill",
        `Deversement detecte sur spill_fall.mp4 (${spillPct}%)`,
      );
      showPopupNotification(
        `Dashboard: Deversement detecte (${spillPct}%)`,
        "critical",
      );
    }

    if (fall) {
      _notifyDetection({
        cam: "dashboard-spill-fall",
        type: "fall",
        detected: fall.fall_detected,
        confidence: fall.confidence,
        details: fall.details || {},
      });
    }
    if (spill) {
      _notifyDetection({
        cam: "dashboard-spill-fall",
        type: "spill",
        detected: spill.spill_detected,
        confidence: spill.confidence,
        details: spill.details || {},
      });
    }
  } catch (err) {
    updateDashboardFallSpillStatus("Erreur detection");
    console.error("[SafeVision] dashboard FALL+SPILL:", err);
  } finally {
    dashboardFallSpillRequestInFlight = false;
  }
}

function initializeDashboardHybrid() {
  const video = document.getElementById("main-video");
  if (video) {
    video.src = DASHBOARD_PPE_SPILL_SRC + "?v=" + Date.now();
    video.load();
    updateDashboardHybridStatus("Pret - ppe_spill.mp4");
    const info = document.getElementById("video-info");
    if (info) info.style.display = "flex";
  }

  const fallSpillVideo = document.getElementById("fall-spill-video");
  if (fallSpillVideo) {
    fallSpillVideo.src = DASHBOARD_SPILL_FALL_SRC + "?v=" + Date.now();
    fallSpillVideo.load();
    updateDashboardFallSpillStatus("Pret - spill_fall.mp4");
    const info = document.getElementById("fall-spill-video-info");
    if (info) info.style.display = "flex";
  }
}

function startAnalysis() {
  const video = document.getElementById("main-video");
  if (video) {
    if (!video.src) video.src = DASHBOARD_PPE_SPILL_SRC + "?v=" + Date.now();
    dashboardHybridActive = true;
    dashboardHybridStartedAt = performance.now();
    video.dataset.shouldRun = "1";
    video.loop = true;
    const start = () => {
      if (!dashboardHybridActive || video.dataset.shouldRun !== "1") return;
      video
        .play()
        .catch((err) =>
          console.error("[SafeVision] dashboard PPE+SPILL play:", err),
        );
      if (!dashboardHybridInterval) {
        analyzeDashboardHybridFrame(video);
        dashboardHybridInterval = setInterval(
          () => analyzeDashboardHybridFrame(video),
          850,
        );
      }
      updateDashboardHybridStatus("Analyse active");
    };
    if (video.readyState >= 2) start();
    else video.addEventListener("loadeddata", start, { once: true });
    const info = document.getElementById("video-info");
    if (info) info.style.display = "flex";
  }

  const fallSpillVideo = document.getElementById("fall-spill-video");
  if (fallSpillVideo) {
    if (!fallSpillVideo.src)
      fallSpillVideo.src = DASHBOARD_SPILL_FALL_SRC + "?v=" + Date.now();
    dashboardFallSpillActive = true;
    fallSpillVideo.dataset.shouldRun = "1";
    fallSpillVideo.loop = true;
    const startFallSpill = () => {
      if (!dashboardFallSpillActive || fallSpillVideo.dataset.shouldRun !== "1")
        return;
      fallSpillVideo
        .play()
        .catch((err) =>
          console.error("[SafeVision] dashboard FALL+SPILL play:", err),
        );
      if (!dashboardFallSpillInterval) {
        analyzeDashboardFallSpillFrame(fallSpillVideo);
        dashboardFallSpillInterval = setInterval(
          () => analyzeDashboardFallSpillFrame(fallSpillVideo),
          900,
        );
      }
      updateDashboardFallSpillStatus("Analyse active");
    };
    if (fallSpillVideo.readyState >= 2) startFallSpill();
    else
      fallSpillVideo.addEventListener("loadeddata", startFallSpill, {
        once: true,
      });
    const info = document.getElementById("fall-spill-video-info");
    if (info) info.style.display = "flex";
  }
}

function stopAnalysis() {
  dashboardFallSpillActive = false;
  const fallSpillVideo = document.getElementById("fall-spill-video");
  if (dashboardFallSpillInterval) {
    clearInterval(dashboardFallSpillInterval);
    dashboardFallSpillInterval = null;
  }
  if (fallSpillVideo) {
    fallSpillVideo.dataset.shouldRun = "0";
    fallSpillVideo.pause();
  }
  clearOverlayCanvas("fall-spill-overlay-canvas");
  updateDashboardFallSpillStatus("Analyse en pause");
  const fallSpillFps = document.getElementById("fall-spill-overlay-fps");
  if (fallSpillFps) fallSpillFps.textContent = "- FPS";

  dashboardHybridActive = false;
  const video = document.getElementById("main-video");
  if (dashboardHybridInterval) {
    clearInterval(dashboardHybridInterval);
    dashboardHybridInterval = null;
  }
  if (video) {
    video.dataset.shouldRun = "0";
    video.pause();
  }
  clearOverlayCanvas("main-overlay-canvas");
  updateDashboardHybridStatus("Analyse en pause");
  const overlayFps = document.getElementById("overlay-fps");
  if (overlayFps) overlayFps.textContent = "- FPS";
}

function loadDashboardVideoSource(
  src,
  label = "ppe_spill.mp4",
  autoplay = false,
) {
  const video = document.getElementById("main-video");
  if (!video) return;
  stopAnalysis();
  video.src = src;
  video.load();
  const overlay = document.getElementById("main-video-overlay");
  if (overlay) {
    const first = overlay.querySelector("span");
    if (first) first.textContent = `Fichier: ${label}`;
  }
  updateDashboardHybridStatus("Pret");
  if (autoplay)
    video.addEventListener("loadeddata", () => startAnalysis(), { once: true });
}

function toggleModule(chip) {
  chip.classList.toggle("on");
  console.log(
    `[SafeVision] Module ${chip.dataset.module}: ${chip.classList.contains("on") ? "on" : "off"}`,
  );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CONTRÔLES GLOBAUX
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function initializeCameras() {
  if (camerasInitialized) return;

  [
    ["cam1-video", LIVE_VIDEO_SRC, updateCam1Status],
    ["cam2-video", LIVE_VIDEO_SRC_CAM2, updateCam2Status],
    ["cam4-video", LIVE_VIDEO_SRC_CAM4, updateCam4Status],
    ["cam5-video", LIVE_VIDEO_SRC_CAM5, updateCam5Status],
    ["cam6-video", LIVE_VIDEO_SRC_CAM6, updateCam6Status],
    ["cam7-video", LIVE_VIDEO_SRC_CAM7, updateCam7Status],
    ["cam8-video", LIVE_VIDEO_SRC_CAM8, updateCam8Status],
    ["cam9-video", LIVE_VIDEO_SRC_CAM9, updateCam9Status],
    ["cam10-video", LIVE_VIDEO_SRC_CAM10, updateCam10Status],
    ["cam12-video", LIVE_VIDEO_SRC_CAM12, updateCam12Status],
  ].forEach(([id, src, statusFn]) => {
    const v = document.getElementById(id);
    if (v) {
      v.src = src + "?v=" + Date.now();
      v.load();
      v.pause();
      statusFn("Camera prete");
    }
  });

  camerasInitialized = true;
}

function _startCamWithLoop(id, src, startFn) {
  const v = document.getElementById(id);
  if (!v) return;
  v.dataset.shouldRun = "1";
  v.src = src + "?v=" + Date.now();
  v.currentTime = 0;
  let detectionStarted = false;
  const startPlaybackAndDetection = () => {
    if (!camerasRunning || v.dataset.shouldRun !== "1") return;
    v.play()
      .then(() => {
        if (detectionStarted) return;
        detectionStarted = true;
        startFn(v);
      })
      .catch((err) => console.error(`[SafeVision] ${id} play:`, err));
  };
  v.load();
  // Try immediately on user click, and retry when data is ready.
  startPlaybackAndDetection();
  v.onloadeddata = startPlaybackAndDetection;
  v.oncanplay = startPlaybackAndDetection;
  v.onended = () => {
    if (!camerasRunning || v.dataset.shouldRun !== "1") return;
    v.currentTime = 0;
    v.play().catch(() => {});
  };
}

function startCameras() {
  console.log("[SafeVision] startCameras()");
  camerasRunning = true;
  initializeCameras();

  const cam1 = document.getElementById("cam1-video");
  if (cam1) {
    cam1.currentTime = 0;
    cam1.play().catch(() => {});
    startLiveDetection(cam1);
  }

  const cam2 = document.getElementById("cam2-video");
  if (cam2) {
    cam2.play().catch(() => {});
    startFatigueDetection(cam2);
  }

  _startCamWithLoop("cam3-video", LIVE_VIDEO_SRC_CAM3, startCam3Detection);
  _startCamWithLoop("cam4-video", LIVE_VIDEO_SRC_CAM4, startCam4Detection);
  _startCamWithLoop("cam5-video", LIVE_VIDEO_SRC_CAM5, startCam5Detection);
  _startCamWithLoop("cam6-video", LIVE_VIDEO_SRC_CAM6, startCam6Detection);
  _startCamWithLoop("cam7-video", LIVE_VIDEO_SRC_CAM7, startCam7Detection);
  _startCamWithLoop("cam8-video", LIVE_VIDEO_SRC_CAM8, startCam8Detection);
  _startCamWithLoop("cam9-video", LIVE_VIDEO_SRC_CAM9, startCam9Detection);
  _startCamWithLoop("cam10-video", LIVE_VIDEO_SRC_CAM10, startCam10Detection);
  _startCamWithLoop("cam12-video", LIVE_VIDEO_SRC_CAM12, startCam12Detection);
  _startCamWithLoop(
    "cam11-video",
    LIVE_VIDEO_SRC_CAM8_PROXIMITY,
    startCam11Detection,
  );
}

function stopCameras() {
  console.log("[SafeVision] stopCameras()");
  camerasRunning = false;

  const cam1 = document.getElementById("cam1-video");
  if (cam1) {
    cam1.pause();
    updateCam1Status("Caméra arrêtée");
  }
  stopLiveDetection();

  const cam2 = document.getElementById("cam2-video");
  if (cam2) {
    cam2.pause();
    updateCam2Status("Caméra arrêtée");
  }
  stopFatigueDetection();

  [
    ["cam3-video", updateCam3Status, stopCam3Detection],
    ["cam4-video", updateCam4Status, stopCam4Detection],
    ["cam5-video", updateCam5Status, stopCam5Detection],
    ["cam6-video", updateCam6Status, stopCam6Detection],
    ["cam7-video", updateCam7Status, stopCam7Detection],
    ["cam8-video", updateCam8Status, stopCam8Detection],
    ["cam9-video", updateCam9Status, stopCam9Detection],
    ["cam10-video", updateCam10Status, stopCam10Detection],
    ["cam11-video", updateCam11Status, stopCam11Detection],
    ["cam12-video", updateCam12Status, stopCam12Detection],
  ].forEach(([id, statusFn, stopFn]) => {
    const v = document.getElementById(id);
    if (v) {
      v.dataset.shouldRun = "0";
      v.onloadeddata = null;
      v.onended = null;
      v.pause();
      statusFn("Camera arretee");
    }
    stopFn();
  });

  clearAllOverlayCanvases();
}

function startAllModules() {
  startCameras();
}
function stopAllModules() {
  stopCameras();
}

function setLayout(cols) {
  const grid = document.getElementById("live-grid");
  if (grid)
    grid.style.gridTemplateColumns = cols === 1 ? "2fr 1fr" : "repeat(2, 1fr)";
}

function filterAlerts(type, btn) {
  document
    .querySelectorAll(".filter-btn")
    .forEach((b) => b.classList.remove("active"));
  if (btn) btn.classList.add("active");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// API HELPERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async function postAlertToAPI(alertData) {
  try {
    const res = await fetch("/api/alerts/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify(alertData),
    });
    return await res.json();
  } catch (e) {
    console.error("[API] POST alerte:", e);
  }
}

async function fetchRecentDetections(moduleId, limit = 20) {
  try {
    const res = await fetch(
      `/api/detections/?module=${moduleId}&limit=${limit}`,
    );
    return await res.json();
  } catch (e) {
    console.error("[API] GET détections:", e);
    return [];
  }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// THÈME
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function applyTheme(theme) {
  document.body.classList.toggle("theme-dark", theme === "dark");
  document.body.classList.toggle("theme-light", theme === "light");
  localStorage.setItem("theme", theme);
  const btn = document.getElementById("theme-toggle");
  if (btn) {
    btn.textContent = theme === "dark" ? "☀️" : "🌙";
    btn.setAttribute(
      "aria-label",
      theme === "dark" ? "Mode clair" : "Mode sombre",
    );
  }
}

function toggleTheme() {
  applyTheme(document.body.classList.contains("theme-dark") ? "light" : "dark");
}

function initTheme() {
  const saved = localStorage.getItem("theme");
  const def = window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
  applyTheme(saved || def);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// KPI
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const KPI = {
  setCritical: (v) => {
    const e = document.getElementById("kpi-critical");
    if (e) e.textContent = v;
  },
  setWarnings: (v) => {
    const e = document.getElementById("kpi-warnings");
    if (e) e.textContent = v;
  },
  setWorkers: (v) => {
    const e = document.getElementById("kpi-workers");
    if (e) e.textContent = v;
  },
  setModules: (v) => {
    const e = document.getElementById("kpi-modules");
    if (e) e.textContent = v;
  },
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DOMContentLoaded
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  document
    .getElementById("theme-toggle")
    ?.addEventListener("click", toggleTheme);

  document
    .querySelectorAll(".stat-card, .module-card, .panel")
    .forEach((el, i) => {
      el.style.opacity = "0";
      el.style.transform = "translateY(8px)";
      setTimeout(() => {
        el.style.transition = "opacity .3s ease, transform .3s ease";
        el.style.opacity = "1";
        el.style.transform = "none";
      }, i * 40);
    });

  const cam1 = document.getElementById("cam1-video");
  if (cam1) {
    cam1.addEventListener("loadeddata", () => {
      if (cam1.paused && !liveVideoAnalysisActive)
        updateCam1Status("Caméra prête");
    });
    cam1.addEventListener("play", () => {
      if (!liveVideoAnalysisActive) startLiveDetection(cam1);
    });
    cam1.addEventListener("pause", () => updateCam1Status("Vidéo en pause"));
  }

  const cam2 = document.getElementById("cam2-video");
  if (cam2) {
    cam2.addEventListener("loadeddata", () => {
      if (cam2.paused && !fatigueVideoAnalysisActive)
        updateCam2Status("Caméra prête");
    });
    cam2.addEventListener("play", () => {
      if (!fatigueVideoAnalysisActive) startFatigueDetection(cam2);
    });
    cam2.addEventListener("pause", () => updateCam2Status("Vidéo en pause"));
  }
});
