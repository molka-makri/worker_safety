// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SAFEVISION AI — main.js  (v5 — 7 cameras, Sign Defect added for CAM 7)
// CAM 1 : Détection de chute   (worker_falling3.mp4)
// CAM 2 : Détection de fatigue (worker_tired.mp4)
// CAM 3 : Spill detection      (spill.mp4)
// CAM 4 : PPE detection        (ppe_video.mp4)
// CAM 5 : Manhole detection    (hole.mp4)
// CAM 6 : Exit emergency       (exit_emergency.mp4)
// CAM 7 : Sign defect detection(construction_signs2.mp4)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ── HORLOGE LIVE ─────────────────────────────────────────
function updateClock() {
  const el = document.getElementById('live-clock');
  if (el) {
    const now = new Date();
    el.textContent = now.toLocaleTimeString('fr-FR', {
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  }
}
setInterval(updateClock, 1000);
updateClock();

// ── CONSTANTES VIDÉO ──────────────────────────────────────
const LIVE_VIDEO_SRC           = '/media/worker_falling3.mp4';
const LIVE_VIDEO_SRC_CAM2      = '/media/worker_tired.mp4';
const LIVE_VIDEO_SRC_CAM3      = '/media/spill.mp4';
const LIVE_VIDEO_SRC_CAM4      = '/media/ppe_video.mp4'; 
const LIVE_VIDEO_SRC_CAM6      = '/media/exit_emergency.mp4';
const LIVE_VIDEO_SRC_CAM5      = '/media/hole.mp4';
const LIVE_VIDEO_SRC_CAM7      = '/media/construction_signs2.mp4';
const VIDEO_CAPTURE_MAX_WIDTH  = 960;
const VIDEO_CAPTURE_MAX_HEIGHT = 960;

// ── ÉTAT DES MODULES ──────────────────────────────────────
let liveDetectionInterval      = null;
let liveVideoAnalysisActive    = false;

let fatigueDetectionInterval   = null;
let fatigueVideoAnalysisActive = false;

let cam3DetectionInterval      = null;
let cam3VideoAnalysisActive    = false;

let cam4DetectionInterval      = null;
let cam4VideoAnalysisActive    = false;

let cam5DetectionInterval      = null;
let cam5VideoAnalysisActive    = false;

let cam6DetectionInterval      = null;
let cam6VideoAnalysisActive    = false;

let cam7DetectionInterval      = null; // ADDED SIGN
let cam7VideoAnalysisActive    = false; // ADDED SIGN

let spillRequestInFlight       = {};
let spillLastAlertAt           = {};

let ppeRequestInFlight         = false;
let ppeLastAlertAt             = 0;
const PPE_ANALYSIS_INTERVAL_MS = 500;
const PPE_ALERT_COOLDOWN_MS    = 10000;

// Sign State Variables
let signRequestInFlight        = false;
let signLastAlertAt            = 0;
const SIGN_ANALYSIS_INTERVAL_MS = 800; // ResNet can be a bit heavier
const SIGN_ALERT_COOLDOWN_MS    = 10000;

let manholeRequestInFlight     = false;
let manholeLastAlertAt         = 0;
let exitRequestInFlight        = false;
let exitLastAlertAt            = 0;
let cam6RunToken               = 0;
let camerasRunning             = false;

const SPILL_ANALYSIS_INTERVAL_MS    = 350;
const SPILL_ALERT_COOLDOWN_MS       = 10000;
const MANHOLE_ANALYSIS_INTERVAL_MS  = 450;
const MANHOLE_ALERT_COOLDOWN_MS     = 12000;
const EXIT_ANALYSIS_INTERVAL_MS     = 450;
const EXIT_ALERT_COOLDOWN_MS        = 10000;

let camerasInitialized = false;
let alertCount         = 0;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// UTILITAIRES COMMUNS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function captureVideoFrame(video, maxWidth = VIDEO_CAPTURE_MAX_WIDTH, maxHeight = VIDEO_CAPTURE_MAX_HEIGHT) {
  try {
    const videoWidth  = video.videoWidth  || maxWidth;
    const videoHeight = video.videoHeight || maxHeight;
    const scale       = Math.min(maxWidth / videoWidth, maxHeight / videoHeight, 1);
    const width       = Math.max(1, Math.round(videoWidth  * scale));
    const height      = Math.max(1, Math.round(videoHeight * scale));

    const canvas  = document.createElement('canvas');
    canvas.width  = width;
    canvas.height = height;
    canvas.getContext('2d').drawImage(video, 0, 0, width, height);

    video.dataset.captureWidth  = width;
    video.dataset.captureHeight = height;

    return canvas.toDataURL('image/jpeg', 0.92).split(',')[1];
  } catch (err) {
    console.error('[SafeVision] captureVideoFrame:', err);
    return null;
  }
}

function getCookie(name) {
  const m = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
  return m ? m[2] : '';
}

function setCanvasSize(canvas, video) {
  if (!canvas || !video) return;
  const w = video.clientWidth  || canvas.parentElement?.clientWidth  || 320;
  const h = video.clientHeight || canvas.parentElement?.clientHeight || 240;
  canvas.width  = w;  canvas.style.width  = w + 'px';
  canvas.height = h;  canvas.style.height = h + 'px';
}

function clearOverlayCanvas(canvasId) {
  const c = document.getElementById(canvasId);
  if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
}

function clearAllOverlayCanvases() {
  ['cam1','cam2','cam3','cam4','cam5','cam6','cam7'].forEach(id => // Added cam7
    clearOverlayCanvas(`${id}-overlay-canvas`)
  );
}

function getContainedVideoRect(canvas, video) {
  const captureWidth  = Number(video.dataset.captureWidth)  || video.videoWidth  || VIDEO_CAPTURE_MAX_WIDTH;
  const captureHeight = Number(video.dataset.captureHeight) || video.videoHeight || VIDEO_CAPTURE_MAX_HEIGHT;
  const scale  = Math.min(canvas.width / captureWidth, canvas.height / captureHeight);
  const width  = captureWidth  * scale;
  const height = captureHeight * scale;
  return {
    x:      (canvas.width  - width)  / 2,
    y:      (canvas.height - height) / 2,
    scaleX: width  / captureWidth,
    scaleY: height / captureHeight,
  };
}

function _notifyDetection(result) {
  if (typeof window.onDetectionResult === 'function') {
    window.onDetectionResult(result);
  }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TOASTS & NOTIFICATIONS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function createToastContainer() {
  if (document.getElementById('toast-container')) return;
  const c = document.createElement('div');
  c.id = 'toast-container';
  c.className = 'toast-container';
  document.body.appendChild(c);
}

function saveNotification(message, severity = 'info') {
  const n = {
    id:        Date.now(),
    message,
    severity,
    timestamp: new Date().toLocaleString('fr-FR'),
    date:      new Date().toISOString(),
  };
  const list = JSON.parse(localStorage.getItem('safevision_notifications') || '[]');
  list.unshift(n);
  if (list.length > 100) list.pop();
  localStorage.setItem('safevision_notifications', JSON.stringify(list));
  return n;
}

function showPopupNotification(message, severity = 'info', persist = true) {
  if (persist) saveNotification(message, severity);
  createToastContainer();
  const container = document.getElementById('toast-container');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `detection-toast detection-toast-${severity}`;
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(() => toast.classList.add('visible'));
  setTimeout(() => {
    toast.classList.remove('visible');
    setTimeout(() => toast.remove(), 300);
  }, 4500);
}

function deleteNotification(id) {
  const list = JSON.parse(localStorage.getItem('safevision_notifications') || '[]');
  localStorage.setItem('safevision_notifications', JSON.stringify(list.filter(n => n.id !== id)));
  const el = document.getElementById('notification-' + id);
  if (el) {
    el.style.opacity   = '0';
    el.style.transform = 'translateX(20px)';
    setTimeout(() => el.remove(), 300);
  }
}

function deleteAllNotifications() {
  if (confirm('Êtes-vous sûr de vouloir supprimer toutes les notifications ?')) {
    localStorage.removeItem('safevision_notifications');
    loadNotifications();
  }
}

function loadNotifications() {
  const container = document.getElementById('alerts-container');
  if (!container) return;
  const list = JSON.parse(localStorage.getItem('safevision_notifications') || '[]');
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
  container.innerHTML = list.map(n => `
    <div id="notification-${n.id}" class="notification-item notification-${n.severity}"
         style="transition:all .3s ease">
      <div class="notification-content">
        <div class="notification-header">
          <span class="notification-severity notification-severity-${n.severity}">
            ${n.severity === 'critical' ? '🔴' : n.severity === 'warning' ? '🟡' : '🔵'} ${n.severity.toUpperCase()}
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
    </div>`).join('');
}

document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('alerts-container')) loadNotifications();
  initializeCameras();
});

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ALERTES GLOBALES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function addAlert(severity, module, message) {
  alertCount++;
  const badge = document.getElementById('alert-badge');
  if (badge) badge.textContent = alertCount;

  const feed = document.getElementById('alerts-feed');
  if (feed) {
    feed.querySelector('.empty-state')?.remove();
    const item = document.createElement('div');
    item.className = 'alert-item fade-in';
    item.innerHTML = `
      <div class="alert-severity sev-${severity}"></div>
      <div class="alert-body">
        <div class="alert-module">${module.toUpperCase()}</div>
        <div class="alert-msg">${message}</div>
        <div class="alert-time">${new Date().toLocaleTimeString('fr-FR')}</div>
      </div>`;
    feed.insertBefore(item, feed.firstChild);
    while (feed.children.length > 50) feed.removeChild(feed.lastChild);
  }

  if (severity === 'critical') playAlertSound();
}

function playAlertSound() {
  try {
    const ctx  = new (window.AudioContext || window.webkitAudioContext)();
    const osc  = ctx.createOscillator();
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
  const el = document.getElementById('cam1-fall-status');
  if (el) el.textContent = 'Détection: ' + text;
}

function drawBoundingBox(bbox, video, isFall) {
  const canvas = document.getElementById('cam1-overlay-canvas');
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!Array.isArray(bbox) || bbox.length !== 4) return;

  const rect = getContainedVideoRect(canvas, video);
  const [x1, y1, x2, y2] = bbox;
  const rx = rect.x + x1 * rect.scaleX;
  const ry = rect.y + y1 * rect.scaleY;

  ctx.strokeStyle = isFall ? 'rgba(255,59,47,0.95)' : 'rgba(74,227,181,0.95)';
  ctx.lineWidth   = 4;
  ctx.setLineDash([10, 6]);
  ctx.strokeRect(rx, ry, (x2 - x1) * rect.scaleX, (y2 - y1) * rect.scaleY);

  ctx.setLineDash([]);
  ctx.fillStyle = isFall ? 'rgba(255,59,47,0.85)' : 'rgba(74,227,181,0.85)';
  ctx.font = 'bold 11px monospace';
  const label = isFall ? '⚠ CHUTE' : '✓ OK';
  const tw    = ctx.measureText(label).width;
  ctx.fillRect(rx, ry - 18, tw + 10, 18);
  ctx.fillStyle = '#fff';
  ctx.fillText(label, rx + 5, ry - 5);
}

async function analyzeVideoFrame(video) {
  if (!video || video.paused || video.ended) return;
  const imageData = captureVideoFrame(video);
  if (!imageData) { updateCam1Status('Capture impossible'); return; }
  updateCam1Status('Analyse en cours…');

  try {
    const res  = await fetch('/api/fall-detection/', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken') },
      body:    JSON.stringify({ image: imageData, camera: 'cam1' }),
    });
    const data = await res.json();

    if (data.status === 'success') {
      const pct = Math.round(data.confidence * 100);
      updateCam1Status(data.fall_detected ? `CHUTE (${pct}%)` : `OK (${pct}%)`);

      if (Array.isArray(data.details?.bbox)) {
        drawBoundingBox(data.details.bbox, video, data.fall_detected);
      } else {
        clearOverlayCanvas('cam1-overlay-canvas');
      }

      if (data.fall_detected) {
        addAlert('critical', 'Fall', 'Chute détectée sur caméra 1');
        showPopupNotification(`Chute détectée (${pct}%)`, 'critical');
      }

      _notifyDetection({
        cam: 'cam1', type: 'fall',
        detected: data.fall_detected, confidence: data.confidence, details: data.details || {},
      });
    } else {
      updateCam1Status('Erreur API');
      clearOverlayCanvas('cam1-overlay-canvas');
    }
  } catch (err) {
    updateCam1Status('Erreur détection');
    console.error('[SafeVision] fetch chute:', err);
  }
}

function startLiveDetection(video) {
  if (!video || liveDetectionInterval) return;
  analyzeVideoFrame(video);
  liveDetectionInterval   = setInterval(() => analyzeVideoFrame(video), 1200);
  liveVideoAnalysisActive = true;
  updateCam1Status('Analyse active');
}

function stopLiveDetection() {
  if (liveDetectionInterval) { clearInterval(liveDetectionInterval); liveDetectionInterval = null; }
  liveVideoAnalysisActive = false;
  clearOverlayCanvas('cam1-overlay-canvas');
  updateCam1Status('Analyse arrêtée');
}

function loadLiveVideo(path) {
  const v = document.getElementById('cam1-video');
  if (!v) return;
  v.src = path; v.load(); v.play().catch(() => {});
  if (!liveVideoAnalysisActive) startLiveDetection(v);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 2 — DÉTECTION DE FATIGUE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam2Status(text) {
  const el = document.getElementById('cam2-fatigue-status');
  if (el) el.textContent = 'Détection: ' + text;
}

function drawFatigueOverlay(bbox, video, isFatigued) {
  const canvas = document.getElementById('cam2-overlay-canvas');
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!Array.isArray(bbox) || bbox.length !== 4) return;

  const rect = getContainedVideoRect(canvas, video);
  const [x1, y1, x2, y2] = bbox;
  const rx = rect.x + x1 * rect.scaleX;
  const ry = rect.y + y1 * rect.scaleY;
  const bw = (x2 - x1) * rect.scaleX;
  const bh = (y2 - y1) * rect.scaleY;

  ctx.strokeStyle = isFatigued ? 'rgba(255,184,0,0.95)' : 'rgba(74,227,181,0.95)';
  ctx.lineWidth   = 4;
  ctx.setLineDash([10, 6]);
  ctx.strokeRect(rx, ry, bw, bh);

  if (isFatigued) {
    ctx.fillStyle = 'rgba(255,184,0,0.10)';
    ctx.fillRect(rx, ry, bw, bh);
  }

  ctx.setLineDash([]);
  ctx.fillStyle = isFatigued ? 'rgba(255,184,0,0.90)' : 'rgba(74,227,181,0.90)';
  ctx.font = 'bold 11px monospace';
  const label = isFatigued ? '😴 FATIGUE' : '✓ OK';
  const tw    = ctx.measureText(label).width;
  ctx.fillRect(rx, ry - 18, tw + 10, 18);
  ctx.fillStyle = '#000';
  ctx.fillText(label, rx + 5, ry - 5);
}

async function analyzeFatigueFrame(video) {
  if (!video || video.paused || video.ended) return;
  const imageData = captureVideoFrame(video);
  if (!imageData) { updateCam2Status('Capture impossible'); return; }
  updateCam2Status('Analyse en cours…');

  try {
    const res  = await fetch('/api/fatigue-detection/', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken') },
      body:    JSON.stringify({ image: imageData, camera: 'cam2' }),
    });
    const data = await res.json();

    if (data.status === 'success') {
      const pct = Math.round(data.confidence * 100);
      if (data.fatigue_detected) {
        const level = data.details?.fatigue_level || 'modérée';
        updateCam2Status(`FATIGUE ${level.toUpperCase()} (${pct}%)`);
        addAlert('warning', 'Fatigue', `Fatigue détectée — niveau ${level}`);
        showPopupNotification(`Fatigue (${pct}%) — niveau ${level}`, 'warning', false);
      } else {
        updateCam2Status(`OK (${pct}%)`);
      }

      if (Array.isArray(data.details?.bbox)) {
        drawFatigueOverlay(data.details.bbox, video, data.fatigue_detected);
      } else {
        clearOverlayCanvas('cam2-overlay-canvas');
      }

      _notifyDetection({
        cam: 'cam2', type: 'fatigue',
        detected: data.fatigue_detected, confidence: data.confidence, details: data.details || {},
      });
    } else {
      updateCam2Status('Erreur API');
      clearOverlayCanvas('cam2-overlay-canvas');
    }
  } catch (err) {
    updateCam2Status('Erreur détection');
    console.error('[SafeVision] fetch fatigue:', err);
  }
}

function startFatigueDetection(video) {
  if (!video || fatigueDetectionInterval) return;
  analyzeFatigueFrame(video);
  fatigueDetectionInterval   = setInterval(() => analyzeFatigueFrame(video), 1500);
  fatigueVideoAnalysisActive = true;
  updateCam2Status('Analyse active');
}

function stopFatigueDetection() {
  if (fatigueDetectionInterval) { clearInterval(fatigueDetectionInterval); fatigueDetectionInterval = null; }
  fatigueVideoAnalysisActive = false;
  clearOverlayCanvas('cam2-overlay-canvas');
  updateCam2Status('Analyse arrêtée');
}

function loadCam2Video(path) {
  const v = document.getElementById('cam2-video');
  if (!v) return;
  v.src = path; v.load(); v.play().catch(() => {});
  if (!fatigueVideoAnalysisActive) startFatigueDetection(v);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 3 — SPILL DETECTION 
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam3Status(text) {
  const el = document.getElementById('cam3-spill-status') || document.getElementById('cam3-fall-status');
  if (el) el.textContent = 'Detection: ' + text;
}

function drawSpillOverlay(canvasId, data, video, isSpill) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect     = getContainedVideoRect(canvas, video);
  const polygons = Array.isArray(data?.polygons) ? data.polygons : [];

  ctx.lineWidth = 3;
  ctx.setLineDash([]);
  polygons.forEach(points => {
    if (!Array.isArray(points) || points.length < 3) return;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = rect.x + point[0] * rect.scaleX;
      const y = rect.y + point[1] * rect.scaleY;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle   = 'rgba(0,194,255,0.28)';
    ctx.strokeStyle = 'rgba(0,194,255,0.95)';
    ctx.fill(); ctx.stroke();
  });

  const bbox = data?.bbox;
  let labelX = 12, labelY = 24;
  if (Array.isArray(bbox) && bbox.length === 4) {
    const [x1, y1, x2, y2] = bbox;
    const rx = rect.x + x1 * rect.scaleX;
    const ry = rect.y + y1 * rect.scaleY;
    ctx.strokeStyle = isSpill ? 'rgba(0,194,255,0.95)' : 'rgba(74,227,181,0.95)';
    ctx.strokeRect(rx, ry, (x2 - x1) * rect.scaleX, (y2 - y1) * rect.scaleY);
    labelX = rx; labelY = Math.max(18, ry);
  }

  if (!isSpill && !polygons.length && !Array.isArray(bbox)) return;
  ctx.fillStyle = isSpill ? 'rgba(0,194,255,0.92)' : 'rgba(74,227,181,0.90)';
  ctx.font = 'bold 11px monospace';
  const label = isSpill ? 'SPILL' : 'OK';
  const tw    = ctx.measureText(label).width;
  ctx.fillRect(labelX, labelY - 18, tw + 10, 18);
  ctx.fillStyle = '#001014';
  ctx.fillText(label, labelX + 5, labelY - 5);
}

async function analyzeSpillFrame(video, cameraId) {
  if (!video || video.paused || video.ended) return;
  if (spillRequestInFlight[cameraId]) return;
  spillRequestInFlight[cameraId] = true;

  const imageData    = captureVideoFrame(video);
  const updateStatus = cameraId === 'cam4' ? updateCam4Status : updateCam3Status; // kept for safety
  const canvasId     = `${cameraId}-overlay-canvas`;

  if (!imageData) {
    updateStatus('Capture impossible');
    spillRequestInFlight[cameraId] = false;
    return;
  }
  updateStatus('Analyse en cours...');

  try {
    const res  = await fetch('/api/spill-detection/', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken') },
      body:    JSON.stringify({ image: imageData, camera: cameraId }),
    });
    const data = await res.json();

    if (data.status === 'success') {
      const pct      = Math.round(data.confidence * 100);
      const holdText = data.details?.temporal_hold ? ' suivi' : '';
      updateStatus(data.spill_detected ? `SPILL${holdText} (${pct}%)` : `OK (${pct}%)`);

      if (Array.isArray(data.details?.bbox) || Array.isArray(data.details?.polygons)) {
        drawSpillOverlay(canvasId, data.details || {}, video, data.spill_detected);
      } else {
        clearOverlayCanvas(canvasId);
      }

      const now      = Date.now();
      const canAlert = now - (spillLastAlertAt[cameraId] || 0) > SPILL_ALERT_COOLDOWN_MS;
      if (data.spill_detected && canAlert) {
        spillLastAlertAt[cameraId] = now;
        addAlert('critical', 'Spill', `Deversement detecte sur ${cameraId.toUpperCase()}`);
        showPopupNotification(`[${cameraId.toUpperCase()}] Deversement detecte (${pct}%)`, 'critical');
      }

      _notifyDetection({
        cam: cameraId, type: 'spill',
        detected: data.spill_detected, confidence: data.confidence, details: data.details || {},
      });
    } else {
      updateStatus('Erreur API');
      clearOverlayCanvas(canvasId);
    }
  } catch (err) {
    updateStatus('Erreur detection');
    console.error(`[SafeVision] fetch ${cameraId}:`, err);
  } finally {
    spillRequestInFlight[cameraId] = false;
  }
}

function analyzeCam3Frame(video) { return analyzeSpillFrame(video, 'cam3'); }

function startCam3Detection(video) {
  if (!video || cam3DetectionInterval) return;
  analyzeCam3Frame(video);
  cam3DetectionInterval   = setInterval(() => analyzeCam3Frame(video), SPILL_ANALYSIS_INTERVAL_MS);
  cam3VideoAnalysisActive = true;
  updateCam3Status('Analyse active');
}

function stopCam3Detection() {
  if (cam3DetectionInterval) { clearInterval(cam3DetectionInterval); cam3DetectionInterval = null; }
  cam3VideoAnalysisActive = false;
  clearOverlayCanvas('cam3-overlay-canvas');
  updateCam3Status('Analyse arretee');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 4 — PPE DETECTION (Motaz)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam4Status(text) {
  const el = document.getElementById('cam4-ppe-status');
  if (el) el.textContent = 'Detection: ' + text;
}

function drawPPEOverlay(canvasId, data, video, isViolation) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect = getContainedVideoRect(canvas, video);
  const detections = Array.isArray(data?.detections) ? data.detections : [];

  ctx.lineJoin = 'round'; ctx.lineCap = 'round';

  detections.forEach(det => {
    const box = det.bbox;
    if (!box || box.length < 4) return;
    const [x1, y1, x2, y2] = box;
    const rx = rect.x + x1 * rect.scaleX;
    const ry = rect.y + y1 * rect.scaleY;
    const rw = (x2 - x1) * rect.scaleX;
    const rh = (y2 - y1) * rect.scaleY;

    let color = 'rgba(74,227,181,0.95)'; 
    let bgColor = 'rgba(74,227,181,0.90)';
    const label = (det.label || '').toLowerCase();

    if (label === 'human' || label === 'person' || label === 'worker') {
        color = 'rgba(255,184,0,0.95)'; 
        bgColor = 'rgba(255,184,0,0.90)';
    } else if (isViolation) {
        color = 'rgba(255,59,47,0.95)'; 
        bgColor = 'rgba(255,59,47,0.90)';
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(rx, ry, rw, rh);

    const confText = Math.round((det.confidence || 0) * 100) + '%';
    const displayLabel = (det.label || 'Object').toUpperCase() + ' ' + confText;
    ctx.font = 'bold 11px monospace';
    const tw = ctx.measureText(displayLabel).width;
    ctx.fillStyle = bgColor;
    ctx.fillRect(rx, ry - 18, tw + 8, 18);
    ctx.fillStyle = '#001014';
    ctx.fillText(displayLabel, rx + 4, ry - 5);
  });

  if (isViolation) {
    ctx.fillStyle = 'rgba(255,59,47,0.92)';
    ctx.font = 'bold 14px monospace';
    const bannerText = '⚠ PPE VIOLATION';
    const bw = ctx.measureText(bannerText).width + 16;
    ctx.fillRect(8, 8, bw, 26);
    ctx.fillStyle = '#FFFFFF';
    ctx.fillText(bannerText, 16, 26);
  }
}

async function analyzePPEFrame(video, cameraId = 'cam4') {
  if (!video || video.paused || video.ended) return;
  if (ppeRequestInFlight) return;
  ppeRequestInFlight = true;

  const imageData = captureVideoFrame(video);
  const updateStatus = updateCam4Status;
  const canvasId = `${cameraId}-overlay-canvas`;

  if (!imageData) {
    updateStatus('Capture impossible');
    ppeRequestInFlight = false;
    return;
  }
  updateStatus('Analyse en cours...');

  try {
    const res  = await fetch('/api/ppe-detection/', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken') },
      body:    JSON.stringify({ image: imageData, camera: cameraId }),
    });
    const data = await res.json();

    if (data.status === 'success') {
      const pct = Math.round(data.confidence * 100);
      const violation = data.ppe_violation;
      updateStatus(violation ? `VIOLATION (${pct}%)` : `OK (${pct}%)`);

      if (Array.isArray(data.details?.detections) && data.details.detections.length > 0) {
        drawPPEOverlay(canvasId, data.details || {}, video, violation);
      } else {
        clearOverlayCanvas(canvasId);
      }

      const now = Date.now();
      const canAlert = now - ppeLastAlertAt > PPE_ALERT_COOLDOWN_MS;
      if (violation && canAlert) {
        ppeLastAlertAt = now;
        addAlert('warning', 'PPE', `Violation EPI detectee sur ${cameraId.toUpperCase()}`);
        showPopupNotification(`[${cameraId.toUpperCase()}] Violation EPI detectee (${pct}%)`, 'warning');
      }

      _notifyDetection({
        cam: cameraId, type: 'ppe',
        detected: violation, confidence: data.confidence, details: data.details || {},
      });
    } else {
      updateStatus('Erreur API');
      clearOverlayCanvas(canvasId);
    }
  } catch (err) {
    updateStatus('Erreur detection');
    console.error(`[SafeVision] fetch ${cameraId} ppe:`, err);
  } finally {
    ppeRequestInFlight = false;
  }
}

function analyzeCam4Frame(video) { return analyzePPEFrame(video, 'cam4'); }

function startCam4Detection(video) {
  if (!video || cam4DetectionInterval) return;
  analyzeCam4Frame(video);
  cam4DetectionInterval   = setInterval(() => analyzeCam4Frame(video), PPE_ANALYSIS_INTERVAL_MS);
  cam4VideoAnalysisActive = true;
  updateCam4Status('Analyse active');
}

function stopCam4Detection() {
  if (cam4DetectionInterval) { clearInterval(cam4DetectionInterval); cam4DetectionInterval = null; }
  cam4VideoAnalysisActive = false;
  clearOverlayCanvas('cam4-overlay-canvas');
  updateCam4Status('Analyse arretee');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 5 — MANHOLE DETECTION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam5Status(text) {
  const el = document.getElementById('cam5-manhole-status');
  if (el) el.textContent = 'Detection: ' + text;
}

function drawManholeOverlay(data, video) {
  const canvas = document.getElementById('cam5-overlay-canvas');
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx      = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect     = getContainedVideoRect(canvas, video);
  const polygons = Array.isArray(data?.polygons) ? data.polygons : [];
  const isOpen   = data?.manhole_state === 'open';

  ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.lineWidth = 4;
  polygons.forEach(points => {
    if (!Array.isArray(points) || points.length < 3) return;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = rect.x + point[0] * rect.scaleX;
      const y = rect.y + point[1] * rect.scaleY;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.shadowColor = isOpen ? 'rgba(255,107,0,0.35)' : 'rgba(74,227,181,0.35)';
    ctx.shadowBlur  = 10;
    ctx.fillStyle   = isOpen ? 'rgba(255,107,0,0.22)'  : 'rgba(74,227,181,0.18)';
    ctx.strokeStyle = isOpen ? 'rgba(255,107,0,0.95)'  : 'rgba(74,227,181,0.92)';
    ctx.fill(); ctx.stroke();
    ctx.shadowBlur = 0;
  });

  const bbox = data?.bbox;
  let labelX = 12, labelY = 24;
  if (Array.isArray(bbox) && bbox.length === 4) {
    const [x1, y1, x2, y2] = bbox;
    const rx = rect.x + x1 * rect.scaleX;
    const ry = rect.y + y1 * rect.scaleY;
    ctx.strokeStyle = isOpen ? 'rgba(255,107,0,0.95)' : 'rgba(74,227,181,0.95)';
    ctx.strokeRect(rx, ry, (x2 - x1) * rect.scaleX, (y2 - y1) * rect.scaleY);
    labelX = rx; labelY = Math.max(18, ry);
  }

  if (!Array.isArray(bbox) && !polygons.length) return;
  const label = (data?.manhole_state || 'unknown').toUpperCase();
  ctx.fillStyle = isOpen ? 'rgba(255,107,0,0.94)' : 'rgba(74,227,181,0.94)';
  ctx.font = 'bold 11px monospace';
  const tw = ctx.measureText(label).width;
  ctx.fillRect(labelX, labelY - 18, tw + 10, 18);
  ctx.fillStyle = '#081014';
  ctx.fillText(label, labelX + 5, labelY - 5);
}

async function analyzeCam5Frame(video) {
  if (!video || video.paused || video.ended) return;
  if (manholeRequestInFlight) return;
  manholeRequestInFlight = true;
  const imageData = captureVideoFrame(video);
  if (!imageData) { updateCam5Status('Capture impossible'); manholeRequestInFlight = false; return; }
  updateCam5Status('Analyse en cours...');

  try {
    const res  = await fetch('/api/manhole-detection/', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken') },
      body:    JSON.stringify({ image: imageData, camera: 'cam5' }),
    });
    const data = await res.json();

    if (data.status === 'success') {
      const pct      = Math.round(data.confidence * 100);
      const holdText = data.details?.temporal_hold ? ' suivi' : '';
      const state    = (data.details?.manhole_state || 'unknown').toUpperCase();
      updateCam5Status(data.manhole_detected ? `${state}${holdText} (${pct}%)` : 'OK (0%)');

      if (Array.isArray(data.details?.bbox) || Array.isArray(data.details?.polygons)) {
        drawManholeOverlay(data.details || {}, video);
      } else {
        clearOverlayCanvas('cam5-overlay-canvas');
      }

      const now      = Date.now();
      const canAlert = now - manholeLastAlertAt > MANHOLE_ALERT_COOLDOWN_MS;
      if (data.manhole_detected && data.details?.manhole_state === 'open' && canAlert) {
        manholeLastAlertAt = now;
        addAlert('critical', 'Manhole', 'Manhole ouvert detecte sur CAM5');
        showPopupNotification(`[CAM5] Manhole ouvert (${pct}%)`, 'critical');
      }

      _notifyDetection({
        cam: 'cam5', type: 'manhole',
        detected: data.manhole_detected, confidence: data.confidence, details: data.details || {},
      });
    } else {
      updateCam5Status('Erreur API');
      clearOverlayCanvas('cam5-overlay-canvas');
    }
  } catch (err) {
    updateCam5Status('Erreur detection');
    console.error('[SafeVision] fetch cam5:', err);
  } finally {
    manholeRequestInFlight = false;
  }
}

function startCam5Detection(video) {
  if (!video || cam5DetectionInterval) return;
  analyzeCam5Frame(video);
  cam5DetectionInterval   = setInterval(() => analyzeCam5Frame(video), MANHOLE_ANALYSIS_INTERVAL_MS);
  cam5VideoAnalysisActive = true;
  updateCam5Status('Analyse active');
}

function stopCam5Detection() {
  if (cam5DetectionInterval) { clearInterval(cam5DetectionInterval); cam5DetectionInterval = null; }
  cam5VideoAnalysisActive = false;
  clearOverlayCanvas('cam5-overlay-canvas');
  updateCam5Status('Analyse arretee');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 6 — EXIT EMERGENCY DETECTION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam6Status(text) {
  ['cam6-exit-status', 'cam6-exit-status-secondary'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.textContent = 'Detection: ' + text;
  });
}

function drawExitOverlay(canvas, video, detections, exitBbox, obstacleBbox, isBlocked) {
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx  = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const rect = getContainedVideoRect(canvas, video);

  (detections || []).forEach(det => {
    const box = det.bbox || det.bbox_xyxy;
    if (!box || box.length < 4) return;
    const x1 = rect.x + box[0] * rect.scaleX;
    const y1 = rect.y + box[1] * rect.scaleY;
    const x2 = rect.x + box[2] * rect.scaleX;
    const y2 = rect.y + box[3] * rect.scaleY;
    const isExit = det.label === 'exit' || det.class_name === 'exit';
    const color  = isExit ? '#388E3C' : '#F57C00';

    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = (det.label || det.class_name || '') +
                  '  ' + Math.round((det.confidence || 0) * 100) + '%';
    ctx.font      = 'bold 11px monospace';
    const tw      = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1, y1 - 18, tw + 8, 18);
    ctx.fillStyle = 'white';
    ctx.fillText(label, x1 + 4, y1 - 4);
  });

  if (isBlocked && exitBbox && exitBbox.length === 4) {
    const x1 = rect.x + exitBbox[0] * rect.scaleX;
    const y1 = rect.y + exitBbox[1] * rect.scaleY;
    const x2 = rect.x + exitBbox[2] * rect.scaleX;
    const y2 = rect.y + exitBbox[3] * rect.scaleY;
    ctx.strokeStyle = '#D32F2F';
    ctx.lineWidth   = 4;
    ctx.setLineDash([]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = 'rgba(211,47,47,0.12)';
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    const txt = 'EXIT BLOCKED';
    ctx.font      = 'bold 13px monospace';
    const tw      = ctx.measureText(txt).width;
    ctx.fillStyle = '#D32F2F';
    ctx.fillRect(x1, y2 + 2, tw + 10, 20);
    ctx.fillStyle = 'white';
    ctx.fillText(txt, x1 + 5, y2 + 16);
  }

  if (isBlocked && obstacleBbox && obstacleBbox.length === 4) {
    const x1 = rect.x + obstacleBbox[0] * rect.scaleX;
    const y1 = rect.y + obstacleBbox[1] * rect.scaleY;
    const x2 = rect.x + obstacleBbox[2] * rect.scaleX;
    const y2 = rect.y + obstacleBbox[3] * rect.scaleY;
    ctx.strokeStyle = '#FF6B00';
    ctx.lineWidth   = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }

  const badgeTxt   = isBlocked ? 'EXIT BLOCKED' : 'EXIT ACCESSIBLE';
  const badgeColor = isBlocked ? '#D32F2F' : '#388E3C';
  ctx.font         = 'bold 12px monospace';
  const bw         = ctx.measureText(badgeTxt).width + 16;
  ctx.fillStyle    = badgeColor;
  ctx.fillRect(canvas.width - bw - 6, 8, bw, 22);
  ctx.fillStyle    = 'white';
  ctx.fillText(badgeTxt, canvas.width - bw, 24);
}

async function analyzeCam6Frame(video, runToken = cam6RunToken) {
  if (!video || video.paused || video.ended || !cam6VideoAnalysisActive) return;
  if (exitRequestInFlight || runToken !== cam6RunToken) return;
  exitRequestInFlight = true;

  const imageData = captureVideoFrame(video);
  if (!imageData) {
    if (runToken === cam6RunToken && cam6VideoAnalysisActive) updateCam6Status('Capture impossible');
    exitRequestInFlight = false;
    return;
  }
  updateCam6Status('Analyse en cours...');

  try {
    const res  = await fetch('/api/blocked-exit-detection/', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken') },
      body:    JSON.stringify({ image: imageData, camera: 'cam6', conf: 0.25 }),
    });
    const data = await res.json();
    if (runToken !== cam6RunToken || !cam6VideoAnalysisActive || video.paused) return;

    if (data.status === 'success') {
      const blocked      = data.blocked_exit_detected || false;
      const conf         = Math.round((data.confidence || 0) * 100);
      const details      = data.details    || {};
      const detections   = details.detections   || [];
      const exitBbox     = details.exit_bbox;
      const obstacleBbox = details.obstacle_bbox;
      const canvas       = document.getElementById('cam6-overlay-canvas');
      const exits        = detections.filter(d => d.label === 'exit'     || d.class_name === 'exit').length;
      const obs          = detections.filter(d => d.label === 'obstacle' || d.class_name === 'obstacle').length;

      if (blocked) {
        updateCam6Status(`BLOQUEE (${conf}%)`);
      } else if (detections.length > 0) {
        updateCam6Status(`exits=${exits}  obstacles=${obs}  LIBRE`);
      } else {
        updateCam6Status('Aucune sortie detectee');
      }

      drawExitOverlay(canvas, video, detections, exitBbox, obstacleBbox, blocked);

      const now      = Date.now();
      const canAlert = now - exitLastAlertAt > EXIT_ALERT_COOLDOWN_MS;
      if (blocked && canAlert) {
        exitLastAlertAt = now;
        addAlert('critical', 'Exit', `Sortie de secours bloquee sur CAM6 (${conf}%)`);
        showPopupNotification(`[CAM6] Sortie bloquee (${conf}%)`, 'critical');
      }

      _notifyDetection({
        cam: 'cam6', type: 'blocked_exit',
        detected: blocked, confidence: data.confidence, details,
      });
    } else {
      updateCam6Status('Erreur API');
      clearOverlayCanvas('cam6-overlay-canvas');
    }
  } catch (err) {
    if (runToken === cam6RunToken && cam6VideoAnalysisActive) {
      updateCam6Status('Erreur detection');
      console.error('[SafeVision] fetch cam6:', err);
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
  cam6DetectionInterval   = setInterval(() => analyzeCam6Frame(video, runToken), EXIT_ANALYSIS_INTERVAL_MS);
  updateCam6Status('Analyse active');
}

function stopCam6Detection() {
  if (cam6DetectionInterval) { clearInterval(cam6DetectionInterval); cam6DetectionInterval = null; }
  cam6VideoAnalysisActive = false;
  cam6RunToken += 1;
  exitRequestInFlight = false;
  clearOverlayCanvas('cam6-overlay-canvas');
  updateCam6Status('Analyse arretee');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CAM 7 — SIGN DEFECT DETECTION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam7Status(text) {
  const el = document.getElementById('cam7-sign-status');
  if (el) el.textContent = 'Detection: ' + text;
}

function drawSignOverlay(canvasId, data, video, isDefective) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !video) return;
  setCanvasSize(canvas, video);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const accentColor = isDefective ? 'rgba(74, 195, 255, 0.95)' : 'rgba(181, 227, 74, 0.95)'; // Soft Orange vs Teal
  
  // 1. Draw Full-Frame Border
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 4;
  ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
  
  // 2. Draw HUD Background
  ctx.fillStyle = 'rgba(30, 30, 30, 0.75)';
  ctx.fillRect(8, 8, 310, 95);
  ctx.strokeStyle = accentColor;
  ctx.lineWidth = 1;
  ctx.strokeRect(8, 8, 310, 95);
  
  // 3. Draw Text
  ctx.font = 'bold 12px monospace';
  ctx.fillStyle = '#F0F0F0';
  ctx.fillText(`CAT: ${data.category} - ${data.category_name}`, 16, 28);
  
  ctx.fillStyle = '#A0A0A0';
  ctx.font = '11px monospace';
  ctx.fillText(`ROUTER CONF: ${Math.round((data.cat_confidence || 0) * 100)}%`, 16, 48);
  ctx.fillText(`DEFECT: ${data.defect_score?.toFixed(3)} (Th: ${data.threshold?.toFixed(3)})`, 16, 64);
  
  ctx.font = 'bold 13px monospace';
  ctx.fillStyle = accentColor;
  ctx.fillText(`VERDICT: ${data.verdict}`, 16, 88);
}

async function analyzeSignFrame(video, cameraId = 'cam7') {
  if (!video || video.paused || video.ended) return;
  if (signRequestInFlight) return;
  signRequestInFlight = true;

  const imageData = captureVideoFrame(video);
  const canvasId = `${cameraId}-overlay-canvas`;

  if (!imageData) {
    updateCam7Status('Capture impossible');
    signRequestInFlight = false;
    return;
  }
  updateCam7Status('Analyse en cours...');

  try {
    const res  = await fetch('/api/sign-detect/', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken') },
      body:    JSON.stringify({ image: imageData, camera: cameraId }),
    });
    const data = await res.json();

    if (data.status === 'success') {
      const result = data.data || {};
      const isDefective = result.is_defective || false;
      const verdict = result.verdict || 'UNKNOWN';
      const conf = Math.round((result.defect_score || 0) * 100);

      updateCam7Status(isDefective ? `${verdict} (${conf}%)` : `${verdict} (${conf}%)`);

      drawSignOverlay(canvasId, result, video, isDefective);

      const now = Date.now();
      const canAlert = now - signLastAlertAt > SIGN_ALERT_COOLDOWN_MS;
      if (isDefective && canAlert) {
        signLastAlertAt = now;
        addAlert('warning', 'Sign', `Panneau défectueux détecté sur ${cameraId.toUpperCase()}`);
        showPopupNotification(`[${cameraId.toUpperCase()}] Panneau Défectueux (${conf}%)`, 'warning');
      }

      _notifyDetection({
        cam: cameraId, type: 'sign_defect',
        detected: isDefective, confidence: result.defect_score, details: result,
      });
    } else {
      updateCam7Status('Erreur API');
      clearOverlayCanvas(canvasId);
    }
  } catch (err) {
    updateCam7Status('Erreur detection');
    console.error(`[SafeVision] fetch ${cameraId} sign:`, err);
  } finally {
    signRequestInFlight = false;
  }
}

function analyzeCam7Frame(video) { return analyzeSignFrame(video, 'cam7'); }

function startCam7Detection(video) {
  if (!video || cam7DetectionInterval) return;
  analyzeCam7Frame(video);
  cam7DetectionInterval   = setInterval(() => analyzeCam7Frame(video), SIGN_ANALYSIS_INTERVAL_MS);
  cam7VideoAnalysisActive = true;
  updateCam7Status('Analyse active');
}

function stopCam7Detection() {
  if (cam7DetectionInterval) { clearInterval(cam7DetectionInterval); cam7DetectionInterval = null; }
  cam7VideoAnalysisActive = false;
  clearOverlayCanvas('cam7-overlay-canvas');
  updateCam7Status('Analyse arretee');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// UPLOAD VIDÉO
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const videoUploadEl = document.getElementById('video-upload');
if (videoUploadEl) {
  videoUploadEl.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    document.title = `[${file.name}] — SafeVision AI`;

    const cam4Feed = document.getElementById('cam-feed-4') || document.getElementById('cam4-feed-active');
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
      const cam4Video = document.getElementById('cam4-video');
      cam4Video.addEventListener('loadeddata', () => {
        cam4Video.play().catch(() => {});
        startCam4Detection(cam4Video);
      }, { once: true });
      cam4Video.load();
      return;
    }

    const liveVideo = document.getElementById('cam1-video');
    if (liveVideo) {
      liveVideo.src = url; liveVideo.load(); liveVideo.play().catch(() => {});
      startLiveDetection(liveVideo);
      return;
    }
    const placeholder = document.getElementById('video-placeholder');
    if (placeholder) {
      placeholder.innerHTML = `<video id="main-video" src="${url}"
        style="width:100%;height:100%;object-fit:contain" controls></video>`;
    }
  });
}

const camSelect = document.getElementById('camera-select');
if (camSelect) {
  camSelect.addEventListener('change', function () {
    if (this.value === 'Fichier vidéo...') {
      document.getElementById('video-upload')?.click();
    }
  });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MODULE TOGGLES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function toggleModule(chip) {
  chip.classList.toggle('on');
  console.log(`[SafeVision] Module ${chip.dataset.module}: ${chip.classList.contains('on') ? 'on' : 'off'}`);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CONTRÔLES GLOBAUX
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function initializeCameras() {
  if (camerasInitialized) return;

  [
    ['cam1-video', LIVE_VIDEO_SRC,       updateCam1Status],
    ['cam2-video', LIVE_VIDEO_SRC_CAM2,  updateCam2Status],
    ['cam4-video', LIVE_VIDEO_SRC_CAM4,  updateCam4Status],
    ['cam5-video', LIVE_VIDEO_SRC_CAM5,  updateCam5Status],
    ['cam6-video', LIVE_VIDEO_SRC_CAM6,  updateCam6Status],
    ['cam7-video', LIVE_VIDEO_SRC_CAM7,  updateCam7Status], // ADDED CAM 7
  ].forEach(([id, src, statusFn]) => {
    const v = document.getElementById(id);
    if (v) { v.src = src + '?v=' + Date.now(); v.load(); v.pause(); statusFn('Camera prete'); }
  });

  camerasInitialized = true;
}

function _startCamWithLoop(id, src, startFn) {
  const v = document.getElementById(id);
  if (!v) return;
  v.dataset.shouldRun = '1';
  v.src         = src + '?v=' + Date.now();
  v.currentTime = 0;
  v.load();
  v.onloadeddata = () => {
    if (!camerasRunning || v.dataset.shouldRun !== '1') return;
    v.play().catch(err => console.error(`[SafeVision] ${id} play:`, err));
    startFn(v);
    v.onloadeddata = null;
  };
  v.onended = () => {
    if (!camerasRunning || v.dataset.shouldRun !== '1') return;
    v.currentTime = 0;
    v.play().catch(() => {});
  };
}

function startCameras() {
  console.log('[SafeVision] startCameras()');
  camerasRunning = true;
  initializeCameras();

  const cam1 = document.getElementById('cam1-video');
  if (cam1) { cam1.currentTime = 0; cam1.play().catch(() => {}); startLiveDetection(cam1); }

  const cam2 = document.getElementById('cam2-video');
  if (cam2) { cam2.play().catch(() => {}); startFatigueDetection(cam2); }

  _startCamWithLoop('cam3-video', LIVE_VIDEO_SRC_CAM3, startCam3Detection);
  _startCamWithLoop('cam4-video', LIVE_VIDEO_SRC_CAM4, startCam4Detection);
  _startCamWithLoop('cam5-video', LIVE_VIDEO_SRC_CAM5, startCam5Detection);
  _startCamWithLoop('cam6-video', LIVE_VIDEO_SRC_CAM6, startCam6Detection);
  _startCamWithLoop('cam7-video', LIVE_VIDEO_SRC_CAM7, startCam7Detection); // ADDED CAM 7
}

function stopCameras() {
  console.log('[SafeVision] stopCameras()');
  camerasRunning = false;

  const cam1 = document.getElementById('cam1-video');
  if (cam1) { cam1.pause(); updateCam1Status('Caméra arrêtée'); }
  stopLiveDetection();

  const cam2 = document.getElementById('cam2-video');
  if (cam2) { cam2.pause(); updateCam2Status('Caméra arrêtée'); }
  stopFatigueDetection();

  [
    ['cam3-video', updateCam3Status, stopCam3Detection],
    ['cam4-video', updateCam4Status, stopCam4Detection],
    ['cam5-video', updateCam5Status, stopCam5Detection],
    ['cam6-video', updateCam6Status, stopCam6Detection],
    ['cam7-video', updateCam7Status, stopCam7Detection], // ADDED CAM 7
  ].forEach(([id, statusFn, stopFn]) => {
    const v = document.getElementById(id);
    if (v) {
      v.dataset.shouldRun = '0';
      v.onloadeddata = null;
      v.onended = null;
      v.pause();
      statusFn('Camera arretee');
    }
    stopFn();
  });

  clearAllOverlayCanvases();
}

function startAllModules() { startCameras(); }
function stopAllModules()  { stopCameras(); }

function setLayout(cols) {
  const grid = document.getElementById('live-grid');
  if (grid) grid.style.gridTemplateColumns = cols === 1 ? '2fr 1fr' : 'repeat(2, 1fr)';
}

function filterAlerts(type, btn) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// API HELPERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async function postAlertToAPI(alertData) {
  try {
    const res = await fetch('/api/alerts/', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken') },
      body:    JSON.stringify(alertData),
    });
    return await res.json();
  } catch (e) { console.error('[API] POST alerte:', e); }
}

async function fetchRecentDetections(moduleId, limit = 20) {
  try {
    const res = await fetch(`/api/detections/?module=${moduleId}&limit=${limit}`);
    return await res.json();
  } catch (e) { console.error('[API] GET détections:', e); return []; }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// THÈME
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function applyTheme(theme) {
  document.body.classList.toggle('theme-dark',  theme === 'dark');
  document.body.classList.toggle('theme-light', theme === 'light');
  localStorage.setItem('theme', theme);
  const btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.textContent = theme === 'dark' ? '☀️' : '🌙';
    btn.setAttribute('aria-label', theme === 'dark' ? 'Mode clair' : 'Mode sombre');
  }
}

function toggleTheme() {
  applyTheme(document.body.classList.contains('theme-dark') ? 'light' : 'dark');
}

function initTheme() {
  const saved = localStorage.getItem('theme');
  const def   = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  applyTheme(saved || def);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// KPI
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const KPI = {
  setCritical: v => { const e = document.getElementById('kpi-critical'); if (e) e.textContent = v; },
  setWarnings: v => { const e = document.getElementById('kpi-warnings'); if (e) e.textContent = v; },
  setWorkers:  v => { const e = document.getElementById('kpi-workers');  if (e) e.textContent = v; },
  setModules:  v => { const e = document.getElementById('kpi-modules');  if (e) e.textContent = v; },
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DOMContentLoaded
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);

  document.querySelectorAll('.stat-card, .module-card, .panel').forEach((el, i) => {
    el.style.opacity   = '0';
    el.style.transform = 'translateY(8px)';
    setTimeout(() => {
      el.style.transition = 'opacity .3s ease, transform .3s ease';
      el.style.opacity    = '1';
      el.style.transform  = 'none';
    }, i * 40);
  });

  const cam1 = document.getElementById('cam1-video');
  if (cam1) {
    cam1.addEventListener('loadeddata', () => {
      if (cam1.paused && !liveVideoAnalysisActive) updateCam1Status('Caméra prête');
    });
    cam1.addEventListener('play',  () => { if (!liveVideoAnalysisActive) startLiveDetection(cam1); });
    cam1.addEventListener('pause', () => updateCam1Status('Vidéo en pause'));
  }

  const cam2 = document.getElementById('cam2-video');
  if (cam2) {
    cam2.addEventListener('loadeddata', () => {
      if (cam2.paused && !fatigueVideoAnalysisActive) updateCam2Status('Caméra prête');
    });
    cam2.addEventListener('play',  () => { if (!fatigueVideoAnalysisActive) startFatigueDetection(cam2); });
    cam2.addEventListener('pause', () => updateCam2Status('Vidéo en pause'));
  }
});