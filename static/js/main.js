// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SAFEVISION AI — main.js  (v2 — intégration module_detail_features)
// Utilitaires partagés, horloge, alertes globales
// CAM 1 : Détection de chute   (worker_falling3.mp4)
// CAM 2 : Détection de fatigue (worker_tired.mp4)
// CAM 3 : Validation chute     (worker_falling3_fall.mp4)
//
// Après chaque résultat d'API, appelle window.onDetectionResult()
// (défini dans module_detail_features.js) pour alimenter :
//   • le tableau de détections en temps réel
//   • le panneau d'alertes droite
//   • l'export PDF
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
const LIVE_VIDEO_SRC          = '/media/worker_falling3.mp4';
const LIVE_VIDEO_SRC_CAM2     = '/media/worker_tired.mp4';
const LIVE_VIDEO_SRC_CAM3     = '/media/worker_falling3_fall.mp4';
const VIDEO_CAPTURE_MAX_WIDTH  = 960;
const VIDEO_CAPTURE_MAX_HEIGHT = 960;

// ── ÉTAT DES MODULES ──────────────────────────────────────
let liveDetectionInterval      = null;   // CAM 1 — chute
let liveVideoAnalysisActive    = false;

let fatigueDetectionInterval   = null;   // CAM 2 — fatigue
let fatigueVideoAnalysisActive = false;

let cam3DetectionInterval      = null;   // CAM 3 — chute test
let cam3VideoAnalysisActive    = false;

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
  clearOverlayCanvas('cam1-overlay-canvas');
  clearOverlayCanvas('cam2-overlay-canvas');
  clearOverlayCanvas('cam3-overlay-canvas');
}

function getContainedVideoRect(canvas, video) {
  const captureWidth = Number(video.dataset.captureWidth) || video.videoWidth || VIDEO_CAPTURE_MAX_WIDTH;
  const captureHeight = Number(video.dataset.captureHeight) || video.videoHeight || VIDEO_CAPTURE_MAX_HEIGHT;
  const scale = Math.min(canvas.width / captureWidth, canvas.height / captureHeight);
  const width = captureWidth * scale;
  const height = captureHeight * scale;
  return {
    x: (canvas.width - width) / 2,
    y: (canvas.height - height) / 2,
    scaleX: width / captureWidth,
    scaleY: height / captureHeight,
  };
}

// Helper : notifie module_detail_features.js si disponible
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
// ALERTES GLOBALES (barre latérale + badge)
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

  // Label
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

      // ── Intégration module_detail_features ──
      _notifyDetection({
        cam:        'cam1',
        type:       'fall',
        detected:   data.fall_detected,
        confidence: data.confidence,
        details:    data.details || {},
      });

    } else {
      updateCam1Status('Erreur API');
      clearOverlayCanvas('cam1-overlay-canvas');
      console.warn('[SafeVision] API chute:', data);
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

  // Label
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

      // ── Intégration module_detail_features ──
      _notifyDetection({
        cam:        'cam2',
        type:       'fatigue',
        detected:   data.fatigue_detected,
        confidence: data.confidence,
        details:    data.details || {},
      });

    } else {
      updateCam2Status('Erreur API');
      clearOverlayCanvas('cam2-overlay-canvas');
      console.warn('[SafeVision] API fatigue:', data);
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
// CAM 3 — DÉTECTION DE CHUTE (TEST)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function updateCam3Status(text) {
  const el = document.getElementById('cam3-fall-status');
  if (el) el.textContent = 'Détection: ' + text;
}

function drawCam3Overlay(bbox, video, isFall) {
  const canvas = document.getElementById('cam3-overlay-canvas');
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
  ctx.lineWidth   = 3;
  ctx.setLineDash([]);
  ctx.strokeRect(rx, ry, (x2 - x1) * rect.scaleX, (y2 - y1) * rect.scaleY);

  ctx.fillStyle = isFall ? 'rgba(255,59,47,0.90)' : 'rgba(74,227,181,0.90)';
  ctx.font = 'bold 11px monospace';
  const label = isFall ? '⚠ CHUTE' : '✓ OK';
  const tw    = ctx.measureText(label).width;
  ctx.fillRect(rx, ry - 18, tw + 10, 18);
  ctx.fillStyle = '#fff';
  ctx.fillText(label, rx + 5, ry - 5);
}

async function analyzeCam3Frame(video) {
  if (!video || video.paused || video.ended) return;
  const imageData = captureVideoFrame(video);
  if (!imageData) { updateCam3Status('Capture impossible'); return; }
  updateCam3Status('Analyse en cours…');

  try {
    const res  = await fetch('/api/fall-detection/', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken') },
      body:    JSON.stringify({ image: imageData, camera: 'cam3' }),
    });
    const data = await res.json();

    if (data.status === 'success') {
      const pct = Math.round(data.confidence * 100);
      updateCam3Status(data.fall_detected ? `CHUTE (${pct}%)` : `OK (${pct}%)`);

      if (Array.isArray(data.details?.bbox)) {
        drawCam3Overlay(data.details.bbox, video, data.fall_detected);
      } else {
        clearOverlayCanvas('cam3-overlay-canvas');
      }

      if (data.fall_detected) {
        addAlert('critical', 'Fall CAM3', 'Chute détectée sur caméra 3 (validation)');
        showPopupNotification(`[CAM3] Chute détectée (${pct}%)`, 'critical');
      }

      // ── Intégration module_detail_features ──
      _notifyDetection({
        cam:        'cam3',
        type:       'fall',
        detected:   data.fall_detected,
        confidence: data.confidence,
        details:    data.details || {},
      });

    } else {
      updateCam3Status('Erreur API');
      clearOverlayCanvas('cam3-overlay-canvas');
      console.warn('[SafeVision] API cam3:', data);
    }
  } catch (err) {
    updateCam3Status('Erreur détection');
    console.error('[SafeVision] fetch cam3:', err);
  }
}

function startCam3Detection(video) {
  if (!video || cam3DetectionInterval) return;
  analyzeCam3Frame(video);
  cam3DetectionInterval   = setInterval(() => analyzeCam3Frame(video), 1200);
  cam3VideoAnalysisActive = true;
  updateCam3Status('Analyse active');
}

function stopCam3Detection() {
  if (cam3DetectionInterval) { clearInterval(cam3DetectionInterval); cam3DetectionInterval = null; }
  cam3VideoAnalysisActive = false;
  clearOverlayCanvas('cam3-overlay-canvas');
  updateCam3Status('Analyse arrêtée');
}

function loadCam3Video(path) {
  const v = document.getElementById('cam3-video');
  if (!v) return;
  v.src = path; v.load(); v.play().catch(() => {});
  if (!cam3VideoAnalysisActive) startCam3Detection(v);
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

    const liveVideo = document.getElementById('cam1-video');
    if (liveVideo) {
      liveVideo.src = url;
      liveVideo.load();
      liveVideo.play().catch(() => {});
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
// CONTRÔLES GLOBAUX (startCameras / stopCameras)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

function initializeCameras() {
  if (camerasInitialized) return;

  const cam1 = document.getElementById('cam1-video');
  if (cam1) {
    cam1.src = LIVE_VIDEO_SRC + '?v=' + Date.now();
    cam1.load(); cam1.pause();
    updateCam1Status('Caméra prête');
  }
  const cam2 = document.getElementById('cam2-video');
  if (cam2) {
    cam2.src = LIVE_VIDEO_SRC_CAM2 + '?v=' + Date.now();
    cam2.load(); cam2.pause();
    updateCam2Status('Caméra prête');
  }
  camerasInitialized = true;
}

function startCameras() {
  console.log('[SafeVision] startCameras()');
  initializeCameras();

  const cam1 = document.getElementById('cam1-video');
  if (cam1) {
    cam1.currentTime = 0;
    cam1.play().catch(() => {});
    startLiveDetection(cam1);
  }

  const cam2 = document.getElementById('cam2-video');
  if (cam2) {
    cam2.play().catch(() => {});
    startFatigueDetection(cam2);
  }

  const cam3 = document.getElementById('cam3-video');
  if (cam3) {
    cam3.src = LIVE_VIDEO_SRC_CAM3 + '?v=' + Date.now();
    cam3.currentTime = 0;
    cam3.load();
    cam3.addEventListener('loadeddata', () => {
      cam3.play().catch(err => console.error('[SafeVision] CAM3 play:', err));
      startCam3Detection(cam3);
    }, { once: true });
    cam3.addEventListener('ended', () => {
      cam3.currentTime = 0;
      cam3.play().catch(() => {});
    });
  }
}

function stopCameras() {
  console.log('[SafeVision] stopCameras()');

  const cam1 = document.getElementById('cam1-video');
  if (cam1) { cam1.pause(); updateCam1Status('Caméra arrêtée'); }
  stopLiveDetection();

  const cam2 = document.getElementById('cam2-video');
  if (cam2) { cam2.pause(); updateCam2Status('Caméra arrêtée'); }
  stopFatigueDetection();

  const cam3 = document.getElementById('cam3-video');
  if (cam3) { cam3.pause(); updateCam3Status('Caméra arrêtée'); }
  stopCam3Detection();
  clearAllOverlayCanvases();
}

function startAllModules() { startCameras(); }
function stopAllModules()  { stopCameras(); }

function setLayout(cols) {
  const grid = document.getElementById('live-grid');
  if (grid) grid.style.gridTemplateColumns = cols === 1 ? '2fr 1fr' : 'repeat(2, 1fr)';
}

// filterAlerts — version de base (peut être surchargée par alerts_features.js)
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
    btn.textContent   = theme === 'dark' ? '☀️' : '🌙';
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
// DOMContentLoaded — INITIALISATION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  document.getElementById('theme-toggle')?.addEventListener('click', toggleTheme);

  // Animation d'entrée des cartes
  document.querySelectorAll('.stat-card, .module-card, .panel').forEach((el, i) => {
    el.style.opacity   = '0';
    el.style.transform = 'translateY(8px)';
    setTimeout(() => {
      el.style.transition = 'opacity .3s ease, transform .3s ease';
      el.style.opacity    = '1';
      el.style.transform  = 'none';
    }, i * 40);
  });

  // ── CAM 1 ──────────────────────────────────────────────
  const cam1 = document.getElementById('cam1-video');
  if (cam1) {
    cam1.addEventListener('loadeddata', () => {
      if (cam1.paused && !liveVideoAnalysisActive) updateCam1Status('Caméra prête');
    });
    cam1.addEventListener('play',  () => { if (!liveVideoAnalysisActive) startLiveDetection(cam1); });
    cam1.addEventListener('pause', () => updateCam1Status('Vidéo en pause'));
  }

  // ── CAM 2 ──────────────────────────────────────────────
  const cam2 = document.getElementById('cam2-video');
  if (cam2) {
    cam2.addEventListener('loadeddata', () => {
      if (cam2.paused && !fatigueVideoAnalysisActive) updateCam2Status('Caméra prête');
    });
    cam2.addEventListener('play',  () => { if (!fatigueVideoAnalysisActive) startFatigueDetection(cam2); });
    cam2.addEventListener('pause', () => updateCam2Status('Vidéo en pause'));
  }
});
