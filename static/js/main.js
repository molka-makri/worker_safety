// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SAFEVISION AI — main.js
// Utilitaires partagés, horloge, alertes globales
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// --- HORLOGE LIVE ---
function updateClock() {
  const el = document.getElementById('live-clock');
  if (el) {
    const now = new Date();
    el.textContent = now.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }
}
setInterval(updateClock, 1000);
updateClock();

// --- UPLOAD VIDÉO ---
const videoUploadEl = document.getElementById('video-upload');
if (videoUploadEl) {
  videoUploadEl.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    console.log('[SafeVision] Vidéo chargée:', file.name, url);
    document.title = `[${file.name}] — SafeVision AI`;

    const placeholder = document.getElementById('video-placeholder');
    if (placeholder) {
      placeholder.innerHTML = `
        <video id="main-video" src="${url}" style="width:100%; height:100%; object-fit:contain" controls></video>
      `;
    }
  });
}

// --- SÉLECTEUR CAMÉRA ---
const camSelect = document.getElementById('camera-select');
if (camSelect) {
  camSelect.addEventListener('change', function() {
    if (this.value === 'Fichier vidéo...') {
      document.getElementById('video-upload').click();
    } else {
      console.log('[SafeVision] Caméra sélectionnée:', this.value);
    }
  });
}

// --- MODULE TOGGLES ---
function toggleModule(chip) {
  chip.classList.toggle('on');
  const mod = chip.dataset.module;
  const active = chip.classList.contains('on');
  console.log(`[SafeVision] Module ${mod}: ${active ? 'activé' : 'désactivé'}`);
}

// --- ALERTES GLOBALES ---
let alertCount = 0;

/**
 * Ajouter une alerte dans le feed global (sidebar + dashboard)
 * @param {string} severity - 'critical' | 'warning' | 'info'
 * @param {string} module   - Nom du module source
 * @param {string} message  - Message de l'alerte
 */
function addAlert(severity, module, message) {
  alertCount++;

  const badge = document.getElementById('alert-badge');
  if (badge) badge.textContent = alertCount;

  const feed = document.getElementById('alerts-feed');
  if (feed) {
    const empty = feed.querySelector('.empty-state');
    if (empty) empty.remove();

    const item = document.createElement('div');
    item.className = 'alert-item fade-in';
    item.innerHTML = `
      <div class="alert-severity sev-${severity}"></div>
      <div class="alert-body">
        <div class="alert-module">${module.toUpperCase()}</div>
        <div class="alert-msg">${message}</div>
        <div class="alert-time">${new Date().toLocaleTimeString('fr-FR')}</div>
      </div>
    `;
    feed.insertBefore(item, feed.firstChild);

    while (feed.children.length > 50) {
      feed.removeChild(feed.lastChild);
    }
  }

  if (severity === 'critical') {
    playAlertSound();
  }

  console.log(`[${severity.toUpperCase()}] ${module}: ${message}`);
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
  } catch(e) {}
}

function startAnalysis() {
  console.log('[SafeVision] startAnalysis() — Connecter vos modèles ici');
  const btn = document.getElementById('btn-start-analysis');
  if (btn) {
    btn.classList.remove('primary');
    btn.style.background = 'var(--accent-green)';
    btn.style.borderColor = 'var(--accent-green)';
  }
  document.getElementById('video-info') && (document.getElementById('video-info').style.display = 'flex');
}

function stopAnalysis() {
  console.log('[SafeVision] stopAnalysis()');
  const btn = document.getElementById('btn-start-analysis');
  if (btn) {
    btn.style.cssText = '';
    btn.classList.add('primary');
  }
}

function startAllModules() {
  console.log('[SafeVision] startAllModules() — Démarrer tous les modèles');
}
function stopAllModules() {
  console.log('[SafeVision] stopAllModules()');
}
function setLayout(cols) {
  const grid = document.getElementById('live-grid');
  if (grid) {
    grid.style.gridTemplateColumns = cols === 1
      ? '2fr 1fr'
      : 'repeat(2, 1fr)';
  }
}

function filterAlerts(type, btn) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  console.log('[SafeVision] Filtre alertes:', type);
}

async function postAlertToAPI(alertData) {
  try {
    const res = await fetch('/api/alerts/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken')
      },
      body: JSON.stringify(alertData)
    });
    return await res.json();
  } catch(e) {
    console.error('[API] Erreur POST alerte:', e);
  }
}

async function fetchRecentDetections(moduleId, limit = 20) {
  try {
    const res = await fetch(`/api/detections/?module=${moduleId}&limit=${limit}`);
    return await res.json();
  } catch(e) {
    console.error('[API] Erreur GET détections:', e);
    return [];
  }
}

function getCookie(name) {
  const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
  return match ? match[2] : '';
}

function applyTheme(theme) {
  document.body.classList.toggle('theme-dark', theme === 'dark');
  document.body.classList.toggle('theme-light', theme === 'light');
  localStorage.setItem('theme', theme);
  const btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.textContent = theme === 'dark' ? '☀️' : '🌙';
    btn.setAttribute('aria-label', theme === 'dark' ? 'Activer mode clair' : 'Activer mode sombre');
  }
}

function toggleTheme() {
  const current = document.body.classList.contains('theme-dark') ? 'dark' : 'light';
  applyTheme(current === 'dark' ? 'light' : 'dark');
}

function initTheme() {
  const savedTheme = localStorage.getItem('theme');
  const defaultTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  applyTheme(savedTheme || defaultTheme);
}

const KPI = {
  setCritical: v => { const el = document.getElementById('kpi-critical'); if(el) el.textContent = v; },
  setWarnings: v => { const el = document.getElementById('kpi-warnings'); if(el) el.textContent = v; },
  setWorkers:  v => { const el = document.getElementById('kpi-workers'); if(el) el.textContent = v; },
  setModules:  v => { const el = document.getElementById('kpi-modules'); if(el) el.textContent = v; }
};

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', toggleTheme);
  }

  document.querySelectorAll('.stat-card, .module-card, .panel').forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(8px)';
    setTimeout(() => {
      el.style.transition = 'opacity .3s ease, transform .3s ease';
      el.style.opacity = '1';
      el.style.transform = 'none';
    }, i * 40);
  });
});