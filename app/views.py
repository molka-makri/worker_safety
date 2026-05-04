from django.shortcuts import render
from django.views.generic import TemplateView, FormView
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth import login
from django.contrib import messages
from django.urls import reverse_lazy
from django.http import JsonResponse, Http404
from django.views.decorators.csrf import csrf_exempt
import json
from datetime import datetime
import random
from .models import Module, Alert, Detection
from .forms import SignUpForm
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from .fall_detector import detect_fall_in_frame
from .fatiguedetector import detect_fatigue_in_frame
from .spill_detector import detect_spill_in_frame
from .manhole_detector import detect_manhole_in_frame, estimate_manhole_depth
from .exit_detector import detect_blocked_exit_in_frame
from .proximity_detector import detect_proximity_in_frame, ProximityDetector
from .ppe_detector import detect_ppe_in_frame
from .sign_detector import detect_sign_in_image
from .posture_detector import detect_posture_in_frame
from .panic_detector import detect_panic_in_frame
from .worker_tracking_detector import detect_worker_tracking_in_frame

MODULE_SLUG_MAP = {
    'ppe': {
        'id': 1,
        'keywords': ('epi', 'ppe', 'signalisation', 'conformite'),
        'fallback_name': 'Conformite EPI + Signalisation',
        'fallback_description': 'Detection casques, gilets, signalisation manquante',
        'icon': 'shield',
        'color': '#4AE3B5',
    },
    'posture': {
        'id': 2,
        'keywords': ('posture', 'comportement'),
        'fallback_name': 'Posture Dangereuse + Comportement',
        'fallback_description': 'Analyse squelette, detection postures anormales',
        'icon': 'activity',
        'color': '#7B61FF',
    },
    'comportement': {'alias': 'posture'},
    'fatigue': {
        'id': 3,
        'keywords': ('fatigue', 'effondrement', 'chute', 'incapacite'),
        'fallback_name': 'Fatigue + Effondrement Travailleur',
        'fallback_description': 'Detection fatigue, chute et immobilite',
        'icon': 'eye',
        'color': '#FFB800',
    },
    'incapacity': {'alias': 'fatigue'},
    'falling': {'alias': 'fatigue'},
    'fall': {'alias': 'fatigue'},
    'tracking': {
        'id': 4,
        'keywords': ('tracking', 'suivi', 'presence', 'objets'),
        'fallback_name': 'Suivi Presence + Objets Tombants',
        'fallback_description': 'Comptage zone, suivi trajectoire objets',
        'icon': 'target',
        'color': '#00C2FF',
    },
    'hazards': {
        'id': 5,
        'keywords': ('hazards', 'risques', 'aleas', 'environnement', 'deversement', 'sorties'),
        'fallback_name': 'Aleas Environnementaux',
        'fallback_description': 'Detection deversements, encombrement, sorties bloquees',
        'icon': 'alert-triangle',
        'color': '#FF6B00',
    },
    'spill': {'alias': 'hazards'},
    'manhole': {'alias': 'hazards'},
    'blocked-exit': {'alias': 'hazards'},
    'fire': {
        'id': 6,
        'keywords': ('feu', 'fumee', 'fire', 'lumiere'),
        'fallback_name': 'Feu/Fumee + Changement Lumiere',
        'fallback_description': 'Detection incendie, fumee, flash/explosion',
        'icon': 'flame',
        'color': '#FF3B2F',
    },
    'machinery': {
        'id': 7,
        'keywords': ('machines', 'machinery', 'proximite', 'anomalies'),
        'fallback_name': 'Anomalies Machines + Proximite',
        'fallback_description': 'Etincelles, fumee, proximite travailleur-machine',
        'icon': 'settings',
        'color': '#E040FB',
    },
    'proximity': {'alias': 'machinery'},
}

MODULE_ALIAS_SLUGS = {slug: cfg.get('alias', slug) for slug, cfg in MODULE_SLUG_MAP.items()}

MODULE_PAGE_CONFIG = {
    'fatigue': {
        'page_title': 'Fatigue & Falling',
        'page_subtitle': 'Analyse caméras fatigue & chute — caméra principale et test',
        'description': 'Page spécifique Fatigue & Falling : surveillance des signes de somnolence et des chutes.',
        'cameras': [
            {
                'name': 'CAM 2 — Zone Machines',
                'role': 'Détection fatigue',
                'source': '/media/worker_tired.mp4',
                'status': 'Active',
                'details': 'Analyse des yeux fermés, PERCLOS et bouche ouverte.',
            },
            {
                'name': 'CAM 1 — Zone Production',
                'role': 'Détection chute',
                'source': '/media/worker_falling3.mp4',
                'status': 'Active',
                'details': 'Surveillance des chutes et des postures dangereuses.',
            },
            {
                'name': 'CAM 3 — Zone Test Chute',
                'role': 'Validation chute',
                'source': '/media/worker_falling3_fall.mp4',
                'status': 'Active',
                'details': 'Scénario de test de chute pour affiner le modèle.',
            },
        ],
    },
}
MODULE_PAGE_CONFIG['falling'] = MODULE_PAGE_CONFIG['fatigue']
MODULE_PAGE_CONFIG['fall'] = MODULE_PAGE_CONFIG['fatigue']

MODULE_PAGE_CONFIG.update({
    'ppe': {
        'page_title': 'EPI & Signalisation',
        'page_subtitle': 'Controle EPI et panneaux de securite - cameras dediees',
        'description': 'Surveillance des equipements de protection et de la signalisation de securite.',
        'cameras': [
            {
                'id': 4,
                'name': 'CAM 4 - PPE Detection',
                'role': 'Detection EPI',
                'source': '/media/ppe_video.mp4',
                'file_label': 'ppe_video.mp4',
                'status_id': 'cam4-ppe-status',
                'chips': [
                    {'label': 'HELMET', 'color': '#E040FB'},
                    {'label': 'VEST', 'color': '#4AE3B5'},
                    {'label': 'BOOTS', 'color': '#FF6B00'},
                ],
            },
            {
                'id': 7,
                'name': 'CAM 7 - Sign Defect Detection',
                'role': 'Controle signalisation',
                'source': '/media/construction_signs2.mp4',
                'file_label': 'construction_signs2.mp4',
                'status_id': 'cam7-sign-status',
                'chips': [
                    {'label': 'ROUTER', 'color': '#00C2FF'},
                    {'label': 'DEFECT', 'color': '#FF6B00'},
                ],
            },
        ],
    },
    'fatigue': {
        'page_title': 'Fatigue & Falling',
        'page_subtitle': 'Analyse cameras fatigue et chute - camera principale et test',
        'description': 'Surveillance des signes de somnolence et des chutes.',
        'cameras': [
            {
                'id': 1,
                'name': 'CAM 1 - Zone Production',
                'role': 'Detection chute',
                'source': '/media/worker_falling3.mp4',
                'file_label': 'worker_falling3.mp4',
                'status_id': 'cam1-fall-status',
                'chips': [{'label': 'CHUTE', 'color': '#FF3B2F'}],
            },
            {
                'id': 2,
                'name': 'CAM 2 - Zone Machines',
                'role': 'Detection fatigue',
                'source': '/media/worker_tired.mp4',
                'file_label': 'worker_tired.mp4',
                'status_id': 'cam2-fatigue-status',
                'chips': [{'label': 'FATIGUE', 'color': '#FFB800'}],
            },
        ],
    },
    'hazards': {
        'page_title': 'Risques Environnement',
        'page_subtitle': 'Analyse deversement, plaques ouvertes et sorties bloquees',
        'description': 'Surveillance des aleas environnementaux critiques: fuite, ouverture dangereuse et issue de secours bloquee.',
        'cameras': [
            {
                'id': 3,
                'name': 'CAM 3 - Spill Detection',
                'role': 'Detection deversement',
                'source': '/media/spill.mp4',
                'file_label': 'spill.mp4',
                'status_id': 'cam3-spill-status',
                'chips': [
                    {'label': 'SPILL', 'color': '#00C2FF'},
                    {'label': 'FUITE', 'color': '#4AE3B5'},
                ],
            },
            {
                'id': 5,
                'name': 'CAM 5 - Manhole Depth Test',
                'role': 'Detection plaque ouverte',
                'source': '/media/hole.mp4',
                'file_label': 'hole.mp4',
                'status_id': 'cam5-manhole-status',
                'chips': [
                    {'label': 'MANHOLE', 'color': '#FF6B00'},
                    {'label': 'DEPTH', 'color': '#00C2FF'},
                ],
            },
            {
                'id': 6,
                'name': 'CAM 6 - Exit Emergency Test',
                'role': 'Issue de secours',
                'source': '/media/exit_emergency.mp4',
                'file_label': 'exit_emergency.mp4',
                'status_id': 'cam6-exit-status',
                'secondary_status_id': 'cam6-exit-status-secondary',
                'chips': [
                    {'label': 'EXIT', 'color': '#FF3B2F'},
                    {'label': 'OBSTACLE', 'color': '#FFB800'},
                ],
            },
        ],
    },
})
MODULE_PAGE_CONFIG['falling'] = MODULE_PAGE_CONFIG['fatigue']
MODULE_PAGE_CONFIG['fall'] = MODULE_PAGE_CONFIG['fatigue']
MODULE_PAGE_CONFIG['spill'] = MODULE_PAGE_CONFIG['hazards']
MODULE_PAGE_CONFIG['manhole'] = MODULE_PAGE_CONFIG['hazards']
MODULE_PAGE_CONFIG['blocked-exit'] = MODULE_PAGE_CONFIG['hazards']

MODULE_PAGE_CONFIG['posture'] = {
    'page_title': 'Posture & Comportement',
    'page_subtitle': 'Analyse posture dangereuse et panique — squelette YOLOv8 (99.2% mAP50)',
    'description': (
        'Detection des postures dangereuses (safe/unsafe) sur CAM 9 '
        'et comportements de panique sur CAM 10 sans overlay squelette. '
        'Pipeline deux etapes : extraction des points cles COCO-17 + classification YOLOv8/BiLSTM.'
    ),
    'cameras': [
        {
            'id': 9,
            'name': 'CAM 9 - Posture Detection',
            'role': 'Detection posture (safe/unsafe)',
            'source': '/media/posture.mp4',
            'file_label': 'posture.mp4',
            'status_id': 'cam9-posture-status',
            'chips': [
                {'label': 'SAFE',      'color': '#4AE3B5'},
                {'label': 'UNSAFE',    'color': '#FF3B2F'},
                {'label': 'SQUELETTE', 'color': '#7B61FF'},
            ],
        },
        {
            'id': 10,
            'name': 'CAM 10 - Panic Detection',
            'role': 'Detection comportement panique',
            'source': '/media/panic.mp4',
            'file_label': 'panic.mp4',
            'status_id': 'cam10-panic-status',
            'chips': [
                {'label': 'PANIC', 'color': '#FF3B2F'},
                {'label': 'CALME', 'color': '#4AE3B5'},
                {'label': 'LSTM', 'color': '#00C2FF'},
            ],
        },
    ],
}
MODULE_PAGE_CONFIG['comportement'] = MODULE_PAGE_CONFIG['posture']


def _normalize_module_slug(slug):
    config = MODULE_SLUG_MAP.get(slug, {})
    return config.get('alias', slug)


def get_module_by_slug(slug):
    canonical_slug = _normalize_module_slug(slug)
    config = MODULE_SLUG_MAP.get(canonical_slug)

    if not config:
        return Module.objects.filter(name__icontains=slug.replace('-', ' ')).first()

    module = Module.objects.filter(id=config['id']).first()
    if module:
        return module

    query = Module.objects.all()
    for keyword in config.get('keywords', ()):
        module = query.filter(name__icontains=keyword).first()
        if module:
            return module

    return Module.objects.create(
        id=config['id'],
        name=config['fallback_name'],
        description=config['fallback_description'],
        icon=config.get('icon', 'module'),
        color=config.get('color', '#3498db'),
        status='active',
    )


# ── Vues génériques ────────────────────────────────────────────────────────────

class IndexView(TemplateView):
    template_name = 'index.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['app_name'] = 'Projet PI'
        return context


class SafetyDashboardView(TemplateView):
    template_name = 'safety_vision/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['app_name']        = 'SafeVision Dashboard'
        context['modules']         = Module.objects.all()
        context['alerts']          = Alert.objects.order_by('-timestamp')[:5]
        context['active_modules']  = Module.objects.filter(status='active').count()
        context['critical_alerts'] = Alert.objects.filter(severity='critical').count()
        return context


class LiveView(TemplateView):
    template_name = 'safety_vision/live.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['app_name'] = 'Live Monitoring'
        context['modules']  = Module.objects.all()
        return context


class AlertsView(TemplateView):
    template_name = 'safety_vision/alerts.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['app_name'] = 'Alertes SafeVision'
        context['alerts']   = Alert.objects.order_by('-timestamp')[:25]
        return context


class SettingsView(TemplateView):
    template_name = 'safety_vision/settings.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['app_name'] = 'Paramètres SafeVision'
        context['cameras']  = [
            {'name': 'Caméra principale', 'status': 'Active',  'location': 'Hall 1'},
            {'name': 'Caméra de chantier','status': 'Active',  'location': 'Zone A'},
        ]
        context['models'] = Module.objects.all()
        return context


class ReportsView(TemplateView):
    template_name = 'safety_vision/reports.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        modules = Module.objects.all()
        alerts = Alert.objects.order_by('-timestamp')[:10]
        context['app_name'] = 'Rapports SafeVision'
        context['modules'] = modules
        context['alerts'] = alerts
        context['total_modules'] = modules.count()
        context['total_alerts'] = Alert.objects.count()
        context['critical_alerts'] = Alert.objects.filter(severity='critical').count()
        context['total_detections'] = Detection.objects.count()
        context['latest_alerts'] = alerts
        return context


class ModuleDetailView(TemplateView):
    template_name = 'safety_vision/module_detail.html'

    def _format_detection_rows(self, detections):
        rows = []
        for det in detections:
            details = det.details or {}
            is_fatigue = 'fatigue_detected' in details
            is_fall = 'fall_detected' in details
            is_spill = 'spill_detected' in details
            is_manhole = 'manhole_detected' in details
            is_blocked_exit = 'blocked_exit_detected' in details
            is_ppe_violation = 'ppe_violation' in details
            is_sign_defect = 'sign_defect_detected' in details or 'is_defective' in details
            is_proximity = 'proximity_detected' in details
            is_posture = 'posture_unsafe' in details
            is_panic = 'panic_detected' in details
            is_worker_tracking = 'worker_tracking_detected' in details
            detected = bool(det.count)
            kind = (
                'Fatigue' if is_fatigue else
                'Chute' if is_fall else
                'Spill' if is_spill else
                'Issue Bloquee' if is_blocked_exit else
                'Manhole' if is_manhole else
                'PPE Violation' if is_ppe_violation else
                'Sign Defect' if is_sign_defect else
                'Proximite' if is_proximity else
                'Posture' if is_posture else
                'Panique' if is_panic else
                'Worker Tracking' if is_worker_tracking else
                details.get('class', 'Detection')
            )
            source = details.get('camera') or details.get('source') or (
                'CAM 2' if is_fatigue else
                'CAM 1' if is_fall else
                'CAM 3' if is_spill else
                'CAM 6' if is_blocked_exit else
                'CAM 5' if is_manhole else
                'CAM 4' if is_ppe_violation else
                'CAM 7' if is_sign_defect else
                'CAM 8' if is_proximity or is_worker_tracking else
                'CAM 9' if is_posture else
                'CAM 10' if is_panic else
                'Module'
            )
            confidence_pct = int(round((det.confidence or 0) * 100))
            rows.append({
                'id': det.id,
                'kind': kind,
                'source': source,
                'confidence_pct': max(0, min(100, confidence_pct)),
                'detected': detected,
                'time': det.timestamp,
            })
        return rows

    def get_context_data(self, **kwargs):
        slug   = kwargs.get('slug')
        module = get_module_by_slug(slug)
        if not module:
            raise Http404('Module non trouvé')

        context = super().get_context_data(**kwargs)
        canonical_slug = _normalize_module_slug(slug)
        config = MODULE_PAGE_CONFIG.get(canonical_slug, MODULE_PAGE_CONFIG.get(slug, {}))

        context['app_name']   = f'{module.name} — Détails'
        context['module']     = module
        context['module_id']  = module.id
        context['detections'] = []
        context['detection_rows'] = []
        context['alerts']     = []
        context['latest_detection_id'] = module.detections.order_by('-id').values_list('id', flat=True).first() or 0
        context['latest_alert_id'] = module.alerts.order_by('-id').values_list('id', flat=True).first() or 0
        context['module_slug'] = slug
        context['module_canonical_slug'] = canonical_slug
        context['page_title'] = config.get('page_title', module.name)
        context['page_subtitle'] = config.get('page_subtitle', module.description)
        context['module_description'] = config.get('description', module.description)
        context['module_cameras'] = config.get('cameras', [])
        return context


class UserLoginView(LoginView):
    template_name            = 'login.html'
    redirect_authenticated_user = True

    def get_success_url(self):
        return reverse_lazy('app:dashboard')


class UserLogoutView(LogoutView):
    next_page = reverse_lazy('app:login')


class SignUpView(FormView):
    template_name = 'signup.html'
    form_class    = SignUpForm
    success_url   = reverse_lazy('app:dashboard')

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        messages.success(self.request, 'Compte créé avec succès. Bienvenue !')
        return super().form_valid(form)


# ── Helpers partagés ──────────────────────────────────────────────────────────

def _decode_image(b64_string: str):
    """Décode une image base64 en frame OpenCV BGR."""
    img_bytes = base64.b64decode(b64_string)
    image     = Image.open(BytesIO(img_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _decode_pil_image(b64_string: str):
    img_bytes = base64.b64decode(b64_string)
    return Image.open(BytesIO(img_bytes)).convert('RGB')


def _get_module_by_name(*names):
    """Cherche un module par plusieurs noms possibles (ordre de priorité)."""
    for name in names:
        module = Module.objects.filter(name__icontains=name).first()
        if module:
            return module
    return None


def _get_module_by_name(*names):
    slug_matches = {
        'ppe': 'ppe',
        'epi': 'ppe',
        'posture': 'posture',
        'fatigue': 'fatigue',
        'fall': 'fatigue',
        'chute': 'fatigue',
        'tracking': 'tracking',
        'suivi': 'tracking',
        'hazards': 'hazards',
        'risques': 'hazards',
        'spill': 'hazards',
        'deversement': 'hazards',
        'chemical': 'hazards',
        'manhole': 'hazards',
        'hole': 'hazards',
        'exit': 'hazards',
        'emergency': 'hazards',
        'fire': 'fire',
        'feu': 'fire',
        'machinery': 'machinery',
        'machines': 'machinery',
        'proximity': 'machinery',
        'proximite': 'machinery',
        'homme-machine': 'machinery',
    }

    for name in names:
        slug = slug_matches.get(str(name).lower())
        if slug:
            module = get_module_by_slug(slug)
            if module:
                return module
        module = Module.objects.filter(name__icontains=name).first()
        if module:
            return module
    return None


def _detection_to_event(det):
    details = det.details or {}
    is_fatigue = 'fatigue_detected' in details
    is_fall = 'fall_detected' in details
    is_spill = 'spill_detected' in details
    is_manhole = 'manhole_detected' in details
    is_blocked_exit = 'blocked_exit_detected' in details
    is_ppe_violation = 'ppe_violation' in details
    is_sign_defect = 'sign_defect_detected' in details or 'is_defective' in details
    is_proximity = 'proximity_detected' in details
    is_posture = 'posture_unsafe' in details
    is_panic = 'panic_detected' in details
    is_worker_tracking = 'worker_tracking_detected' in details
    event_type = (
        'fatigue' if is_fatigue else
        'fall' if is_fall else
        'spill' if is_spill else
        'manhole' if is_manhole else
        'blocked_exit' if is_blocked_exit else
        'ppe_violation' if is_ppe_violation else
        'sign_defect' if is_sign_defect else
        'proximity' if is_proximity else
        'posture' if is_posture else
        'panic' if is_panic else
        'worker_tracking' if is_worker_tracking else
        'detection'
    )
    cam = details.get('camera') or details.get('source') or (
        'cam2' if is_fatigue else
        'cam1' if is_fall else
        'cam3' if is_spill else
        'cam5' if is_manhole else
        'cam6' if is_blocked_exit else
        'cam4' if is_ppe_violation else
        'cam7' if is_sign_defect else
        'cam8' if is_proximity or is_worker_tracking else
        'cam9' if is_posture else
        'cam10' if is_panic else
        'module'
    )
    return {
        'id': det.id,
        'type': event_type,
        'cam': cam,
        'detected': bool(det.count),
        'confidence': float(det.confidence or 0),
        'time': det.timestamp.strftime('%H:%M:%S'),
        'details': details,
    }


def _alert_to_event(alert):
    return {
        'id': alert.id,
        'severity': alert.severity,
        'module': alert.module.name,
        'message': alert.title,
        'time': alert.timestamp.strftime('%H:%M:%S'),
    }


# ── API endpoints ──────────────────────────────────────────────────────────────

def api_test(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            return JsonResponse({'status': 'success', 'message': 'Données reçues avec succès', 'data': data})
        except json.JSONDecodeError:
            return JsonResponse({'status': 'error', 'message': 'Erreur de décodage JSON'}, status=400)
    return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)


# ── /api/fall-detection/ ──────────────────────────────────────────────────────

def api_module_events(request, slug):
    module = get_module_by_slug(slug)
    if not module:
        return JsonResponse({'status': 'error', 'message': 'Module non trouve'}, status=404)

    try:
        after_detection_id = int(request.GET.get('after_detection_id', '0') or 0)
        after_alert_id = int(request.GET.get('after_alert_id', '0') or 0)
    except ValueError:
        after_detection_id = 0
        after_alert_id = 0

    detections = module.detections.filter(id__gt=after_detection_id).order_by('id')[:50]
    alerts = module.alerts.filter(id__gt=after_alert_id).order_by('id')[:50]

    return JsonResponse({
        'status': 'success',
        'detections': [_detection_to_event(det) for det in detections],
        'alerts': [_alert_to_event(alert) for alert in alerts],
    })


def api_clear_alerts(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Methode non autorisee'}, status=405)

    deleted_count, _ = Alert.objects.all().delete()
    return JsonResponse({'status': 'success', 'deleted_count': deleted_count})


def api_fall_detection(request):
    """API endpoint pour la détection de chute (image unique)."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)

    try:
        data = json.loads(request.body)

        if 'image' not in data:
            return JsonResponse(
                {'status': 'error', 'message': 'Image requise pour la détection de chute'}, status=400
            )

        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Erreur de décodage image : {e}'}, status=400)

        fall_detected, confidence, details = detect_fall_in_frame(frame)

        module = _get_module_by_name('fall', 'fatigue')
        if module:
            Detection.objects.create(
                module     = module,
                confidence = confidence,
                count      = 1 if fall_detected else 0,
                details    = {
                    'fall_detected': fall_detected,
                    'confidence':    confidence,
                    'camera':        data.get('camera', 'cam1'),
                    'bbox':          details.get('bbox'),
                    'detections':    details.get('detections', []),
                    'frame_shape':   list(map(int, frame.shape)),
                    'model_version': '1.0.0',
                },
            )

            if fall_detected:
                Alert.objects.create(
                    module      = module,
                    severity    = 'critical',
                    title       = 'Chute détectée !',
                    description = f'Chute détectée par le modèle avec une confiance de {confidence:.2%}',
                    details     = {
                        'confidence': confidence,
                        'location':   details.get('location', 'Zone principale'),
                        'camera':     data.get('camera', 'CAM 1'),
                        'event_type': 'fall',
                        'timestamp':  datetime.now().isoformat(),
                    },
                )

        return JsonResponse({
            'status':        'success',
            'fall_detected': bool(fall_detected),
            'confidence':    float(confidence),
            'details':       details,
            'message':       'Analyse de chute terminée',
        })

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de décodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur lors de la détection : {e}'}, status=500)


# ── /api/fall-detection/batch/ ────────────────────────────────────────────────

def api_fall_detection_batch(request):
    """API endpoint pour traiter plusieurs images (batch)."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)

    try:
        data = json.loads(request.body)

        if 'images' not in data or not isinstance(data['images'], list):
            return JsonResponse({'status': 'error', 'message': "Liste 'images' requise"}, status=400)

        results    = []
        total_falls = 0

        for i, image_b64 in enumerate(data['images']):
            try:
                frame = _decode_image(image_b64)
                fall_detected, confidence, details = detect_fall_in_frame(frame)
                results.append({
                    'frame_id':     i,
                    'fall_detected': bool(fall_detected),
                    'confidence':    float(confidence),
                    'details':       details,
                })
                if fall_detected:
                    total_falls += 1
            except Exception as e:
                results.append({'frame_id': i, 'error': str(e)})

        return JsonResponse({
            'status':               'success',
            'total_frames':         len(results),
            'total_falls_detected': total_falls,
            'results':              results,
            'message':              f'Analyse batch terminée : {total_falls} chute(s) détectée(s)',
        })

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de décodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur lors du traitement batch : {e}'}, status=500)


# ── /api/fatigue-detection/ ───────────────────────────────────────────────────

def api_fatigue_detection(request):
    """API endpoint pour la détection de fatigue (image unique — CAM 2)."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)

    try:
        data = json.loads(request.body)

        if 'image' not in data:
            return JsonResponse(
                {'status': 'error', 'message': 'Image requise pour la détection de fatigue'}, status=400
            )

        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Erreur de décodage image : {e}'}, status=400)

        fatigue_detected, confidence, details = detect_fatigue_in_frame(frame)

        # Persistance en base
        module = _get_module_by_name('fatigue')
        if module:
            Detection.objects.create(
                module     = module,
                confidence = confidence,
                count      = 1 if fatigue_detected else 0,
                details    = {
                    'fatigue_detected': fatigue_detected,
                    'confidence':       confidence,
                    'camera':           data.get('camera', 'cam2'),
                    'bbox':             details.get('bbox'),
                    'model_detections': details.get('model_detections', []),
                    'fatigue_level':    details.get('fatigue_level', 'faible'),
                    'frame_shape':      list(map(int, frame.shape)),
                    'model_version':    '1.0.0',
                    'model_available':  details.get('model_available', False),
                    'pytorch_available': details.get('pytorch_available', False),
                },
            )

            if fatigue_detected:
                level = details.get('fatigue_level', 'modérée')
                Alert.objects.create(
                    module      = module,
                    severity    = 'critical' if confidence > 0.8 else 'warning',
                    title       = f'Fatigue {level} détectée !',
                    description = (
                        f'Fatigue de niveau « {level} » détectée sur caméra 2 '
                        f'(confiance {confidence:.2%})'
                    ),
                    details     = {
                        'confidence':    confidence,
                        'fatigue_level': level,
                        'location':      'Zone Machines — CAM 2',
                        'timestamp':     datetime.now().isoformat(),
                    },
                )

        return JsonResponse({
            'status':            'success',
            'fatigue_detected':  bool(fatigue_detected),
            'confidence':        float(confidence),
            'details':           details,
            'model_available':   details.get('model_available', False),
            'pytorch_available': details.get('pytorch_available', False),
            'message':           'Analyse de fatigue terminée',
        })

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de décodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur lors de la détection fatigue : {e}'}, status=500)


# ── /api/fatigue-detection/batch/ ─────────────────────────────────────────────

def api_fatigue_detection_batch(request):
    """API endpoint pour traiter plusieurs images en batch (fatigue)."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)

    try:
        data = json.loads(request.body)

        if 'images' not in data or not isinstance(data['images'], list):
            return JsonResponse({'status': 'error', 'message': "Liste 'images' requise"}, status=400)

        results        = []
        total_fatigues = 0

        for i, image_b64 in enumerate(data['images']):
            try:
                frame = _decode_image(image_b64)
                fatigue_detected, confidence, details = detect_fatigue_in_frame(frame)
                results.append({
                    'frame_id':        i,
                    'fatigue_detected': bool(fatigue_detected),
                    'confidence':       float(confidence),
                    'details':          details,
                })
                if fatigue_detected:
                    total_fatigues += 1
            except Exception as e:
                results.append({'frame_id': i, 'error': str(e)})

        return JsonResponse({
            'status':                  'success',
            'total_frames':            len(results),
            'total_fatigues_detected': total_fatigues,
            'results':                 results,
            'message':                 f'Analyse batch terminée : {total_fatigues} fatigue(s) détectée(s)',
        })

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de décodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur batch fatigue : {e}'}, status=500)


# ── /api/proximity-detection/ ─────────────────────────────────────────────────

def api_ppe_detection(request):
    """API endpoint pour la detection EPI (CAM 4)."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Methode non autorisee'}, status=405)

    try:
        data = json.loads(request.body)
        if 'image' not in data:
            return JsonResponse({'status': 'error', 'message': 'Image requise pour la detection EPI'}, status=400)

        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Erreur de decodage image : {e}'}, status=400)

        camera = data.get('camera', 'cam4')
        ppe_violation, confidence, details = detect_ppe_in_frame(frame, camera=camera)
        module = get_module_by_slug('ppe')

        if module:
            Detection.objects.create(
                module=module,
                confidence=confidence,
                count=1 if ppe_violation else 0,
                details={
                    'ppe_violation': ppe_violation,
                    'confidence': confidence,
                    'camera': camera,
                    'detections': details.get('detections', []),
                    'has_human': details.get('has_human', False),
                    'has_helmet': details.get('has_helmet', False),
                    'has_vest': details.get('has_vest', False),
                    'has_boots': details.get('has_boots', False),
                    'frame_shape': list(map(int, frame.shape)),
                    'model_version': '1.0.0',
                    'model_available': details.get('model_available', False),
                },
            )

            if ppe_violation:
                Alert.objects.create(
                    module=module,
                    severity='warning',
                    title='Violation EPI detectee !',
                    description=f'Equipement de protection manquant sur {camera.upper()} (confiance {confidence:.2%})',
                    details={
                        'confidence': confidence,
                        'camera': camera,
                        'event_type': 'ppe',
                        'timestamp': datetime.now().isoformat(),
                    },
                )

        return JsonResponse({
            'status': 'success',
            'ppe_violation': bool(ppe_violation),
            'confidence': float(confidence),
            'details': details,
            'model_available': details.get('model_available', False),
            'message': 'Analyse EPI terminee',
        })
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de decodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur detection EPI : {e}'}, status=500)


def api_sign_detect(request):
    """API endpoint pour la detection de panneaux defectueux (CAM 7)."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Methode non autorisee'}, status=405)

    try:
        data = json.loads(request.body)
        if 'image' not in data:
            return JsonResponse({'status': 'error', 'message': 'Image requise pour la detection signalisation'}, status=400)

        try:
            pil_image = _decode_pil_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Erreur de decodage image : {e}'}, status=400)

        camera = data.get('camera', 'cam7')
        result = detect_sign_in_image(pil_image)
        is_defective = bool(result.get('is_defective', False))
        confidence = float(result.get('defect_score') or result.get('cat_confidence') or 0)
        module = get_module_by_slug('ppe')

        if module:
            Detection.objects.create(
                module=module,
                confidence=confidence,
                count=1 if is_defective else 0,
                details={
                    'sign_defect_detected': is_defective,
                    'confidence': confidence,
                    'camera': camera,
                    'category': result.get('category'),
                    'category_name': result.get('category_name'),
                    'verdict': result.get('verdict'),
                    'defect_score': result.get('defect_score'),
                    'threshold': result.get('threshold'),
                    'model_version': '1.0.0',
                    'model_available': True,
                },
            )

            if is_defective:
                Alert.objects.create(
                    module=module,
                    severity='warning',
                    title='Panneau defectueux detecte !',
                    description=f'Signalisation defectueuse sur {camera.upper()} (score {confidence:.2%})',
                    details={
                        'confidence': confidence,
                        'camera': camera,
                        'event_type': 'sign_defect',
                        'category': result.get('category'),
                        'verdict': result.get('verdict'),
                        'timestamp': datetime.now().isoformat(),
                    },
                )

        return JsonResponse({
            'status': 'success',
            'data': result,
            'message': 'Analyse signalisation terminee',
        })
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de decodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur detection signalisation : {e}'}, status=500)


@csrf_exempt
def api_proximity_detection(request):
    """API endpoint pour la détection de proximité homme-machine (image unique)."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)

    try:
        data = json.loads(request.body)

        if 'image' not in data:
            return JsonResponse(
                {'status': 'error', 'message': 'Image requise pour la détection de proximité'}, status=400
            )

        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Erreur de décodage image : {e}'}, status=400)

        proximity_detected, confidence, details = detect_proximity_in_frame(frame)

        module = _get_module_by_name('machinery', 'proximity', 'homme-machine')
        if module:
            Detection.objects.create(
                module     = module,
                confidence = confidence,
                count      = len(details.get('proximity_alerts', [])),
                details    = {
                    'proximity_detected': proximity_detected,
                    'confidence':         confidence,
                    'camera':             data.get('camera', 'cam1'),
                    'proximity_alerts':   details.get('proximity_alerts', []),
                    'detections':         details.get('detections', []),
                    'frame_shape':        list(map(int, frame.shape)),
                    'model_version':      '1.0.0',
                },
            )

            if proximity_detected:
                for alert in details.get('proximity_alerts', []):
                    Alert.objects.create(
                        module      = module,
                        severity    = alert.get('severity', 'warning'),
                        title       = f'Proximité dangereuse ! ({alert.get("distance", 0):.1f}m)',
                        description = f'Proximité détectée entre ouvrier et machine à {alert.get("distance", 0):.1f}m',
                        details     = {
                            'distance':   alert.get('distance', 0),
                            'severity':   alert.get('severity', 'warning'),
                            'worker_id':  alert.get('worker_id'),
                            'machine_id': alert.get('machine_id'),
                            'location':   details.get('location', 'Zone principale'),
                            'camera':     data.get('camera', 'CAM 1'),
                            'event_type': 'proximity',
                            'timestamp':  datetime.now().isoformat(),
                        },
                    )

        return JsonResponse({
            'status':             'success',
            'proximity_detected': bool(proximity_detected),
            'confidence':         float(confidence),
            'details':            details,
            'message':            'Analyse de proximité terminée',
        })

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de décodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur lors de la détection de proximité : {e}'}, status=500)


@csrf_exempt
def api_proximity_detection_batch(request):
    """API endpoint pour traiter plusieurs images en batch (proximité)."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)

    try:
        data = json.loads(request.body)

        if 'images' not in data or not isinstance(data['images'], list):
            return JsonResponse({'status': 'error', 'message': "Liste 'images' requise"}, status=400)

        results          = []
        total_proximities = 0

        for i, image_b64 in enumerate(data['images']):
            try:
                frame = _decode_image(image_b64)
                proximity_detected, confidence, details = detect_proximity_in_frame(frame)
                proximity_count = len(details.get('proximity_alerts', []))
                results.append({
                    'frame_id':          i,
                    'proximity_detected': bool(proximity_detected),
                    'confidence':         float(confidence),
                    'proximity_count':    proximity_count,
                    'details':            details,
                })
                if proximity_detected:
                    total_proximities += proximity_count
            except Exception as e:
                results.append({'frame_id': i, 'error': str(e)})

        return JsonResponse({
            'status':                    'success',
            'total_frames':              len(results),
            'total_proximities_detected': total_proximities,
            'results':                   results,
            'message':                   f'Analyse batch terminée : {total_proximities} proximité(s) détectée(s)',
        })

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de décodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur batch proximité : {e}'}, status=500)


def api_spill_detection(request):
    """API endpoint pour la detection de deversement chimique (segmentation)."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Methode non autorisee'}, status=405)

    try:
        data = json.loads(request.body)

        if 'image' not in data:
            return JsonResponse(
                {'status': 'error', 'message': 'Image requise pour la detection de deversement'}, status=400
            )

        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Erreur de decodage image : {e}'}, status=400)

        camera = data.get('camera', 'cam3')
        spill_detected, confidence, details = detect_spill_in_frame(frame, camera=camera)

        module = _get_module_by_name('hazards', 'risques', 'spill', 'chemical')
        if module:
            Detection.objects.create(
                module=module,
                confidence=confidence,
                count=1 if spill_detected else 0,
                details={
                    'spill_detected': spill_detected,
                    'confidence': confidence,
                    'camera': camera,
                    'bbox': details.get('bbox'),
                    'polygons': details.get('polygons', []),
                    'detections': details.get('detections', []),
                    'spill_area_ratio': details.get('spill_area_ratio', 0),
                    'frame_shape': list(map(int, frame.shape)),
                    'model_version': '1.0.0',
                    'model_available': details.get('model_available', False),
                    'temporal_hold': details.get('temporal_hold', False),
                    'missed_frames': details.get('missed_frames', 0),
                },
            )

            if spill_detected:
                Alert.objects.create(
                    module=module,
                    severity='critical',
                    title='Deversement chimique detecte !',
                    description=(
                        f'Deversement ou fuite detecte sur {camera.upper()} '
                        f'avec une confiance de {confidence:.2%}'
                    ),
                    details={
                        'confidence': confidence,
                        'camera': camera,
                        'event_type': 'spill',
                        'spill_area_ratio': details.get('spill_area_ratio', 0),
                        'timestamp': datetime.now().isoformat(),
                    },
                )

        return JsonResponse({
            'status': 'success',
            'spill_detected': bool(spill_detected),
            'confidence': float(confidence),
            'details': details,
            'model_available': details.get('model_available', False),
            'message': 'Analyse de deversement terminee',
        })

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de decodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur detection deversement : {e}'}, status=500)


def api_manhole_detection(request):
    """API endpoint pour la detection temps reel de plaques d'egout."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Methode non autorisee'}, status=405)

    try:
        data = json.loads(request.body)
        if 'image' not in data:
            return JsonResponse({'status': 'error', 'message': 'Image requise pour la detection de manhole'}, status=400)

        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Erreur de decodage image : {e}'}, status=400)

        camera = data.get('camera', 'cam5')
        manhole_detected, confidence, details = detect_manhole_in_frame(frame, camera=camera, include_depth=False)

        module = _get_module_by_name('hazards', 'risques', 'manhole', 'hole')
        if module:
            Detection.objects.create(
                module=module,
                confidence=confidence,
                count=1 if manhole_detected else 0,
                details={
                    'manhole_detected': manhole_detected,
                    'confidence': confidence,
                    'camera': camera,
                    'bbox': details.get('bbox'),
                    'polygons': details.get('polygons', []),
                    'detections': details.get('detections', []),
                    'manhole_state': details.get('manhole_state', 'unknown'),
                    'depth_available': details.get('depth_available', False),
                    'depth_source': details.get('depth_source', 'deferred_report'),
                    'depth_score': details.get('depth_score'),
                    'depth_level': details.get('depth_level', 'pending'),
                    'risk_level': details.get('risk_level', 'low'),
                    'frame_shape': list(map(int, frame.shape)),
                    'model_version': '1.0.0',
                    'model_available': details.get('model_available', False),
                    'temporal_hold': details.get('temporal_hold', False),
                    'missed_frames': details.get('missed_frames', 0),
                },
            )

            if manhole_detected and details.get('manhole_state') == 'open':
                Alert.objects.create(
                    module=module,
                    severity='critical' if details.get('risk_level') == 'critical' else 'warning',
                    title='Manhole ouvert detecte !',
                    description=(
                        f"Ouverture detectee sur {camera.upper()} "
                        f"(confiance {confidence:.2%})"
                    ),
                    details={
                        'confidence': confidence,
                        'camera': camera,
                        'event_type': 'manhole',
                        'depth_score': details.get('depth_score'),
                        'depth_level': details.get('depth_level', 'pending'),
                        'risk_level': details.get('risk_level', 'low'),
                        'depth_source': details.get('depth_source', 'deferred_report'),
                        'timestamp': datetime.now().isoformat(),
                    },
                )

        return JsonResponse({
            'status': 'success',
            'manhole_detected': bool(manhole_detected),
            'confidence': float(confidence),
            'details': details,
            'model_available': details.get('model_available', False),
            'message': 'Analyse manhole terminee',
        })

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de decodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur detection manhole : {e}'}, status=500)


def api_manhole_depth(request):
    """API endpoint pour estimer la profondeur d'un manhole a la demande."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Methode non autorisee'}, status=405)

    try:
        data = json.loads(request.body)
        if 'image' not in data:
            return JsonResponse({'status': 'error', 'message': 'Image requise pour l estimation de profondeur'}, status=400)

        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Erreur de decodage image : {e}'}, status=400)

        polygon = None
        polygons = data.get('polygons')
        if isinstance(polygons, list) and polygons:
            polygon = polygons[0]
        bbox = data.get('bbox') if isinstance(data.get('bbox'), list) else None
        state = data.get('manhole_state', 'unknown')

        if polygon is None and bbox is None:
            detected, confidence, details = detect_manhole_in_frame(frame, camera=data.get('camera', 'report-depth'), include_depth=False)
            if not detected:
                return JsonResponse({
                    'status': 'success',
                    'depth_available': False,
                    'message': 'Aucun manhole detecte pour estimer la profondeur',
                })
            polygon = (details.get('polygons') or [None])[0]
            bbox = details.get('bbox')
            state = details.get('manhole_state', state)

        depth_details = estimate_manhole_depth(frame, polygon=polygon, bbox=bbox, state=state)
        return JsonResponse({
            'status': 'success',
            'depth_available': depth_details.get('depth_available', False),
            'details': depth_details,
            'message': 'Estimation de profondeur terminee',
        })
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de decodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur estimation profondeur manhole : {e}'}, status=500)


def api_blocked_exit_detection(request):
    """API endpoint pour detecter une issue de secours bloquee."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Methode non autorisee'}, status=405)

    try:
        data = json.loads(request.body)
        if 'image' not in data:
            return JsonResponse({'status': 'error', 'message': 'Image requise pour la detection de sortie'}, status=400)

        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Erreur de decodage image : {e}'}, status=400)

        camera = data.get('camera', 'cam6')
        blocked_detected, confidence, details = detect_blocked_exit_in_frame(frame, camera=camera)

        module = _get_module_by_name('hazards', 'risques', 'exit', 'emergency')
        if module:
            Detection.objects.create(
                module=module,
                confidence=confidence,
                count=1 if blocked_detected else 0,
                details={
                    'blocked_exit_detected': blocked_detected,
                    'confidence': confidence,
                    'camera': camera,
                    'bbox': details.get('bbox'),
                    'exit_bbox': details.get('exit_bbox'),
                    'obstacle_bbox': details.get('obstacle_bbox'),
                    'distance_pixels': details.get('distance_pixels'),
                    'distance_ratio': details.get('distance_ratio'),
                    'threshold_pixels': details.get('threshold_pixels'),
                    'detections': details.get('detections', []),
                    'frame_shape': list(map(int, frame.shape)),
                    'model_version': '1.0.0',
                    'model_available': details.get('model_available', False),
                    'temporal_hold': details.get('temporal_hold', False),
                    'missed_frames': details.get('missed_frames', 0),
                },
            )

            if blocked_detected:
                Alert.objects.create(
                    module=module,
                    severity='critical',
                    title='Issue de secours bloquee !',
                    description=(
                        f'Obstacle trop proche d une issue de secours sur {camera.upper()} '
                        f'(confiance {confidence:.2%})'
                    ),
                    details={
                        'confidence': confidence,
                        'camera': camera,
                        'event_type': 'blocked_exit',
                        'distance_pixels': details.get('distance_pixels'),
                        'distance_ratio': details.get('distance_ratio'),
                        'threshold_pixels': details.get('threshold_pixels'),
                        'timestamp': datetime.now().isoformat(),
                    },
                )

        return JsonResponse({
            'status': 'success',
            'blocked_exit_detected': bool(blocked_detected),
            'confidence': float(confidence),
            'details': details,
            'model_available': details.get('model_available', False),
            'message': 'Analyse sortie de secours terminee',
        })
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de decodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur detection sortie de secours : {e}'}, status=500)


# ── /api/posture-detection/ — CAM 9 ──────────────────────────────────────────

def api_posture_detection(request):
    """Detection posture dangereuse (safe/unsafe) avec squelette COCO-17 — CAM 9."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Methode non autorisee'}, status=405)

    try:
        data = json.loads(request.body)
        if 'image' not in data:
            return JsonResponse({'status': 'error', 'message': 'Image requise pour la detection de posture'}, status=400)

        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Erreur de decodage image : {e}'}, status=400)

        camera = data.get('camera', 'cam9')
        unsafe_detected, confidence, details = detect_posture_in_frame(frame, camera=camera)

        module = get_module_by_slug('posture')
        if module:
            Detection.objects.create(
                module=module,
                confidence=confidence,
                count=1 if unsafe_detected else 0,
                details={
                    'posture_detected': unsafe_detected,
                    'posture': details.get('posture', 'safe'),
                    'confidence': confidence,
                    'camera': camera,
                    'bbox': details.get('bbox'),
                    'keypoints': details.get('keypoints', details.get('skeleton_keypoints', [])),
                    'reasons': details.get('reasons', []),
                    'persons_detected': details.get('persons_detected', 0),
                    'frame_shape': list(map(int, frame.shape)),
                    'model_version': '1.0.0',
                    'model_available': details.get('model_available', False),
                },
            )

            if unsafe_detected:
                reasons_str = ', '.join(details.get('reasons', [])[:2]) or 'posture incorrecte'
                Alert.objects.create(
                    module=module,
                    severity='warning',
                    title='Posture dangereuse detectee !',
                    description=f'Posture UNSAFE sur {camera.upper()} ({confidence:.2%}) — {reasons_str}',
                    details={
                        'confidence': confidence,
                        'camera': camera,
                        'event_type': 'posture',
                        'reasons': details.get('reasons', []),
                        'timestamp': datetime.now().isoformat(),
                    },
                )

        return JsonResponse({
            'status': 'success',
            'unsafe_posture_detected': bool(unsafe_detected),
            'posture': details.get('posture', 'safe'),
            'confidence': float(confidence),
            'details': details,
            'model_available': details.get('model_available', False),
            'message': 'Analyse posture terminee',
        })
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de decodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur detection posture : {e}'}, status=500)


# ── /api/worker-tracking-detection/ ─────────────────────────────────────────

def api_worker_tracking_detection(request):
    """API endpoint pour le tracking des travailleurs et comptage des franchissements."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)

    try:
        data = json.loads(request.body)
        if 'image' not in data:
            return JsonResponse({'status': 'error', 'message': 'Image requise pour le tracking'}, status=400)

        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Erreur de décodage image : {e}'}, status=400)

        camera = data.get('camera', 'cam8')
        reset_state = bool(data.get('reset', False))
        tracking_detected, confidence, details = detect_worker_tracking_in_frame(
            frame, camera=camera, reset=reset_state
        )

        module = _get_module_by_name('tracking', 'objets', 'worker')
        if module:
            Detection.objects.create(
                module=module,
                confidence=confidence,
                count=1 if tracking_detected else 0,
                details={
                    'worker_tracking_detected': tracking_detected,
                    'confidence': confidence,
                    'camera': camera,
                    'count_in': details.get('count_in', 0),
                    'count_out': details.get('count_out', 0),
                    'total_crossings': details.get('total_crossings', 0),
                    'crossed_ids': details.get('crossed_ids', []),
                    'crossed_count': details.get('crossed_count', 0),
                    'tracks': details.get('tracks', []),
                    'line_start': details.get('line_start'),
                    'line_end': details.get('line_end'),
                    'frame_shape': list(map(int, frame.shape)),
                    'model_version': '1.0.0',
                    'model_available': details.get('model_available', False),
                    'processing_method': details.get('processing_method', 'yolo_deepsort_tracking_line_crossing'),
                },
            )

            if tracking_detected:
                Alert.objects.create(
                    module=module,
                    severity='warning',
                    title='Franchissement détecté',
                    description=(
                        f"Franchissement de ligne détecté sur {camera.upper()} "
                        f"(IN={details.get('count_in', 0)} / OUT={details.get('count_out', 0)})"
                    ),
                    details={
                        'confidence': confidence,
                        'camera': camera,
                        'event_type': 'worker_tracking',
                        'count_in': details.get('count_in', 0),
                        'count_out': details.get('count_out', 0),
                        'total_crossings': details.get('total_crossings', 0),
                        'crossed_ids': details.get('crossed_ids', []),
                        'timestamp': datetime.now().isoformat(),
                    },
                )

        return JsonResponse({
            'status': 'success',
            'worker_tracking_detected': bool(tracking_detected),
            'confidence': float(confidence),
            'details': details,
            'message': 'Analyse tracking workers terminée',
        })

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de décodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur tracking workers : {e}'}, status=500)


# ── /api/panic-detection/ — CAM 10 ───────────────────────────────────────────

def api_panic_detection(request):
    """Detection comportement de panique avec extraction de points cles COCO-17 — CAM 10."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Methode non autorisee'}, status=405)

    try:
        data = json.loads(request.body)
        if 'image' not in data:
            return JsonResponse({'status': 'error', 'message': 'Image requise pour la detection de panique'}, status=400)

        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Erreur de decodage image : {e}'}, status=400)

        camera = data.get('camera', 'cam10')
        panic_detected, confidence, details = detect_panic_in_frame(frame, camera=camera)

        module = get_module_by_slug('posture')
        if module:
            Detection.objects.create(
                module=module,
                confidence=confidence,
                count=1 if panic_detected else 0,
                details={
                    'panic_detected': panic_detected,
                    'label': details.get('label', 'NORMAL'),
                    'confidence': confidence,
                    'camera': camera,
                    'p_panic': details.get('p_panic', 0.0),
                    'p_normal': details.get('p_normal', 1.0),
                    'frames_collected': details.get('frames_collected', 0),
                    'persons_detected': details.get('persons_detected', 0),
                    'frame_shape': list(map(int, frame.shape)),
                    'model_version': '1.0.0',
                    'model_available': details.get('model_available', False),
                },
            )

            if panic_detected:
                Alert.objects.create(
                    module=module,
                    severity='critical',
                    title='Comportement de panique detecte !',
                    description=f'Panique detectee sur {camera.upper()} (confiance {confidence:.2%})',
                    details={
                        'confidence': confidence,
                        'camera': camera,
                        'p_panic': details.get('p_panic', 0.0),
                        'event_type': 'panic',
                        'timestamp': datetime.now().isoformat(),
                    },
                )

        return JsonResponse({
            'status': 'success',
            'panic_detected': bool(panic_detected),
            'label': details.get('label', 'NORMAL'),
            'confidence': float(confidence),
            'details': details,
            'model_available': details.get('model_available', False),
            'message': 'Analyse panique terminee',
        })
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de decodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur detection panique : {e}'}, status=500)
