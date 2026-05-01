from django.shortcuts import render
from django.views.generic import TemplateView, FormView
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth import login
from django.contrib import messages
from django.urls import reverse_lazy
from django.http import JsonResponse, Http404
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

MODULE_SLUG_MAP = {
    'ppe':        'PPE',
    'posture':    'Posture',
    'fatigue':    'Fatigue',
    'incapacity': 'Fatigue',
    'falling':    'Fatigue',
    'fall':       'Fatigue',
    'tracking':   'Tracking',
    'hazards':    'Hazards',
    'fire':       'Fire',
    'machinery':  'Machinery',
}

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


def get_module_by_slug(slug):
    name = MODULE_SLUG_MAP.get(slug, slug.replace('-', ' ').title())
    return Module.objects.filter(name__icontains=name).first()


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
        context['app_name'] = 'Rapports SafeVision'
        context['modules']  = Module.objects.all()
        context['alerts']   = Alert.objects.order_by('-timestamp')[:10]
        return context


class ModuleDetailView(TemplateView):
    template_name = 'safety_vision/module_detail.html'

    def _format_detection_rows(self, detections):
        rows = []
        for det in detections:
            details = det.details or {}
            is_fatigue = 'fatigue_detected' in details
            is_fall = 'fall_detected' in details
            detected = bool(det.count)
            kind = 'Fatigue' if is_fatigue else 'Chute' if is_fall else details.get('class', 'Detection')
            source = details.get('camera') or details.get('source') or ('CAM 2' if is_fatigue else 'CAM 1' if is_fall else 'Module')
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
        config = MODULE_PAGE_CONFIG.get(slug, {})

        context['app_name']   = f'{module.name} — Détails'
        context['module']     = module
        context['module_id']  = module.id
        context['detections'] = []
        context['detection_rows'] = []
        context['alerts']     = []
        context['latest_detection_id'] = module.detections.order_by('-id').values_list('id', flat=True).first() or 0
        context['latest_alert_id'] = module.alerts.order_by('-id').values_list('id', flat=True).first() or 0
        context['module_slug'] = slug
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


def _get_module_by_name(*names):
    """Cherche un module par plusieurs noms possibles (ordre de priorité)."""
    for name in names:
        module = Module.objects.filter(name__icontains=name).first()
        if module:
            return module
    return None


def _detection_to_event(det):
    details = det.details or {}
    is_fatigue = 'fatigue_detected' in details
    is_fall = 'fall_detected' in details
    event_type = 'fatigue' if is_fatigue else 'fall' if is_fall else 'detection'
    cam = details.get('camera') or details.get('source') or ('cam2' if is_fatigue else 'cam1' if is_fall else 'module')
    return {
        'id': det.id,
        'type': event_type,
        'cam': cam,
        'detected': bool(det.count),
        'confidence': float(det.confidence or 0),
        'time': det.timestamp.strftime('%H:%M:%S'),
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
