from django.shortcuts import render
from django.views.generic import TemplateView, FormView
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth import login
from django.contrib import messages
from django.urls import reverse_lazy
from django.http import JsonResponse, Http404
import json
from datetime import datetime, timedelta
import random
from .models import Module, Alert, Detection
from .forms import SignUpForm
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from .fall_detector import detect_fall_in_frame

MODULE_SLUG_MAP = {
    'ppe': 'PPE',
    'posture': 'Posture',
    'fatigue': 'Fatigue',
    'tracking': 'Tracking',
    'hazards': 'Hazards',
    'fire': 'Fire',
    'machinery': 'Machinery',
}


def get_module_by_slug(slug):
    name = MODULE_SLUG_MAP.get(slug, slug.replace('-', ' ').title())
    return Module.objects.filter(name__icontains=name).first()


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
        context['app_name'] = 'SafeVision Dashboard'
        context['modules'] = Module.objects.all()
        context['alerts'] = Alert.objects.order_by('-timestamp')[:5]
        context['active_modules'] = Module.objects.filter(status='active').count()
        context['critical_alerts'] = Alert.objects.filter(severity='critical').count()
        return context


class LiveView(TemplateView):
    template_name = 'safety_vision/live.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['app_name'] = 'Live Monitoring'
        context['modules'] = Module.objects.all()
        return context


class AlertsView(TemplateView):
    template_name = 'safety_vision/alerts.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['app_name'] = 'Alertes SafeVision'
        context['alerts'] = Alert.objects.order_by('-timestamp')[:25]
        return context


class SettingsView(TemplateView):
    template_name = 'safety_vision/settings.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['app_name'] = 'Paramètres SafeVision'
        context['cameras'] = [
            {'name': 'Caméra principale', 'status': 'Active', 'location': 'Hall 1'},
            {'name': 'Caméra de chantier', 'status': 'Active', 'location': 'Zone A'},
        ]
        context['models'] = Module.objects.all()
        return context


class ReportsView(TemplateView):
    template_name = 'safety_vision/reports.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['app_name'] = 'Rapports SafeVision'
        context['modules'] = Module.objects.all()
        context['alerts'] = Alert.objects.order_by('-timestamp')[:10]
        return context


class ModuleDetailView(TemplateView):
    template_name = 'safety_vision/module_detail.html'

    def get_context_data(self, **kwargs):
        slug = kwargs.get('slug')
        module = get_module_by_slug(slug)
        if not module:
            raise Http404('Module non trouvé')

        context = super().get_context_data(**kwargs)
        context['app_name'] = f'{module.name} — Détails'
        context['module'] = module
        context['module_id'] = module.id
        context['detections'] = module.detections.order_by('-timestamp')[:20]
        context['alerts'] = module.alerts.order_by('-timestamp')[:10]
        return context


class UserLoginView(LoginView):
    template_name = 'login.html'
    redirect_authenticated_user = True

    def get_success_url(self):
        return reverse_lazy('app:dashboard')


class UserLogoutView(LogoutView):
    next_page = reverse_lazy('app:login')


class SignUpView(FormView):
    template_name = 'signup.html'
    form_class = SignUpForm
    success_url = reverse_lazy('app:dashboard')

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        messages.success(self.request, 'Compte créé avec succès. Bienvenue !')
        return super().form_valid(form)


def api_test(request):
    """API test endpoint"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            return JsonResponse({
                'status': 'success',
                'message': 'Données reçues avec succès',
                'data': data
            })
        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error',
                'message': 'Erreur de décodage JSON'
            }, status=400)

    return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)


def api_fall_detection(request):
    """API endpoint pour la détection de chute"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Vérifier si on reçoit une image en base64
            if 'image' not in data:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Image requise pour la détection de chute'
                }, status=400)

            # Décoder l'image base64
            try:
                image_data = base64.b64decode(data['image'])
                image = Image.open(BytesIO(image_data))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                return JsonResponse({
                    'status': 'error',
                    'message': f'Erreur de décodage d\'image: {str(e)}'
                }, status=400)

            # Détecter les chutes avec le modèle avancé
            fall_detected, confidence, details = detect_fall_in_frame(frame)

            # Créer une détection dans la base de données
            module = Module.objects.filter(name__icontains='fatigue').first()
            if module:
                Detection.objects.create(
                    module=module,
                    confidence=confidence,
                    count=1 if fall_detected else 0,
                    details={
                        'fall_detected': fall_detected,
                        'confidence': confidence,
                        'frame_shape': frame.shape,
                        'processing_time': random.uniform(100, 300),
                        'model_version': '1.0.0'
                    }
                )

            # Créer une alerte si chute détectée
            if fall_detected and confidence > 0.8:
                Alert.objects.create(
                    module=module,
                    severity='critical',
                    title='Chute détectée!',
                    description=f'Une chute a été détectée avec une confiance de {confidence:.2%}',
                    details={
                        'confidence': confidence,
                        'location': details.get('location', 'Zone principale'),
                        'timestamp': datetime.now().isoformat()
                    }
                )

            return JsonResponse({
                'status': 'success',
                'fall_detected': fall_detected,
                'confidence': confidence,
                'details': details,
                'message': 'Analyse de chute terminée'
            })

        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error',
                'message': 'Erreur de décodage JSON'
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Erreur lors de la détection: {str(e)}'
            }, status=500)

    return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)


def simulate_fall_detection(frame):
    """
    Simulation de la détection de chute
    Remplacer par un vrai modèle de ML (YOLO, MediaPipe, etc.)
    """
    # Simulation basée sur des critères simples
    height, width = frame.shape[:2]

    # Analyse basique de la couleur et du mouvement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)

    # Simulation de détection aléatoire avec biais
    # En production, remplacer par un vrai modèle
    confidence = random.uniform(0.1, 0.95)

    # Seuil de détection (ajuster selon le modèle réel)
    fall_detected = confidence > 0.7

    details = {
        'frame_analysis': {
            'brightness': mean_brightness,
            'dimensions': [width, height],
            'processing_method': 'simulation'
        },
        'detection_params': {
            'model': 'FallDetection_v1.0',
            'threshold': 0.7,
            'algorithm': 'Pose_Estimation_Simulation'
        },
        'location': 'Zone de surveillance principale'
    }

    return fall_detected, confidence, details


def api_fall_detection_batch(request):
    """API endpoint pour traiter plusieurs images de détection de chute"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            if 'images' not in data or not isinstance(data['images'], list):
                return JsonResponse({
                    'status': 'error',
                    'message': 'Liste d\'images requise'
                }, status=400)

            results = []
            total_falls = 0

            for i, image_data in enumerate(data['images']):
                try:
                    # Décoder l'image
                    img_bytes = base64.b64decode(image_data)
                    image = Image.open(BytesIO(img_bytes))
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    # Détecter les chutes
                    fall_detected, confidence, details = detect_fall_in_frame(frame)

                    results.append({
                        'frame_id': i,
                        'fall_detected': fall_detected,
                        'confidence': confidence,
                        'details': details
                    })

                    if fall_detected:
                        total_falls += 1

                except Exception as e:
                    results.append({
                        'frame_id': i,
                        'error': str(e)
                    })

            return JsonResponse({
                'status': 'success',
                'total_frames': len(results),
                'total_falls_detected': total_falls,
                'results': results,
                'message': f'Analyse batch terminée: {total_falls} chutes détectées'
            })

        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error',
                'message': 'Erreur de décodage JSON'
            }, status=400)
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'Erreur lors du traitement batch: {str(e)}'
            }, status=500)

    return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)
