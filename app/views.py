from django.shortcuts import render
from django.views.generic import TemplateView, FormView
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth import login
from django.contrib import messages
from django.urls import reverse_lazy
from django.http import JsonResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg, Count, Max, Sum
from django.conf import settings
from django.utils import timezone
import json
from datetime import datetime, timedelta
import random
import os
from urllib.parse import urljoin
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
from .fire_smoke_detector import detect_fire_smoke_in_frame
from decouple import config
import requests
from django.views.decorators.csrf import csrf_exempt

SMS_EVENT_CACHE = {}

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

MODULE_SLUG_TO_ID = {
    'ppe': 1,
    'posture': 2,
    'fatigue': 3,
    'incapacity': 3,
    'falling': 3,
    'fall': 3,
    'tracking': 4,
    'hazards': 5,
    'spill': 5,
    'manhole': 5,
    'blocked-exit': 5,
    'fire': 6,
    'machinery': 7,
    'proximity': 7,
}

MODULE_SLUG_ALIASES = {
    'incapacity': 'fatigue',
    'falling': 'fatigue',
    'fall': 'fatigue',
    'spill': 'hazards',
    'manhole': 'hazards',
    'blocked-exit': 'hazards',
    'proximity': 'machinery',
}

MODULE_FALLBACKS = {
    'ppe': {
        'id': 1,
        'keywords': ('epi', 'ppe', 'signalisation', 'conformite'),
        'name': 'Conformite EPI + Signalisation',
        'description': 'Detection casques, gilets, signalisation manquante',
        'icon': 'shield',
        'color': '#4AE3B5',
    },
    'posture': {
        'id': 2,
        'keywords': ('posture', 'comportement'),
        'name': 'Posture Dangereuse + Comportement',
        'description': 'Analyse squelette, detection postures anormales',
        'icon': 'activity',
        'color': '#7B61FF',
    },
    'fatigue': {
        'id': 3,
        'keywords': ('fatigue', 'effondrement', 'chute', 'incapacite'),
        'name': 'Fatigue + Effondrement Travailleur',
        'description': 'Detection fatigue, chute et immobilite',
        'icon': 'eye',
        'color': '#FFB800',
    },
    'tracking': {
        'id': 4,
        'keywords': ('tracking', 'suivi', 'presence', 'objets'),
        'name': 'Suivi Presence + Objets Tombants',
        'description': 'Comptage zone, suivi trajectoire objets',
        'icon': 'target',
        'color': '#00C2FF',
    },
    'hazards': {
        'id': 5,
        'keywords': ('hazards', 'risques', 'aleas', 'environnement', 'deversement', 'sorties'),
        'name': 'Aleas Environnementaux',
        'description': 'Detection deversements, encombrement, sorties bloquees',
        'icon': 'alert-triangle',
        'color': '#FF6B00',
    },
    'fire': {
        'id': 6,
        'keywords': ('feu', 'fumee', 'fire', 'lumiere'),
        'name': 'Feu/Fumee + Changement Lumiere',
        'description': 'Detection incendie, fumee, flash/explosion',
        'icon': 'flame',
        'color': '#FF3B2F',
    },
    'machinery': {
        'id': 7,
        'keywords': ('machines', 'machinery', 'proximite', 'anomalies'),
        'name': 'Anomalies Machines + Proximite',
        'description': 'Etincelles, fumee, proximite travailleur-machine',
        'icon': 'settings',
        'color': '#E040FB',
    },
}

MODULE_PAGE_CONFIG.update({
    'fatigue': {
        'page_title': 'Fatigue & Falling',
        'page_subtitle': 'Analyse cameras fatigue et chute - camera principale et test',
        'description': 'Surveillance des signes de somnolence et des chutes.',
        'cameras': [
            {
                'id': 1,
                'name': 'CAM 1 - Zone Production',
                'source': '/media/worker_falling3.mp4',
                'file_label': 'worker_falling3.mp4',
                'status_id': 'cam1-fall-status',
                'chips': [{'label': 'CHUTE', 'color': '#FF3B2F', 'rgb': '255,59,47'}],
            },
            {
                'id': 2,
                'name': 'CAM 2 - Zone Machines',
                'source': '/media/worker_tired.mp4',
                'file_label': 'worker_tired.mp4',
                'status_id': 'cam2-fatigue-status',
                'chips': [{'label': 'FATIGUE', 'color': '#FFB800', 'rgb': '255,184,0'}],
            },
        ],
    },
    'ppe': {
        'page_title': 'EPI & Signalisation',
        'page_subtitle': 'Controle EPI et panneaux de securite - cameras dediees',
        'description': 'Surveillance des equipements de protection et de la signalisation de securite.',
        'cameras': [
            {
                'id': 4,
                'name': 'CAM 4 - PPE Detection',
                'source': '/media/ppe_video.mp4',
                'file_label': 'ppe_video.mp4',
                'status_id': 'cam4-ppe-status',
                'chips': [
                    {'label': 'HELMET', 'color': '#E040FB', 'rgb': '224,64,251'},
                    {'label': 'VEST', 'color': '#4AE3B5', 'rgb': '74,227,181'},
                    {'label': 'BOOTS', 'color': '#FF6B00', 'rgb': '255,107,0'},
                ],
            },
            {
                'id': 7,
                'name': 'CAM 7 - Sign Defect Detection',
                'source': '/media/construction_signs2.mp4',
                'file_label': 'construction_signs2.mp4',
                'status_id': 'cam7-sign-status',
                'chips': [
                    {'label': 'ROUTER', 'color': '#00C2FF', 'rgb': '0,194,255'},
                    {'label': 'DEFECT', 'color': '#FF6B00', 'rgb': '255,107,0'},
                ],
            },
        ],
    },
    'hazards': {
        'page_title': 'Risques Environnement',
        'page_subtitle': 'Analyse deversement, plaques ouvertes et sorties bloquees',
        'description': 'Surveillance des aleas environnementaux critiques.',
        'cameras': [
            {
                'id': 3,
                'name': 'CAM 3 - Spill Detection',
                'source': '/media/spill.mp4',
                'file_label': 'spill.mp4',
                'status_id': 'cam3-spill-status',
                'chips': [{'label': 'SPILL', 'color': '#00C2FF', 'rgb': '0,194,255'}],
            },
            {
                'id': 5,
                'name': 'CAM 5 - Manhole Depth Test',
                'source': '/media/hole.mp4',
                'file_label': 'hole.mp4',
                'status_id': 'cam5-manhole-status',
                'chips': [
                    {'label': 'MANHOLE', 'color': '#FF6B00', 'rgb': '255,107,0'},
                    {'label': 'DEPTH', 'color': '#00C2FF', 'rgb': '0,194,255'},
                ],
            },
            {
                'id': 6,
                'name': 'CAM 6 - Exit Emergency Test',
                'source': '/media/exit_emergency.mp4',
                'file_label': 'exit_emergency.mp4',
                'status_id': 'cam6-exit-status',
                'secondary_status_id': 'cam6-exit-status-secondary',
                'chips': [
                    {'label': 'EXIT', 'color': '#FF3B2F', 'rgb': '255,59,47'},
                    {'label': 'OBSTACLE', 'color': '#FFB800', 'rgb': '255,184,0'},
                ],
            },
        ],
    },
    'tracking': {
        'page_title': 'Tracking & Objets',
        'page_subtitle': 'Suivi presence, trajectoires et objets tombants',
        'description': 'Module de suivi, de comptage et de visualisation des objets en mouvement.',
        'cameras': [
            {
                'id': 8,
                'name': 'CAM 8 - Worker Tracking',
                'source': '/media/tracking_workers.mp4',
                'file_label': 'tracking_workers.mp4',
                'status_id': 'cam8-worker-status',
                'chips': [
                    {'label': 'TRACKING', 'color': '#00C2FF', 'rgb': '0,194,255'},
                    {'label': 'WORKERS', 'color': '#4AE3B5', 'rgb': '74,227,181'},
                    {'label': 'COUNTING', 'color': '#FF6B00', 'rgb': '255,107,0'},
                ],
            },
            {
                'id': 13,
                'name': 'CAM 13 - Debris Tracking',
                'source': '/media/debris_tracking.mp4',
                'file_label': 'debris_tracking.mp4',
                'status_id': 'cam13-debris-status',
                'status_text': 'Flux video uniquement',
                'chips': [
                    {'label': 'TRACKING', 'color': '#00C2FF', 'rgb': '0,194,255'},
                    {'label': 'DEBRIS', 'color': '#FFB800', 'rgb': '255,184,0'},
                    {'label': 'VIDEO', 'color': '#4AE3B5', 'rgb': '74,227,181'},
                ],
            },
        ],
    },
    'fire': {
        'page_title': 'Feu & Fumee',
        'page_subtitle': 'Detection incendie, fumee et changement lumiere',
        'description': 'Module incendie et changements lumineux. Flux configurable.',
        'cameras': [
            {
                'id': 12,
                'name': 'CAM 12 - Detection Feu & Fumee',
                'source': '/media/fire.mp4',
                'file_label': 'fire.mp4',
                'status_id': 'cam12-fire-status',
                'chips': [
                    {'label': 'FIRE', 'color': '#FF3B2F', 'rgb': '255,59,47'},
                    {'label': 'SMOKE', 'color': '#FFB800', 'rgb': '255,184,0'},
                ],
            },
        ],
    },
    'machinery': {
        'page_title': 'Machines & Proximite',
        'page_subtitle': 'Detection de proximite homme-machine',
        'description': 'Surveillance des zones dangereuses autour des machines.',
        'cameras': [
            {
                'id': 11,
                'name': 'CAM 11 - Proximite Homme-Machine',
                'source': '/media/Media_Proximity/vid3.mp4',
                'file_label': 'vid3.mp4',
                'status_id': 'cam11-proximity-status',
                'chips': [
                    {'label': 'WORKER', 'color': '#4AE3B5', 'rgb': '74,227,181'},
                    {'label': 'MACHINE', 'color': '#FF6B00', 'rgb': '255,107,0'},
                    {'label': 'PROXIMITY', 'color': '#00C2FF', 'rgb': '0,194,255'},
                ],
            },
        ],
    },
})
MODULE_PAGE_CONFIG['spill'] = MODULE_PAGE_CONFIG['hazards']
MODULE_PAGE_CONFIG['manhole'] = MODULE_PAGE_CONFIG['hazards']
MODULE_PAGE_CONFIG['blocked-exit'] = MODULE_PAGE_CONFIG['hazards']
MODULE_PAGE_CONFIG['proximity'] = MODULE_PAGE_CONFIG['machinery']
MODULE_PAGE_CONFIG['falling'] = MODULE_PAGE_CONFIG['fatigue']
MODULE_PAGE_CONFIG['fall'] = MODULE_PAGE_CONFIG['fatigue']


def normalize_module_slug(slug):
    return MODULE_SLUG_ALIASES.get(slug, slug)

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
    return normalize_module_slug(slug)


def get_module_by_slug(slug):
    canonical_slug = normalize_module_slug(slug)
    module_id = MODULE_SLUG_TO_ID.get(slug) or MODULE_SLUG_TO_ID.get(canonical_slug)
    fallback = MODULE_FALLBACKS.get(canonical_slug)
    if module_id:
        module = Module.objects.filter(id=module_id).first()
        if module:
            return module

    if fallback:
        for keyword in fallback.get('keywords', ()):
            module = Module.objects.filter(name__icontains=keyword).first()
            if module:
                return module
        return Module.objects.create(
            id=fallback['id'],
            name=fallback['name'],
            description=fallback['description'],
            icon=fallback.get('icon', 'module'),
            color=fallback.get('color', '#3498db'),
            status='active',
        )

    name = MODULE_SLUG_MAP.get(canonical_slug, slug.replace('-', ' ').title())
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
        context['modules']  = Module.objects.all()
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

    MODEL_BLUEPRINTS = {
        1: {
            'slug': 'ppe',
            'short_name': 'EPI & Signalisation',
            'axis': 'Axes 1 + 13',
            'stack': 'YOLO PPE + ResNet50 signalisation',
            'goal': 'Verifier casque, gilet, bottes, presence humaine et qualite des panneaux de securite.',
            'cameras': ['CAM 4 - ppe_video.mp4', 'CAM 7 - construction_signs2.mp4'],
            'model_files': ['ppe.pt', 'resnet50_router.pt', 'resnet50_E.pt', 'resnet50_F.pt', 'resnet50_M.pt', 'resnet50_P.pt', 'resnet50_W.pt'],
            'pipeline': ['Capture video', 'Detection EPI par bbox', 'Classification signalisation', 'Alerte violation'],
            'strengths': ['Detection multi-classes', 'Controle EPI + panneaux', 'Bounding boxes interpretables'],
            'impact': 'Reduction des oublis EPI et verification visuelle des zones a risque.',
            'accent': '#4AE3B5',
        },
        2: {
            'slug': 'posture',
            'short_name': 'Posture & Comportement',
            'axis': 'Axe 2 + 12',
            'stack': 'Analyse posture / squelette',
            'goal': 'Identifier les postures dangereuses et les comportements anormaux.',
            'cameras': ['Camera posture - source configurable'],
            'model_files': [],
            'pipeline': ['Extraction frame', 'Analyse posture', 'Score risque ergonomique', 'Rapport comportement'],
            'strengths': ['Vision ergonomique', 'Interpretation metier', 'Extensible a MediaPipe'],
            'impact': 'Prevention des accidents musculaires et des gestes dangereux.',
            'accent': '#7B61FF',
        },
        3: {
            'slug': 'fatigue',
            'short_name': 'Fatigue & Chute',
            'axis': 'Axes 3 + 4',
            'stack': 'YOLO fatigue + YOLO fall',
            'goal': 'Detecter somnolence, yeux fermes, baillement, chute et immobilite.',
            'cameras': ['CAM 1 - worker_falling3.mp4', 'CAM 2 - worker_tired.mp4'],
            'model_files': ['fatigue_detection.pt', 'fall_detection.pt'],
            'pipeline': ['Frame video', 'YOLO closed/open eye', 'YOLO down/up/bending', 'Confirmation temporelle', 'Alerte critique'],
            'strengths': ['Detection temps reel', 'Filtrage faux positifs', 'Notifications + tableau dynamique'],
            'impact': 'Reaction rapide aux chutes et signes de fatigue conducteur/travailleur.',
            'accent': '#FFB800',
        },
        4: {
            'slug': 'tracking',
            'short_name': 'Tracking & Objets',
            'axis': 'Axes 6 + 11',
            'stack': 'Tracking multi-objets',
            'goal': 'Suivre presence, objets tombants, mouvements et occupation des zones.',
            'cameras': ['Camera zone production - source configurable'],
            'model_files': [],
            'pipeline': ['Detection personne/objet', 'Association ID', 'Trajectoires', 'Comptage zone'],
            'strengths': ['Historique trajectoire', 'Vision supervision', 'Pret pour DeepSORT/ByteTrack'],
            'impact': 'Meilleure lecture des flux travailleurs et objets dangereux.',
            'accent': '#00C2FF',
        },
        5: {
            'slug': 'hazards',
            'short_name': 'Risques Environnement',
            'axis': 'Axe 5',
            'stack': 'YOLOv8 segmentation + detection sortie',
            'goal': 'Detecter deversement, plaque ouverte et issue de secours bloquee.',
            'cameras': ['CAM 3 - spill.mp4', 'CAM 5 - hole.mp4', 'CAM 6 - exit_emergency.mp4'],
            'model_files': ['spill_detection_model.pt', 'manhole_seg.pt', 'exit_emergency.pth'],
            'pipeline': ['Segmentation fuite', 'Segmentation manhole', 'Detection sortie/obstacle', 'Alerte critique'],
            'strengths': ['Masques/polygones', 'Analyse risque environnement', 'Multi-camera dediee'],
            'impact': 'Detection proactive des dangers qui bloquent ou contaminent la zone.',
            'accent': '#FF6B00',
        },
        6: {
            'slug': 'fire',
            'short_name': 'Feu & Fumee',
            'axis': 'Axes 7 + 8',
            'stack': 'Detection incendie / changement lumiere',
            'goal': 'Identifier fumee, feu, flash et changement brutal de luminosite.',
            'cameras': ['CAM 12 - fire.mp4'],
            'model_files': ['fire_smoke_detection.pt'],
            'pipeline': ['Capture scene', 'Analyse fumee/feu', 'Filtrage lumiere', 'Alerte incendie'],
            'strengths': ['Couverture securite vitale', 'Extension optique possible', 'Integration alertes'],
            'impact': 'Acceleration du temps de reaction en cas d incendie.',
            'accent': '#FF3B2F',
        },
        7: {
            'slug': 'proximity',
            'short_name': 'Machines & Proximite',
            'axis': 'Axe 9 + 10',
            'stack': 'YOLO proximite homme-machine',
            'goal': 'Detecter distance dangereuse entre travailleur et machine.',
            'cameras': ['CAM 8 - proximite configurable'],
            'model_files': [],
            'pipeline': ['Detection worker/machine', 'Calcul distance relative', 'Seuil risque', 'Alerte proximite'],
            'strengths': ['Alerte collision', 'Visualisation distance', 'Adaptable aux machines chantier'],
            'impact': 'Prevention des collisions et zones d ecrasement.',
            'accent': '#E040FB',
        },
    }

    def _models_dir(self):
        return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))

    def _detection_kind(self, details):
        checks = [
            ('fatigue_detected', 'Fatigue'),
            ('fall_detected', 'Chute'),
            ('spill_detected', 'Deversement'),
            ('manhole_detected', 'Manhole'),
            ('blocked_exit_detected', 'Issue bloquee'),
            ('ppe_violation', 'Violation EPI'),
            ('sign_defect_detected', 'Signalisation'),
            ('proximity_detected', 'Proximite'),
            ('fire_detected', 'Feu/Fumee'),
            ('smoke_detected', 'Feu/Fumee'),
        ]
        for key, label in checks:
            if key in details:
                return label
        return details.get('class', 'Detection')

    def _capture_visual(self, details, module=None):
        if 'fall_detected' in details:
            return 'fall'
        if 'fatigue_detected' in details:
            return 'fatigue'
        if 'spill_detected' in details:
            return 'spill'
        if 'manhole_detected' in details:
            return 'manhole'
        if 'blocked_exit_detected' in details:
            return 'exit'
        if 'ppe_violation' in details:
            return 'ppe'
        if 'sign_defect_detected' in details:
            return 'sign'
        if 'proximity_detected' in details or (module and module.id == 7):
            return 'proximity'
        if 'fire_detected' in details or 'smoke_detected' in details:
            return 'fire'
        return 'generic'

    def _capture_card(self, det):
        details = det.details or {}
        kind = self._detection_kind(details)
        visual = self._capture_visual(details, det.module)
        confidence = max(0, min(100, int(round((det.confidence or details.get('confidence') or 0) * 100))))
        camera = str(details.get('camera') or details.get('source') or f'Module {det.module_id}').upper()
        bbox = details.get('bbox') or details.get('exit_bbox')
        detections = details.get('detections') or details.get('model_detections') or []

        title_map = {
            'fall': 'Chute detectee',
            'fatigue': 'Fatigue detectee',
            'spill': 'Deversement detecte',
            'manhole': 'Plaque ouverte detectee',
            'exit': 'Issue de secours bloquee',
            'ppe': 'Violation EPI detectee',
            'sign': 'Signalisation defectueuse',
            'proximity': 'Proximite machine',
            'fire': 'Feu/Fumee detecte',
            'generic': kind,
        }
        stack_map = {
            'fall': 'YOLO fall_detection.pt',
            'fatigue': 'YOLO fatigue_detection.pt',
            'spill': 'YOLOv8 spill_detection_model.pt',
            'manhole': 'YOLOv8 manhole_seg.pt',
            'exit': 'exit_emergency.pth',
            'ppe': 'YOLO PPE ppe.pt',
            'sign': 'ResNet50 signalisation',
            'proximity': 'Detection proximite',
            'fire': 'YOLO fire_smoke_detection.pt',
            'generic': 'Analyse SafeVision',
        }
        accent_map = {
            'fall': '#FF3B2F',
            'fatigue': '#FFB800',
            'spill': '#00C2FF',
            'manhole': '#FF6B00',
            'exit': '#FF3B2F',
            'ppe': '#4AE3B5',
            'sign': '#7B61FF',
            'proximity': '#E040FB',
            'fire': '#FF3B2F',
            'generic': '#2979FF',
        }

        badges = []
        if visual == 'ppe':
            missing = []
            if not details.get('has_helmet', True):
                missing.append('Casque')
            if not details.get('has_vest', True):
                missing.append('Gilet')
            if not details.get('has_boots', True):
                missing.append('Bottes')
            badges = ['EPI absent'] + (missing[:2] or ['Controle incomplet'])
        elif visual == 'sign':
            badges = [details.get('category_name') or details.get('category') or 'Panneau', details.get('verdict', 'DEFECTIVE')]
        elif visual == 'fall':
            labels = [item.get('label') for item in detections if item.get('label')]
            badges = ['CHUTE'] + labels[:2]
        elif visual == 'fatigue':
            level = details.get('fatigue_level', 'elevee')
            badges = ['Yeux fermes', f'Niveau {level}']
        elif visual == 'spill':
            area = details.get('spill_area_ratio')
            badges = ['SPILL', f'Zone {int(area * 100)}%' if isinstance(area, (int, float)) else 'Masque detecte']
        elif visual == 'manhole':
            badges = ['MANHOLE', details.get('manhole_state', 'open'), f"Risque {details.get('risk_level', 'high')}"]
        elif visual == 'exit':
            badges = ['EXIT', 'Obstacle', f"{int(details.get('confidence', det.confidence or 0) * 100)}%"]
        elif visual == 'proximity':
            badges = ['Machine', 'Distance critique']
        elif visual == 'fire':
            if details.get('fire_detected'):
                badges = ['FIRE', f"{int((details.get('confidence', det.confidence or 0)) * 100)}%"]
            elif details.get('smoke_detected'):
                badges = ['SMOKE', f"{int((details.get('confidence', det.confidence or 0)) * 100)}%"]
            else:
                badges = ['Alerte incendie']
        else:
            badges = [kind, det.module.name[:18]]

        return {
            'id': det.id,
            'module': det.module.name,
            'kind': kind,
            'visual': visual,
            'title': title_map.get(visual, kind),
            'camera': camera,
            'confidence': confidence,
            'timestamp': det.timestamp,
            'stack': stack_map.get(visual, 'Analyse SafeVision'),
            'accent': accent_map.get(visual, '#2979FF'),
            'bbox': bbox,
            'detections_count': len(detections),
            'badges': [badge for badge in badges if badge][:3],
            'capture_url': details.get('capture_url'),
            'capture_path': details.get('capture_path'),
        }

    def _build_module_report(self, module, blueprint):
        detections = module.detections.all()
        alerts = module.alerts.all()
        total = detections.count()
        positives = detections.filter(count__gt=0).count()
        avg_conf = detections.aggregate(value=Avg('confidence'))['value'] or 0
        last_detection = detections.order_by('-timestamp').first()
        critical = alerts.filter(severity='critical').count()
        warnings = alerts.exclude(severity='critical').count()
        model_files = blueprint.get('model_files', [])
        models_dir = self._models_dir()
        available_files = [name for name in model_files if os.path.exists(os.path.join(models_dir, name))]
        missing_files = [name for name in model_files if name not in available_files]
        coverage = int((len(available_files) / len(model_files)) * 100) if model_files else 0
        detection_rate = int((positives / total) * 100) if total else 0
        risk_score = min(100, int(critical * 18 + warnings * 7 + positives * 1.4))
        return {
            'module': module,
            'slug': blueprint['slug'],
            'short_name': blueprint['short_name'],
            'axis': blueprint['axis'],
            'stack': blueprint['stack'],
            'goal': blueprint['goal'],
            'cameras': blueprint['cameras'],
            'pipeline': blueprint['pipeline'],
            'strengths': blueprint['strengths'],
            'impact': blueprint['impact'],
            'accent': blueprint['accent'],
            'total_detections': total,
            'positive_detections': positives,
            'detection_rate': detection_rate,
            'alerts_count': alerts.count(),
            'critical_count': critical,
            'warning_count': warnings,
            'avg_confidence': int(avg_conf * 100),
            'risk_score': risk_score,
            'model_files': model_files,
            'available_files': available_files,
            'missing_files': missing_files,
            'coverage': coverage,
            'last_detection': last_detection,
        }

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        modules = list(Module.objects.order_by('id'))
        detections = Detection.objects.select_related('module').order_by('-timestamp')
        alerts = Alert.objects.select_related('module').order_by('-timestamp')
        total_detections = detections.count()
        positive_detections = detections.filter(count__gt=0).count()
        total_alerts = alerts.count()
        critical_alerts = alerts.filter(severity='critical').count()
        avg_conf = detections.aggregate(value=Avg('confidence'))['value'] or 0
        active_modules = sum(1 for module in modules if module.status == 'active')

        module_reports = [
            self._build_module_report(module, self.MODEL_BLUEPRINTS.get(module.id, self.MODEL_BLUEPRINTS[1]))
            for module in modules
        ]
        top_module = max(module_reports, key=lambda item: item['risk_score'], default=None)
        model_files_total = sum(len(item['model_files']) for item in module_reports)
        model_files_ready = sum(len(item['available_files']) for item in module_reports)

        detection_types = {}
        for det in detections[:300]:
            kind = self._detection_kind(det.details or {})
            detection_types[kind] = detection_types.get(kind, 0) + 1
        detection_mix = [
            {'label': key, 'count': value}
            for key, value in sorted(detection_types.items(), key=lambda item: item[1], reverse=True)[:8]
        ]
        max_mix = max([item['count'] for item in detection_mix] or [1])
        for item in detection_mix:
            item['width'] = int((item['count'] / max_mix) * 100)

        capture_cards = []
        captured_visuals = set()
        for det in detections.filter(count__gt=0)[:800]:
            visual = self._capture_visual(det.details or {}, det.module)
            if visual in captured_visuals:
                continue
            capture_cards.append(self._capture_card(det))
            captured_visuals.add(visual)
            if len(capture_cards) >= 8:
                break

        recent_detections = []
        for det in detections[:25]:
            details = det.details or {}
            recent_detections.append({
                'id': det.id,
                'module': det.module.name,
                'kind': self._detection_kind(details),
                'camera': details.get('camera', details.get('source', 'module')),
                'confidence': int((det.confidence or 0) * 100),
                'status': 'DANGER' if det.count else 'OK',
                'timestamp': det.timestamp,
            })

        export_rows = [
            {
                'module': item['short_name'],
                'stack': item['stack'],
                'detections': item['total_detections'],
                'events': item['positive_detections'],
                'alerts': item['alerts_count'],
                'critical': item['critical_count'],
                'confidence': item['avg_confidence'],
                'coverage': item['coverage'],
                'risk': item['risk_score'],
            }
            for item in module_reports
        ]

        context.update({
            'app_name': 'Rapports SafeVision',
            'modules': modules,
            'alerts': alerts[:12],
            'module_reports': module_reports,
            'capture_cards': capture_cards,
            'recent_detections': recent_detections,
            'detection_mix': detection_mix,
            'report_export': export_rows,
            'generated_at': datetime.now(),
            'kpis': {
                'active_modules': active_modules,
                'total_modules': len(modules),
                'total_detections': total_detections,
                'positive_detections': positive_detections,
                'total_alerts': total_alerts,
                'critical_alerts': critical_alerts,
                'avg_confidence': int(avg_conf * 100),
                'model_files_ready': model_files_ready,
                'model_files_total': model_files_total,
                'readiness': int((model_files_ready / model_files_total) * 100) if model_files_total else 0,
                'risk_focus': top_module['short_name'] if top_module else 'N/A',
            },
        })
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
            is_fire = 'fire_detected' in details or 'smoke_detected' in details
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
                'Feu/Fumee' if is_fire else
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
                'CAM 12' if is_fire else
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
        canonical_slug = normalize_module_slug(slug)
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


def _safe_file_part(value):
    value = str(value or 'capture').strip().lower()
    clean = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in value)
    clean = '_'.join(part for part in clean.split('_') if part)
    return (clean or 'capture')[:48]


def _draw_bbox(frame, bbox, color, label=None):
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    if label:
        cv2.rectangle(frame, (x1, max(0, y1 - 24)), (min(w - 1, x1 + 190), y1), color, -1)
        cv2.putText(frame, str(label)[:24], (x1 + 6, max(16, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def _save_detection_capture(frame, event_type, camera, confidence, details=None, label=None):
    """Sauvegarde une vraie frame detectee avec overlay et retourne ses URLs media."""
    if frame is None:
        return {}

    details = details or {}
    annotated = frame.copy()
    color_map = {
        'fall': (47, 59, 255),
        'fatigue': (0, 184, 255),
        'spill': (255, 194, 0),
        'manhole': (0, 107, 255),
        'blocked_exit': (47, 59, 255),
        'ppe': (181, 227, 74),
        'sign_defect': (255, 97, 123),
        'proximity': (251, 64, 224),
        'fire': (47, 59, 255),
    }
    color = color_map.get(event_type, (255, 194, 0))

    for polygon in details.get('polygons') or []:
        if isinstance(polygon, list) and len(polygon) >= 3:
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, color, 2)

    main_label = label or event_type.replace('_', ' ').upper()
    for box_key in ('bbox', 'exit_bbox', 'obstacle_bbox'):
        _draw_bbox(annotated, details.get(box_key), color, main_label if box_key == 'bbox' else box_key.upper())

    for det in (details.get('detections') or details.get('model_detections') or [])[:8]:
        det_label = det.get('label') or det.get('class') or main_label
        if det.get('confidence') is not None:
            det_label = f"{det_label} {float(det.get('confidence')):.0%}"
        _draw_bbox(annotated, det.get('bbox'), color, det_label)

    header = f"{main_label} | {str(camera).upper()} | {float(confidence or 0):.1%}"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(annotated, header[:70], (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)

    capture_dir = os.path.join(settings.MEDIA_ROOT, 'detection_captures')
    os.makedirs(capture_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f"{stamp}_{_safe_file_part(event_type)}_{_safe_file_part(camera)}.jpg"
    abs_path = os.path.join(capture_dir, filename)
    saved = cv2.imwrite(abs_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not saved:
        return {}

    relative_path = f"detection_captures/{filename}"
    return {
        'capture_path': relative_path,
        'capture_url': f"{settings.MEDIA_URL}{relative_path}",
    }


def _public_media_url(relative_or_media_url):
    public_base_url = config('PUBLIC_BASE_URL', default='').strip()
    if not public_base_url or not relative_or_media_url:
        return ''

    clean_path = str(relative_or_media_url).lstrip('/')
    return urljoin(public_base_url.rstrip('/') + '/', clean_path)


def _sms_cooldown_active(event_key):
    cooldown_seconds = int(config('TWILIO_SMS_COOLDOWN_SECONDS', default='300') or 300)
    now = timezone.now()
    last_sent_at = SMS_EVENT_CACHE.get(event_key)
    if last_sent_at and (now - last_sent_at).total_seconds() < cooldown_seconds:
        return True
    return False


def _mark_sms_sent(event_key):
    SMS_EVENT_CACHE[event_key] = timezone.now()


def _send_twilio_sms(body, media_url=None):
    account_sid = config('TWILIO_ACCOUNT_SID', default='').strip()
    auth_token = config('TWILIO_AUTH_TOKEN', default='').strip()
    to_number = config('TWILIO_TO_NUMBER', default='').strip()
    from_number = config('TWILIO_FROM_NUMBER', default='').strip()
    messaging_service_sid = config('TWILIO_MESSAGING_SERVICE_SID', default='').strip()

    if not account_sid or not auth_token or not to_number:
        return {
            'sent': False,
            'reason': 'Twilio non configure: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN et TWILIO_TO_NUMBER sont requis.',
        }
    if not from_number and not messaging_service_sid:
        return {
            'sent': False,
            'reason': 'Twilio non configure: TWILIO_FROM_NUMBER ou TWILIO_MESSAGING_SERVICE_SID est requis.',
        }

    payload = {
        'To': to_number,
        'Body': body,
    }
    if messaging_service_sid:
        payload['MessagingServiceSid'] = messaging_service_sid
    else:
        payload['From'] = from_number
    if media_url and config('TWILIO_SEND_MEDIA_URL', default='False').lower() in ('1', 'true', 'yes', 'on'):
        payload['MediaUrl'] = media_url

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    try:
        response = requests.post(url, data=payload, auth=(account_sid, auth_token), timeout=8)
        if response.status_code >= 400:
            return {
                'sent': False,
                'status_code': response.status_code,
                'reason': response.text[:500],
            }
        result = response.json()
        return {
            'sent': True,
            'sid': result.get('sid'),
            'status': result.get('status'),
            'media_attached': bool(payload.get('MediaUrl')),
        }
    except requests.exceptions.RequestException as exc:
        return {'sent': False, 'reason': str(exc)}


def _notify_fall_sms(alert, camera, confidence, capture_meta):
    event_key = f"fall:{camera}"
    if _sms_cooldown_active(event_key):
        alert.details = {
            **(alert.details or {}),
            'sms': {'sent': False, 'reason': 'cooldown actif'},
        }
        alert.save(update_fields=['details'])
        return alert.details['sms']

    capture_url = capture_meta.get('capture_url')
    public_capture_url = _public_media_url(capture_url)
    message = (
        f"ALERTE CHUTE SafeVision: chute detectee sur {str(camera).upper()} "
        f"(confiance {confidence:.0%})."
    )
    if public_capture_url:
        message = f"{message} Capture: {public_capture_url}"

    sms_result = _send_twilio_sms(message, media_url=public_capture_url)
    if sms_result.get('sent'):
        _mark_sms_sent(event_key)
    alert.details = {
        **(alert.details or {}),
        'sms': sms_result,
        'public_capture_url': public_capture_url,
    }
    alert.save(update_fields=['details'])
    return sms_result


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
        'hazards': 'hazards',
        'risques': 'hazards',
        'spill': 'hazards',
        'deversement': 'hazards',
        'chemical': 'hazards',
        'manhole': 'hazards',
        'hole': 'hazards',
        'exit': 'hazards',
        'emergency': 'hazards',
        'tracking': 'tracking',
        'suivi': 'tracking',
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
    is_fire = 'fire_detected' in details or 'smoke_detected' in details
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
        'fire' if is_fire else
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
        'cam12' if is_fire else
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

        sms_result = None
        capture_meta = {}
        module = _get_module_by_name('fall', 'fatigue')
        if module:
            capture_meta = _save_detection_capture(
                frame, 'fall', data.get('camera', 'cam1'), confidence, details, 'CHUTE'
            ) if fall_detected else {}
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
                    **capture_meta,
                },
            )

            if fall_detected:
                alert = Alert.objects.create(
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
                        **capture_meta,
                    },
                )
                sms_result = _notify_fall_sms(alert, data.get('camera', 'cam1'), confidence, capture_meta)

        return JsonResponse({
            'status':        'success',
            'fall_detected': bool(fall_detected),
            'confidence':    float(confidence),
            'details':       {**details, **capture_meta} if fall_detected else details,
            'capture_url':   capture_meta.get('capture_url') if fall_detected else None,
            'public_capture_url': _public_media_url(capture_meta.get('capture_url')) if fall_detected else None,
            'sms':           sms_result,
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
            capture_meta = _save_detection_capture(
                frame, 'fatigue', data.get('camera', 'cam2'), confidence, details, 'FATIGUE'
            ) if fatigue_detected else {}
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
                    **capture_meta,
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
                        **capture_meta,
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
            capture_meta = _save_detection_capture(
                frame, 'ppe', camera, confidence, details, 'EPI'
            ) if ppe_violation else {}
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
                    'model_available': details.get('model_available', False),
                    **capture_meta,
                },
            )
            if ppe_violation:
                Alert.objects.create(
                    module=module,
                    severity='warning',
                    title='Violation EPI detectee !',
                    description=f'Equipement de protection manquant sur {camera.upper()} ({confidence:.2%})',
                    details={'confidence': confidence, 'camera': camera, 'event_type': 'ppe', 'timestamp': datetime.now().isoformat(), **capture_meta},
                )

        return JsonResponse({
            'status': 'success',
            'ppe_violation': bool(ppe_violation),
            'confidence': float(confidence),
            'details': details,
            'model_available': details.get('model_available', False),
        })
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de decodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur detection EPI : {e}'}, status=500)


def api_sign_detect(request):
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
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            capture_meta = _save_detection_capture(
                frame, 'sign_defect', camera, confidence, result, 'SIGN DEFECT'
            ) if is_defective else {}
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
                    'model_available': True,
                    **capture_meta,
                },
            )
            if is_defective:
                Alert.objects.create(
                    module=module,
                    severity='warning',
                    title='Panneau defectueux detecte !',
                    description=f'Signalisation defectueuse sur {camera.upper()} ({confidence:.2%})',
                    details={'confidence': confidence, 'camera': camera, 'event_type': 'sign_defect', 'timestamp': datetime.now().isoformat(), **capture_meta},
                )

        return JsonResponse({'status': 'success', 'data': result})
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de decodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur detection signalisation : {e}'}, status=500)


@csrf_exempt
def api_proximity_detection(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)
    try:
        data = json.loads(request.body)
        if 'image' not in data:
            return JsonResponse({'status': 'error', 'message': 'Image requise'}, status=400)
        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Décodage: {e}'}, status=400)

        # ── Détection ─────────────────────────────────────────────
        from .proximity_detector import proximity_detector as _pd
        proximity_detected, confidence, details = _pd.detect_proximity(frame)

        # ── Annotation ────────────────────────────────────────────
        annotated     = _pd.annotate_frame(frame.copy(), details)
        _, buf        = cv2.imencode('.jpg', annotated,
                                     [cv2.IMWRITE_JPEG_QUALITY, 88])
        annotated_b64 = ('data:image/jpeg;base64,'
                         + base64.b64encode(buf).decode('utf-8'))

        # ── Sévérité max ──────────────────────────────────────────
        incidents = details.get('incident_logs', [])
        order     = ['safe','vigilance','alert','critical']
        top       = max(
            [i.get('severity','safe') for i in incidents] or ['safe'],
            key=lambda s: order.index(s) if s in order else 0
        )

        # ── Persistance ───────────────────────────────────────────
        module = _get_module_by_name('machinery','proximity','homme-machine')
        if module:
            Detection.objects.create(
                module=module, confidence=confidence,
                count=len(incidents),
                details={
                    'proximity_detected': proximity_detected,
                    'confidence'        : confidence,
                    'camera'            : data.get('camera','cam8'),
                    'workers_count'     : details.get('workers_count',0),
                    'machines_count'    : details.get('machines_count',0),
                    'incidents'         : len(incidents),
                },
            )
            if proximity_detected and incidents:
                sev  = incidents[0].get('severity','alert')
                dist = incidents[0].get('distance_m', 0)
                Alert.objects.create(
                    module=module, severity=sev,
                    title=f'Proximité dangereuse ! ({dist:.1f}m)',
                    description=f"Ouvrier à {dist:.1f}m d'un engin",
                    details={
                        'distance'  : dist, 'severity': sev,
                        'camera'    : data.get('camera','CAM 8'),
                        'event_type': 'proximity',
                        'timestamp' : datetime.now().isoformat(),
                    },
                )

        return JsonResponse({
            'status'            : 'success',
            'proximity_detected': bool(proximity_detected),
            'confidence'        : round(float(confidence), 3),
            'annotated_image'   : annotated_b64,
            'details'           : {
                'workers_count'  : details.get('workers_count', 0),
                'machines_count' : details.get('machines_count', 0),
                'severity'       : top,
                'incidents_count': len(incidents),
                'incident_logs'  : incidents,
            },
        })

    except json.JSONDecodeError:
        return JsonResponse({'status':'error','message':'JSON invalide'},status=400)
    except Exception as e:
        return JsonResponse({'status':'error','message':str(e)},status=500)
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)

    try:
        data = json.loads(request.body)

        if 'image' not in data:
            return JsonResponse(
                {'status': 'error', 'message': 'Image requise'}, status=400)

        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse(
                {'status': 'error', 'message': f'Erreur décodage : {e}'}, status=400)

        # ── Détection ────────────────────────────────────────────────────────
        from .proximity_detector import proximity_detector as _pd
        proximity_detected, confidence, details = _pd.detect_proximity(frame)

        # ── Annotation visuelle ───────────────────────────────────────────────
        annotated = _pd.annotate_frame(frame.copy(), details)
        _, buf     = cv2.imencode('.jpg', annotated,
                                  [cv2.IMWRITE_JPEG_QUALITY, 88])
        annotated_b64 = ('data:image/jpeg;base64,'
                         + base64.b64encode(buf).decode('utf-8'))

        # ── Persistance ───────────────────────────────────────────────────────
        incidents = details.get('incident_logs', [])
        module    = _get_module_by_name('machinery', 'proximity')
        if module:
            capture_meta = _save_detection_capture(
                frame, 'proximity', data.get('camera', 'cam8'), confidence, details, 'PROXIMITE'
            ) if proximity_detected else {}
            Detection.objects.create(
                module     = module,
                confidence = confidence,
                count      = len(incidents),
                details    = {
                    'proximity_detected': proximity_detected,
                    'confidence'        : confidence,
                    'camera'            : data.get('camera', 'cam8'),
                    'workers_count'     : details.get('workers_count', 0),
                    'machines_count'    : details.get('machines_count', 0),
                    'incidents'         : len(incidents),
                },
            )
            if proximity_detected and incidents:
                sev  = incidents[0].get('severity', 'alert')
                dist = incidents[0].get('distance_m', 0)
                Alert.objects.create(
                    module      = module,
                    severity    = sev,
                    title       = f'Proximité dangereuse ! ({dist:.1f}m)',
                    description = (f'Ouvrier à {dist:.1f}m d\'un engin'),
                    details     = {
                        'distance'  : dist,
                        'severity'  : sev,
                        'camera'    : data.get('camera', 'CAM 8'),
                        'event_type': 'proximity',
                        'timestamp' : datetime.now().isoformat(),
                    },
                )

        # ── Sévérité maximale ─────────────────────────────────────────────────
        order = ['safe', 'vigilance', 'alert', 'critical']
        top   = max(
            [i.get('severity', 'safe') for i in incidents],
            key=lambda s: order.index(s) if s in order else 0,
            default='safe'
        )

        return JsonResponse({
            'status'            : 'success',
            'proximity_detected': bool(proximity_detected),
            'confidence'        : round(float(confidence), 3),
            'annotated_image'   : annotated_b64,
            'details'           : {
                'workers_count' : details.get('workers_count', 0),
                'machines_count': details.get('machines_count', 0),
                'severity'      : top,
                'incidents_count': len(incidents),
                'incident_logs' : incidents,
            },
        })

    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'JSON invalide'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
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
            capture_meta = _save_detection_capture(
                frame, 'spill', camera, confidence, details, 'SPILL'
            ) if spill_detected else {}
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
                    **capture_meta,
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
                        **capture_meta,
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
            capture_meta = _save_detection_capture(
                frame, 'manhole', camera, confidence, details, 'MANHOLE'
            ) if manhole_detected else {}
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
                    **capture_meta,
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
                        **capture_meta,
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
            capture_meta = _save_detection_capture(
                frame, 'blocked_exit', camera, confidence, details, 'SORTIE BLOQUEE'
            ) if blocked_detected else {}
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
                    **capture_meta,
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
                        **capture_meta,
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


def api_fire_detection(request):
    """API endpoint pour la detection de feu et fumee (CAM 12)."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Methode non autorisee'}, status=405)

    try:
        data = json.loads(request.body)
        if 'image' not in data:
            return JsonResponse({'status': 'error', 'message': 'Image requise pour la detection incendie'}, status=400)

        try:
            frame = _decode_image(data['image'])
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': f'Erreur de decodage image : {e}'}, status=400)

        camera = data.get('camera', 'cam12')
        incident_detected, confidence, details = detect_fire_smoke_in_frame(frame, camera=camera)
        fire_detected = bool(details.get('fire_detected', False))
        smoke_detected = bool(details.get('smoke_detected', False))

        module = get_module_by_slug('fire')
        if module:
            capture_meta = _save_detection_capture(
                frame, 'fire', camera, confidence, details, 'FEU/FUMEE'
            ) if incident_detected else {}
            Detection.objects.create(
                module=module,
                confidence=confidence,
                count=1 if incident_detected else 0,
                details={
                    'fire_detected': fire_detected,
                    'smoke_detected': smoke_detected,
                    'confidence': confidence,
                    'camera': camera,
                    'bbox': details.get('bbox'),
                    'detections': details.get('detections', []),
                    'primary_type': details.get('primary_type', 'other'),
                    'frame_shape': list(map(int, frame.shape)),
                    'model_version': '1.0.0',
                    'model_available': details.get('model_available', False),
                    **capture_meta,
                },
            )

            if incident_detected:
                severity = 'critical' if fire_detected else 'warning'
                event_label = 'Feu detecte' if fire_detected else 'Fumee detectee'
                Alert.objects.create(
                    module=module,
                    severity=severity,
                    title='Alerte incendie detectee !',
                    description=f'{event_label} sur {camera.upper()} (confiance {confidence:.2%})',
                    details={
                        'confidence': confidence,
                        'camera': camera,
                        'event_type': 'fire',
                        'fire_detected': fire_detected,
                        'smoke_detected': smoke_detected,
                        'timestamp': datetime.now().isoformat(),
                        **capture_meta,
                    },
                )

        return JsonResponse({
            'status': 'success',
            'fire_detected': bool(incident_detected),
            'smoke_detected': smoke_detected,
            'confidence': float(confidence),
            'details': details,
            'model_available': details.get('model_available', False),
            'message': 'Analyse feu/fumee terminee',
        })
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Erreur de decodage JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur detection incendie : {e}'}, status=500)


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
    


# ── /api/chat/ ──────────────────────────────────────────────────────────────
@csrf_exempt
def api_chat(request):
    """API endpoint pour le chatbot SafeBot (OpenRouter)."""
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'message': 'Méthode non autorisée'}, status=405)

    try:
        data = json.loads(request.body)
        user_message = data.get('message', '')

        if not user_message:
            return JsonResponse({'status': 'error', 'message': 'Message vide'}, status=400)

        # On lit la clé ICI, avec un défaut vide pour ne pas crasher si elle manque
        api_key = config('OPENROUTER_API_KEY', default='')
        
        if not api_key:
            return JsonResponse({'status': 'error', 'message': 'Clé API OpenRouter non configurée dans le fichier .env'}, status=500)

        # Appel à l'API OpenRouter
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://127.0.0.1:8000",
            "X-Title": "SafeVision AI"
        }
        
        system_prompt = """
        Tu es SafeBot, un assistant expert en santé et sécurité au travail, spécifiquement en Tunisie. 
        Tu connais parfaitement le Code du Travail tunisien, les normes de la CNAMPH, et les réglementations de l'INORPI.
        Tu réponds de manière professionnelle, claire et concise. Si on te pose une question hors sujet, 
        rappelle poliment que tu es spécialisé dans la sécurité au travail en Tunisie.
        """

        payload = {
            "model": "meta-llama/llama-3-8b-instruct", 
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        bot_reply = result['choices'][0]['message']['content']

        return JsonResponse({'status': 'success', 'reply': bot_reply})

    except requests.exceptions.RequestException as e:
        print(f"[SafeBot] OpenRouter API Error: {e}")
        return JsonResponse({'status': 'error', 'message': 'Erreur de connexion à l\'IA'}, status=500)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': f'Erreur interne : {e}'}, status=500)
