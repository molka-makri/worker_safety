"""
fatiguedetector.py — SafeVision AI
====================================
Détection de fatigue des travailleurs à partir d'une frame OpenCV (BGR).

Stratégie :
  1. Détection du visage via Haarcascade OpenCV
  2. Analyse des yeux :
     - EAR  (Eye Aspect Ratio)  → détection de clignement / yeux mi-clos
     - PERCLOS (% du temps les yeux fermés) → indicateur de somnolence
     - Absence de détection d'yeux = yeux fermés (CORRECTION CRITIQUE)
  3. Détection bouche ouverte par analyse de variance pixel (fonctionne en IR)
  4. Détection de la posture de tête (inclinaison vers le bas = somnolence)
  5. Si le modèle PyTorch fatigue_detection.pt est présent, il est utilisé
     en complément pour confirmer la détection.

Corrections v2 :
  - Yeux non détectés par Haar → considérés fermés (et non ouverts)
  - Détection bouche par variance pixel zone basse du visage (robuste IR/nuit)
  - Score de fatigue révisé : bâillement (yeux fermés + bouche ouverte) = 90 %
  - Séparation des cas : yeux fermés seuls vs tête inclinée seule

Retour : (fatigue_detected: bool, confidence: float, details: dict)
"""

import os
import math
import logging
import traceback
from typing import Tuple, Dict, Any, Optional, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))

FATIGUE_MODEL_PATH  = os.path.join(MODELS_DIR, 'fatigue_detection.pt')
CASCADE_FACE_PATHS  = [
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
]
CASCADE_EYE_PATHS   = [
    cv2.data.haarcascades + 'haarcascade_eye.xml',
    cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml',
]

# ── Seuils ─────────────────────────────────────────────────────────────────────
EAR_THRESHOLD           = 0.25   # En dessous : œil considéré fermé
PERCLOS_THRESHOLD       = 0.30   # > 30 % du temps les yeux fermés = somnolence
HEAD_TILT_THRESHOLD     = 20     # Degrés d'inclinaison de la tête vers le bas
MIN_CONFIDENCE          = 0.50   # Confiance minimale pour déclencher une alerte
HISTORY_SIZE            = 20     # Nombre de frames conservées pour PERCLOS
MOUTH_DARK_RATIO        = 0.15   # Seuil de pixels sombres pour bouche ouverte
MOUTH_DARK_PIXEL_MAX    = 80     # Valeur max d'un pixel considéré "sombre"
MOUTH_ROI_TOP           = 0.55   # Début ROI bouche (% hauteur visage)
MOUTH_ROI_BOTTOM        = 0.85   # Fin ROI bouche (% hauteur visage)

# ── Niveaux de fatigue ─────────────────────────────────────────────────────────
FATIGUE_LEVELS = {
    (0.0,  0.4):  'faible',
    (0.4,  0.65): 'modérée',
    (0.65, 0.85): 'élevée',
    (0.85, 1.01): 'critique',
}


def _get_fatigue_level(confidence: float) -> str:
    for (low, high), label in FATIGUE_LEVELS.items():
        if low <= confidence < high:
            return label
    return 'inconnue'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chargement paresseux du modèle PyTorch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_fatigue_model      = None
_fatigue_model_ok   = False
_pytorch_available  = False
_yolo_available     = False

try:
    import torch
    import torchvision.transforms as T
    _pytorch_available = True
except ImportError:
    logger.warning('[FatigueDetector] PyTorch non disponible — mode heuristique seul.')


try:
    from ultralytics import YOLO
    _yolo_available = True
except ImportError:
    logger.warning('[FatigueDetector] Ultralytics non disponible - mode heuristique seul.')


def _load_fatigue_model():
    global _fatigue_model, _fatigue_model_ok
    if _fatigue_model_ok or not _yolo_available:
        return _fatigue_model_ok

    if not os.path.exists(FATIGUE_MODEL_PATH):
        logger.warning('[FatigueDetector] Modèle introuvable: %s', FATIGUE_MODEL_PATH)
        return False

    try:
        _fatigue_model    = YOLO(FATIGUE_MODEL_PATH)
        _fatigue_model_ok = True
        logger.info('[FatigueDetector] Modèle chargé depuis %s', FATIGUE_MODEL_PATH)
    except Exception as exc:
        logger.error('[FatigueDetector] Erreur chargement modèle: %s', exc)
        _fatigue_model_ok = False

    return _fatigue_model_ok


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Etat PERCLOS (persistant entre appels)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_eye_closure_history: List[bool] = []


def _update_perclos(eyes_closed: bool) -> float:
    """Met à jour l'historique et retourne le taux PERCLOS courant."""
    _eye_closure_history.append(eyes_closed)
    if len(_eye_closure_history) > HISTORY_SIZE:
        _eye_closure_history.pop(0)
    if not _eye_closure_history:
        return 0.0
    return sum(_eye_closure_history) / len(_eye_closure_history)


def reset_perclos() -> None:
    """Remet à zéro l'historique PERCLOS (utile entre deux travailleurs)."""
    _eye_closure_history.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Calcul EAR (Eye Aspect Ratio)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _eye_aspect_ratio(eye_rect) -> float:
    """
    Approximation EAR à partir d'un rectangle (x, y, w, h).
    EAR = hauteur / largeur  (simplifié ; sans landmarks 68 pts).
    Un œil ouvert a un ratio ~0.30-0.40 ; fermé < 0.25.
    """
    x, y, w, h = eye_rect
    if w == 0:
        return 1.0
    return h / w


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Détection bouche ouverte par analyse pixel (robuste IR)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _is_mouth_open_by_pixel(face_roi_gray: np.ndarray) -> Tuple[bool, float]:
    """
    Détecte une bouche ouverte en analysant la région basse du visage.

    Principes :
      - La cavité buccale ouverte crée une zone sombre (< MOUTH_DARK_PIXEL_MAX)
        à fort contraste, visible même en infrarouge.
      - On analyse la bande verticale [55 % – 85 %] de la hauteur du visage,
        sur toute la largeur (la bouche est centrée horizontalement).

    Retourne (mouth_open: bool, dark_ratio: float)
    """
    h = face_roi_gray.shape[0]
    y_top    = int(h * MOUTH_ROI_TOP)
    y_bottom = int(h * MOUTH_ROI_BOTTOM)
    mouth_roi = face_roi_gray[y_top:y_bottom, :]

    if mouth_roi.size == 0:
        return False, 0.0

    dark_pixels = int(np.sum(mouth_roi < MOUTH_DARK_PIXEL_MAX))
    dark_ratio  = dark_pixels / mouth_roi.size

    mouth_open = dark_ratio > MOUTH_DARK_RATIO
    return mouth_open, round(dark_ratio, 3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Détecteurs OpenCV (Haar Cascades)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_face_cascade: Optional[List[cv2.CascadeClassifier]] = None
_eye_cascade:  Optional[List[cv2.CascadeClassifier]] = None


def _get_cascades():
    global _face_cascade, _eye_cascade
    if _face_cascade is None:
        _face_cascade = [cv2.CascadeClassifier(p) for p in CASCADE_FACE_PATHS]
    if _eye_cascade is None:
        _eye_cascade  = [cv2.CascadeClassifier(p) for p in CASCADE_EYE_PATHS]
    return _face_cascade, _eye_cascade


def _detect_faces(gray: np.ndarray):
    face_cascades, _ = _get_cascades()
    for cascade in face_cascades:
        if cascade.empty():
            continue
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) > 0:
            return faces
    return np.empty((0, 4), dtype=int)


def _detect_eyes_in_roi(face_roi_gray: np.ndarray):
    """
    Cherche les yeux dans le ROI supérieur du visage (60 % haut).
    Limiter la recherche au haut du visage évite les faux positifs
    (narines, bouche ouverte détectée comme un œil).
    """
    _, eye_cascades = _get_cascades()
    h = face_roi_gray.shape[0]
    upper_roi = face_roi_gray[:int(h * 0.60), :]  # Seulement la moitié supérieure

    for cascade in eye_cascades:
        if cascade.empty():
            continue
        eyes = cascade.detectMultiScale(
            upper_roi,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(16, 16),
        )
        if len(eyes) > 0:
            return eyes
    return np.empty((0, 4), dtype=int)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analyse heuristique d'une frame
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _analyze_heuristic(frame: np.ndarray) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Retourne (fatigue_detected, confidence, details).
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.equalizeHist(gray)
    faces = _detect_faces(gray)

    details: Dict[str, Any] = {
        'method':        'heuristic',
        'faces_found':   0,
        'eyes_found':    0,
        'eyes_closed':   False,
        'mouth_open':    False,
        'mouth_dark_ratio': None,
        'ear':           None,
        'perclos':       None,
        'head_tilt':     None,
        'fatigue_level': 'faible',
        'bbox':          None,
    }

    # ── Aucun visage détecté ───────────────────────────────
    if len(faces) == 0:
        perclos = _update_perclos(False)
        details['perclos'] = round(perclos, 3)
        return False, 0.0, details

    # On prend le visage le plus grand (plus proche de la caméra)
    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    details['faces_found'] = len(faces)
    details['bbox']        = [int(fx), int(fy), int(fx + fw), int(fy + fh)]

    face_roi_gray = gray[fy:fy + fh, fx:fx + fw]

    # ── Analyse des yeux ───────────────────────────────────
    eyes = _detect_eyes_in_roi(face_roi_gray)
    details['eyes_found'] = len(eyes)

    # CORRECTION CRITIQUE : si aucun œil détecté dans la zone supérieure
    # du visage, les yeux sont très probablement fermés (bâillement, somnolence).
    # L'ancien code supposait les yeux OUVERTS si non détectés — erreur majeure.
    if len(eyes) == 0:
        avg_ear     = 0.0   # EAR minimal = yeux complètement fermés
        eyes_closed = True
    else:
        ears = [_eye_aspect_ratio(e) for e in eyes]
        avg_ear     = float(np.mean(ears))
        eyes_closed = avg_ear < EAR_THRESHOLD

    perclos = _update_perclos(eyes_closed)

    details['ear']         = round(avg_ear, 3)
    details['perclos']     = round(perclos, 3)
    details['eyes_closed'] = eyes_closed

    # ── Détection bouche ouverte (analyse pixel, robuste IR) ──
    mouth_open, dark_ratio = _is_mouth_open_by_pixel(face_roi_gray)
    details['mouth_open']       = mouth_open
    details['mouth_dark_ratio'] = dark_ratio

    # ── Inclinaison de la tête ─────────────────────────────
    # Approx : position verticale du centre du visage dans l'image
    frame_h       = frame.shape[0]
    head_position = (fy + fh / 2) / frame_h   # 0 = haut, 1 = bas
    head_tilt     = max(0.0, (head_position - 0.5) * 100)   # 0..50
    details['head_tilt'] = round(head_tilt, 1)

    # ━━ Score de fatigue (logique révisée) ━━━━━━━━━━━━━━━━━
    #
    #  Cas 1 — Bâillement confirmé : yeux fermés ET bouche ouverte
    #           → confiance élevée (0.90), alerte immédiate
    #
    #  Cas 2 — Yeux seuls fermés (PERCLOS ou EAR bas)
    #           → score pondéré PERCLOS + EAR + tête
    #
    #  Cas 3 — Pas de yeux fermés, tête inclinée seulement
    #           → score faible, PERCLOS + tête
    #
    if eyes_closed and mouth_open:
        # ── Cas 1 : Bâillement ────────────────────────────────
        confidence       = 0.90
        fatigue_detected = True

    elif eyes_closed:
        # ── Cas 2 : Yeux fermés (sans bâillement visible) ─────
        score_ear     = max(0.0, (EAR_THRESHOLD - avg_ear) / EAR_THRESHOLD) \
                        if avg_ear < EAR_THRESHOLD else 0.0
        score_perclos = min(perclos / PERCLOS_THRESHOLD, 1.0)
        score_head    = min(head_tilt / HEAD_TILT_THRESHOLD, 1.0)

        confidence = (
            0.40 * score_perclos +
            0.40 * score_ear     +
            0.20 * score_head
        )
        confidence       = float(np.clip(confidence, 0.0, 1.0))
        fatigue_detected = confidence >= MIN_CONFIDENCE

    else:
        # ── Cas 3 : Yeux ouverts, évaluation posture/PERCLOS ──
        score_perclos = min(perclos / PERCLOS_THRESHOLD, 1.0)
        score_head    = min(head_tilt / HEAD_TILT_THRESHOLD, 1.0)

        confidence = (
            0.55 * score_perclos +
            0.45 * score_head
        )
        confidence       = float(np.clip(confidence, 0.0, 1.0))
        fatigue_detected = confidence >= MIN_CONFIDENCE

    details['fatigue_level'] = _get_fatigue_level(confidence)

    return fatigue_detected, confidence, details


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Inférence avec le modèle PyTorch (optionnel)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _analyze_with_model(frame: np.ndarray) -> Optional[Tuple[bool, float]]:
    """
    Lance l'inférence PyTorch si le modèle est disponible.
    Retourne (fatigue_detected, confidence) ou None si indisponible.
    """
    if not _load_fatigue_model():
        return None

    try:
        import torch
        import torchvision.transforms as T

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std =[0.229, 0.224, 0.225]),
        ])

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0)

        with torch.no_grad():
            output = _fatigue_model(tensor)
            # Supporte sortie (batch, 1) ou (batch, 2)
            if output.shape[-1] == 1:
                prob = float(torch.sigmoid(output)[0, 0])
            else:
                prob = float(torch.softmax(output, dim=1)[0, 1])

        return prob >= MIN_CONFIDENCE, prob

    except Exception as exc:
        logger.warning('[FatigueDetector] Inférence modèle échouée: %s', exc)
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Point d'entrée public
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _analyze_with_model(frame: np.ndarray) -> Optional[Tuple[bool, float, Dict[str, Any]]]:
    """Inference YOLO fatigue_detection.pt (closed_eye/open_mouth)."""
    if not _load_fatigue_model():
        return None

    try:
        results = _fatigue_model(frame, conf=0.25, verbose=False)
        detections = []
        closed_eye_boxes = []
        open_mouth_boxes = []
        closed_eye_conf = 0.0
        open_eye_conf = 0.0
        open_mouth_conf = 0.0

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = result.names.get(cls_id, str(cls_id)).lower().strip()
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                detections.append({
                    'class_id': cls_id,
                    'label': label,
                    'confidence': round(conf, 3),
                    'bbox': [x1, y1, x2, y2],
                })

                if label == 'closed_eye':
                    closed_eye_conf = max(closed_eye_conf, conf)
                    closed_eye_boxes.append([x1, y1, x2, y2])
                elif label == 'open_eye':
                    open_eye_conf = max(open_eye_conf, conf)
                elif label == 'open_mouth':
                    open_mouth_conf = max(open_mouth_conf, conf)
                    open_mouth_boxes.append([x1, y1, x2, y2])

        eyes_closed = closed_eye_conf >= 0.35 and (closed_eye_conf + 0.05) >= open_eye_conf
        yawn_without_open_eyes = open_mouth_conf >= 0.55 and (eyes_closed or open_eye_conf < 0.45)

        confidence = closed_eye_conf if eyes_closed else 0.0
        fatigue_boxes = closed_eye_boxes if eyes_closed else []

        if yawn_without_open_eyes:
            confidence = max(confidence, open_mouth_conf)
            fatigue_boxes.extend(open_mouth_boxes)

        if eyes_closed and open_mouth_conf >= 0.35:
            confidence = min(1.0, 0.55 * closed_eye_conf + 0.45 * open_mouth_conf + 0.10)

        bbox = None
        fatigue_detected = eyes_closed or yawn_without_open_eyes
        if fatigue_detected and fatigue_boxes:
            bbox = [
                min(b[0] for b in fatigue_boxes),
                min(b[1] for b in fatigue_boxes),
                max(b[2] for b in fatigue_boxes),
                max(b[3] for b in fatigue_boxes),
            ]

        return fatigue_detected, float(confidence), {
            'model': 'YOLO_fatigue_detection.pt',
            'model_detections': detections,
            'closed_eye_confidence': round(closed_eye_conf, 3),
            'open_eye_confidence': round(open_eye_conf, 3),
            'open_mouth_confidence': round(open_mouth_conf, 3),
            'eyes_closed_by_model': eyes_closed,
            'yawn_without_open_eyes': yawn_without_open_eyes,
            'bbox': bbox,
        }

    except Exception as exc:
        logger.warning('[FatigueDetector] Inference YOLO fatigue echouee: %s', exc)
        return None


def detect_fatigue_in_frame(
    frame: np.ndarray,
) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Détecte la fatigue dans une frame BGR OpenCV.

    Paramètres
    ----------
    frame : np.ndarray
        Image BGR (ex. lue par OpenCV ou convertie depuis PIL).

    Retour
    ------
    fatigue_detected : bool
    confidence       : float  (0.0 – 1.0)
    details          : dict   (informations de diagnostic)

    Exemple de details retourné
    ---------------------------
    {
        'method':             'heuristic' | 'hybrid',
        'faces_found':        1,
        'eyes_found':         0,          # 0 = yeux fermés (non détectés)
        'eyes_closed':        True,
        'mouth_open':         True,
        'mouth_dark_ratio':   0.21,       # ratio pixels sombres dans zone bouche
        'ear':                0.0,
        'perclos':            0.45,
        'head_tilt':          12.3,
        'fatigue_level':      'critique',
        'bbox':               [x1, y1, x2, y2],
        'model_available':    False,
        'pytorch_available':  False,
    }
    """
    if frame is None or frame.size == 0:
        return False, 0.0, {'error': 'Frame vide ou nulle'}

    try:
        # 1) Analyse heuristique (toujours exécutée)
        h_detected, h_conf, details = _analyze_heuristic(frame)

        # 2) Modèle PyTorch (si disponible)
        model_result = _analyze_with_model(frame)

        if model_result is not None:
            m_detected, m_conf, model_details = model_result
            # Fusion : moyenne pondérée (heuristique 40 % / modèle 60 %)
            final_conf     = m_conf
            final_detected = m_detected
            details.update(model_details)
            details['model_confidence']     = round(m_conf, 3)
            details['heuristic_confidence'] = round(h_conf, 3)
            details['method']               = 'yolo_hybrid'
            details['model_available']      = True
            details['pytorch_available']    = _pytorch_available
            details['yolo_available']       = _yolo_available
        else:
            final_conf     = h_conf
            final_detected = h_detected
            details['model_available']   = False
            details['pytorch_available'] = _pytorch_available
            details['yolo_available']    = _yolo_available

        details['fatigue_level'] = _get_fatigue_level(final_conf)

        logger.debug(
            '[FatigueDetector] fatigue=%s conf=%.2f level=%s eyes_closed=%s mouth_open=%s',
            final_detected, final_conf, details['fatigue_level'],
            details.get('eyes_closed'), details.get('mouth_open'),
        )

        return final_detected, round(final_conf, 4), details

    except Exception:
        logger.error('[FatigueDetector] Erreur inattendue:\n%s', traceback.format_exc())
        return False, 0.0, {'error': traceback.format_exc()}
