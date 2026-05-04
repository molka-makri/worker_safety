#!/usr/bin/env python
"""
Module de détection de proximité entre travailleurs et machines de construction
Utilise YOLOv11 pour détecter les objets et calculer les distances
"""
import cv2
import numpy as np
import math
import os
from typing import Tuple, Dict, Any, List

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Ultralytics/YOLO non disponible, utilisation de la simulation")

class ProximityDetector:
    """Détecteur de proximité utilisant YOLOv11"""

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), '..', 'models', 'model_axe1_F.pt')

        if YOLO_AVAILABLE and os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                print(f"Modèle YOLO chargé depuis {self.model_path}")
            except Exception as e:
                print(f"Erreur lors du chargement du modèle: {e}")
                self.model = None
        else:
            print("Modèle YOLO non disponible, utilisation de la simulation")

    def detect_proximity(self, frame: np.ndarray, threshold_distance: float = 100.0) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Détecte les proximités dangereuses entre travailleurs et machines

        Args:
            frame: Image OpenCV (BGR)
            threshold_distance: Distance seuil en pixels pour considérer une proximité dangereuse

        Returns:
            Tuple (proximity_detected, confidence, details)
        """
        if not YOLO_AVAILABLE or self.model is None:
            return self._simulate_detection(frame, threshold_distance)

        try:
            # Effectuer l'inférence YOLO
            results = self.model(frame, conf=0.5)

            if len(results) == 0 or len(results[0].boxes) == 0:
                return False, 0.0, {
                    'error': 'Aucun objet détecté',
                    'detections_count': 0,
                    'model': 'YOLOv11_Proximity',
                    'processing_method': 'object_detection'
                }

            # Extraire les détections
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())

                    # Calculer le centre de la boîte
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'center': [center_x, center_y],
                        'confidence': conf,
                        'class': cls,
                        'class_name': self._get_class_name(cls)
                    })

            # Analyser les proximités
            proximity_detected, confidence, details = self._analyze_proximities(detections, threshold_distance, frame.shape)

            return proximity_detected, confidence, details

        except Exception as e:
            print(f"Erreur lors de la détection YOLO: {e}")
            return self._simulate_detection(frame, threshold_distance)

    def _get_class_name(self, class_id: int) -> str:
        """Mappe l'ID de classe au nom"""
        class_names = {
            0: 'worker',
            1: 'machine',
            2: 'excavator',
            3: 'truck',
            4: 'crane',
            5: 'person'
        }
        return class_names.get(class_id, f'class_{class_id}')

    def _analyze_proximities(self, detections: List[Dict], threshold_distance: float, frame_shape) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Analyse les détections pour identifier les proximités dangereuses
        """
        workers = [d for d in detections if 'worker' in d['class_name'].lower() or 'person' in d['class_name'].lower()]
        machines = [d for d in detections if any(m in d['class_name'].lower() for m in ['machine', 'excavator', 'truck', 'crane'])]

        proximity_alerts = []
        max_confidence = 0.0

        # Calculer les distances entre travailleurs et machines
        for worker in workers:
            for machine in machines:
                distance = self._calculate_distance(worker['center'], machine['center'])

                if distance < threshold_distance:
                    confidence = min(1.0, (threshold_distance - distance) / threshold_distance)
                    max_confidence = max(max_confidence, confidence)

                    proximity_alerts.append({
                        'worker': worker,
                        'machine': machine,
                        'distance': distance,
                        'confidence': confidence,
                        'severity': 'high' if distance < threshold_distance * 0.5 else 'medium'
                    })

        proximity_detected = len(proximity_alerts) > 0

        details = {
            'detections_count': len(detections),
            'workers_count': len(workers),
            'machines_count': len(machines),
            'proximity_alerts': proximity_alerts,
            'threshold_distance': threshold_distance,
            'model': 'YOLOv11_Proximity',
            'processing_method': 'object_detection_distance_analysis',
            'frame_shape': frame_shape
        }

        return proximity_detected, max_confidence, details

    def _calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """Calcule la distance euclidienne entre deux points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def _simulate_detection(self, frame: np.ndarray, threshold_distance: float) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Simulation de détection quand YOLO n'est pas disponible
        """
        height, width = frame.shape[:2]

        # Simulation basée sur des critères simples
        # Créer des détections fictives
        workers = []
        machines = []

        # Simuler quelques travailleurs et machines
        if np.random.random() > 0.7:  # 30% de chance d'avoir des détections
            num_workers = np.random.randint(1, 4)
            num_machines = np.random.randint(1, 3)

            for i in range(num_workers):
                workers.append({
                    'bbox': [np.random.randint(0, width//2), np.random.randint(0, height//2),
                            np.random.randint(width//2, width), np.random.randint(height//2, height)],
                    'center': [np.random.randint(width//4, 3*width//4), np.random.randint(height//4, 3*height//4)],
                    'confidence': np.random.uniform(0.6, 0.9),
                    'class': 0,
                    'class_name': 'worker'
                })

            for i in range(num_machines):
                machines.append({
                    'bbox': [np.random.randint(0, width//2), np.random.randint(0, height//2),
                            np.random.randint(width//2, width), np.random.randint(height//2, height)],
                    'center': [np.random.randint(width//4, 3*width//4), np.random.randint(height//4, 3*height//4)],
                    'confidence': np.random.uniform(0.6, 0.9),
                    'class': 1,
                    'class_name': 'machine'
                })

        # Analyser les proximités simulées
        proximity_alerts = []
        max_confidence = 0.0

        for worker in workers:
            for machine in machines:
                distance = self._calculate_distance(worker['center'], machine['center'])
                if distance < threshold_distance:
                    confidence = min(1.0, (threshold_distance - distance) / threshold_distance)
                    max_confidence = max(max_confidence, confidence)

                    proximity_alerts.append({
                        'worker': worker,
                        'machine': machine,
                        'distance': distance,
                        'confidence': confidence,
                        'severity': 'high' if distance < threshold_distance * 0.5 else 'medium'
                    })

        proximity_detected = len(proximity_alerts) > 0

        details = {
            'detections_count': len(workers) + len(machines),
            'workers_count': len(workers),
            'machines_count': len(machines),
            'proximity_alerts': proximity_alerts,
            'threshold_distance': threshold_distance,
            'model': 'YOLOv11_Proximity_Simulation',
            'processing_method': 'simulated_object_detection',
            'frame_shape': frame.shape,
            'simulation_mode': True
        }

        return proximity_detected, max_confidence, details

    def annotate_frame(self, frame: np.ndarray, details: dict) -> np.ndarray:

        PIXELS_PAR_METRE  = 140.0 / 1.75   # = 80.0
        RATIO_PERSPECTIVE = 0.20
        SEUIL_CRITIQUE    = 2.0
        SEUIL_ALERTE      = 5.0
        SEUIL_VIGILANCE   = 6.0

        COLORS = {
            'worker'    : (0,   200,   0),
            'machine'   : (0,   165, 255),
            'critical'  : (0,     0, 220),
            'alert'     : (0,   165, 255),
            'vigilance' : (0,   220, 220),
            'safe'      : (0,   200,   0),
        }

        def shrink_box(box):
            x1,y1,x2,y2 = [int(v) for v in box]
            w,h  = x2-x1, y2-y1
            size = max(w,h)
            r    = 0.13 if size>300 else 0.10 if size>150 else 0.08
            dx,dy = int(w*r), int(h*r)
            return [x1+dx, y1+dy, x2-dx, y2-dy]

        def get_8_points(box):
            x1,y1,x2,y2 = [int(v) for v in box]
            mx,my = (x1+x2)//2, (y1+y2)//2
            return [(x1,y1),(mx,y1),(x2,y1),(x2,my),
                    (x2,y2),(mx,y2),(x1,y2),(x1,my)]

        def worker_ref_point(box):
            x1,y1,x2,y2 = [int(v) for v in box]
            return ((x1+x2)//2, y2)

        def min_dist_and_point(pt_w, box_shrunk):
            pts = get_8_points(box_shrunk)
            best = min(pts, key=lambda p: math.hypot(pt_w[0]-p[0], pt_w[1]-p[1]))
            return math.hypot(pt_w[0]-best[0], pt_w[1]-best[1]), best

        def perspective_dist(dist_px, box_worker):
            x1,y1,x2,y2 = [int(v) for v in box_worker]
            h = max(1, y2-y1)
            ratio = 140.0 / h
            ppm   = PIXELS_PAR_METRE / ratio
            return dist_px / ppm

        def get_severity(dist_m):
            if dist_m < SEUIL_CRITIQUE:  return 'critical'
            if dist_m < SEUIL_ALERTE:    return 'alert'
            if dist_m < SEUIL_VIGILANCE: return 'vigilance'
            return 'safe'

        def draw_bb(img, box, label, conf, color):
            x1,y1,x2,y2 = [int(v) for v in box]
            cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
            txt = f'{label} {conf:.2f}'
            (tw,th),_ = cv2.getTextSize(
                txt,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
            cv2.rectangle(img,(x1,y1-th-8),(x1+tw+4,y1),color,-1)
            cv2.putText(img,txt,(x1+2,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            ref = worker_ref_point(box)
            cv2.circle(img, ref, 4, color, -1)

        def draw_zones(img, box_shrunk,
                       w_in_crit, w_in_alert, w_in_vig):
            x1m,y1m,x2m,y2m = [int(v) for v in box_shrunk]
            cx = (x1m+x2m)//2
            cy = y2m
            for seuil, color, a_on, a_off, is_active in [
                (SEUIL_VIGILANCE, (0,220,220), 0.35, 0.12, w_in_vig),
                (SEUIL_ALERTE,    (0,165,255), 0.45, 0.18, w_in_alert),
                (SEUIL_CRITIQUE,  (0,  0,220), 0.55, 0.25, w_in_crit),
            ]:
                px = int(seuil * PIXELS_PAR_METRE)
                rx = px
                ry = int(px * RATIO_PERSPECTIVE)
                alpha = a_on if is_active else a_off
                ov = img.copy()
                cv2.ellipse(ov,(cx,cy),(rx,ry),0,0,180,color,-1)
                cv2.addWeighted(ov,alpha,img,1-alpha,0,img)
                th = 2 if seuil==SEUIL_CRITIQUE else 1
                cv2.ellipse(img,(cx,cy),(rx,ry),0,0,180,color,th)

        def draw_line(img, pt_w, pt_m, dist_m, severity):
            color = COLORS.get(severity, COLORS['safe'])
            cv2.line(img,
                     (int(pt_w[0]),int(pt_w[1])),
                     (int(pt_m[0]),int(pt_m[1])),
                     color, 2, cv2.LINE_AA)
            cv2.circle(img,(int(pt_w[0]),int(pt_w[1])),
                       6, COLORS['worker'], -1)
            cv2.circle(img,(int(pt_m[0]),int(pt_m[1])),
                       6, COLORS['machine'], -1)
            mid = (int((pt_w[0]+pt_m[0])/2),
                   int((pt_w[1]+pt_m[1])/2))
            lbl = f'{dist_m:.1f}m'
            (tw,th),_ = cv2.getTextSize(
                lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(img,
                          (mid[0]-tw//2-3, mid[1]-th-3),
                          (mid[0]+tw//2+3, mid[1]+3),
                          (255,255,255),-1)
            cv2.putText(img, lbl,
                        (mid[0]-tw//2, mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2)

        def draw_banner(img, severity, dist_m, w_id, m_id):
            if severity == 'safe': return
            H,W = img.shape[:2]
            bcolor = {'critical':(0,0,220),
                      'alert':(0,120,255),
                      'vigilance':(0,180,180)}.get(severity,(0,120,255))
            blabel = {'critical':'DANGER',
                      'alert':'ALERTE',
                      'vigilance':'VIGILANCE'}.get(severity,'')
            txt = f'{blabel} : Worker#{w_id} a {dist_m:.1f}m de Machine#{m_id}'
            cv2.rectangle(img,(0,0),(W,50),bcolor,-1)
            (tw,_),_ = cv2.getTextSize(
                txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(img, txt,
                        ((W-tw)//2, 33),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255,255,255), 2)

        # ── Extract detections ────────────────────────────────────────
        detections = details.get('detections', [])
        incidents  = details.get('incident_logs', [])
        workers    = [d for d in detections
                      if d.get('class_name') == 'worker' or
                         d.get('class') == 0]
        machines   = [d for d in detections
                      if d.get('class_name') != 'worker' and
                         d.get('class') != 0]

        # ── Compute per-machine zone activation ───────────────────────
        machine_flags = {}
        for i, m in enumerate(machines):
            machine_flags[i] = {
                'crit' : False,
                'alert': False,
                'vig'  : False
            }

        for inc in incidents:
            for i, m in enumerate(machines):
                if m.get('center') == inc['machine'].get('center'):
                    sev = inc['severity']
                    if sev == 'critical':
                        machine_flags[i]['crit']  = True
                    elif sev == 'alert':
                        machine_flags[i]['alert'] = True
                    elif sev == 'vigilance':
                        machine_flags[i]['vig']   = True

        # ── STEP 1: Draw zones (background) ───────────────────────────
        for i, machine in enumerate(machines):
            box_s = shrink_box(machine['bbox'])
            flags = machine_flags.get(i, {})
            draw_zones(frame, box_s,
                       flags.get('crit',  False),
                       flags.get('alert', False),
                       flags.get('vig',   False))

        # ── STEP 2: Draw bounding boxes ───────────────────────────────
        for w in workers:
            draw_bb(frame, w['bbox'], 'worker',
                    w.get('confidence', 0.0), COLORS['worker'])
        for m in machines:
            draw_bb(frame, m['bbox'], 'machine',
                    m.get('confidence', 0.0), COLORS['machine'])

        # ── STEP 3: Draw distance lines ───────────────────────────────
        max_severity = 'safe'
        for w_id, worker in enumerate(workers):
            pt_w = worker_ref_point(worker['bbox'])
            best_dist_m  = float('inf')
            best_pt_m    = None
            best_severity = 'safe'
            best_m_id    = 0

            for m_id, machine in enumerate(machines):
                box_s = shrink_box(machine['bbox'])
                dist_px, pt_m = min_dist_and_point(pt_w, box_s)
                dist_m  = perspective_dist(dist_px, worker['bbox'])
                severity = get_severity(dist_m)

                if dist_m < best_dist_m:
                    best_dist_m   = dist_m
                    best_pt_m     = pt_m
                    best_severity = severity
                    best_m_id     = m_id

            if best_pt_m is not None:
                draw_line(frame, pt_w, best_pt_m,
                          best_dist_m, best_severity)
                if best_severity != 'safe':
                    draw_banner(frame, best_severity,
                                best_dist_m, w_id, best_m_id)
                    if (best_severity == 'critical' or
                        max_severity != 'critical'):
                        max_severity = best_severity

        return frame

# Instance globale du détecteur
proximity_detector = ProximityDetector()

def detect_proximity_in_frame(frame: np.ndarray, threshold_distance: float = 100.0) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Fonction principale pour détecter les proximités dangereuses
    """
    return proximity_detector.detect_proximity(frame, threshold_distance)

if __name__ == '__main__':
    # Test du module
    print("Test du détecteur de proximité...")

    # Créer une image de test
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:, :] = [200, 200, 200]

    # Ajouter des formes qui pourraient ressembler à des travailleurs et machines
    cv2.rectangle(test_frame, (100, 100), (150, 200), (0, 255, 0), -1)  # Travailleur
    cv2.rectangle(test_frame, (200, 150), (300, 250), (255, 0, 0), -1)  # Machine

    proximity_detected, confidence, details = detect_proximity_in_frame(test_frame, threshold_distance=150.0)

    print(f"Proximité détectée: {proximity_detected}")
    print(f"Confiance: {confidence:.2%}")
    print("Détails:", details)