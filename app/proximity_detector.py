#!/usr/bin/env python
"""
proximity_detector.py
=====================
Détection de proximité Ouvrier / Engin de chantier
Modèle : YOLO11l  —  Classes : worker (0) · machine (1)

Pipeline visuel :
  1. Zones concentriques elliptiques au sol (fond)
  2. Bounding boxes OUVRIER (vert) / ENGIN (couleur selon danger)
  3. Flèche de distance colorée avec métrage
  4. Bannière d'alerte pleine largeur (haut)
  5. Barre d'information (bas)
"""

import os
import math
import base64

import cv2
import numpy as np

# ── Ultralytics YOLO ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[ProximityDetector] WARNING: ultralytics non installé → mode simulation")

# ── Chemins modèle ────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH          = os.path.join(_BASE, '..', 'models', 'proximity.pt')
MODEL_PATH_FALLBACK = os.path.join(_BASE, '..', 'models', 'proximity_detection.pt')

# ── Calibration & seuils ─────────────────────────────────────────────────────
HAUTEUR_REF_PIXELS = 140      # hauteur moyenne ouvrier en pixels
HAUTEUR_REF_METRES = 1.75     # hauteur réelle en mètres
PIXELS_PAR_METRE   = HAUTEUR_REF_PIXELS / HAUTEUR_REF_METRES   # 80.0 px/m

SEUIL_CRITIQUE  = 2.0   # mètres — danger immédiat
SEUIL_ALERTE    = 5.0   # mètres — zone d'alerte
SEUIL_VIGILANCE = 6.0   # mètres — zone de vigilance

RATIO_PERSPECTIVE = 0.20   # aplatissement ellipse au sol


# ═════════════════════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE
# ═════════════════════════════════════════════════════════════════════════════

class ProximityDetector:
    """Détecteur de proximité ouvrier / engin basé sur YOLO11l."""

    def __init__(self, model_path: str = None):
        self.model      = None
        self.model_path = os.path.abspath(model_path or MODEL_PATH)

        if not YOLO_AVAILABLE:
            return

        # Fallback si proximity.pt absent
        if not os.path.exists(self.model_path):
            fb = os.path.abspath(MODEL_PATH_FALLBACK)
            if os.path.exists(fb):
                self.model_path = fb
                print(f"[ProximityDetector] Fallback → {self.model_path}")

        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                print(f"[ProximityDetector] ✅ Modèle chargé : {self.model_path}")
            except Exception as exc:
                print(f"[ProximityDetector] ❌ Erreur chargement : {exc}")
        else:
            print(f"[ProximityDetector] ❌ Modèle introuvable : {self.model_path}")

    # ──────────────────────────────────────────────────────────────────────────
    #  DÉTECTION
    # ──────────────────────────────────────────────────────────────────────────

    def detect_proximity(self, frame: np.ndarray,
                         threshold_distance: float = 100.0):
        """
        Détecte les proximités dangereuses.

        Returns
        -------
        (proximity_detected: bool, confidence: float, details: dict)
        """
        if self.model is None:
            return self._simulate(frame, threshold_distance)

        try:
            results = self.model(frame, conf=0.25, verbose=False)
        except Exception as exc:
            print(f"[ProximityDetector] Erreur inférence : {exc}")
            return self._simulate(frame, threshold_distance)

        detections = self._extract(results)
        return self._analyze(detections, frame.shape)

    # ──────────────────────────────────────────────────────────────────────────
    #  ANNOTATION VISUELLE
    # ──────────────────────────────────────────────────────────────────────────

    def annotate_frame(self, frame: np.ndarray, details: dict) -> np.ndarray:
        """
        Dessine sur la frame toutes les annotations visuelles :
          • Zones concentriques elliptiques au sol
          • Bounding boxes OUVRIER / ENGIN
          • Flèche de distance colorée + métrage
          • Bannière d'alerte pleine largeur
          • Barre d'info bas de frame
        """
        H, W = frame.shape[:2]

        # ── Palette ──────────────────────────────────────────────────────────
        C = {
            'worker'    : (50,  220,  50),
            'safe'      : (50,  220,  50),
            'vigilance' : (0,   210, 210),
            'alert'     : (0,   140, 255),
            'critical'  : (0,    30, 220),
            'white'     : (255, 255, 255),
            'black'     : (0,     0,   0),
            'dark'      : (20,   20,  20),
        }

        # ── Helpers ──────────────────────────────────────────────────────────

        def sev_color(sev):
            return C.get(sev, C['safe'])

        def shrink(box):
            x1, y1, x2, y2 = [int(v) for v in box]
            w, h = x2 - x1, y2 - y1
            s = max(w, h)
            r = 0.13 if s > 300 else 0.10 if s > 150 else 0.08
            dx, dy = int(w * r), int(h * r)
            return [x1 + dx, y1 + dy, x2 - dx, y2 - dy]

        def eight_pts(box):
            x1, y1, x2, y2 = [int(v) for v in box]
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            return [(x1,y1),(mx,y1),(x2,y1),(x2,my),
                    (x2,y2),(mx,y2),(x1,y2),(x1,my)]

        def worker_ref(box):
            x1, y1, x2, y2 = [int(v) for v in box]
            return ((x1 + x2) // 2, y2)

        def min_dist_to_machine(pt, box_s):
            pts  = eight_pts(box_s)
            best = min(pts, key=lambda p: math.hypot(pt[0]-p[0], pt[1]-p[1]))
            return math.hypot(pt[0]-best[0], pt[1]-best[1]), best

        def px_to_metres(dpx, box_w):
            x1, y1, x2, y2 = [int(v) for v in box_w]
            h   = max(1, y2 - y1)
            ppm = PIXELS_PAR_METRE / (HAUTEUR_REF_PIXELS / h)
            return dpx / ppm

        def get_sev(dm):
            if dm < SEUIL_CRITIQUE:  return 'critical'
            if dm < SEUIL_ALERTE:    return 'alert'
            if dm < SEUIL_VIGILANCE: return 'vigilance'
            return 'safe'

        # ── Fn : Bounding box avec coins en L ────────────────────────────────
        def draw_bb(img, box, color, label):
            x1, y1, x2, y2 = [int(v) for v in box]
            corner = min(18, (x2-x1)//4, (y2-y1)//4)
            lw = 2

            # Rectangle principal (semi-épais)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)

            # Coins en L (plus épais)
            for (px, py), (dx, dy) in [
                ((x1, y1), (+corner, 0)),
                ((x1, y1), (0, +corner)),
                ((x2, y1), (-corner, 0)),
                ((x2, y1), (0, +corner)),
                ((x1, y2), (+corner, 0)),
                ((x1, y2), (0, -corner)),
                ((x2, y2), (-corner, 0)),
                ((x2, y2), (0, -corner)),
            ]:
                cv2.line(img, (px, py), (px+dx, py+dy), color, lw + 2)

            # Étiquette
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
            pad = 5
            ty = max(y1 - 1, th + pad * 2)
            cv2.rectangle(img,
                          (x1, ty - th - pad * 2),
                          (x1 + tw + pad * 2, ty),
                          color, -1)
            cv2.putText(img, label,
                        (x1 + pad, ty - pad),
                        font, scale, C['white'], thick)

            # Point référence bas-centre
            ref = worker_ref(box)
            cv2.circle(img, ref, 5, color, -1)

        # ── Fn : Zones concentriques elliptiques ──────────────────────────────
        def draw_zones(img, box_s, sev):
            x1m, y1m, x2m, y2m = [int(v) for v in box_s]
            cx, cy = (x1m + x2m) // 2, y2m   # ancrage sol

            order = ['safe', 'vigilance', 'alert', 'critical']
            sev_rank = order.index(sev) if sev in order else 0

            for seuil, zone_sev, a_idle, a_active, lw in [
                (SEUIL_VIGILANCE, 'vigilance', 0.10, 0.30, 1),
                (SEUIL_ALERTE,    'alert',     0.12, 0.40, 1),
                (SEUIL_CRITIQUE,  'critical',  0.18, 0.55, 2),
            ]:
                col   = sev_color(zone_sev)
                px    = int(seuil * PIXELS_PAR_METRE)
                rx    = px
                ry    = int(px * RATIO_PERSPECTIVE)
                z_rank = order.index(zone_sev)
                active = (sev_rank >= z_rank)
                alpha  = a_active if active else a_idle

                ov = img.copy()
                cv2.ellipse(ov, (cx, cy), (rx, ry), 0, 0, 180, col, -1)
                cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)
                cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 180, col, lw)

                # Label de zone discret
                lx = min(W - 60, cx + int(rx * 0.55))
                ly = cy + int(ry * 0.6) + 14
                if 0 < lx < W - 40 and 0 < ly < H:
                    z_labels = {
                        'vigilance': f'<{SEUIL_VIGILANCE:.0f}m',
                        'alert':     f'<{SEUIL_ALERTE:.0f}m',
                        'critical':  f'<{SEUIL_CRITIQUE:.0f}m',
                    }
                    zt = z_labels.get(zone_sev, '')
                    cv2.putText(img, zt, (lx, ly),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.40, col, 1)

        # ── Fn : Flèche de distance ────────────────────────────────────────────
        def draw_arrow(img, pt_w, pt_m, dm, sev):
            col = sev_color(sev)
            pw  = (int(pt_w[0]), int(pt_w[1]))
            pm  = (int(pt_m[0]), int(pt_m[1]))

            # Ligne épaisse
            cv2.line(img, pw, pm, col, 3, cv2.LINE_AA)
            # Pointe flèche vers la machine
            cv2.arrowedLine(img, pw, pm, col, 3,
                            cv2.LINE_AA, tipLength=0.18)

            # Points extrémités
            cv2.circle(img, pw, 7, C['worker'],  -1)
            cv2.circle(img, pw, 7, C['white'],    1)
            cv2.circle(img, pm, 7, col,          -1)
            cv2.circle(img, pm, 7, C['white'],    1)

            # Étiquette distance (grande, lisible)
            mid   = ((pw[0] + pm[0]) // 2, (pw[1] + pm[1]) // 2)
            lbl   = f'{dm:.1f} m'
            font  = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.85
            thick = 2
            (tw, th), _ = cv2.getTextSize(lbl, font, scale, thick)
            pad = 8
            # Fond blanc avec bordure colorée
            cv2.rectangle(img,
                          (mid[0] - tw//2 - pad, mid[1] - th - pad),
                          (mid[0] + tw//2 + pad, mid[1] + pad),
                          C['white'], -1)
            cv2.rectangle(img,
                          (mid[0] - tw//2 - pad, mid[1] - th - pad),
                          (mid[0] + tw//2 + pad, mid[1] + pad),
                          col, 2)
            cv2.putText(img, lbl,
                        (mid[0] - tw//2, mid[1]),
                        font, scale, col, thick)

        # ── Fn : Bannière d'alerte ─────────────────────────────────────────────
        def draw_banner(img, sev, dm, w_idx, m_idx):
            if sev == 'safe':
                return

            cfg = {
    'critical': {
        'bg'    : (0,  0, 200),
        'height': 85,
        'title' : '!  DANGER CRITIQUE  !',
        'msg'   : f'OUVRIER #{w_idx+1} TROP PROCHE DE L ENGIN #{m_idx+1}',
        'sub'   : f'Distance : {dm:.1f} m   |   Seuil de securite : {SEUIL_CRITIQUE:.0f} m',
    },
    'alert': {
        'bg'    : (0, 100, 220),
        'height': 60,
        'title' : '!  ZONE D ALERTE  !',
        'msg'   : f'Ouvrier #{w_idx+1} a {dm:.1f} m de l engin #{m_idx+1}',
        'sub'   : f'Seuil alerte : {SEUIL_ALERTE:.0f} m',
    },
    'vigilance': {
        'bg'    : (0, 145, 145),
        'height': 50,
        'title' : 'VIGILANCE',
        'msg'   : f'Ouvrier #{w_idx+1} a {dm:.1f} m - Restez vigilant',
        'sub'   : '',
    },
}.get(sev)

            if not cfg:
                return

            bh   = cfg['height']
            bg   = cfg['bg']
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Fond de bannière
            cv2.rectangle(img, (0, 0), (W, bh), bg, -1)
            # Ligne séparatrice
            cv2.line(img, (0, bh), (W, bh), C['white'], 2)

            # Titre (grand, centré, gras simulé)
            ts = 0.82 if sev == 'critical' else 0.74
            (tw, _), _ = cv2.getTextSize(cfg['title'], font, ts, 2)
            cv2.putText(img, cfg['title'],
                        ((W - tw) // 2, 28),
                        font, ts, C['white'], 2)

            # Message (moyen)
            (tw2, _), _ = cv2.getTextSize(cfg['msg'], font, 0.60, 2)
            cv2.putText(img, cfg['msg'],
                        ((W - tw2) // 2, 46),
                        font, 0.60, (230, 230, 255), 2)

            # Sous-message (petite police, seulement critical)
            if sev == 'critical' and cfg['sub']:
                (tw3, _), _ = cv2.getTextSize(cfg['sub'], font, 0.52, 1)
                cv2.putText(img, cfg['sub'],
                            ((W - tw3) // 2, 64),
                            font, 0.52, (255, 255, 180), 1)

        # ── Fn : Barre info bas ────────────────────────────────────────────────
        def draw_info_bar(img, n_w, n_m, n_alerts, top_sev):
            bh  = 34
            ov  = img.copy()
            cv2.rectangle(ov, (0, H - bh), (W, H), C['dark'], -1)
            cv2.addWeighted(ov, 0.78, img, 0.22, 0, img)
            cv2.line(img, (0, H - bh), (W, H - bh), (60, 60, 60), 1)

            dot_col = sev_color(top_sev)
            cy_dot  = H - bh // 2
            cv2.circle(img, (14, cy_dot), 7, dot_col, -1)
            cv2.circle(img, (14, cy_dot), 7, C['white'], 1)

            txt = (f'   Ouvriers : {n_w}      '
                   f'Engins : {n_m}      '
                   f'Alertes : {n_alerts}')
            cv2.putText(img, txt,
                        (26, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (180, 180, 180), 1)

            brand = 'SafetyVision - Proximite Homme-Machine'

            (tw, _), _ = cv2.getTextSize(
                brand, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.putText(img, brand,
                        (W - tw - 10, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (90, 90, 90), 1)

        # ══════════════════════════════════════════════════════════════════════
        #  PIPELINE DE RENDU
        # ══════════════════════════════════════════════════════════════════════

        detections = details.get('detections', [])
        incidents  = details.get('incident_logs', [])

        workers  = [d for d in detections
                    if d.get('class_name') == 'worker' or d.get('class') == 0]
        machines = [d for d in detections
                    if d.get('class_name') != 'worker' and d.get('class') != 0]

        # Pire sévérité par machine
        sev_order  = ['safe', 'vigilance', 'alert', 'critical']
        mach_sev   = {i: 'safe' for i in range(len(machines))}

        for inc in incidents:
            for i, m in enumerate(machines):
                if m.get('center') == inc['machine'].get('center'):
                    cur = mach_sev[i]
                    new = inc['severity']
                    if sev_order.index(new) > sev_order.index(cur):
                        mach_sev[i] = new

        top_sev = max(mach_sev.values(),
                      key=lambda s: sev_order.index(s),
                      default='safe')

        # ── ÉTAPE 1 : Zones de danger (couche de fond) ────────────────────────
        for i, machine in enumerate(machines):
            draw_zones(frame, shrink(machine['bbox']), mach_sev[i])

        # ── ÉTAPE 2 : Bounding boxes ──────────────────────────────────────────
        for w in workers:
            draw_bb(frame, w['bbox'], C['worker'], 'OUVRIER')

        for i, m in enumerate(machines):
            col = sev_color(mach_sev[i]) if mach_sev[i] != 'safe' \
                  else (0, 165, 255)
            draw_bb(frame, m['bbox'], col, 'ENGIN')

        # ── ÉTAPE 3 : Flèches de distance ─────────────────────────────────────
        n_alerts        = 0
        banner_drawn    = False

        for w_idx, worker in enumerate(workers):
            pt_w      = worker_ref(worker['bbox'])
            best_dm   = float('inf')
            best_pt_m = None
            best_sev  = 'safe'
            best_m    = 0

            for m_idx, machine in enumerate(machines):
                bs         = shrink(machine['bbox'])
                dpx, pt_m  = min_dist_to_machine(pt_w, bs)
                dm         = px_to_metres(dpx, worker['bbox'])
                sv         = get_sev(dm)

                if dm < best_dm:
                    best_dm   = dm
                    best_pt_m = pt_m
                    best_sev  = sv
                    best_m    = m_idx

            if best_pt_m is not None:
                draw_arrow(frame, pt_w, best_pt_m, best_dm, best_sev)
                if best_sev != 'safe':
                    n_alerts += 1
                if not banner_drawn and best_sev != 'safe':
                    draw_banner(frame, best_sev, best_dm, w_idx, best_m)
                    banner_drawn = True

        # ── ÉTAPE 4 : Barre info ──────────────────────────────────────────────
        draw_info_bar(frame, len(workers), len(machines), n_alerts, top_sev)

        return frame

    # ──────────────────────────────────────────────────────────────────────────
    #  EXTRACTION DES DÉTECTIONS
    # ──────────────────────────────────────────────────────────────────────────

    def _extract(self, results) -> list:
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls  = int(box.cls[0].cpu().numpy())
                detections.append({
                    'bbox'      : [float(x1), float(y1),
                                   float(x2), float(y2)],
                    'center'    : [float((x1+x2)/2), float((y1+y2)/2)],
                    'confidence': conf,
                    'class'     : cls,
                    'class_name': 'worker' if cls == 0 else 'machine',
                })
        return detections

    # ──────────────────────────────────────────────────────────────────────────
    #  ANALYSE DES PROXIMITÉS
    # ──────────────────────────────────────────────────────────────────────────

    def _analyze(self, detections: list, frame_shape):
        workers  = [d for d in detections if d['class'] == 0]
        machines = [d for d in detections if d['class'] == 1]

        incident_logs = []
        max_conf      = 0.0

        for worker in workers:
            x1, y1, x2, y2 = [int(v) for v in worker['bbox']]
            h   = max(1, y2 - y1)
            ppm = PIXELS_PAR_METRE / (HAUTEUR_REF_PIXELS / h)
            pt_w = ((x1 + x2) // 2, y2)

            for machine in machines:
                # Shrink machine BB
                mx1, my1, mx2, my2 = [int(v) for v in machine['bbox']]
                mw, mh = mx2 - mx1, my2 - my1
                ms = max(mw, mh)
                mr = 0.13 if ms > 300 else 0.10 if ms > 150 else 0.08
                mdx, mdy = int(mw * mr), int(mh * mr)
                bs = [mx1+mdx, my1+mdy, mx2-mdx, my2-mdy]

                # 8 points
                bx1,by1,bx2,by2 = bs
                bmx,bmy = (bx1+bx2)//2, (by1+by2)//2
                pts = [(bx1,by1),(bmx,by1),(bx2,by1),(bx2,bmy),
                       (bx2,by2),(bmx,by2),(bx1,by2),(bx1,bmy)]

                best = min(pts,
                           key=lambda p: math.hypot(pt_w[0]-p[0],
                                                    pt_w[1]-p[1]))
                dpx = math.hypot(pt_w[0]-best[0], pt_w[1]-best[1])
                dm  = dpx / ppm

                sev = ('critical'  if dm < SEUIL_CRITIQUE  else
                       'alert'     if dm < SEUIL_ALERTE     else
                       'vigilance' if dm < SEUIL_VIGILANCE  else
                       'safe')

                if sev != 'safe':
                    conf = min(1.0, (SEUIL_ALERTE - dm) / SEUIL_ALERTE)
                    max_conf = max(max_conf, conf)
                    incident_logs.append({
                        'worker'       : worker,
                        'machine'      : machine,
                        'distance_m'   : round(dm, 2),
                        'severity'     : sev,
                        'confidence'   : round(conf, 3),
                        'worker_point' : list(pt_w),
                    })

        proximity_detected = len(incident_logs) > 0
        details = {
            'detections'      : detections,
            'detections_count': len(detections),
            'workers_count'   : len(workers),
            'machines_count'  : len(machines),
            'incident_logs'   : incident_logs,
            'model'           : 'YOLO11l_Proximity',
            'thresholds'      : {
                'critique_m'  : SEUIL_CRITIQUE,
                'alerte_m'    : SEUIL_ALERTE,
                'vigilance_m' : SEUIL_VIGILANCE,
            },
        }
        return proximity_detected, max_conf, details

    # ──────────────────────────────────────────────────────────────────────────
    #  SIMULATION (mode sans modèle)
    # ──────────────────────────────────────────────────────────────────────────

    def _simulate(self, frame: np.ndarray, _threshold):
        H, W = frame.shape[:2]
        detections = []

        if np.random.random() > 0.5:
            # Worker fictif
            detections.append({
                'bbox'      : [float(W*0.1), float(H*0.2),
                               float(W*0.25), float(H*0.75)],
                'center'    : [float(W*0.175), float(H*0.475)],
                'confidence': 0.82,
                'class'     : 0,
                'class_name': 'worker',
            })
            # Machine fictive
            detections.append({
                'bbox'      : [float(W*0.35), float(H*0.15),
                               float(W*0.85), float(H*0.85)],
                'center'    : [float(W*0.60), float(H*0.50)],
                'confidence': 0.91,
                'class'     : 1,
                'class_name': 'machine',
            })

        return self._analyze(detections, frame.shape)


# ── Instance globale ──────────────────────────────────────────────────────────
proximity_detector = ProximityDetector()


# ── Fonction utilitaire exposée à views.py ───────────────────────────────────
def detect_proximity_in_frame(frame: np.ndarray,
                               threshold_distance: float = 100.0):
    """Point d'entrée principal appelé depuis views.py."""
    return proximity_detector.detect_proximity(frame, threshold_distance)


# ── Test rapide ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Test ProximityDetector…")
    test = np.zeros((480, 640, 3), dtype=np.uint8)
    test[:] = [60, 60, 60]
    detected, conf, details = detect_proximity_in_frame(test)
    annotated = proximity_detector.annotate_frame(test.copy(), details)
    cv2.imwrite('test_proximity_output.jpg', annotated)
    print(f"Détecté : {detected} | Conf : {conf:.2%}")
    print(f"Workers : {details['workers_count']} | "
          f"Machines : {details['machines_count']}")
    print("Image sauvegardée → test_proximity_output.jpg")