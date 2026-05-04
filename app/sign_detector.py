import os
import time
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np

# Checkpoints saved with NumPy 2 may reference numpy._core while older NumPy
# exposes the same internals under numpy.core.
if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)
    for name in ('multiarray', 'numeric', 'fromnumeric', 'umath'):
        if hasattr(np.core, name):
            sys.modules.setdefault(f'numpy._core.{name}', getattr(np.core, name))

# ── Path Configuration ──────────────────────────────────────
# Points to the root 'models/' folder, exactly like ppe_detector.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'models'))

# ── Mapping Categories ──────────────────────────────────────
CATEGORY_NAMES = {
    'E': 'Safe Condition',
    'F': 'Fire Protection',
    'P': 'Prohibition',
    'M': 'Mandatory',
    'W': 'Warning'
}

# ── Device Configuration ────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Preprocessing Pipeline ──────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class SignDetector:
    def __init__(self):
        self.device = DEVICE
        self.router = None
        self.defect_models = {}
        self.idx_to_category = {}
        self.load_models()

    def _create_resnet(self, num_classes):
        """Creates ResNet-50 with a custom head matching the training architecture."""
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        return model

    def load_models(self):
        """Loads all 6 models once at startup."""
        # ── STAGE 1: Load Router ──
        router_path = os.path.join(MODELS_DIR, 'resnet50_router.pt')
        if os.path.exists(router_path):
            try:
                checkpoint = torch.load(router_path, map_location=self.device, weights_only=False)
                self.router = self._create_resnet(5)
                self.router.load_state_dict(checkpoint['state_dict'])
                self.router.to(self.device).eval()
                
                # Handle idx mapping (PyTorch sometimes converts int keys to strings in dicts)
                self.idx_to_category = checkpoint.get('idx_to_category', {0: 'E', 1: 'F', 2: 'P', 3: 'M', 4: 'W'})
                print(f"[SignDetector] OK: Router loaded from {router_path}")
            except Exception as e:
                print(f"[SignDetector] ERROR: Failed to load Router: {e}")
        else:
            print(f"[SignDetector] WARNING: Router model not found at {router_path}")

        # ── STAGE 2: Load Defect Models ──
        for cat in ['E', 'F', 'P', 'M', 'W']:
            path = os.path.join(MODELS_DIR, f'resnet50_{cat}.pt')
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    model = self._create_resnet(1)
                    model.load_state_dict(checkpoint['state_dict'])
                    model.to(self.device).eval()
                    
                    # Extract the specific threshold saved in the checkpoint
                    threshold = float(checkpoint.get('threshold', 0.5))
                    
                    self.defect_models[cat] = {
                        'model': model,
                        'threshold': threshold
                    }
                    print(f"[SignDetector] OK: Defect model for {cat} loaded (Threshold: {threshold:.3f})")
                except Exception as e:
                    print(f"[SignDetector] ERROR: Failed to load Defect model for {cat}: {e}")
            else:
                print(f"[SignDetector] WARNING: Defect model for {cat} not found at {path}")

    def predict(self, pil_image):
        """Runs the two-stage pipeline on a PIL Image."""
        start_time = time.time()
        
        if self.router is None:
            raise RuntimeError("Router model is not loaded.")
        
        # Preprocess
        img_tensor = TRANSFORM(pil_image).unsqueeze(0).to(self.device)
        
        # ── STAGE 1: Category Router ──
        with torch.no_grad():
            outputs = self.router(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            
            # Safely get category string from prediction index
            idx_key = pred_idx.item()
            category = self.idx_to_category.get(str(idx_key), self.idx_to_category.get(idx_key, 'Unknown'))
            cat_confidence = conf.item()
        
        # ── STAGE 2: Defect Detector ──
        if category not in self.defect_models:
            raise RuntimeError(f"Defect model for category '{category}' is not loaded.")
            
        defect_model_info = self.defect_models[category]
        defect_model = defect_model_info['model']
        threshold = defect_model_info['threshold']
        
        with torch.no_grad():
            logit = defect_model(img_tensor)
            defect_score = torch.sigmoid(logit).item()
        
        verdict = "DEFECTIVE" if defect_score >= threshold else "NORMAL"
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            "category": category,
            "category_name": CATEGORY_NAMES.get(category, "Unknown"),
            "cat_confidence": round(cat_confidence, 4),
            "defect_score": round(defect_score, 4),
            "threshold": round(threshold, 4),
            "verdict": verdict,
            "is_defective": verdict == "DEFECTIVE",
            "latency_ms": round(latency_ms, 2)
        }

# ── Module-level instantiation (Runs once at server startup) ──
sign_detector = SignDetector()

def detect_sign_in_image(pil_image):
    return sign_detector.predict(pil_image)
