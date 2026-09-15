"""
FaceTrack Skincare Advisor Engine
=================================
Core inference pipeline:
1. YuNet face detection (OpenCV DNN)
2. MobileNetV2 Skin Type classifier (4 classes)
3. MobileNetV2 Lesion Severity classifier (4 classes)
4. Lifestyle factor fusion (Sleep, Hydration, Stress) -> 7-dim User Profile Vector
5. Cosine Similarity search over 1,138 products in Product Knowledge Base
"""

import os
import sys
from pathlib import Path
import pickle
import numpy as np
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# Labels
SKIN_LABELS = ["Combination", "Dry", "Normal", "Oily"]
LESION_LABELS = ["Normal", "Mild", "Moderate", "High Concern"]

# Resolve paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models" if (PROJECT_ROOT / "models").exists() else BASE_DIR / "models"

YUNET_PATH = MODELS_DIR / "face_detection_yunet.onnx"
SKIN_H5_PATH = MODELS_DIR / "skin_type_model.h5"
LESION_H5_PATH = MODELS_DIR / "lesion_severity_model.h5"
KB_PKL_PATH = MODELS_DIR / "product_kb.pkl"


class SkincareAdvisor:
    def __init__(self):
        self.face_detector = None
        self.skin_model = None
        self.lesion_model = None
        self.kb_vectors = None
        self.kb_meta = []
        self._load_all()

    def _load_all(self):
        # 1. Face Detector (YuNet)
        if YUNET_PATH.exists():
            try:
                self.face_detector = cv2.FaceDetectorYN.create(
                    str(YUNET_PATH),
                    "",
                    (640, 480),
                    score_threshold=0.5,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                print(f"[OK] YuNet face detector loaded: {YUNET_PATH.name}")
            except Exception as e:
                print(f"[WARN] Failed to initialize YuNet detector: {e}")
        else:
            print(f"[WARN] YuNet ONNX model not found at {YUNET_PATH}")

        # 2. Skin Type Model
        if SKIN_H5_PATH.exists():
            try:
                self.skin_model = tf.keras.models.load_model(str(SKIN_H5_PATH), compile=False)
                print(f"[OK] Skin Type model loaded: {SKIN_H5_PATH.name}")
            except Exception as e:
                print(f"[ERROR] Failed to load Skin Type model: {e}")

        # 3. Lesion Severity Model
        if LESION_H5_PATH.exists():
            try:
                self.lesion_model = tf.keras.models.load_model(str(LESION_H5_PATH), compile=False)
                print(f"[OK] Lesion Severity model loaded: {LESION_H5_PATH.name}")
            except Exception as e:
                print(f"[ERROR] Failed to load Lesion Severity model: {e}")

        # 4. Product Knowledge Base
        if KB_PKL_PATH.exists():
            try:
                with open(KB_PKL_PATH, "rb") as f:
                    kb = pickle.load(f)
                self.kb_vectors = kb["vectors"].astype(np.float32)
                self.kb_meta = kb["meta"]
                print(f"[OK] Product KB loaded: {self.kb_vectors.shape[0]} products")
            except Exception as e:
                print(f"[WARN] Failed to load product_kb.pkl: {e}")
        else:
            print(f"[WARN] product_kb.pkl not found at {KB_PKL_PATH}")

    def detect_face(self, bgr_image):
        """Detect face using YuNet. Returns (x, y, w, h) of best face or None."""
        if self.face_detector is None or bgr_image is None:
            return None

        h, w = bgr_image.shape[:2]
        self.face_detector.setInputSize((w, h))
        _, faces = self.face_detector.detect(bgr_image)

        if faces is None or len(faces) == 0:
            return None

        # Sort faces by bounding box area (largest face)
        best_face = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        x = max(0, int(best_face[0]))
        y = max(0, int(best_face[1]))
        fw = min(int(best_face[2]), w - x)
        fh = min(int(best_face[3]), h - y)

        if fw > 20 and fh > 20:
            return (x, y, fw, fh)
        return None

    def preprocess_crop(self, crop_bgr):
        """Resize to 224x224 and apply MobileNetV2 preprocessing."""
        img = cv2.resize(crop_bgr, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        return np.expand_dims(img, 0).astype(np.float32)

    def predict_skin_type(self, img_batch):
        """Returns dict of class probabilities and predicted class."""
        if self.skin_model is None:
            probs = [0.25, 0.25, 0.25, 0.25]
        else:
            raw = self.skin_model.predict(img_batch, verbose=0)[0]
            probs = [float(p) for p in raw]

        pred_idx = int(np.argmax(probs))
        return {
            "predicted": SKIN_LABELS[pred_idx],
            "confidence": round(probs[pred_idx] * 100, 1),
            "probabilities": {
                SKIN_LABELS[i]: round(probs[i] * 100, 1) for i in range(4)
            },
            "raw": probs,
        }

    def predict_lesion_severity(self, img_batch):
        """Returns dict of severity class probabilities and predicted class."""
        if self.lesion_model is None:
            probs = [0.25, 0.25, 0.25, 0.25]
        else:
            raw = self.lesion_model.predict(img_batch, verbose=0)[0]
            probs = [float(p) for p in raw]

        pred_idx = int(np.argmax(probs))
        return {
            "predicted": LESION_LABELS[pred_idx],
            "confidence": round(probs[pred_idx] * 100, 1),
            "probabilities": {
                LESION_LABELS[i]: round(probs[i] * 100, 1) for i in range(4)
            },
            "raw": probs,
        }

    def build_user_vector(self, skin_probs, lesion_probs, sleep_hours=7.0, water_liters=2.0, stress_level=5.0):
        """
        Derive 7-dim profile vector:
        0: Oily  = P(oily) + 0.5 * P(comb)
        1: Dry   = P(dry) + 0.5 * P(comb) + lifestyle
        2: Norm  = P(norm)
        3: Acne  = P(high)*1.0 + P(mod)*0.6 + P(mild)*0.3 + lifestyle
        4: Paula = 0.5 (neutral)
        5: Type  = 0.5 (neutral)
        6: Price = 0.5 (neutral)
        """
        # SKIN_LABELS = ["Combination", "Dry", "Normal", "Oily"]
        # LESION_LABELS = ["Normal", "Mild", "Moderate", "High Concern"]
        p_comb, p_dry, p_norm, p_oily = skin_probs
        p_les_norm, p_mild, p_mod, p_high = lesion_probs

        v = np.zeros(7, dtype=np.float32)
        v[0] = p_oily + 0.5 * p_comb
        v[1] = p_dry + 0.5 * p_comb
        v[2] = p_norm
        v[3] = p_high * 1.0 + p_mod * 0.6 + p_mild * 0.3
        v[4] = 0.5
        v[5] = 0.5
        v[6] = 0.5

        # Lifestyle adjustments
        lifestyle_notes = []
        if sleep_hours < 6.0:
            v[1] += 0.15
            v[3] += 0.10
            lifestyle_notes.append("Sleep < 6h: Boosted dryness & acne concern")
        if water_liters < 1.5:
            v[1] += 0.15
            lifestyle_notes.append("Water < 1.5L: Boosted dryness factor")
        if stress_level > 7.0:
            v[3] += 0.20
            v[0] += 0.10
            lifestyle_notes.append("High Stress (>7): Elevated acne & oiliness factors")

        v = np.clip(v, 0.0, 1.5)
        return v, lifestyle_notes

    def recommend_products(self, user_vec, top_k=5):
        """Perform Cosine Similarity search against product KB."""
        if self.kb_vectors is None or len(self.kb_meta) == 0:
            return []

        u_norm = np.linalg.norm(user_vec)
        if u_norm < 1e-8:
            return []

        kb_norms = np.linalg.norm(self.kb_vectors, axis=1)
        kb_norms[kb_norms < 1e-8] = 1e-8

        # Cosine similarity
        sims = np.dot(self.kb_vectors, user_vec) / (kb_norms * u_norm)
        top_indices = np.argsort(-sims)[:top_k]

        results = []
        for idx in top_indices:
            score = float(sims[idx])
            meta_item = self.kb_meta[idx]
            match_pct = round(max(0.0, min(100.0, score * 100)), 1)
            
            # Format product fields
            name = meta_item.get("product_name", meta_item.get("name", f"Product #{idx}"))
            ptype = meta_item.get("product_type", meta_item.get("type", "Skincare"))
            price = meta_item.get("price", "N/A")
            if price != "N/A" and not str(price).startswith("$"):
                price = f"${price}"
                
            ingreds = meta_item.get("clean_ingreds", meta_item.get("ingredients", ""))
            if isinstance(ingreds, list):
                ingreds = ", ".join(ingreds[:6])
            elif isinstance(ingreds, str) and len(ingreds) > 100:
                ingreds = ingreds[:97] + "..."

            paula_rating = meta_item.get("paula_rating", meta_item.get("rating", "Good"))

            results.append({
                "product_name": name,
                "product_type": ptype,
                "price": price,
                "match_score": match_pct,
                "paula_rating": paula_rating,
                "key_ingredients": ingreds,
            })

        return results

    def analyze(self, image_bgr, sleep_hours=7.0, water_liters=2.0, stress_level=5.0):
        """Full pipeline execution on a BGR image."""
        if image_bgr is None:
            raise ValueError("Invalid image provided")

        h, w = image_bgr.shape[:2]
        face_bbox = self.detect_face(image_bgr)

        if face_bbox is not None:
            x, y, fw, fh = face_bbox
            # Add subtle padding around face
            pad_x = int(fw * 0.1)
            pad_y = int(fh * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + fw + pad_x)
            y2 = min(h, y + fh + pad_y)
            face_crop = image_bgr[y1:y2, x1:x2]
            face_detected = True
        else:
            # Fallback to center crop if face detector didn't find any face
            cx, cy = w // 2, h // 2
            size = min(w, h) // 2
            face_crop = image_bgr[cy - size:cy + size, cx - size:cx + size]
            face_bbox = (cx - size, cy - size, size * 2, size * 2)
            face_detected = False

        # Preprocess and inference
        batch = self.preprocess_crop(face_crop)
        skin_res = self.predict_skin_type(batch)
        lesion_res = self.predict_lesion_severity(batch)

        # 7-dim fusion
        user_vec, lifestyle_notes = self.build_user_vector(
            skin_res["raw"],
            lesion_res["raw"],
            sleep_hours,
            water_liters,
            stress_level
        )

        # Product matching
        recommendations = self.recommend_products(user_vec, top_k=5)

        return {
            "face_detected": face_detected,
            "face_bbox": {
                "x": int(face_bbox[0]),
                "y": int(face_bbox[1]),
                "w": int(face_bbox[2]),
                "h": int(face_bbox[3]),
            },
            "skin_type": skin_res,
            "lesion_severity": lesion_res,
            "user_vector": [round(float(v), 3) for v in user_vec],
            "lifestyle_notes": lifestyle_notes,
            "recommendations": recommendations,
        }
