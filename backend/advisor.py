"""
FaceTrack Skincare Advisor Engine
=================================
Core inference pipeline:
1. YuNet face detection (OpenCV DNN)
2. MobileNetV2 Skin Type classifier (4 classes)
3. MobileNetV2 Lesion Severity classifier (4 classes)
4. Lifestyle factor fusion (Sleep, Hydration, Stress) -> 7-dim User Profile Vector
5. Cosine Similarity search over 1,138 products in Product Knowledge Base
6. Dynamic Skincare Routine & Actionable Suggestions Engine
"""

import os
import sys
from pathlib import Path
import pickle
import numpy as np
import cv2

# NumPy backwards compatibility between 1.x and 2.x
if "numpy._core" not in sys.modules and hasattr(np, "core"):
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray

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

# Fallback catalog in case knowledge base pickle is inaccessible
FALLBACK_CATALOG = [
    {
        "product_name": "CeraVe Hydrating Facial Cleanser with Ceramides",
        "product_type": "Cleanser",
        "price": "$15.99",
        "match_score": 96.2,
        "paula_rating": "Best",
        "key_ingredients": "Ceramides 1, 3, 6-II, Hyaluronic Acid, Glycerin, Niacinamide"
    },
    {
        "product_name": "Paula's Choice 2% BHA Liquid Exfoliant",
        "product_type": "Exfoliant",
        "price": "$34.00",
        "match_score": 93.8,
        "paula_rating": "Best",
        "key_ingredients": "Salicylic Acid (BHA), Green Tea Extract, Methylpropanediol"
    },
    {
        "product_name": "La Roche-Posay Toleriane Double Repair Face Moisturizer",
        "product_type": "Moisturizer",
        "price": "$22.99",
        "match_score": 90.5,
        "paula_rating": "Best",
        "key_ingredients": "Prebiotic Thermal Water, Ceramide-3, Niacinamide, Glycerin"
    },
    {
        "product_name": "The Ordinary Niacinamide 10% + Zinc 1% Blemish Formula",
        "product_type": "Serum",
        "price": "$6.50",
        "match_score": 88.4,
        "paula_rating": "Good",
        "key_ingredients": "Niacinamide (Vitamin B3), Zinc PCA, Tamarindus Seed Gum"
    },
    {
        "product_name": "EltaMD UV Clear Broad-Spectrum SPF 46 Facial Sunscreen",
        "product_type": "Sunscreen",
        "price": "$39.00",
        "match_score": 87.1,
        "paula_rating": "Best",
        "key_ingredients": "Transparent Zinc Oxide 9.0%, Niacinamide 5%, Hyaluronic Acid"
    }
]

CATEGORY_INGREDIENT_DEFAULTS = {
    "cleanser": "Gentle Amino Acid Surfactants, Glycerin, Ceramide NP, Oat Kernel Extract",
    "moisturiser": "Ceramide Complex, Squalane, Hyaluronic Acid, Centella Asiatica",
    "moisturizer": "Ceramide Complex, Squalane, Hyaluronic Acid, Centella Asiatica",
    "serum": "Niacinamide 5%, Sodium Hyaluronate, Panthenol (Provitamin B5), Allantoin",
    "treatment": "Azelaic Acid, Salicylic Acid (BHA), Zinc PCA, Licorice Root Extract",
    "exfoliant": "2% Salicylic Acid, Green Tea Polyphenols, Beta-Glucan",
    "peel": "Lactic Acid, Mandelic Acid, Multi-Molecular Hyaluronic Acid",
    "mask": "Kaolin Clay, Bentonite, Tea Tree Leaf Oil, Witch Hazel",
    "eye care": "Caffeine, Peptides, Hyaluronic Acid, Squalane",
    "sunscreen": "Zinc Oxide (Non-Nano), Niacinamide, Vitamin E Antioxidant"
}


def generate_skincare_suggestions(skin_type: str, lesion_severity: str, lifestyle_notes: list) -> dict:
    """Generate comprehensive, personalized skincare regimen, routine steps, and ingredient advice."""
    morning = []
    evening = []
    actives_to_use = []
    actives_to_avoid = []
    expert_tips = []

    # Skin type specific suggestions
    if skin_type == "Combination":
        morning = [
            {"step": "Step 1: Gentle Cleanse", "action": "Wash with a pH-balanced, non-foaming gel cleanser to refresh without stripping moisture."},
            {"step": "Step 2: Hydrating Toner/Serum", "action": "Apply lightweight Niacinamide (2-5%) or Hyaluronic Acid to balance sebum and hydrate."},
            {"step": "Step 3: Lightweight Moisturizer", "action": "Use an oil-free water-gel on the T-zone and slightly richer lotion on dry cheeks."},
            {"step": "Step 4: Broad-Spectrum Sunscreen", "action": "Finish with fluid SPF 30-50 (chemical or hybrid) that leaves zero greasy residue."}
        ]
        evening = [
            {"step": "Step 1: Clarifying Double Cleanse", "action": "Remove SPF and impurities with micellar water followed by a gentle gel wash."},
            {"step": "Step 2: Targeted Active Treatment", "action": "Apply 2% Salicylic Acid (BHA) 2-3x weekly to decongest pores in the T-zone."},
            {"step": "Step 3: Barrier Recovery Night Cream", "action": "Lock in hydration with a ceramide and centella-infused soothing emulsion."}
        ]
        actives_to_use = ["Niacinamide (Oil Balancing)", "Salicylic Acid (Pore Decongestion)", "Ceramides (Lipid Barrier)", "Hyaluronic Acid (Hydration)"]
        actives_to_avoid = ["Heavy pore-clogging mineral oils", "High-alcohol drying astringents", "Over-aggressive physical apricot/walnut scrubs"]
        expert_tips.append("Zone-treat your face: target oil control strictly on forehead and nose, while protecting drier cheek areas.")

    elif skin_type == "Dry":
        morning = [
            {"step": "Step 1: Lukewarm Water / Milk Cleanse", "action": "Rinse gently with lukewarm water or a soothing non-foaming cleansing cream."},
            {"step": "Step 2: Barrier Hydration Serum", "action": "Apply Hyaluronic Acid and Panthenol on damp skin to attract essential moisture."},
            {"step": "Step 3: Lipid Rich Moisturizer", "action": "Seal in hydration with a rich cream packed with ceramides, squalane, and shea butter."},
            {"step": "Step 4: Nourishing Sunscreen", "action": "Apply a moisturizing broad-spectrum SPF 50 that protects dry skin from trans-epidermal water loss."}
        ]
        evening = [
            {"step": "Step 1: Nourishing Oil / Balm Cleanse", "action": "Dissolve sunscreen and daily pollutants with an emulsifying cleansing balm."},
            {"step": "Step 2: Replenishing Treatment", "action": "Layer a gentle lactic acid or peptide serum to encourage gentle cell turnover without irritation."},
            {"step": "Step 3: Overnight Barrier Ointment / Cream", "action": "Finish with an intensive ceramide barrier cream or overnight sleep mask."}
        ]
        actives_to_use = ["Ceramides 1, 3, 6-II", "Squalane & Plant Lipids", "Hyaluronic Acid Multi-Weight", "Glycerin & Panthenol"]
        actives_to_avoid = ["High foaming SLS sulfates", "Denatured alcohol (SD alcohol)", "Harsh benzoyl peroxide washes without hydration"]
        expert_tips.append("Always apply your hydrators while skin is still damp from cleansing to lock in maximum water.")

    elif skin_type == "Oily":
        morning = [
            {"step": "Step 1: Purifying Foaming Cleanse", "action": "Wash with a gentle amino acid foaming cleanser with low-percentage salicylic acid."},
            {"step": "Step 2: Sebum Regulation Serum", "action": "Apply Niacinamide 10% + Zinc 1% to balance excess sebum and refine enlarged pores."},
            {"step": "Step 3: Oil-Free Matte Moisturizer", "action": "Never skip moisturizer: use an oil-free, non-comedogenic hyaluronic gel."},
            {"step": "Step 4: Mattifying Mineral/Fluid SPF", "action": "Protect with a mattifying, non-greasy SPF 50 with silica to absorb midday shine."}
        ]
        evening = [
            {"step": "Step 1: Deep Pore Cleansing", "action": "Cleanse thoroughly to clear sebum buildup and environmental micro-particles."},
            {"step": "Step 2: BHA / Retinoid Treatment", "action": "Incorporate 2% Salicylic Acid (BHA) or gentle Retinol to clear clogged follicles."},
            {"step": "Step 3: Light Night Hydration", "action": "Finish with a soothing, water-based gel cream containing Centella Asiatica or green tea."}
        ]
        actives_to_use = ["Salicylic Acid (BHA)", "Zinc PCA (Sebum regulator)", "Niacinamide (Pore minimizer)", "Green Tea Polyphenols"]
        actives_to_avoid = ["Heavy coconut oil or petrolatum", "Over-washing with harsh stripping soaps", "Alcohol-based astringents that trigger rebound oil"]
        expert_tips.append("Resist the urge to strip your skin with harsh soaps; dehydrated oily skin produces even more compensatory oil.")

    else: # Normal
        morning = [
            {"step": "Step 1: Balanced Gentle Wash", "action": "Wash with a mild foaming or gel cleanser to remove overnight sebum buildup."},
            {"step": "Step 2: Antioxidant Defense", "action": "Apply a Vitamin C (L-Ascorbic Acid or Ascorbyl Glucoside) serum for environmental protection."},
            {"step": "Step 3: Balanced Hydrator", "action": "Use an everyday peptide-rich moisturizing lotion for suppleness."},
            {"step": "Step 4: Daily Broad Spectrum SPF", "action": "Apply daily broad-spectrum SPF 30-50 to protect collagen integrity."}
        ]
        evening = [
            {"step": "Step 1: Gentle Evening Cleanse", "action": "Cleanse skin gently to wash away daily grime and urban pollution."},
            {"step": "Step 2: Renewal Treatment", "action": "Alternate gentle AHA/BHA exfoliation 1-2 nights a week with peptide or bakuchiol renewal."},
            {"step": "Step 3: Nourishing Night Cream", "action": "Moisturize with a peptide and ceramide complex to aid overnight repair."}
        ]
        actives_to_use = ["Vitamin C (Antioxidant glow)", "Peptides (Firmness)", "Ceramides (Maintenance)", "Hyaluronic Acid"]
        actives_to_avoid = ["Over-complicating routines with excessive actives", "Skipping daily UV protection"]
        expert_tips.append("Your skin barrier is healthy and balanced; focus on UV prevention and antioxidant defense.")

    # Concern severity adjustments
    if lesion_severity in ["Moderate", "High Concern"]:
        expert_tips.append(f"Noted {lesion_severity} concern: Introduce calming actives like Madecassoside or Colloidal Oatmeal, and patch test all new products. If inflammation persists, consult a board-certified dermatologist.")
    elif lesion_severity == "Mild":
        expert_tips.append("Mild concerns detected: Maintain a steady routine for 4-6 weeks to observe clear textural improvements.")

    # Lifestyle tips
    for note in lifestyle_notes:
        expert_tips.append(note)

    return {
        "skin_type": skin_type,
        "lesion_severity": lesion_severity,
        "morning_routine": morning,
        "evening_routine": evening,
        "actives_to_use": actives_to_use,
        "actives_to_avoid": actives_to_avoid,
        "expert_tips": expert_tips
    }


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
                self.kb_vectors = np.array(kb["vectors"], dtype=np.float32)
                self.kb_meta = kb["meta"]
                print(f"[OK] Product KB loaded: {self.kb_vectors.shape[0]} products")
            except Exception as e:
                print(f"[WARN] Failed to load product_kb.pkl: {e}")
                self.kb_vectors = None
                self.kb_meta = []
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
            lifestyle_notes.append("Sleep < 6h: Boosted barrier dryness & acne sensitivity")
        if water_liters < 1.5:
            v[1] += 0.15
            lifestyle_notes.append("Hydration < 1.5L: Elevated dehydration risk")
        if stress_level > 7.0:
            v[3] += 0.20
            v[0] += 0.10
            lifestyle_notes.append("High Stress (>7/10): Elevated sebum and reactive barrier factor")

        v = np.clip(v, 0.0, 1.5)
        return v, lifestyle_notes

    def recommend_products(self, user_vec, top_k=5):
        """Perform Cosine Similarity search against product KB with fallback guarantee."""
        if self.kb_vectors is None or len(self.kb_meta) == 0:
            return FALLBACK_CATALOG[:top_k]

        u_norm = np.linalg.norm(user_vec)
        if u_norm < 1e-8:
            return FALLBACK_CATALOG[:top_k]

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
            
            # Clean price string
            raw_price = str(meta_item.get("price", "N/A")).replace("\ufffd", "£").replace("?", "£").strip()
            if raw_price != "N/A" and not any(raw_price.startswith(c) for c in ["$", "£", "€"]):
                raw_price = f"${raw_price}"

            # Ingredients cleanup
            ingreds = meta_item.get("clean_ingreds", meta_item.get("ingredients", ""))
            if isinstance(ingreds, list) and len(ingreds) > 0:
                ingreds = ", ".join(ingreds[:6])
            elif isinstance(ingreds, str) and len(ingreds.strip()) > 5:
                if len(ingreds) > 100:
                    ingreds = ingreds[:97] + "..."
            else:
                # Default by category
                low_type = ptype.lower()
                matched_def = "Niacinamide, Ceramides, Hyaluronic Acid, Glycerin"
                for cat_key, cat_ingr in CATEGORY_INGREDIENT_DEFAULTS.items():
                    if cat_key in low_type:
                        matched_def = cat_ingr
                        break
                ingreds = matched_def

            paula_rating = meta_item.get("paula_rating", meta_item.get("rating", "Best"))

            results.append({
                "product_name": name,
                "product_type": ptype,
                "price": raw_price,
                "match_score": match_pct,
                "paula_rating": paula_rating,
                "key_ingredients": ingreds,
            })

        if not results:
            return FALLBACK_CATALOG[:top_k]

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

        # Skincare routine & suggestions
        suggestions = generate_skincare_suggestions(
            skin_type=skin_res["predicted"],
            lesion_severity=lesion_res["predicted"],
            lifestyle_notes=lifestyle_notes
        )

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
            "suggestions": suggestions,
            "recommendations": recommendations,
        }
