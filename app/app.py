"""
FaceTrack -- Real-Time AI Skincare Advisor
=============================================
Upload or webcam -> YuNet face detection -> MobileNetV2 Skin Type (4 classes)
+ MobileNetV2 Lesion Severity (4 classes) + Lifestyle fusion -> 7-dim user
vector -> Cosine similarity -> Top-5 product recommendations.

Run:  python app/app.py
"""

import os
import sys
import io

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TF warnings

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import pickle
import numpy as np
import cv2
import gradio as gr

# ── Paths ────────────────────────────────────────────────────────────────────
# Try multiple root locations to find models/
for _base in [Path("."), Path(".."), Path(__file__).parent.parent, Path("/kaggle/working")]:
    if (_base / "models" / "skin_type_model.h5").exists() or \
       (_base / "models" / "skin_type_model_int8.tflite").exists():
        ROOT = _base
        break
else:
    ROOT = Path(__file__).parent.parent  # fallback to project root

MODELS_DIR = ROOT / "models"

# Model paths — prefer TFLite INT8, fallback to H5
SKIN_TFLITE = MODELS_DIR / "skin_type_model_int8.tflite"
LESION_TFLITE = MODELS_DIR / "lesion_severity_model_int8.tflite"
SKIN_H5 = MODELS_DIR / "skin_type_model.h5"
LESION_H5 = MODELS_DIR / "lesion_severity_model.h5"
KB_PKL = MODELS_DIR / "product_kb.pkl"

# YuNet face detection model
YUNET_MODEL = MODELS_DIR / "face_detection_yunet.onnx"

# Training artifacts for display
TRAINING_CURVES = MODELS_DIR / "combined_training_curves.png"
SKIN_CONFUSION = MODELS_DIR / "skin_type_confusion.png"
LESION_CONFUSION = MODELS_DIR / "lesion_severity_confusion.png"
ACCURACY_CHART = MODELS_DIR / "accuracy_comparison.png"
SKIN_HISTORY = MODELS_DIR / "skin_type_training_history.png"
LESION_HISTORY = MODELS_DIR / "lesion_severity_training_history.png"

# Labels
SKIN_LABELS = ["Combination", "Dry", "Normal", "Oily"]
LESION_LABELS = ["Normal", "Mild", "Moderate", "High Concern"]

# ── Face Detection (OpenCV 5 compatible) ─────────────────────────────────────
def create_face_detector(width=640, height=480):
    """Create YuNet face detector (OpenCV 5 DNN-based, replaces deprecated Haar)."""
    if YUNET_MODEL.exists():
        detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL),
            "",
            (width, height),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000,
        )
        print(f"[OK] YuNet face detector loaded: {YUNET_MODEL.name}")
        return detector
    else:
        print(f"[WARN] YuNet model not found at {YUNET_MODEL}")
        return None


face_detector = create_face_detector()


def detect_faces(image_bgr):
    """Detect faces using YuNet. Returns list of (x, y, w, h) tuples."""
    global face_detector
    if face_detector is None:
        return []

    h, w = image_bgr.shape[:2]
    face_detector.setInputSize((w, h))

    _, faces = face_detector.detect(image_bgr)
    if faces is None:
        return []

    results = []
    for face in faces:
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        # Clamp to image bounds
        x = max(0, x)
        y = max(0, y)
        fw = min(fw, w - x)
        fh = min(fh, h - y)
        if fw > 20 and fh > 20:
            results.append((x, y, fw, fh))
    return results


# ── Model Loading ────────────────────────────────────────────────────────────
import tensorflow as tf
tf.get_logger().setLevel("ERROR")


def load_model(tflite_path, h5_path):
    """Load TFLite model if available, otherwise fall back to H5 Keras model."""
    # Try TFLite first
    if tflite_path.exists():
        try:
            interp = tf.lite.Interpreter(model_path=str(tflite_path))
            interp.allocate_tensors()
            inp = interp.get_input_details()[0]
            out = interp.get_output_details()[0]
            print(f"[OK] Loaded TFLite: {tflite_path.name} | "
                  f"in={inp['shape']} {inp['dtype'].__name__} -> "
                  f"out={out['shape']} {out['dtype'].__name__}")
            return ("tflite", interp)
        except Exception as e:
            print(f"[WARN] TFLite load failed ({tflite_path.name}): {e}")

    # Fallback to H5
    if h5_path.exists():
        try:
            model = tf.keras.models.load_model(str(h5_path), compile=False)
            print(f"[OK] Loaded H5: {h5_path.name} | "
                  f"params={model.count_params():,}")
            return ("keras", model)
        except Exception as e:
            print(f"[WARN] H5 load failed ({h5_path.name}): {e}")

    print(f"[ERROR] No model found for {tflite_path.stem}")
    return None


print("\n" + "=" * 60)
print("  FaceTrack -- Loading Models")
print("=" * 60)
skin_model = load_model(SKIN_TFLITE, SKIN_H5)
lesion_model = load_model(LESION_TFLITE, LESION_H5)

# ── Product Knowledge Base ───────────────────────────────────────────────────
if KB_PKL.exists():
    with open(KB_PKL, "rb") as f:
        kb = pickle.load(f)
    vectors = kb["vectors"]  # [N, 7] float32
    meta = kb["meta"]        # list of dicts
    print(f"[OK] Product KB: {vectors.shape[0]} products x {vectors.shape[1]}-dim")
else:
    print("[WARN] product_kb.pkl not found -- using dummy KB")
    vectors = np.random.rand(700, 7).astype(np.float32)
    meta = [
        {"product_name": f"Sample Product {i}", "product_type": "Moisturiser", "price": "10"}
        for i in range(700)
    ]
print("=" * 60 + "\n")


# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_face(bgr_crop):
    """Resize face crop to 224x224 and apply MobileNetV2 preprocessing."""
    img = cv2.resize(bgr_crop, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return np.expand_dims(img, 0).astype(np.float32)


def predict(model_tuple, img_pre):
    """Run inference through TFLite or Keras model. Returns softmax probs."""
    if model_tuple is None:
        return np.ones(4) / 4  # uniform fallback

    kind, m = model_tuple
    if kind == "keras":
        return m.predict(img_pre, verbose=0)[0]

    # TFLite path
    interp = m
    inp_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    if inp_det["dtype"] == np.uint8:
        scale, zp = inp_det["quantization"]
        scale = scale if scale != 0 else 1.0
        q = np.clip((img_pre / scale + zp), 0, 255).astype(np.uint8)
        interp.set_tensor(inp_det["index"], q)
    else:
        interp.set_tensor(inp_det["index"], img_pre)

    interp.invoke()
    pred = interp.get_tensor(out_det["index"])[0]

    if out_det["dtype"] == np.uint8:
        scale, zp = out_det["quantization"]
        pred = scale * (pred.astype(np.float32) - zp)

    pred = np.array(pred, dtype=np.float32)
    # Apply softmax if output doesn't sum to ~1
    if not np.isclose(pred.sum(), 1.0, atol=0.1):
        e = np.exp(pred - pred.max())
        pred = e / e.sum()
    return pred


# ── User Profile Vector ─────────────────────────────────────────────────────
def build_user_vector(skin_probs, lesion_probs, sleep, water, stress):
    """
    Late-fusion: combine vision outputs + lifestyle into 7-dim user vector.
    Schema: [oily, dry, normal, acne, avg_paula, type, price]
    """
    p_comb, p_dry, p_norm, p_oily = (
        skin_probs if len(skin_probs) == 4
        else (0.25, 0.25, 0.25, 0.25)
    )

    oily = float(p_oily + 0.5 * p_comb)
    dry = float(p_dry + 0.5 * p_comb)
    normal = float(p_norm)

    if len(lesion_probs) == 4:
        _, p_mild, p_mod, p_high = lesion_probs
        acne = float(p_high * 1.0 + p_mod * 0.6 + p_mild * 0.3)
    else:
        acne = 0.3

    # Lifestyle adjustments
    if sleep < 6:
        dry = min(1.0, dry + 0.15)
        acne = min(1.0, acne + 0.1)
    if water < 1.5:
        dry = min(1.0, dry + 0.15)
    if stress > 7:
        acne = min(1.0, acne + 0.2)
        oily = min(1.0, oily + 0.1)

    vec = np.array([oily, dry, normal, acne, 0.5, 0.5, 0.5], dtype=np.float32)
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


# ── Recommendation Engine ───────────────────────────────────────────────────
def recommend(user_vec, topk=5):
    """Cosine similarity search over product KB."""
    prod_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    sims = prod_norm @ user_vec
    idx = np.argsort(sims)[::-1][:topk]
    return idx, sims[idx]


# ── Skincare Tips based on analysis ──────────────────────────────────────────
SKIN_TIPS = {
    "Combination": [
        "Use a gentle, non-stripping cleanser",
        "Apply lightweight moisturizer on oily zones, richer cream on dry areas",
        "Use niacinamide to balance oil production",
        "Try clay masks on T-zone only, hydrating masks on cheeks",
    ],
    "Dry": [
        "Use cream-based or oil-based cleansers (avoid foam)",
        "Layer hydrating toner + serum + rich moisturizer",
        "Look for hyaluronic acid, ceramides, and squalane",
        "Avoid hot water when washing face; use lukewarm",
    ],
    "Normal": [
        "Maintain your routine -- your skin is well-balanced",
        "Focus on prevention with SPF 30+ daily",
        "Use antioxidant serums (Vitamin C) for glow",
        "Gentle exfoliation 1-2x per week",
    ],
    "Oily": [
        "Use gel or foam cleanser with salicylic acid",
        "Lightweight gel moisturizer (still moisturize!)",
        "Niacinamide + zinc to control sebum",
        "Blotting papers for midday shine; avoid over-washing",
    ],
}

CONCERN_TIPS = {
    "Normal": "Your skin looks healthy! Focus on maintenance and sun protection.",
    "Mild": "Minor concerns detected. Consistent routine with gentle actives should help.",
    "Moderate": "Moderate concerns present. Consider targeted treatments and consult a dermatologist.",
    "High Concern": "Significant concerns detected. Please consult a dermatologist for professional advice.",
}


# ── Main Inference ───────────────────────────────────────────────────────────
def analyze_skin(image, sleep, water, stress):
    """Full pipeline: face detect -> classify -> fuse -> recommend."""
    if image is None:
        return (
            None,
            "Please upload an image or use webcam.",
            {},  # skin probs
            {},  # lesion probs
            "No image provided.",
            "",
            "No tips available without analysis.",
        )

    # Convert Gradio RGB->BGR for OpenCV
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # YuNet face detection
    faces = detect_faces(bgr)

    vis = bgr.copy()
    if len(faces) == 0:
        face_crop = bgr  # use entire image as fallback
        cv2.putText(vis, "No face detected - using full image", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        # Use largest face
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        face_crop = bgr[y:y + h, x:x + w]
        # Draw boxes on all detected faces
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
            cv2.putText(vis, "Face", (fx, fy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if face_crop.size == 0:
        return None, "Invalid face crop.", {}, {}, "", "", ""

    # Predict
    img_pre = preprocess_face(face_crop)
    skin_probs = predict(skin_model, img_pre)
    lesion_probs = predict(lesion_model, img_pre)

    # Build user vector and recommend
    user_vec = build_user_vector(skin_probs, lesion_probs, sleep, water, stress)
    idx_arr, scores = recommend(user_vec, topk=5)

    # ── Format outputs ───────────────────────────────────────────────────────

    # Skin type label dict for gr.Label
    skin_dict = {SKIN_LABELS[i]: float(skin_probs[i]) for i in range(len(SKIN_LABELS))}

    # Lesion severity label dict for gr.Label
    lesion_dict = {LESION_LABELS[i]: float(lesion_probs[i]) for i in range(len(LESION_LABELS))}

    # Primary diagnosis text
    top_skin = SKIN_LABELS[np.argmax(skin_probs)]
    top_lesion = LESION_LABELS[np.argmax(lesion_probs)]
    skin_conf = float(np.max(skin_probs)) * 100
    lesion_conf = float(np.max(lesion_probs)) * 100

    diagnosis = (
        f"### Skin Analysis Results\n\n"
        f"**Skin Type:** {top_skin} ({skin_conf:.1f}% confidence)\n\n"
        f"**Concern Level:** {top_lesion} ({lesion_conf:.1f}% confidence)\n\n"
    )

    # Lifestyle impact notes
    notes = []
    if sleep < 6:
        notes.append("Low sleep (<6h) -- increased dryness & acne risk")
    if water < 1.5:
        notes.append("Low hydration (<1.5L) -- increased dryness risk")
    if stress > 7:
        notes.append("High stress (>7/10) -- increased oiliness & acne risk")
    if notes:
        diagnosis += "---\n**Lifestyle Flags:**\n" + "\n".join(f"- {n}" for n in notes) + "\n"

    # Annotate output image
    top_label = f"{top_skin} | {top_lesion}"
    cv2.putText(vis, top_label, (10, vis.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Product recommendations
    rec_lines = []
    for rank, (pidx, sim) in enumerate(zip(idx_arr, scores), 1):
        m = meta[pidx] if pidx < len(meta) else {"product_name": f"Product {pidx}", "product_type": "", "price": ""}
        name = m.get("product_name", "Unknown")[:65]
        ptype = m.get("product_type", "")
        price = m.get("price", "")
        match_pct = float(sim) * 100
        rec_lines.append(
            f"**{rank}. {name}**\n"
            f"   Type: {ptype} | Price: {price} | Match: {match_pct:.1f}%\n"
        )
    recommendations = "### Top 5 Recommended Products\n\n" + "\n".join(rec_lines)

    # Skincare tips
    tips = SKIN_TIPS.get(top_skin, [])
    concern_tip = CONCERN_TIPS.get(top_lesion, "")
    tips_text = f"### Tips for {top_skin} Skin\n\n"
    tips_text += "\n".join(f"- {t}" for t in tips)
    tips_text += f"\n\n### Concern Level: {top_lesion}\n{concern_tip}"

    return (
        cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
        diagnosis,
        skin_dict,
        lesion_dict,
        recommendations,
        f"User vector: [{', '.join(f'{v:.3f}' for v in user_vec)}]",
        tips_text,
    )


# ── Gather available training artifacts ──────────────────────────────────────
def get_training_images():
    """Return list of available training visualization images."""
    images = []
    for path, label in [
        (TRAINING_CURVES, "Combined Training Curves"),
        (SKIN_HISTORY, "Skin Type Training History"),
        (LESION_HISTORY, "Lesion Severity Training History"),
        (SKIN_CONFUSION, "Skin Type Confusion Matrix"),
        (LESION_CONFUSION, "Lesion Severity Confusion Matrix"),
        (ACCURACY_CHART, "Accuracy Comparison"),
    ]:
        if path.exists():
            images.append((str(path), label))
    return images


# ── CSS Theme ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.app-header {
    text-align: center;
    padding: 24px 16px 16px 16px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
}
.app-header h1 {
    color: white !important;
    font-size: 2.2em !important;
    margin-bottom: 6px !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}
.app-header p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.05em !important;
    margin-bottom: 10px !important;
}
.model-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(10px);
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.85em;
    margin: 3px;
    color: white;
    border: 1px solid rgba(255,255,255,0.2);
}
footer { display: none !important; }
"""


# ── Build Gradio UI ─────────────────────────────────────────────────────────
def build_app():
    """Construct and return the Gradio Blocks app."""

    # Determine model status for header
    skin_status = "[OK] H5" if skin_model and skin_model[0] == "keras" else (
        "[OK] TFLite" if skin_model and skin_model[0] == "tflite" else "[X] Missing"
    )
    lesion_status = "[OK] H5" if lesion_model and lesion_model[0] == "keras" else (
        "[OK] TFLite" if lesion_model and lesion_model[0] == "tflite" else "[X] Missing"
    )
    kb_status = f"[OK] {vectors.shape[0]} products" if KB_PKL.exists() else "[!] Dummy"

    with gr.Blocks(
        title="FaceTrack - AI Skincare Advisor",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.purple,
            secondary_hue=gr.themes.colors.indigo,
        ),
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────
        gr.HTML(f"""
        <div class="app-header">
            <h1>FaceTrack - AI Skincare Advisor</h1>
            <p>Real-time skin analysis powered by MobileNetV2 + Cosine similarity recommendations</p>
            <div>
                <span class="model-badge">Skin Model: {skin_status}</span>
                <span class="model-badge">Lesion Model: {lesion_status}</span>
                <span class="model-badge">Product KB: {kb_status}</span>
            </div>
        </div>
        """)

        with gr.Tabs():
            # ── Tab 1: Skin Analysis ─────────────────────────────────────
            with gr.Tab("Skin Analysis", id="analysis"):
                gr.Markdown(
                    "Upload a face photo or use your webcam. "
                    "Adjust the lifestyle sliders to personalize your recommendations."
                )

                with gr.Row(equal_height=False):
                    # Left column -- inputs
                    with gr.Column(scale=1):
                        inp_img = gr.Image(
                            label="Face Image",
                            sources=["upload", "webcam"],
                            type="numpy",
                            height=360,
                        )
                        gr.Markdown("#### Lifestyle Factors")
                        sleep = gr.Slider(
                            3, 12, value=7.5, step=0.5,
                            label="Sleep (hours last night)",
                        )
                        water = gr.Slider(
                            0.5, 5.0, value=2.0, step=0.1,
                            label="Water Intake (L/day)",
                        )
                        stress = gr.Slider(
                            1, 10, value=3, step=1,
                            label="Stress Level (1-10)",
                        )
                        btn = gr.Button(
                            "Analyze & Recommend",
                            variant="primary",
                            size="lg",
                        )

                    # Right column -- outputs
                    with gr.Column(scale=1):
                        out_img = gr.Image(
                            label="Face Detection",
                            height=300,
                        )
                        diagnosis_md = gr.Markdown(
                            label="Diagnosis",
                            value="*Upload an image to begin analysis...*",
                        )
                        with gr.Row():
                            skin_label = gr.Label(
                                label="Skin Type Probabilities",
                                num_top_classes=4,
                            )
                            lesion_label = gr.Label(
                                label="Concern Severity",
                                num_top_classes=4,
                            )

                with gr.Row():
                    with gr.Column():
                        rec_md = gr.Markdown(
                            value="*Recommendations will appear here after analysis...*",
                            label="Recommendations",
                        )
                    with gr.Column():
                        tips_md = gr.Markdown(
                            value="*Skincare tips will appear here after analysis...*",
                            label="Skincare Tips",
                        )

                debug_txt = gr.Textbox(
                    label="Debug - User Profile Vector",
                    interactive=False,
                    lines=1,
                    visible=False,
                )

                btn.click(
                    fn=analyze_skin,
                    inputs=[inp_img, sleep, water, stress],
                    outputs=[out_img, diagnosis_md, skin_label, lesion_label, rec_md, debug_txt, tips_md],
                )

            # ── Tab 2: Training Results ──────────────────────────────────
            with gr.Tab("Training Results", id="training"):
                gr.Markdown(
                    "### Model Training Visualizations\n"
                    "These plots were generated during the MobileNetV2 training pipeline "
                    "(Phase 1: Skin Type, Phase 2: Lesion Severity)."
                )

                training_imgs = get_training_images()
                if training_imgs:
                    gr.Gallery(
                        value=training_imgs,
                        label="Training Artifacts",
                        columns=2,
                        height=500,
                        object_fit="contain",
                    )
                else:
                    gr.Markdown("*No training artifacts found in models/ folder.*")

                # Metrics table
                gr.Markdown("### Model Performance Metrics")
                gr.Dataframe(
                    headers=["Model", "Test Accuracy", "Best Per-Class", "Worst Per-Class", "H5 Size"],
                    value=[
                        ["Skin Type (MobileNetV2)", "63.0%", "Combination 70%", "Normal 50%", "21.7 MB"],
                        ["Lesion Severity (MobileNetV2)", "77.6%", "Normal 87.4%", "Mild 55.2%", "21.7 MB"],
                    ],
                    interactive=False,
                )

            # ── Tab 3: How It Works ──────────────────────────────────────
            with gr.Tab("How It Works", id="info"):
                gr.Markdown("""
### System Architecture

```
Webcam / Upload
  -> YuNet Face Detection -> face crop 224x224 -> MobileNetV2 preprocess
      |---> Skin Type Model    -> [P_combination, P_dry, P_normal, P_oily]
      |---> Lesion Model       -> [P_normal, P_mild, P_moderate, P_high]
           + Lifestyle (sleep, water, stress)
           -> Late Fusion -> 7-dim User Vector -> L2 normalize
           -> Cosine Similarity Search over 1138 products -> Top 5
```

### Models

| Model | Backbone | Input | Output | Training |
|-------|----------|-------|--------|----------|
| Skin Type | MobileNetV2 | 224x224x3 | Softmax(4) | 20ep frozen + 15ep fine-tune top-30 |
| Lesion Severity | MobileNetV2 | 224x224x3 | Softmax(4) | 20ep + 15ep, class-weighted |

### User Vector Schema (7 dimensions)

| Dim | Name | Derivation |
|-----|------|-----------|
| 0 | Oily score | P(oily) + 0.5 x P(combination) |
| 1 | Dry score | P(dry) + 0.5 x P(combination) + lifestyle |
| 2 | Normal score | P(normal) |
| 3 | Acne concern | P(high) x 1.0 + P(mod) x 0.6 + P(mild) x 0.3 + lifestyle |
| 4 | Avg Paula quality | Neutral 0.5 |
| 5 | Product type | Neutral 0.5 |
| 6 | Price pref | Neutral 0.5 |

### Lifestyle Modifiers

- **Sleep < 6h**: dry +0.15, acne +0.1
- **Water < 1.5L**: dry +0.15
- **Stress > 7**: acne +0.2, oily +0.1

### Product Knowledge Base

- **1,138 products** with 7-dim feature vectors
- Keyword scoring (oily/dry/normal/acne ingredient keywords)
- Paula's Choice ingredient ratings (BEST=3, GOOD=2, AVERAGE=1, POOR=0)
- Cosine similarity for scale-invariant matching
                """)

        return demo


# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_app()
    print("\nStarting FaceTrack Skincare Advisor...")
    print("   Open: http://localhost:7860\n")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
