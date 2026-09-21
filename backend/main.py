"""
FaceTrack FastAPI Backend Server
================================
Lightweight REST API for FaceTrack Skincare Advisor,
designed for Render Free Tier deployment.
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from .advisor import SkincareAdvisor

app = FastAPI(
    title="FaceTrack AI Skincare Advisor API",
    description="Real-time Face Detection, Skin Type Classification, Severity Grading, and Product Recommendation.",
    version="1.0.0"
)

# Enable CORS for Vercel frontend and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global advisor instance
advisor: Optional[SkincareAdvisor] = None

def get_advisor() -> SkincareAdvisor:
    global advisor
    if advisor is None:
        print("\n[FaceTrack] Initializing SkincareAdvisor engine...")
        advisor = SkincareAdvisor()
        print("[FaceTrack] SkincareAdvisor engine ready.\n")
    return advisor

@app.on_event("startup")
def startup_event():
    get_advisor()


@app.get("/")
def root():
    return {
        "service": "FaceTrack AI Skincare Advisor API",
        "status": "online",
        "endpoints": {
            "health": "/api/health",
            "analyze": "POST /api/analyze",
            "sample": "/api/sample",
            "kb_stats": "/api/kb/stats"
        }
    }


@app.get("/api/health")
def health():
    adv = get_advisor()
    return {
        "status": "healthy",
        "models": {
            "face_detector": adv.face_detector is not None,
            "skin_model": adv.skin_model is not None,
            "lesion_model": adv.lesion_model is not None,
            "product_kb": adv.kb_vectors is not None,
            "total_products": len(adv.kb_meta)
        }
    }


@app.get("/api/kb/stats")
def kb_stats():
    adv = get_advisor()
    if adv.kb_vectors is None:
        return {"error": "Knowledge base not loaded"}
    return {
        "total_products": int(adv.kb_vectors.shape[0]),
        "feature_dimensions": int(adv.kb_vectors.shape[1]),
        "sample_categories": list(set([p.get("product_type", "Unknown") for p in adv.kb_meta[:50]]))
    }


def _read_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes into OpenCV BGR numpy array with EXIF auto-rotation and safe resizing."""
    try:
        from PIL import ImageOps
        pil_img = Image.open(io.BytesIO(image_bytes))
        
        # Auto-rotate smartphone photos based on EXIF tag (prevents sideways faces)
        pil_img = ImageOps.exif_transpose(pil_img)
        pil_img = pil_img.convert("RGB")
        
        # Downscale ultra-high resolution smartphone photos (e.g. 12-48 MP) to safe size
        max_dimension = 1280
        if max(pil_img.size) > max_dimension:
            pil_img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            
        rgb_arr = np.array(pil_img)
        bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        return bgr_arr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    sleep_hours: float = Form(7.0),
    water_liters: float = Form(2.0),
    stress_level: float = Form(5.0),
):
    """
    Analyze uploaded face image:
    1. Detects face with YuNet
    2. Classifies Skin Type
    3. Grades Lesion Severity
    4. Combines with Lifestyle factors into 7-dim profile
    5. Recommends Top 5 skincare products via Cosine Similarity
    """
    adv = get_advisor()
    content = await file.read()
    bgr_img = _read_image_bytes(content)

    try:
        results = adv.analyze(
            image_bgr=bgr_img,
            sleep_hours=sleep_hours,
            water_liters=water_liters,
            stress_level=stress_level
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


class Base64AnalyzeRequest(BaseModel):
    image_base64: str
    sleep_hours: float = 7.0
    water_liters: float = 2.0
    stress_level: float = 5.0


@app.post("/api/analyze-base64")
def analyze_base64(req: Base64AnalyzeRequest):
    """Analyze image passed as base64 string from webcam canvas."""
    adv = get_advisor()

    try:
        data = req.image_base64
        if "," in data:
            data = data.split(",")[1]
        raw_bytes = base64.b64decode(data)
        bgr_img = _read_image_bytes(raw_bytes)
        return adv.analyze(
            image_bgr=bgr_img,
            sleep_hours=req.sleep_hours,
            water_liters=req.water_liters,
            stress_level=req.stress_level
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process base64 image: {str(e)}")


@app.get("/api/sample")
def sample_analysis():
    """Returns sample pre-calculated analysis data for testing the UI without an image."""
    return {
        "face_detected": True,
        "face_bbox": {"x": 120, "y": 80, "w": 280, "h": 280},
        "skin_type": {
            "predicted": "Combination",
            "confidence": 78.4,
            "probabilities": {
                "Combination": 78.4,
                "Dry": 11.2,
                "Normal": 6.8,
                "Oily": 3.6
            }
        },
        "lesion_severity": {
            "predicted": "Mild",
            "confidence": 64.1,
            "probabilities": {
                "Normal": 22.5,
                "Mild": 64.1,
                "Moderate": 10.4,
                "High Concern": 3.0
            }
        },
        "lifestyle_notes": [
            "Sleep < 6h: Boosted barrier dryness & acne sensitivity",
            "High Stress (>7/10): Elevated sebum and reactive barrier factor"
        ],
        "suggestions": {
            "skin_type": "Combination",
            "lesion_severity": "Mild",
            "morning_routine": [
                {"step": "Step 1: Gentle Cleanse", "action": "Wash with a pH-balanced, non-foaming gel cleanser to refresh without stripping moisture."},
                {"step": "Step 2: Hydrating Serum", "action": "Apply lightweight Niacinamide (2-5%) or Hyaluronic Acid to balance sebum and hydrate."},
                {"step": "Step 3: Lightweight Moisturizer", "action": "Use an oil-free water-gel on the T-zone and slightly richer lotion on dry cheeks."},
                {"step": "Step 4: Broad-Spectrum Sunscreen", "action": "Finish with fluid SPF 30-50 that leaves zero greasy residue."}
            ],
            "evening_routine": [
                {"step": "Step 1: Clarifying Cleanse", "action": "Remove SPF and impurities with micellar water followed by a gentle gel wash."},
                {"step": "Step 2: Targeted Active Treatment", "action": "Apply 2% Salicylic Acid (BHA) 2-3x weekly to decongest pores in the T-zone."},
                {"step": "Step 3: Barrier Recovery Cream", "action": "Lock in hydration with a ceramide and centella-infused soothing emulsion."}
            ],
            "actives_to_use": [
                "Niacinamide (Oil Balancing)",
                "Salicylic Acid (Pore Decongestion)",
                "Ceramides (Lipid Barrier)",
                "Hyaluronic Acid (Hydration)"
            ],
            "actives_to_avoid": [
                "Heavy pore-clogging mineral oils",
                "High-alcohol drying astringents",
                "Over-aggressive physical apricot/walnut scrubs"
            ],
            "expert_tips": [
                "Zone-treat your face: target oil control strictly on forehead and nose, while protecting drier cheek areas.",
                "Mild concerns detected: Maintain a steady routine for 4-6 weeks to observe clear textural improvements.",
                "Sleep < 6h: Boosted barrier dryness & acne sensitivity",
                "High Stress (>7/10): Elevated sebum and reactive barrier factor"
            ]
        },
        "recommendations": [
            {
                "product_name": "CeraVe Hydrating Facial Cleanser",
                "product_type": "Cleanser",
                "price": "$15.99",
                "match_score": 94.8,
                "paula_rating": "Best",
                "key_ingredients": "Ceramides 1, 3, 6-II, Hyaluronic Acid, Glycerin, Niacinamide"
            },
            {
                "product_name": "Paula's Choice 2% BHA Liquid Exfoliant",
                "product_type": "Exfoliant",
                "price": "$34.00",
                "match_score": 91.2,
                "paula_rating": "Best",
                "key_ingredients": "Salicylic Acid, Green Tea Extract, Methylpropanediol"
            },
            {
                "product_name": "La Roche-Posay Toleriane Double Repair",
                "product_type": "Moisturizer",
                "price": "$21.99",
                "match_score": 88.5,
                "paula_rating": "Best",
                "key_ingredients": "Ceramide-3, Niacinamide, Glycerin, Thermal Water"
            },
            {
                "product_name": "The Ordinary Niacinamide 10% + Zinc 1%",
                "product_type": "Serum",
                "price": "$6.00",
                "match_score": 86.3,
                "paula_rating": "Good",
                "key_ingredients": "Niacinamide, Zinc PCA, Tamarindus Indica Seed Gum"
            },
            {
                "product_name": "EltaMD UV Clear Broad-Spectrum SPF 46",
                "product_type": "Sunscreen",
                "price": "$39.00",
                "match_score": 84.7,
                "paula_rating": "Best",
                "key_ingredients": "Zinc Oxide 9.0%, Niacinamide 5%, Hyaluronic Acid"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    print("\nStarting FaceTrack FastAPI Server on http://localhost:8000...")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
