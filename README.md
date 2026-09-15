# FaceTrack — Real-Time AI Skincare Advisor

> Real-time facial diagnostics, deep-learning skin barrier classification, lesion concern severity grading, and Paula's Choice-backed product formulation recommendations.

[![Render](https://img.shields.io/badge/Backend-Render-blue?logo=render)](https://render.com)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel-black?logo=vercel)](https://vercel.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](LICENSE)

---

## System Architecture

```
User (Browser / Mobile)
  │
  ├──► Vercel Frontend (HTML5 / Modern Dark Glassmorphic UI)
  │      ├── Live Webcam Stream / Photo Upload
  │      ├── Face Framing Guide & Instant Canvas Capture
  │      └── Lifestyle Controls (Sleep hrs, Hydration L, Stress level 1-10)
  │
  └──► Render Backend (FastAPI + Uvicorn)
         ├── 1. OpenCV YuNet Face Detection (ONNX)
         ├── 2. MobileNetV2 Skin Type Classification (Combination / Dry / Normal / Oily)
         ├── 3. MobileNetV2 Lesion Severity Grading (Normal / Mild / Moderate / High Concern)
         ├── 4. 7-Dimensional User Profile Vector Fusion (+ Lifestyle modifiers)
         └── 5. Cosine Similarity Vector Search across 1,138 Products (Paula's Choice Ratings)
```

---

## Project Structure

```
FaceTrack/
├── backend/                      # Render Web Service (FastAPI)
│   ├── main.py                   # REST API routes (/api/health, /api/analyze, /api/sample)
│   ├── advisor.py                # YuNet + MobileNetV2 + 7-dim fusion + Top 5 matcher
│   ├── requirements.txt          # Lightweight CPU dependencies for Render 512MB RAM
│   └── __init__.py
├── frontend/                     # Vercel Deployment (Static Web App)
│   ├── index.html                # Responsive web interface
│   ├── style.css                 # Dark glassmorphic styling & neon accents
│   ├── app.js                    # Camera capture & API client
│   └── vercel.json               # Vercel routing configuration
├── models/                       # Core Inference Weights (~44MB)
│   ├── face_detection_yunet.onnx # 232 KB (YuNet DNN Face Detector)
│   ├── skin_type_model.h5        # 21.6 MB (MobileNetV2 Skin Classifier)
│   ├── lesion_severity_model.h5  # 21.6 MB (MobileNetV2 Lesion Severity)
│   └── product_kb.pkl            # 806 KB (1,138 enriched skincare products)
├── render.yaml                   # Optional Render Blueprint
├── .gitignore                    # Excludes datasets (>3.6GB) and temp files
└── README.md
```

---

## 1. Deploying the Backend on Render (100% Free)

1. **Sign Up / Log In**: Go to [render.com](https://render.com) and log in with your GitHub account.
2. Click **New +** → Select **Web Service**.
3. Connect your repository: `https://github.com/SifatAhmed28/FaceTrack.git`.
4. Configure the Web Service settings:
   - **Name**: `facetrack-api` (or your chosen name)
   - **Region**: Choose closest to you (e.g., Oregon, Frankfurt, Singapore)
   - **Branch**: `main`
   - **Root Directory**: leave blank (or `.`)
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r backend/requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn backend.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: Select **Free** (0.1 CPU, 512 MB RAM).
5. **Environment Variables** (under *Advanced*):
   - Add `PYTHON_VERSION` = `3.11.9`
   - Add `TF_CPP_MIN_LOG_LEVEL` = `3`
6. Click **Create Web Service**.
7. Once deployed, Render will provide a public URL like:
   `https://facetrack-api.onrender.com`
   *(Test it by opening `https://facetrack-api.onrender.com/api/health` in your browser)*.

> [!NOTE]
> Render free web services spin down after 15 minutes of inactivity. When a new request arrives, it may take 30–50 seconds for the first cold start.

---

## 2. Deploying the Frontend on Vercel (100% Free)

1. **Sign Up / Log In**: Go to [vercel.com](https://vercel.com) and log in with your GitHub account.
2. Click **Add New...** → **Project**.
3. Import your repository: `FaceTrack`.
4. In the **Configure Project** screen:
   - **Framework Preset**: Select **Other**
   - **Root Directory**: Click **Edit** and choose `frontend` *(Important!)*
   - **Build and Output Settings**: Leave default (no build command needed, it's instant pure static web).
5. Click **Deploy**.
6. In ~5 seconds, Vercel will give you a live production URL:
   `https://facetrack-xxxx.vercel.app`

### Linking Frontend to your Render Backend:
- Open your live Vercel app.
- Click the **Gear (⚙️) Settings button** in the top-right navbar.
- Enter your Render URL (e.g. `https://facetrack-api.onrender.com`).
- Click **Test Connection** → **Save & Connect**.
- That's it! Your Vercel frontend is now live and talking to your Render backend!

---

## 3. Running Locally

### Backend (Terminal 1)
```bash
# From the project root:
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API docs available at: `http://localhost:8000/docs`

### Frontend (Terminal 2 or Live Server)
You can open `frontend/index.html` directly in any browser, or use Python's built-in HTTP server:
```bash
cd frontend
python -m http.server 3000
```
Open: `http://localhost:3000`

---

## License
MIT License. Built for educational and AI skincare research purposes.
