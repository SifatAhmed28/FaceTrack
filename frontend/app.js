/**
 * FaceTrack AI Skincare Advisor — Client Application
 * Handles webcam stream, mobile photo resizing & compression,
 * API communication, responsive tab navigation, and diagnostic rendering.
 */

// Permanent Backend API URL (Render Free Tier)
const API_BASE_URL = "https://facetrack-7ex1.onrender.com";

// State
let activeMode = "webcam"; // "webcam" | "upload"
let mediaStream = null;
let currentImageBlob = null; // Lightweight Blob ready for analysis
let currentBase64 = null;
let healthCheckTimer = null;

// Primary View Navigation
const navTabAdvisor = document.getElementById("nav-tab-advisor");
const navTabTech = document.getElementById("nav-tab-tech");
const viewAdvisor = document.getElementById("view-advisor");
const viewModelTech = document.getElementById("view-model-tech");

// DOM Elements
const backendStatusBadge = document.getElementById("backend-status-badge");
const backendStatusText = document.getElementById("backend-status-text");

// Input Mode Tabs
const tabWebcam = document.getElementById("tab-webcam");
const tabUpload = document.getElementById("tab-upload");
const webcamView = document.getElementById("webcam-view");
const uploadView = document.getElementById("upload-view");

// Webcam
const videoStream = document.getElementById("video-stream");
const detectionCanvas = document.getElementById("detection-canvas");
const cameraOverlay = document.getElementById("camera-overlay");
const btnStartCamera = document.getElementById("btn-start-camera");
const btnSnapCamera = document.getElementById("btn-snap-camera");

// Upload
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const uploadPreviewWrap = document.getElementById("upload-preview-wrap");
const uploadPreviewImg = document.getElementById("upload-preview-img");
const btnRemoveImage = document.getElementById("btn-remove-image");

// Sliders
const sliderSleep = document.getElementById("slider-sleep");
const valSleep = document.getElementById("val-sleep");
const sliderWater = document.getElementById("slider-water");
const valWater = document.getElementById("val-water");
const sliderStress = document.getElementById("slider-stress");
const valStress = document.getElementById("val-stress");

// Actions
const btnAnalyze = document.getElementById("btn-analyze");
const btnDemo = document.getElementById("btn-demo");

// Results
const resultsEmpty = document.getElementById("results-empty");
const resultsLoading = document.getElementById("results-loading");
const resultsContent = document.getElementById("results-content");
const loaderStatus = document.getElementById("loader-status");

// Results details
const resSkinBadge = document.getElementById("res-skin-badge");
const resSkinConf = document.getElementById("res-skin-conf");
const resSkinBars = document.getElementById("res-skin-bars");

const resLesionBadge = document.getElementById("res-lesion-badge");
const resLesionConf = document.getElementById("res-lesion-conf");
const resLesionBars = document.getElementById("res-lesion-bars");

const resLifestyleWrap = document.getElementById("res-lifestyle-wrap");
const resLifestyleTags = document.getElementById("res-lifestyle-tags");

// Skincare suggestions & routine elements
const resSuggestionsWrap = document.getElementById("res-suggestions-wrap");
const resMorningRoutine = document.getElementById("res-morning-routine");
const resEveningRoutine = document.getElementById("res-evening-routine");
const resActivesUse = document.getElementById("res-actives-use");
const resActivesAvoid = document.getElementById("res-actives-avoid");
const resExpertTips = document.getElementById("res-expert-tips");

// Products list
const resProductsList = document.getElementById("res-products-list");

// ==========================================================================
// Initialization
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  setupEventListeners();
  setupNavSwitcher();
});

// ==========================================================================
// Primary View Switcher (Diagnostics vs Model Training Tech)
// ==========================================================================
function setupNavSwitcher() {
  if (!navTabAdvisor || !navTabTech) return;

  navTabAdvisor.addEventListener("click", () => {
    navTabAdvisor.classList.add("active");
    navTabTech.classList.remove("active");
    viewAdvisor.classList.add("active");
    viewModelTech.classList.remove("active");
    window.scrollTo({ top: 0, behavior: "smooth" });
  });

  navTabTech.addEventListener("click", () => {
    navTabTech.classList.add("active");
    navTabAdvisor.classList.remove("active");
    viewModelTech.classList.add("active");
    viewAdvisor.classList.remove("active");
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

// ==========================================================================
// Backend Health Checker with Auto-Retry
// ==========================================================================
async function checkBackendHealth() {
  backendStatusBadge.className = "status-badge status-checking";
  backendStatusText.textContent = "Connecting to API...";

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000);
    const res = await fetch(`${API_BASE_URL}/api/health`, {
      method: "GET",
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      backendStatusBadge.className = "status-badge status-connected";
      backendStatusText.textContent = `Online (${data.models?.total_products || 1138} prods)`;
      if (healthCheckTimer) clearTimeout(healthCheckTimer);
      return true;
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    backendStatusBadge.className = "status-badge status-checking";
    backendStatusText.textContent = "Waking up API (~30s)...";
    healthCheckTimer = setTimeout(checkBackendHealth, 4000);
    return false;
  }
}

// ==========================================================================
// Event Listeners
// ==========================================================================
function setupEventListeners() {
  // Input Tabs
  tabWebcam.addEventListener("click", () => switchMode("webcam"));
  tabUpload.addEventListener("click", () => switchMode("upload"));

  // Camera
  btnStartCamera.addEventListener("click", startCamera);
  btnSnapCamera.addEventListener("click", captureSnapshot);

  // File Upload
  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileSelect);
  btnRemoveImage.addEventListener("click", clearUploadedImage);

  // Drag and Drop
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      processImageFile(e.dataTransfer.files[0]);
    }
  });

  // Sliders
  sliderSleep.addEventListener("input", (e) => valSleep.textContent = `${e.target.value} hrs`);
  sliderWater.addEventListener("input", (e) => valWater.textContent = `${e.target.value} L`);
  sliderStress.addEventListener("input", (e) => valStress.textContent = `${e.target.value} / 10`);

  // Actions
  btnAnalyze.addEventListener("click", runAnalysis);
  btnDemo.addEventListener("click", runDemoAnalysis);
}

// ==========================================================================
// Mode Switcher
// ==========================================================================
function switchMode(mode) {
  activeMode = mode;
  if (mode === "webcam") {
    tabWebcam.classList.add("active");
    tabUpload.classList.remove("active");
    webcamView.classList.add("active");
    uploadView.classList.remove("active");
    updateAnalyzeButtonState();
  } else {
    tabUpload.classList.add("active");
    tabWebcam.classList.remove("active");
    uploadView.classList.add("active");
    webcamView.classList.remove("active");
    stopCamera();
    updateAnalyzeButtonState();
  }
}

// ==========================================================================
// Smart Image Resizing & Compression for Mobile / Phone Photos
// Ensures phone camera 12-48MP photos (15MB+) are smoothly downscaled to ~200KB
// ==========================================================================
function resizeAndCompressImage(fileOrBlob, maxDimension = 1024, quality = 0.85) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const objectUrl = URL.createObjectURL(fileOrBlob);

    img.onload = () => {
      URL.revokeObjectURL(objectUrl);
      let { width, height } = img;

      // Scale down proportionally if larger than maxDimension
      if (width > maxDimension || height > maxDimension) {
        if (width > height) {
          height = Math.round((height * maxDimension) / width);
          width = maxDimension;
        } else {
          width = Math.round((width * maxDimension) / height);
          height = maxDimension;
        }
      }

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");

      // Draw onto smooth canvas
      ctx.drawImage(img, 0, 0, width, height);

      const base64Data = canvas.toDataURL("image/jpeg", quality);
      canvas.toBlob(
        (blob) => {
          resolve({ blob, base64: base64Data, width, height });
        },
        "image/jpeg",
        quality
      );
    };

    img.onerror = (err) => {
      URL.revokeObjectURL(objectUrl);
      reject(err);
    };

    img.src = objectUrl;
  });
}

// ==========================================================================
// Webcam Handling with Natural Mirroring & Mirrored Snapshot Capture
// ==========================================================================
async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      },
      audio: false
    });
    videoStream.srcObject = mediaStream;
    btnStartCamera.textContent = "Camera Ready";
    btnStartCamera.disabled = true;
    btnSnapCamera.disabled = false;
  } catch (err) {
    alert("Could not access camera: " + err.message + "\nPlease allow camera permissions or upload a selfie photo.");
  }
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
    btnStartCamera.textContent = "Start Camera";
    btnStartCamera.disabled = false;
    btnSnapCamera.disabled = true;
  }
}

async function captureSnapshot() {
  if (!videoStream.videoWidth) return;

  const canvas = document.createElement("canvas");
  canvas.width = videoStream.videoWidth;
  canvas.height = videoStream.videoHeight;
  const ctx = canvas.getContext("2d");

  // Mirror snapshot horizontally so it matches the mirrored preview exactly
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(videoStream, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (rawBlob) => {
    try {
      // Compress and optimize snapshot
      const processed = await resizeAndCompressImage(rawBlob, 1024, 0.85);
      currentImageBlob = processed.blob;
      currentBase64 = processed.base64;

      btnSnapCamera.textContent = "Snapshot Captured!";
      setTimeout(() => { btnSnapCamera.textContent = "Capture Snapshot"; }, 1600);
      updateAnalyzeButtonState();
    } catch (e) {
      console.error("Failed to process snapshot:", e);
    }
  }, "image/jpeg", 0.9);
}

// ==========================================================================
// File Upload Handling (With instant client-side resize for mobile photos)
// ==========================================================================
function handleFileSelect(e) {
  if (e.target.files.length > 0) {
    processImageFile(e.target.files[0]);
  }
}

async function processImageFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please select a valid image file (JPG, PNG, HEIC, or WebP)");
    return;
  }

  try {
    dropZone.classList.add("hidden");
    uploadPreviewWrap.classList.remove("hidden");
    uploadPreviewImg.alt = "Processing image...";

    // Smoothly downscale phone photo if large (handles 12-48MP smartphone photos)
    const processed = await resizeAndCompressImage(file, 1024, 0.85);
    currentImageBlob = processed.blob;
    currentBase64 = processed.base64;

    uploadPreviewImg.src = currentBase64;
    uploadPreviewImg.alt = "Uploaded Face Diagnostic";
    updateAnalyzeButtonState();
  } catch (err) {
    console.error("Error processing mobile image:", err);
    alert("Could not process image file: " + err.message);
    clearUploadedImage();
  }
}

function clearUploadedImage() {
  currentImageBlob = null;
  currentBase64 = null;
  fileInput.value = "";
  uploadPreviewWrap.classList.add("hidden");
  dropZone.classList.remove("hidden");
  updateAnalyzeButtonState();
}

function updateAnalyzeButtonState() {
  btnAnalyze.disabled = !currentImageBlob && !currentBase64;
}

// ==========================================================================
// Analysis Execution
// ==========================================================================
async function runAnalysis() {
  if (!currentImageBlob && !currentBase64) return;

  showLoading(true);

  const sleepHours = parseFloat(sliderSleep.value);
  const waterLiters = parseFloat(sliderWater.value);
  const stressLevel = parseFloat(sliderStress.value);

  try {
    let response;
    // Prefer multipart upload with compressed mobile-safe blob
    if (currentImageBlob) {
      const formData = new FormData();
      formData.append("file", currentImageBlob, "face_diagnostic.jpg");
      formData.append("sleep_hours", sleepHours);
      formData.append("water_liters", waterLiters);
      formData.append("stress_level", stressLevel);

      response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        body: formData,
      });
    } else {
      // Base64 fallback
      response = await fetch(`${API_BASE_URL}/api/analyze-base64`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_base64: currentBase64,
          sleep_hours: sleepHours,
          water_liters: waterLiters,
          stress_level: stressLevel,
        }),
      });
    }

    if (!response.ok) {
      const errJson = await response.json().catch(() => ({ detail: "Server response error" }));
      throw new Error(errJson.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    console.warn("Live analysis error:", err.message);
    alert(`Diagnostics Note: ${err.message}\n\nIf the server is waking up from idle, please try again in ~30 seconds, or click 'Load Demo Sample' to view instant results.`);
  } finally {
    showLoading(false);
  }
}

async function runDemoAnalysis() {
  showLoading(true);
  try {
    // Try live sample endpoint
    const res = await fetch(`${API_BASE_URL}/api/sample`);
    if (res.ok) {
      const data = await res.json();
      renderResults(data);
      showLoading(false);
      return;
    }
  } catch (e) {
    // Fallback if Render instance is spinning up
  }

  // Instant fallback benchmark sample
  const sampleData = {
    face_detected: true,
    skin_type: {
      predicted: "Combination",
      confidence: 76.0,
      probabilities: { Combination: 76.0, Dry: 12.5, Normal: 7.2, Oily: 4.3 }
    },
    lesion_severity: {
      predicted: "Mild",
      confidence: 76.0,
      probabilities: { Normal: 22.0, Mild: 76.0, Moderate: 14.2, "High Concern": 3.8 }
    },
    lifestyle_notes: [
      "Sleep < 6h: Boosted barrier dryness & acne sensitivity",
      "High Stress (>7/10): Elevated sebum and reactive barrier factor"
    ],
    suggestions: {
      skin_type: "Combination",
      lesion_severity: "Mild",
      morning_routine: [
        { step: "Step 1: Gentle Refresh Cleanse", action: "Wash with a pH-balanced amino acid gel cleanser to clear overnight sebum without stripping lipid moisture." },
        { step: "Step 2: Hydrating Barrier Serum", action: "Apply Niacinamide 3-5% and multi-molecular Hyaluronic Acid on damp skin to balance oil production." },
        { step: "Step 3: Lightweight Fluid Hydrator", action: "Smooth an oil-free water-gel on the T-zone and slightly richer barrier cream on dry cheek areas." },
        { step: "Step 4: Fluid Broad-Spectrum SPF 50", action: "Protect with non-greasy, non-comedogenic sunscreen to safeguard barrier recovery." }
      ],
      evening_routine: [
        { step: "Step 1: Clarifying Double Cleanse", action: "Dissolve daily sunscreen and pollution with micellar water followed by your gentle gel wash." },
        { step: "Step 2: Targeted Active Treatment", action: "Apply 2% Salicylic Acid (BHA) 2-3 nights per week to gently dissolve pore debris in the T-zone." },
        { step: "Step 3: Barrier Recovery Night Cream", action: "Seal hydration with a ceramide-rich moisturizer packed with Centella Asiatica and panthenol." }
      ],
      actives_to_use: [
        "Niacinamide (Sebum Balancing)",
        "Salicylic Acid 2% (Pore Clarity)",
        "Ceramide NP, AP, EOP (Lipid Repair)",
        "Hyaluronic Acid (Multi-depth Hydration)"
      ],
      actives_to_avoid: [
        "Abrasive walnut/apricot physical scrubs",
        "High-percentage denatured alcohol toners",
        "Pore-clogging heavy mineral oils on T-zone"
      ],
      expert_tips: [
        "Zone-treat your face: target oil control strictly on forehead and nose, while protecting drier cheek areas.",
        "Mild concerns detected: Maintain a steady routine for 4-6 weeks to observe clear textural improvements.",
        "Ensure consistent 7+ hours sleep to lower inflammatory cortisol and restore barrier lipid synthesis."
      ]
    },
    recommendations: [
      {
        product_name: "CeraVe Hydrating Facial Cleanser with Ceramides",
        product_type: "Cleanser",
        price: "$15.99",
        match_score: 95.8,
        paula_rating: "Best",
        key_ingredients: "Ceramides 1, 3, 6-II, Hyaluronic Acid, Glycerin, Niacinamide"
      },
      {
        product_name: "Paula's Choice 2% BHA Liquid Exfoliant",
        product_type: "Exfoliant",
        price: "$34.00",
        match_score: 93.4,
        paula_rating: "Best",
        key_ingredients: "Salicylic Acid (BHA), Camellia Oleifera Green Tea, Methylpropanediol"
      },
      {
        product_name: "La Roche-Posay Toleriane Double Repair Face Moisturizer",
        product_type: "Moisturizer",
        price: "$22.99",
        match_score: 90.2,
        paula_rating: "Best",
        key_ingredients: "Prebiotic Thermal Water, Ceramide-3, Niacinamide, Glycerin"
      },
      {
        product_name: "The Ordinary Niacinamide 10% + Zinc 1% Blemish Formula",
        product_type: "Serum",
        price: "$6.50",
        match_score: 88.7,
        paula_rating: "Good",
        key_ingredients: "Niacinamide (Vitamin B3), Zinc PCA, Tamarindus Indica Seed Extract"
      },
      {
        product_name: "EltaMD UV Clear Broad-Spectrum SPF 46 Facial Sunscreen",
        product_type: "Sunscreen",
        price: "$39.00",
        match_score: 87.5,
        paula_rating: "Best",
        key_ingredients: "Transparent Zinc Oxide 9.0%, Niacinamide 5%, Hyaluronic Acid"
      }
    ]
  };

  setTimeout(() => {
    renderResults(sampleData);
    showLoading(false);
  }, 350);
}

function showLoading(isLoading) {
  if (isLoading) {
    resultsEmpty.classList.add("hidden");
    resultsContent.classList.add("hidden");
    resultsLoading.classList.remove("hidden");
  } else {
    resultsLoading.classList.add("hidden");
  }
}

// ==========================================================================
// Results Rendering (Diagnostics, Personalized Routine & Top 5 Matches)
// ==========================================================================
function renderResults(data) {
  resultsEmpty.classList.add("hidden");
  resultsLoading.classList.add("hidden");
  resultsContent.classList.remove("hidden");

  // 1. Skin Type Diagnostics
  const skin = data.skin_type || { predicted: "Combination", confidence: 76.0, probabilities: { Combination: 76.0, Dry: 12.0, Normal: 7.0, Oily: 5.0 } };
  resSkinBadge.textContent = skin.predicted;
  resSkinConf.textContent = `${skin.confidence}%`;

  resSkinBars.innerHTML = "";
  if (skin.probabilities) {
    for (const [cls, pct] of Object.entries(skin.probabilities)) {
      resSkinBars.innerHTML += `
        <div class="bar-row">
          <div class="bar-labels">
            <span>${cls}</span>
            <span>${pct}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
    }
  }

  // 2. Lesion Concern Severity
  const lesion = data.lesion_severity || { predicted: "Mild", confidence: 76.0, probabilities: { Normal: 22.0, Mild: 76.0, Moderate: 14.0, "High Concern": 4.0 } };
  resLesionBadge.textContent = `${lesion.predicted} Concern`;
  resLesionConf.textContent = `${lesion.confidence}%`;

  // Badge severity color
  if (lesion.predicted === "Normal") {
    resLesionBadge.className = "badge-pill badge-success";
  } else if (lesion.predicted === "Mild") {
    resLesionBadge.className = "badge-pill badge-primary";
  } else if (lesion.predicted === "Moderate") {
    resLesionBadge.className = "badge-pill badge-warning";
  } else {
    resLesionBadge.className = "badge-pill badge-danger";
  }

  resLesionBars.innerHTML = "";
  if (lesion.probabilities) {
    for (const [cls, pct] of Object.entries(lesion.probabilities)) {
      resLesionBars.innerHTML += `
        <div class="bar-row">
          <div class="bar-labels">
            <span>${cls}</span>
            <span>${pct}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
    }
  }

  // 3. Lifestyle Diagnostic Pills
  resLifestyleTags.innerHTML = "";
  if (data.lifestyle_notes && data.lifestyle_notes.length > 0) {
    data.lifestyle_notes.forEach(note => {
      resLifestyleTags.innerHTML += `<span class="lifestyle-tag">${note}</span>`;
    });
    resLifestyleWrap.style.display = "block";
  } else {
    resLifestyleTags.innerHTML = `<span class="lifestyle-tag">Balanced Lifestyle: Sleep, hydration, and stress metrics within healthy equilibrium</span>`;
    resLifestyleWrap.style.display = "block";
  }

  // 4. Personalized Skincare Routine & Suggestions
  renderSkincareSuggestions(data);

  // 5. Products Recommendations (Curated Top 5)
  renderProductRecommendations(data);
}

// ==========================================================================
// Skincare Suggestions & Regimen Renderer
// ==========================================================================
function renderSkincareSuggestions(data) {
  const skinType = data.skin_type?.predicted || "Combination";
  const lesionSeverity = data.lesion_severity?.predicted || "Mild";
  const suggestions = data.suggestions || generateClientFallbackSuggestions(skinType, lesionSeverity, data.lifestyle_notes);

  // Morning Routine Steps
  resMorningRoutine.innerHTML = "";
  if (suggestions.morning_routine) {
    suggestions.morning_routine.forEach(stepItem => {
      resMorningRoutine.innerHTML += `
        <div class="routine-step-item">
          <div class="step-badge">${stepItem.step}</div>
          <div class="step-action">${stepItem.action}</div>
        </div>
      `;
    });
  }

  // Evening Routine Steps
  resEveningRoutine.innerHTML = "";
  if (suggestions.evening_routine) {
    suggestions.evening_routine.forEach(stepItem => {
      resEveningRoutine.innerHTML += `
        <div class="routine-step-item">
          <div class="step-badge pm-step">${stepItem.step}</div>
          <div class="step-action">${stepItem.action}</div>
        </div>
      `;
    });
  }

  // Recommended Actives
  resActivesUse.innerHTML = "";
  if (suggestions.actives_to_use) {
    suggestions.actives_to_use.forEach(act => {
      resActivesUse.innerHTML += `<span class="active-tag tag-use">+ ${act}</span>`;
    });
  }

  // Actives to Avoid
  resActivesAvoid.innerHTML = "";
  if (suggestions.actives_to_avoid) {
    suggestions.actives_to_avoid.forEach(act => {
      resActivesAvoid.innerHTML += `<span class="active-tag tag-avoid">✕ ${act}</span>`;
    });
  }

  // Expert Tips
  resExpertTips.innerHTML = "";
  if (suggestions.expert_tips) {
    suggestions.expert_tips.forEach(tip => {
      resExpertTips.innerHTML += `<li>${tip}</li>`;
    });
  }
}

// Client fallback suggestions if API response lacked suggestions object
function generateClientFallbackSuggestions(skinType, lesionSeverity, lifestyleNotes = []) {
  const commonNotes = lifestyleNotes.length > 0 ? lifestyleNotes : ["Ensure minimum 7h sleep and 2L water to support natural skin barrier repair."];

  if (skinType === "Dry") {
    return {
      morning_routine: [
        { step: "Step 1: Lukewarm Cleanse", action: "Rinse with lukewarm water or a non-foaming hydrating milk cleanser." },
        { step: "Step 2: Moisture Serum", action: "Apply Hyaluronic Acid and Panthenol to damp skin to lock in moisture." },
        { step: "Step 3: Ceramide Cream", action: "Layer a barrier cream rich in ceramides, squalane, and shea butter." },
        { step: "Step 4: Nourishing SPF 50", action: "Finish with a moisturizing broad-spectrum SPF to prevent trans-epidermal moisture loss." }
      ],
      evening_routine: [
        { step: "Step 1: Nourishing Balm Cleanse", action: "Dissolve SPF and daily grit with an emulsifying cleansing balm." },
        { step: "Step 2: Gentle Renewal", action: "Use a mild lactic acid or peptide serum 2 nights a week." },
        { step: "Step 3: Barrier Repair Mask", action: "Seal with an intensive ceramide sleep cream or barrier lipid ointment." }
      ],
      actives_to_use: ["Ceramides 1, 3, 6-II", "Squalane", "Hyaluronic Acid", "Centella Asiatica"],
      actives_to_avoid: ["High-foaming SLS washes", "Denatured alcohol astringents", "Harsh scrubbing beads"],
      expert_tips: [
        "Always apply humectants to damp skin immediately after cleansing.",
        "Avoid hot showers on the face to protect fragile epidermal lipids.",
        ...commonNotes
      ]
    };
  } else if (skinType === "Oily") {
    return {
      morning_routine: [
        { step: "Step 1: Purifying Cleanse", action: "Wash with a gentle amino-acid foaming wash with low-percentage BHA." },
        { step: "Step 2: Sebum Control Serum", action: "Apply Niacinamide 5-10% + Zinc PCA to refine pores and regulate oil." },
        { step: "Step 3: Oil-Free Water Gel", action: "Hydrate with a lightweight, non-comedogenic hyaluronic gel." },
        { step: "Step 4: Matte Fluid SPF 50", action: "Protect with a mattifying sunscreen with silica for midday shine control." }
      ],
      evening_routine: [
        { step: "Step 1: Deep Pore Cleanse", action: "Cleanse thoroughly to clear sebum buildup and urban pollution particles." },
        { step: "Step 2: BHA / Retinoid Treatment", action: "Apply 2% Salicylic Acid 3x weekly to clear deep follicular congestion." },
        { step: "Step 3: Soothing Night Hydration", action: "Finish with a water-based green tea or cica emulsion." }
      ],
      actives_to_use: ["Salicylic Acid (BHA)", "Zinc PCA", "Niacinamide", "Green Tea Polyphenols"],
      actives_to_avoid: ["Heavy coconut oil / petrolatum", "Stripping alcohol astringents", "Over-washing face"],
      expert_tips: [
        "Do not skip moisturizer: dehydrated oily skin overcompensates by producing more sebum.",
        "Use blotting sheets rather than re-washing skin during the day.",
        ...commonNotes
      ]
    };
  }

  // Combination & Normal default
  return {
    morning_routine: [
      { step: "Step 1: Gentle Cleanser", action: "Wash with a pH-balanced gel wash to refresh without stripping moisture." },
      { step: "Step 2: Balancing Serum", action: "Apply lightweight Niacinamide 3-5% or Hyaluronic Acid for even tone." },
      { step: "Step 3: Dual Moisturizing", action: "Use light water-gel on T-zone and richer moisturizer on dry cheek areas." },
      { step: "Step 4: Broad-Spectrum SPF", action: "Finish with a fluid broad-spectrum SPF 30-50 that leaves zero residue." }
    ],
    evening_routine: [
      { step: "Step 1: Double Cleanse", action: "Remove sunscreen and impurities with micellar water followed by a gentle wash." },
      { step: "Step 2: Targeted Exfoliation", action: "Apply 2% BHA Salicylic Acid 2-3x weekly to keep pores clear." },
      { step: "Step 3: Barrier Recovery", action: "Lock in hydration with a soothing ceramide and panthenol night cream." }
    ],
    actives_to_use: ["Niacinamide (Oil Balancing)", "Salicylic Acid (Pores)", "Ceramides (Barrier)", "Hyaluronic Acid"],
    actives_to_avoid: ["Abrasive physical scrubs", "High-alcohol drying toners", "Heavy pore-clogging waxes"],
    expert_tips: [
      "Zone-treat your face: target sebum control strictly on the T-zone while nourishing cheeks.",
      "Consistency is key: allow active ingredients 4-6 weeks to manifest structural results.",
      ...commonNotes
    ]
  };
}

// ==========================================================================
// Product Recommendations Renderer (with Fallback Catalog)
// ==========================================================================
function renderProductRecommendations(data) {
  resProductsList.innerHTML = "";

  const products = (data.recommendations && data.recommendations.length > 0)
    ? data.recommendations
    : getClientFallbackProducts(data.skin_type?.predicted);

  products.forEach((prod, index) => {
    let paulaClass = "paula-best";
    const pRating = (prod.paula_rating || "").toLowerCase();
    if (pRating.includes("good")) paulaClass = "paula-good";
    else if (pRating.includes("average")) paulaClass = "paula-average";

    // Clean price display
    let cleanPrice = String(prod.price || "$18.00").replace(/\ufffd/g, "£").replace(/\?/g, "£").trim();
    if (!cleanPrice.startsWith("$") && !cleanPrice.startsWith("£") && !cleanPrice.startsWith("€")) {
      cleanPrice = `$${cleanPrice}`;
    }

    resProductsList.innerHTML += `
      <div class="product-card">
        <div class="rank-pill">#${index + 1}</div>
        <div class="prod-details">
          <div class="prod-meta">
            <span class="prod-type">${prod.product_type || "Skincare"}</span>
            <span class="paula-tag ${paulaClass}">Paula's: ${prod.paula_rating || "Best"}</span>
          </div>
          <div class="prod-name">${prod.product_name}</div>
          <div class="prod-ingreds">${prod.key_ingredients || "Niacinamide, Ceramides, Hyaluronic Acid, Glycerin"}</div>
        </div>
        <div class="prod-stats">
          <span class="match-badge">${prod.match_score}% Match</span>
          <span class="prod-price">${cleanPrice}</span>
        </div>
      </div>
    `;
  });
}

function getClientFallbackProducts(skinType = "Combination") {
  return [
    {
      product_name: "CeraVe Hydrating Facial Cleanser with Ceramides",
      product_type: "Cleanser",
      price: "$15.99",
      match_score: 96.2,
      paula_rating: "Best",
      key_ingredients: "Ceramides 1, 3, 6-II, Hyaluronic Acid, Glycerin, Niacinamide"
    },
    {
      product_name: "Paula's Choice 2% BHA Liquid Exfoliant",
      product_type: "Exfoliant",
      price: "$34.00",
      match_score: 93.8,
      paula_rating: "Best",
      key_ingredients: "Salicylic Acid (BHA), Camellia Oleifera Green Tea, Methylpropanediol"
    },
    {
      product_name: "La Roche-Posay Toleriane Double Repair Face Moisturizer",
      product_type: "Moisturizer",
      price: "$22.99",
      match_score: 90.5,
      paula_rating: "Best",
      key_ingredients: "Prebiotic Thermal Water, Ceramide-3, Niacinamide, Glycerin"
    },
    {
      product_name: "The Ordinary Niacinamide 10% + Zinc 1% Blemish Formula",
      product_type: "Serum",
      price: "$6.50",
      match_score: 88.4,
      paula_rating: "Good",
      key_ingredients: "Niacinamide (Vitamin B3), Zinc PCA, Tamarindus Seed Gum"
    },
    {
      product_name: "EltaMD UV Clear Broad-Spectrum SPF 46 Facial Sunscreen",
      product_type: "Sunscreen",
      price: "$39.00",
      match_score: 87.1,
      paula_rating: "Best",
      key_ingredients: "Transparent Zinc Oxide 9.0%, Niacinamide 5%, Hyaluronic Acid"
    }
  ];
}
