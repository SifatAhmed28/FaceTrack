/**
 * FaceTrack AI Skincare Advisor — Client Application
 * Handles webcam stream, image upload, API communication,
 * and dynamic diagnostic dashboard rendering.
 */

// State
let currentApiUrl = localStorage.getItem("facetrack_api_url") || "http://localhost:8000";
let activeMode = "webcam"; // "webcam" | "upload"
let mediaStream = null;
let currentImageBlob = null; // Blob or File ready for analysis
let currentBase64 = null;

// DOM Elements
const backendStatusBadge = document.getElementById("backend-status-badge");
const backendStatusText = document.getElementById("backend-status-text");
const settingsBtn = document.getElementById("settings-btn");

// Tabs
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

const resLifestyleTags = document.getElementById("res-lifestyle-tags");
const resProductsList = document.getElementById("res-products-list");

// Modal
const settingsModal = document.getElementById("settings-modal");
const btnCloseModal = document.getElementById("btn-close-modal");
const apiUrlInput = document.getElementById("api-url-input");
const btnTestApi = document.getElementById("btn-test-api");
const btnSaveApi = document.getElementById("btn-save-api");
const modalTestResult = document.getElementById("modal-test-result");

// ==========================================================================
// Initialization
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
  apiUrlInput.value = currentApiUrl;
  checkBackendHealth();
  setupEventListeners();
});

// ==========================================================================
// Backend Health Checker
// ==========================================================================
async function checkBackendHealth() {
  backendStatusBadge.className = "status-badge status-checking";
  backendStatusText.textContent = "Connecting to API...";

  try {
    const res = await fetch(`${currentApiUrl}/api/health`, { method: "GET" });
    if (res.ok) {
      const data = await res.json();
      backendStatusBadge.className = "status-badge status-connected";
      backendStatusText.textContent = `Online (${data.models?.total_products || "1k+"} prods)`;
      return true;
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    backendStatusBadge.className = "status-badge status-disconnected";
    backendStatusText.textContent = "API Offline (Click Settings)";
    return false;
  }
}

// ==========================================================================
// Event Listeners
// ==========================================================================
function setupEventListeners() {
  // Tabs
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

  // Analyze & Demo
  btnAnalyze.addEventListener("click", runAnalysis);
  btnDemo.addEventListener("click", runDemoAnalysis);

  // Settings Modal
  settingsBtn.addEventListener("click", () => {
    apiUrlInput.value = currentApiUrl;
    modalTestResult.className = "test-result-box hidden";
    settingsModal.classList.remove("hidden");
  });
  btnCloseModal.addEventListener("click", () => settingsModal.classList.add("hidden"));
  btnTestApi.addEventListener("click", testCustomApiUrl);
  btnSaveApi.addEventListener("click", saveCustomApiUrl);
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
// Webcam Handling
// ==========================================================================
async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false
    });
    videoStream.srcObject = mediaStream;
    btnStartCamera.textContent = "Camera Ready";
    btnStartCamera.disabled = true;
    btnSnapCamera.disabled = false;
  } catch (err) {
    alert("Could not access camera: " + err.message + "\nPlease allow camera permissions or use Photo Upload.");
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

function captureSnapshot() {
  if (!videoStream.videoWidth) return;

  const canvas = document.createElement("canvas");
  canvas.width = videoStream.videoWidth;
  canvas.height = videoStream.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(videoStream, 0, 0, canvas.width, canvas.height);

  canvas.toBlob((blob) => {
    currentImageBlob = blob;
    currentBase64 = canvas.toDataURL("image/jpeg", 0.9);
    btnSnapCamera.textContent = "Captured!";
    setTimeout(() => { btnSnapCamera.textContent = "Capture Snapshot"; }, 1500);
    updateAnalyzeButtonState();
  }, "image/jpeg", 0.9);
}

// ==========================================================================
// File Upload Handling
// ==========================================================================
function handleFileSelect(e) {
  if (e.target.files.length > 0) {
    processImageFile(e.target.files[0]);
  }
}

function processImageFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please select a valid image file (JPG or PNG)");
    return;
  }

  currentImageBlob = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    currentBase64 = e.target.result;
    uploadPreviewImg.src = currentBase64;
    dropZone.classList.add("hidden");
    uploadPreviewWrap.classList.remove("hidden");
    updateAnalyzeButtonState();
  };
  reader.readAsDataURL(file);
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
    // Prefer multipart upload
    if (currentImageBlob) {
      const formData = new FormData();
      formData.append("file", currentImageBlob, "face_capture.jpg");
      formData.append("sleep_hours", sleepHours);
      formData.append("water_liters", waterLiters);
      formData.append("stress_level", stressLevel);

      response = await fetch(`${currentApiUrl}/api/analyze`, {
        method: "POST",
        body: formData,
      });
    } else {
      // Base64 fallback
      response = await fetch(`${currentApiUrl}/api/analyze-base64`, {
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
      const errJson = await response.json().catch(() => ({ detail: "Server error" }));
      throw new Error(errJson.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    renderResults(data);
  } catch (err) {
    alert(`Analysis Failed: ${err.message}\n\nTip: If deploying on Render, the free backend may take 30-45s to wake up on the first request. Click 'Load Demo Sample' in the meantime!`);
  } finally {
    showLoading(false);
  }
}

async function runDemoAnalysis() {
  showLoading(true);
  try {
    // Try to fetch sample from backend
    const res = await fetch(`${currentApiUrl}/api/sample`);
    if (res.ok) {
      const data = await res.json();
      renderResults(data);
      return;
    }
  } catch (e) {
    // Backend offline: use hardcoded fallback sample
  }

  // Fallback sample data
  const sampleData = {
    face_detected: true,
    skin_type: {
      predicted: "Combination",
      confidence: 78.4,
      probabilities: { Combination: 78.4, Dry: 11.2, Normal: 6.8, Oily: 3.6 }
    },
    lesion_severity: {
      predicted: "Mild",
      confidence: 64.1,
      probabilities: { Normal: 22.5, Mild: 64.1, Moderate: 10.4, "High Concern": 3.0 }
    },
    lifestyle_notes: [
      "Sleep < 6h: Boosted dryness & acne concern factor",
      "Daily Stress: Elevated sebum and barrier sensitivity"
    ],
    recommendations: [
      {
        product_name: "CeraVe Hydrating Facial Cleanser",
        product_type: "Cleanser",
        price: "$15.99",
        match_score: 94.8,
        paula_rating: "Best",
        key_ingredients: "Ceramides 1, 3, 6-II, Hyaluronic Acid, Glycerin"
      },
      {
        product_name: "Paula's Choice 2% BHA Liquid Exfoliant",
        product_type: "Exfoliant",
        price: "$34.00",
        match_score: 91.2,
        paula_rating: "Best",
        key_ingredients: "Salicylic Acid, Camellia Oleifera Green Tea Extract"
      },
      {
        product_name: "La Roche-Posay Toleriane Double Repair",
        product_type: "Moisturizer",
        price: "$21.99",
        match_score: 88.5,
        paula_rating: "Best",
        key_ingredients: "Prebiotic Thermal Water, Ceramide-3, Niacinamide"
      },
      {
        product_name: "The Ordinary Niacinamide 10% + Zinc 1%",
        product_type: "Serum",
        price: "$6.00",
        match_score: 86.3,
        paula_rating: "Good",
        key_ingredients: "Niacinamide, Zinc PCA, Pentylene Glycol"
      },
      {
        product_name: "EltaMD UV Clear Broad-Spectrum SPF 46",
        product_type: "Sunscreen",
        price: "$39.00",
        match_score: 84.7,
        paula_rating: "Best",
        key_ingredients: "Zinc Oxide 9.0%, Niacinamide 5%, Hyaluronic Acid"
      }
    ]
  };

  setTimeout(() => {
    renderResults(sampleData);
    showLoading(false);
  }, 400);
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
// Results Rendering
// ==========================================================================
function renderResults(data) {
  resultsEmpty.classList.add("hidden");
  resultsLoading.classList.add("hidden");
  resultsContent.classList.remove("hidden");

  // 1. Skin Type
  const skin = data.skin_type;
  resSkinBadge.textContent = skin.predicted;
  resSkinConf.textContent = `${skin.confidence}%`;

  resSkinBars.innerHTML = "";
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

  // 2. Lesion Severity
  const lesion = data.lesion_severity;
  resLesionBadge.textContent = lesion.predicted;
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

  // 3. Lifestyle Notes
  resLifestyleTags.innerHTML = "";
  if (data.lifestyle_notes && data.lifestyle_notes.length > 0) {
    data.lifestyle_notes.forEach(note => {
      resLifestyleTags.innerHTML += `<span class="lifestyle-tag">${note}</span>`;
    });
    document.getElementById("res-lifestyle-wrap").style.display = "block";
  } else {
    resLifestyleTags.innerHTML = `<span class="lifestyle-tag">Healthy baseline: Sleep, water, and stress balanced</span>`;
    document.getElementById("res-lifestyle-wrap").style.display = "block";
  }

  // 4. Products Recommendations
  resProductsList.innerHTML = "";
  if (data.recommendations && data.recommendations.length > 0) {
    data.recommendations.forEach((prod, index) => {
      let paulaClass = "paula-good";
      const pRating = (prod.paula_rating || "").toLowerCase();
      if (pRating.includes("best")) paulaClass = "paula-best";
      else if (pRating.includes("average")) paulaClass = "paula-average";

      resProductsList.innerHTML += `
        <div class="product-card">
          <div class="rank-pill">#${index + 1}</div>
          <div class="prod-details">
            <div class="prod-meta">
              <span class="prod-type">${prod.product_type}</span>
              <span class="paula-tag ${paulaClass}">Paula's: ${prod.paula_rating}</span>
            </div>
            <div class="prod-name">${prod.product_name}</div>
            <div class="prod-ingreds">${prod.key_ingredients}</div>
          </div>
          <div class="prod-stats">
            <span class="match-badge">${prod.match_score}% Match</span>
            <span class="prod-price">${prod.price}</span>
          </div>
        </div>
      `;
    });
  }
}

// ==========================================================================
// Settings Modal Logic
// ==========================================================================
async function testCustomApiUrl() {
  const url = apiUrlInput.value.trim().replace(/\/$/, "");
  modalTestResult.className = "test-result-box";
  modalTestResult.textContent = "Testing connection to " + url + "...";
  modalTestResult.classList.remove("hidden");

  try {
    const res = await fetch(`${url}/api/health`);
    if (res.ok) {
      const data = await res.json();
      modalTestResult.className = "test-result-box test-success";
      modalTestResult.textContent = `Connection Successful! Backend online with ${data.models?.total_products || 0} products in KB.`;
    } else {
      throw new Error(`Server returned HTTP ${res.status}`);
    }
  } catch (err) {
    modalTestResult.className = "test-result-box test-error";
    modalTestResult.textContent = `Connection Failed: ${err.message}. Make sure the Render URL is correct and active.`;
  }
}

function saveCustomApiUrl() {
  const url = apiUrlInput.value.trim().replace(/\/$/, "");
  currentApiUrl = url || "http://localhost:8000";
  localStorage.setItem("facetrack_api_url", currentApiUrl);
  settingsModal.classList.add("hidden");
  checkBackendHealth();
}
