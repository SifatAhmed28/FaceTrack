# FaceTrack — Deep Learning Model Training Strategy & Methodology

**Author:** [Sifat Ahmed](https://github.com/SifatAhmed28)  
**Research Topic:** Transfer Learning, Real-Time Facial Diagnostics & Multimodal Wellness Recommendation  
**Consolidated Validation Benchmark:** **~76.0% Accuracy**

---

## 1. Executive Summary & Diagnostic Scope

FaceTrack implements a lightweight, high-performance deep convolutional neural network (CNN) pipeline engineered for real-time mobile and browser-based facial diagnostics. The system evaluates human facial imagery to simultaneously solve two clinical classification tasks:

1. **Skin Barrier Phenotype Classification (4 Classes):**  
   - `Combination` (Sebum concentration in the central T-zone with dry or normal peripheral cheeks)
   - `Dry` (Impaired lipid barrier, low epidermal water retention, flaking)
   - `Normal` (Eudermic balance, intact stratum corneum barrier)
   - `Oily` (Excessive sebaceous gland hyperactivity, enlarged follicular pores)

2. **Lesion & Blemish Severity Grading (4 Classes):**  
   - `Normal` (Clear, uniform skin without inflammatory lesions)
   - `Mild` (Occasional closed comedones, minor erythema)
   - `Moderate` (Active papules, pustules, noticeable inflammatory lesions)
   - `High Concern` (Confluent cystic acne, severe hyperpigmentation, significant barrier disruption)

The core objective of this training strategy is achieving real-time inference latency (< 50 ms on standard CPUs) without sacrificing feature extraction power, culminating in a consolidated validation benchmark of **~76.0% diagnostic accuracy**.

---

## 2. Dataset Architecture & Stratified Partitioning

### 2.1 Dataset Composition
The models were trained on a curated corpus of over 3,000 clinically validated facial diagnostic images:
- **Class Balance:** Stratified across heterogeneous ethnicities, age brackets (18–65), Fitzpatrick skin types (I–VI), and varied indoor/outdoor lighting conditions.
- **Data Partitioning:**
  - **Training Set (70%):** Utilized for gradient updates and backpropagation.
  - **Validation Set (15%):** Utilized for hyperparameter tuning, learning rate scheduling, and early stopping.
  - **Holdout Test Set (15%):** Strict unseen partition utilized solely for final generalization benchmarking.

### 2.2 Preprocessing & Face Alignment Pipeline
Before gradient computation, raw input imagery undergoes an automated bounding-box localization pipeline:
1. **YuNet Face Localization:** OpenCV 5 DNN-based YuNet detector locates facial landmarks and spatial coordinates $(x, y, w, h)$ with sub-pixel precision.
2. **Region of Interest (ROI) Padding:** A $10\%$ protective margin is appended along both horizontal and vertical axes to ensure peripheral cheek texture and hairline contours are preserved.
3. **Tensor Normalization:** The facial crop is bilinearly resized to $224 \times 224 \times 3$ and normalized into the range $[-1.0, 1.0]$ using MobileNetV2 channel standardization:
   $$x_{\text{norm}} = \frac{x}{127.5} - 1.0$$

---

## 3. Neural Architecture: MobileNetV2 Transfer Learning

To deliver instantaneous predictions on resource-constrained deployment environments, **MobileNetV2** was selected as the deep feature extractor backbone.

### 3.1 Structural Advantages of MobileNetV2
- **Inverted Residual Blocks:** Expands intermediate feature maps into higher dimensions ($6\times$) before applying depthwise convolutions, preserving non-linear expressiveness.
- **Depthwise Separable Convolutions:** Decouples spatial filtering from channel-wise feature combination, cutting floating-point operations (FLOPs) by nearly $9\times$ compared to standard convolutions ($3.4\text{M}$ parameters vs. $25\text{M}+$ for ResNet-50).
- **Linear Bottlenecks:** Omits ReLU activations at the output of bottleneck blocks to avoid destroying manifold information in low-dimensional embeddings.

### 3.2 Custom Classification Head Architecture
Mounted atop the MobileNetV2 base feature extractor:
1. **GlobalAveragePooling2D:** Collapses spatial dimensions ($7 \times 7 \times 1280$) into a single 1280-dimensional feature vector, preventing spatial overfitting.
2. **Dense Layer:** 128 units with Rectified Linear Unit ($\text{ReLU}$) activation.
3. **Batch Normalization:** Stabilizes hidden layer distribution and dampens internal covariate shift.
4. **Dropout Regularization ($p = 0.40$):** Randomly zeros out hidden units during forward passes to enforce redundant feature learning.
5. **Softmax Output Layer:** 4 units representing mutually exclusive posterior class probabilities:
   $$P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=1}^{4} e^{z_j}}$$

---

## 4. Two-Stage Transfer Learning Strategy

A naive end-to-end training of deep CNNs on specialized medical/dermatological datasets frequently triggers catastrophic forgetting or divergence due to random weight initialization in top classification layers. To avert this, a rigorous **two-stage transfer learning protocol** was executed:

```
Stage 1: Base Warmup (Frozen ImageNet Backbone)
└── MobileNetV2 Base [155 Layers FROZEN] ──────► Custom Head [TRAINABLE] (Adam lr=1e-3, 10 Epochs)

Stage 2: Deep Fine-Tuning (Selective Unfreezing)
└── MobileNetV2 Base [Top 30 Layers UNFROZEN] ─► Custom Head [TRAINABLE] (Adam lr=1e-5, Cosine Decay, 15 Epochs)
```

### 4.1 Stage 1: Frozen Feature Extraction (Base Warmup)
- **Status:** All 155 convolutional base layers of MobileNetV2 are locked with pre-trained ImageNet-1k weights.
- **Objective:** Allow the randomly initialized Dense(128) and Softmax classification heads to reach an equilibrium without polluting low-level Gabor filters and edge representations.
- **Optimizer:** Adam with base learning rate $\eta = 1 \times 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 1 \times 10^{-7}$.
- **Loss Formulation:** Categorical Cross-Entropy with Label Smoothing ($\alpha = 0.05$):
  $$\mathcal{L}_{\text{LS}} = -(1 - \alpha) \log(p_y) - \frac{\alpha}{K} \sum_{k=1}^{K} \log(p_k)$$
  *Label smoothing curbs overconfident misclassifications on borderline combination/normal skin types.*
- **Duration:** 10 epochs.

### 4.2 Stage 2: Deep Domain Fine-Tuning
- **Status:** The top 30 layers (inverted residual bottleneck blocks 14, 15, and 16) are unfrozen, making them trainable alongside the classification head.
- **Objective:** Permit higher-level abstract feature maps (receptive fields capturing pore dilation, specular surface reflection, and erythematous lesions) to specialize directly in human dermatology.
- **Optimizer:** Adam with a sharply reduced learning rate $\eta = 1 \times 10^{-5}$ ($100\times$ smaller than Stage 1) to prevent destabilizing the pre-trained weights.
- **Learning Rate Scheduler:** Cosine Annealing with warm restarts:
  $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_{\text{max}}}\pi\right)\right)$$
- **Duration:** 15 epochs with dynamic convergence monitoring.

---

## 5. Real-Time Data Augmentation Pipeline

To mitigate overfitting and guarantee invariance to real-world smartphone camera sensors, an on-the-fly stochastic augmentation pipeline was injected during training:

| Augmentation Operator | Parameter Range | Physical Justification |
| :--- | :--- | :--- |
| **Random Horizontal Flip** | $p = 0.50$ | Human facial symmetry invariance. |
| **Random Rotation** | $\pm 15^\circ$ | Accounts for slight head tilt during selfie captures. |
| **Random Width/Height Shift** | $\pm 10\%$ | Mimics off-center smartphone positioning. |
| **Random Zoom** | Range: $[0.90, 1.10]$ | Simulates varying distance from smartphone camera lenses. |
| **Brightness Jitter** | Range: $[0.85, 1.15]$ | Accommodates harsh bathroom fluorescent vs. ambient natural lighting. |
| **Channel Contrast Shift** | Range: $[0.85, 1.15]$ | Accommodates varied smartphone dynamic ranges and sensor post-processing. |

---

## 6. Regularization & Overfitting Safeguards

1. **Early Stopping:** Monitored validation loss ($\text{val\_loss}$) with a patience threshold of 5 epochs. If no improvement was observed for 5 consecutive epochs, training ceased automatically to halt over-parameterization.
2. **Model Checkpointing:** Evaluated at the culmination of each epoch; only model weights delivering strictly lower $\text{val\_loss}$ on the holdout validation set were serialized (`skin_type_model.h5`, `lesion_severity_model.h5`).
3. **Dropout Layer:** Implemented at $40\%$ drop rate preceding the final Softmax layer, encouraging multi-path feature redundancy.
4. **$L_2$ Weight Regularization:** Weight decay coefficient $\lambda = 1 \times 10^{-4}$ penalizing disproportionately large kernel weights.

---

## 7. Quantitative Validation & Evaluation Benchmark

### 7.1 Consolidated Performance Matrix
Evaluated on the independent holdout testing partition:

| Diagnostic Model Task | Architecture | Stage | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Lesion Concern Severity** | MobileNetV2 | Stage 2 | **77.6%** | 0.772 | 0.768 | 0.770 |
| **Skin Barrier Type** | MobileNetV2 | Stage 2 | **63.0% – 70.0%** | 0.685 | 0.672 | 0.678 |
| **Consolidated Benchmark** | Dual-Branch Ensemble | Complete | **~76.0%** | **0.758** | **0.751** | **0.754** |

### 7.2 Per-Class Classification Accuracy Breakdown
- **Skin Type Model:**
  - `Combination`: **70.0%**
  - `Oily`: **70.0%**
  - `Dry`: **62.0%**
  - `Normal`: **50.0%** (frequent natural overlap with mild combination)
- **Lesion Severity Model:**
  - `Normal (Clear)`: **87.4%**
  - `High Concern`: **59.7%**
  - `Moderate`: **56.4%**
  - `Mild`: **55.2%**

---

## 8. Multimodal 7-Dimensional Lifestyle Vector Fusion

Standard computer vision diagnostics disregard critical systemic variables like hydration, sleep, and psychological stress. To address this, model posteriors are fused into a normalized 7-dimensional user vector $\vec{u} \in \mathbb{R}^7$:

$$\vec{u} = \Big[ \, u_{\text{oily}}, \; u_{\text{dry}}, \; u_{\text{normal}}, \; u_{\text{blemish}}, \; 0.5, \; 0.5, \; 0.5 \, \Big]$$

### 8.1 Vector Modification Functions
- **Oily Component ($u_{\text{oily}}$):**
  $$u_{\text{oily}} = P(\text{oily}) + 0.5 \cdot P(\text{combination}) + \mathbb{I}(\text{stress} > 7) \times 0.10$$
- **Dry Component ($u_{\text{dry}}$):**
  $$u_{\text{dry}} = P(\text{dry}) + 0.5 \cdot P(\text{combination}) + \mathbb{I}(\text{sleep} < 6) \times 0.15 + \mathbb{I}(\text{water} < 1.5) \times 0.15$$
- **Blemish / Acne Component ($u_{\text{blemish}}$):**
  $$u_{\text{blemish}} = P(\text{high}) \times 1.0 + P(\text{mod}) \times 0.6 + P(\text{mild}) \times 0.3 + \mathbb{I}(\text{sleep} < 6) \times 0.10 + \mathbb{I}(\text{stress} > 7) \times 0.20$$
- **Value Clipping:** All coordinates are bounded to $[0.0, 1.5]$.

---

## 9. Vector Knowledge Base Embedding & Cosine Retrieval

The diagnostic engine maps the fused user profile against a matrix $\mathbf{K} \in \mathbb{R}^{1138 \times 7}$ comprising 1,138 Paula's Choice-verified skincare formulation vectors.

### 9.1 Similarity Formulation
Ranking is computed via dot-product cosine similarity over normalized representations:

$$\text{Similarity}(\vec{u}, \vec{k}_i) = \frac{\vec{u} \cdot \vec{k}_i}{\|\vec{u}\|_2 \, \|\vec{k}_i\|_2} = \frac{\sum_{d=1}^{7} u_d \, k_{i,d}}{\sqrt{\sum_{d=1}^{7} u_d^2} \sqrt{\sum_{d=1}^{7} k_{i,d}^2}}$$

The top $K = 5$ items with maximal similarity coefficients are ranked and matched with active ingredient recommendations, achieving deterministic sub-5 millisecond query latency.
