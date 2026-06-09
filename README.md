# Deepfake Detection on FaceForensics++
> A two-stage deep learning system for detecting and classifying facial manipulation using MobileNetV2 and EfficientNet-B0 on the FaceForensics++ benchmark dataset.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Notebooks](#notebooks)
  - [Notebook 1 - Binary Detection](#notebook-1--binary-detection-real-vs-fake)
  - [Notebook 2 - Manipulation Classifier](#notebook-2--manipulation-type-classifier-3-class)
- [Baseline vs Final Binary Notebook](#-baseline-vs-final-binary-notebook)
- [Architecture](#architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Visualizations](#visualizations)
- [Setup & Requirements](#setup--requirements)
- [How to Run](#how-to-run)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)

## Overview

This project implements a **complete deepfake detection pipeline** built on top of the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset. It consists of two complementary notebooks:

| Notebook | Task | Model | Classes |
|---|---|---|---|
| `deepfake_detection_tuned.ipynb` | Binary detection | MobileNetV2 | Real vs Fake |
| `deepfake_multiclass.ipynb` | Manipulation classification | EfficientNet-B0 + FrequencyBranch | Deepfakes / Face2Face / FaceSwap |

**Key highlights:**
- Runs fully on **CPU** (no GPU required)
- Uses the **official FF++ train/val/test splits** - no data leakage
- Two-stage transfer learning with progressive backbone unfreezing
- Frequency-domain features via FFT for artifact detection
- MixUp augmentation to reduce overfitting
- Video-level (group-level) evaluation in addition to frame-level
- Robustness evaluation under JPEG compression and Gaussian noise
- Grad-CAM visualizations to interpret model decisions

## Project Structure

```
your_project_folder/
│
├── deepfake_detection_tuned.ipynb        ← Notebook 1: Binary (Real vs Fake)
├── deepfake_multiclass.ipynb       ← Notebook 2: 3-Class manipulation type
│
├── dataset_300/                    ← Extracted face frames
│   ├── real/
│   ├── Deepfakes/
│   ├── Face2Face/
│   ├── FaceSwap/
│   └── NeuralTextures/
│
├── FaceForensics_300/              ← Raw downloaded videos
│   ├── original_sequences/
│   │   └── youtube/c23/
│   └── manipulated_sequences/
│       ├── Deepfakes/c23/
│       ├── Face2Face/c23/
│       ├── FaceSwap/c23/
│       └── NeuralTextures/c23/
│
├── train.json                      ← Official FF++ split files
├── val.json
├── test.json
│
├── final_officialsplit_index_1.csv ← Auto-generated frame index (binary)
├── multiclass_index.csv            ← Auto-generated frame index (multiclass)
│
├── best_model_officialsplit_1.pth  ← Best binary model checkpoint
├── best_multiclass.pth             ← Best multiclass model checkpoint
│
└── multiclass_results/             ← All output visualizations
    ├── training_curves.png
    ├── confusion_matrix_frame.png
    ├── confusion_matrix_group.png
    ├── per_class_metrics.png
    ├── robustness_plot.png
    ├── error_analysis.png
    └── gradcam.png
```

---

## Dataset

**FaceForensics++** is the standard benchmark for deepfake detection research.

| Property | Value |
|---|---|
| Videos per class | 300 |
| Manipulation types | Deepfakes, Face2Face, FaceSwap, NeuralTextures |
| Compression | c23 (high quality) |
| Frames extracted per video | 20 |
| Frame resolution | 224 × 224 px |
| Split protocol | Official FF++ JSON files |

### Split Protocol

The official split files contain **pairs of (source, target) video IDs**:

```
train.json  →  720 pairs  →  used for training
val.json    →   90 pairs  →  used for validation + threshold tuning
test.json   →   90 pairs  →  used for final evaluation only
```

**Real videos** are assigned to splits based on whether their ID appears in the split's pair list. This ensures zero video-level overlap between splits: a critical requirement for unbiased evaluation.

```python
# Leakage check output (should always be 0)
train ∩ val  pairs: 0
train ∩ test pairs: 0
val   ∩ test pairs: 0
```

### Frame Extraction

Frames are extracted using OpenCV with **Haar Cascade face detection**:
1. Sample 20 frames uniformly across each video
2. Detect the largest face with 40% padding
3. Fall back to the center square crop if no face is detected
4. Resize to 224 × 224 and save as JPEG

## Notebooks

### Notebook 1  Binary Detection (Real vs Fake)

**Goal:** Classify any video frame as either real (authentic) or fake (manipulated by any method).

#### Pipeline

```
Raw Videos → Frame Extraction → Index Building → Training → Threshold Tuning → Evaluation
```

#### Key Components

**Model - MobileNetV2:**
```
Backbone (frozen in Stage 1)
    ↓
Custom Classifier:
    Dropout(0.3)
    Linear(1280 → 256)
    ReLU
    Dropout(0.2)
    Linear(256 → 2)
```

**Two-Stage Training:**
- **Stage 1 (5 epochs):** Only the classifier head is trained. Backbone is fully frozen. High learning rate (`1e-3`).
- **Stage 2 (up to 15 epochs):** Last 4 backbone blocks are unfrozen. Low learning rate (`1e-5`). Early stopping with patience=4.

**Threshold Tuning:**
Instead of using the default 0.5 threshold, the model searches 39 thresholds on the validation set to find the best accuracy and F1 separately. This is especially important given the class imbalance (1 real: 4 fake).

**Ensemble:**
Two MobileNetV2 variants are trained:
- `v1`  simple head (Dropout → Linear → 2)
- `v2`  deep head (Dropout → Linear(256) → ReLU → Dropout → Linear(2))

Their predictions are combined in two ways:
- **Simple ensemble:** `(p_v1 + p_v2) / 2`
- **Weighted ensemble:** Weights assigned proportional to validation AUC

### Notebook 2  Manipulation Type Classifier (3-Class)

**Goal:** Given a fake frame, identify which manipulation method was used: Deepfakes, Face2Face, or FaceSwap.

> Note: This is a **fake-only** classifier. Real frames are excluded because they carry no manipulation signal and would make the task trivial, while degrading class separation.

#### Pipeline
```
Fake Frames (3 classes) → Index Building → Training → Group-Level Eval → Robustness Testing
```

#### Key Components

**Model  EfficientNet-B0 + FrequencyBranch:**

```
Input Image (224×224×3)
       │
       ├──────────────────────┐
       │                      │
  Spatial Path           Frequency Path
  EfficientNet-B0        FFT → log magnitude
  features[-1:]          3-layer CNN
  AdaptiveAvgPool        Linear projection
  [1280-dim]             [128-dim]
       │                      │
       └──────────┬───────────┘
                  │
            Concatenate [1408-dim]
                  │
            Dropout(0.4)
            Linear(1408 → 512)
            BatchNorm1D
            ReLU
            Dropout(0.2)
            Linear(512 → 3)
```

**FrequencyBranch  Why it matters:**

Each manipulation method leaves different artifacts in the frequency domain:
- **Deepfakes** (GAN-based): periodic patterns from the decoder upsampling
- **Face2Face** (expression transfer): high-frequency boundary mismatches at blending edges
- **FaceSwap** (region swap): color histogram discontinuities visible as DC shifts in FFT

A spatial CNN learns these slowly and imperfectly. The FrequencyBranch computes the FFT magnitude directly, applies log-scale compression (equivalent to log-mel in audio), and processes it with a lightweight 3-layer CNN (~50K params).

**Loss Function- SoftCrossEntropyLoss:**

A custom loss that accepts both hard integer labels and soft MixUp labels:
```python
loss = -(soft_targets * log_softmax(logits)).sum(dim=1).mean()
```
With `label_smoothing=0.10` - higher than the binary task, because Face2Face and FaceSwap are visually similar.

**Two-Stage Training:**
- **Stage 1 (5 epochs):** Head + FrequencyBranch trained. EfficientNet backbone frozen. Cosine LR annealing.
- **Stage 2 (up to 25 epochs):** Last 3 EfficientNet blocks unfrozen. 10× lower LR. CosineAnnealingWarmRestarts. Early stopping with patience=6.

##  Baseline vs Final Binary Notebook

The binary detection system (`deepfake_detection_tuned.ipynb`) evolved from a minimal prototype (`deepfake_detection_local_windows.ipynb`). The table below shows **exactly** what changed and what stayed the same based on direct code comparison between the two files.

### At a Glance

| Property | Baseline | Final |
|---|---|---|
| Videos per class | **100** | **300** |
| Frames per video | **10** | **20** |
| Total training frames | ~5,000 | ~30,000 |
| `label_smoothing` | 0.05 | 0.05 *(unchanged)* |
| Class weights in loss | ✖️ | ✅ `[4.0, 1.0]` |
| Ensemble (v1 + v2) | ✖️ | ✅ simple + weighted |
| Error analysis plot | ✖️ | ✅ |
| Grad-CAM plot | ✖️ | ✅ |
| Ablation study table | ✖️ | ✅ *(hardcoded numbers)* |
| Model search utility | ✖️ | ✅|

> **Note:** MixUp, stronger augmentation, label_smoothing=0.10, gap monitor, and dynamic ablation table were improvements planned and discussed but applied in the **fixed version** (`multiclass_fake.ipynb`), not in this intermediate version.

### 1. Dataset Scale  100 → 300 Videos, 10 → 20 Frames

**Baseline:**
```python
max_videos_per_folder: int = 100
frames_per_video:      int = 10
dataset_dir: str = os.path.join(BASE_DIR, 'dataset_100')
root_dir:    str = os.path.join(BASE_DIR, 'FaceForensics_100')
```

**Final:**
```python
max_videos_per_folder: int = 300
frames_per_video:      int = 20
dataset_dir: str = os.path.join(BASE_DIR, 'dataset_300')
root_dir:    str = r'C:\...\FaceForensics_300'   # absolute path
```

**Why it matters:** Tripling the videos and doubling the frames per video gives **6× more training data**. The baseline had too few samples to learn generalizable features, the model memorized identities instead of manipulation artifacts. At 300 videos, enough identity diversity exists that the model is forced to learn the actual manipulation signal.

### 2. Class Weights in Loss  Added

**Baseline:** Unweighted cross-entropy. The 1:4 real/fake imbalance was handled only by `WeightedRandomSampler`:
```python
criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
```

**Final:** Explicit class weights added to the loss function:
```python
class_w   = torch.tensor([4.0, 1.0], dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=cfg.label_smoothing)
```

**Why it matters:** `WeightedRandomSampler` balances at the batch-sampling level- each batch sees roughly equal real/fake counts. But the loss function still treats misclassifying real as equally costly as misclassifying fake. Adding `weight=[4.0, 1.0]` makes misclassifying a real frame 4× more costly in the loss, which directly reduces false positives (real frames predicted as fake), the most common error in the baseline confusion matrix.

### 3. Ensemble (v1 + v2) - Added

**Baseline:** Single model, only  one architecture, one checkpoint.

**Final:** Two MobileNetV2 variants trained separately and combined:

```python
# v1 : simple head (trained on ~50 videos, loaded from external checkpoint)
classifier: Dropout(0.2) → Linear(1280 → 2)

# v2: deep head (trained on 300 videos, current session)
classifier: Dropout(0.3) → Linear(1280 → 256) → ReLU → Dropout(0.2) → Linear(256 → 2)
```

**Simple Ensemble:**
```python
p_ens = (p_v1 + p_v2) / 2
threshold_ens, _, _ = tune_threshold(y_val, p_val_ens)
```

**Weighted Ensemble  weights from val AUC:**
```python
w1 = auc_v1 / (auc_v1 + auc_v2)   # ≈ 0.452
w2 = auc_v2 / (auc_v1 + auc_v2)   # ≈ 0.548
p_weighted = w1 * p_v1 + w2 * p_v2
```

**Results on test set:**
```
MobileNetV2 v1              66.38%   F1 0.613   AUC 0.662
MobileNetV2 v2 (ours)       73.70%   F1 0.710   AUC 0.812
Simple Ensemble             74.06%   F1 0.708   AUC 0.792
Weighted Ensemble           72.86%   F1 0.709   AUC 0.796
```

The simple ensemble wins on accuracy (74.06%). v2 wins on AUC (0.812); the weighted ensemble trades a little accuracy for a marginally higher AUC (0.796).

### 4. Visualizations - None → 5 Plots

**Baseline:** Only a printed robustness table (`print` statements), no saved figures.

**Final:** Five complete visualization functions added:

| Plot | What It Shows |
|---|---|
| **Confusion Matrix** | TP/TN/FP/FN with normalized % per cell, threshold annotated, metrics sidebar |
| **Robustness Plot** | Accuracy/F1/AUC vs JPEG quality (q=80,60,40,20) and Gaussian noise (σ=5,10,20,40) |
| **Error Analysis** | Top-16 highest-confidence wrong predictions shown as images with true/predicted labels |
| **Grad-CAM** | Heatmap overlays on real and fake samples- shows which face regions trigger the decision |
| **Ablation Study Table** | Side-by-side model comparison (v1, v2, simple ensemble, weighted ensemble) |

>  `plot_training_curves` is defined but **commented out** in the run cell - this is because the `history` dict was not saved during training in this version. Fixed in `deepfake_detection_fixed.ipynb`.

### 5. Model Search Utility - Added

Two helper cells added to locate and inspect saved checkpoints on disk:

```python
# Cell 11: find all .pth files recursively
results = glob.glob(os.path.join(search_dir, "**", "*.pth"), recursive=True)

# Cell 12: inspect classifier layer shapes of each checkpoint
state = torch.load(path, map_location='cpu')
classifier_keys = [k for k in state.keys() if 'classifier' in k]
```

This was added because loading v1 from a different folder required verifying its architecture before building the matching `build_model_v1()` function.

### What Was NOT Changed (Same as Baseline)

These properties are **identical** in both versions, worth noting because they are common points of confusion:

| Property | Value in Both |
|---|---|
| `label_smoothing` | 0.05 |
| Augmentation pipeline | Resize → Flip → ColorJitter only |
| Training loop | No gap monitor, no MixUp |
| Model architecture | MobileNetV2, same classifier structure |
| Stage 1/2 LR | `1e-3` / `1e-5` |
| `patience` | 4 |
| `num_workers` | 0 |

These unchanged properties are exactly what `deepfake_detection_fixed.ipynb` addresses, adding MixUp, stronger augmentation, label_smoothing=0.10, gap monitor, and dynamic ablation table on top of this version.

### Three-Version Evolution Summary

```
deepfake_detection_local_windows.ipynb   ← Baseline
    │  Changes: data scale only
    │
deepfake_detection_tuned.ipynb                 ← Intermediate (this file)
    │  Changes: class weights, ensemble, 5 visualizations
    │
deepfake_multiclass.ipynb           ← Final
       Changes: MixUp, stronger augmentation,
                label_smoothing=0.10, gap monitor,
                dynamic ablation table
```

## Architecture

### Why EfficientNet-B0 over MobileNetV2 for Multiclass?

| Property | MobileNetV2 | EfficientNet-B0 |
|---|---|---|
| Parameters | 3.4M | 5.3M |
| ImageNet Top-1 | 71.8% | 77.1% |
| Texture sensitivity | Medium | High |
| Compound scaling | No | Yes (depth + width + resolution) |
| CPU inference speed | Fast | Slightly slower |
| Suitable for fine-grained texture | ✖️ | ✅ |

EfficientNet-B0's compound scaling makes it significantly better at detecting subtle texture differences exactly what separates Face2Face from FaceSwap at the blending boundary level.

### Why MobileNetV2 for Binary?

The binary task (real vs fake) is coarser; it doesn't need to discriminate between manipulation types, just detect the presence of any manipulation. MobileNetV2 is faster to train on CPU and sufficient for this task.

---

## Training Strategy

### MixUp Augmentation

MixUp creates convex combinations of training samples and their labels:

```
x_mix = λ·x_i + (1-λ)·x_j       where λ ~ Beta(α, α)
y_mix = λ·y_i + (1-λ)·y_j
```

**Why it's used here:**
- Face2Face and FaceSwap are visually similar → hard labels cause overconfident wrong predictions
- MixUp forces smooth decision boundaries between classes
- Side effect: `train_acc < val_acc,` this is expected and healthy, not underfitting

```
With MixUp (expected behavior):
  Train acc: 62%  (trains on hard blended images)
  Val acc:   86%  (evaluates on clean images)
  Gap:      +24%  MixUp working correctly

Overfitting signal (NOT seen here):
  Train acc: 86%
  Val acc:   70%
  Gap:      -16%  overfitting
```

### Threshold Tuning

For binary classification, the default 0.5 threshold is suboptimal under class imbalance. The notebooks search 39 thresholds in `[0.05, 0.95]` on the validation set:

```python
for t in np.linspace(0.05, 0.95, 39):
    pred = (prob_fake >= t).astype(int)
    # evaluate accuracy and F1
```

Two thresholds are saved: one optimizing accuracy, one optimizing Macro-F1.

### Group-Level Evaluation

Frame-level metrics can be misleading - a single hard frame can lower the score unfairly. Video-level evaluation aggregates frame probabilities per video:

```python
video_score = mean(frame_probs)   # average over all frames
video_pred  = argmax(video_score)
```

This evaluation protocol, used in the original FF++ paper, is more meaningful for real-world deployment.

## Results

### Binary Detection (MobileNetV2)

| Model | Accuracy | Macro-F1 | AUC |
|---|---|---|---|
| MobileNetV2 v1 (simple head) | 66.38% | 0.613 | 0.662 |
| MobileNetV2 v2 (deep head) | 73.70% | 0.710 | 0.812 |
| Simple Ensemble (v1+v2) | 74.06% | 0.708 | 0.792 |
| Weighted Ensemble | 72.86% | 0.709 | 0.796 |

**Group-level (video-level) results:**
```
VAL  acc 74.36% | f1 0.742 | auc 0.964 | groups 78
TEST acc 75.36% | f1 0.750 | auc 0.899 | groups 69
```

> The high group-level AUC (0.964 val, 0.899 test) compared to frame-level AUC (0.812) shows the model is more reliable when aggregating evidence across multiple frames.

### 3-Class Manipulation Classifier (EfficientNet-B0)

| Metric | Frame-Level | Video-Level |
|---|---|---|
| Accuracy | 91.38% | 97.10% |
| Macro-F1 | 0.913 | 0.971 |
| Macro-AUC | 0.984 | 0.993 |

**Training curve highlights:**
```
Stage 1: Val acc 44% → 64% (head + freq branch only)
Stage 2, Ep01: Val acc jumps to 73% (backbone unlocked)
Stage 2, Ep09: Val acc 86%, F1 0.862, AUC 0.971 ← best
```
## Visualizations

Both notebooks generate the following plots automatically:

| Plot | Description |
|---|---|
| `training_curves.png` | Train vs Val accuracy, F1, AUC per epoch with stage boundary marked |
| `confusion_matrix_frame.png` | Frame-level confusion matrix with normalized percentages |
| `confusion_matrix_group.png` | Video-level confusion matrix *(multiclass only)* |
| `per_class_metrics.png` | Precision, Recall, F1 per class as grouped bar chart *(multiclass only)* |
| `robustness_plot.png` | Performance under JPEG compression (q=80,60,40,20) and Gaussian noise (σ=5,10,20,40) |
| `error_analysis.png` | Top-N most confidently wrong predictions with true/predicted labels |
| `gradcam.png` | Grad-CAM overlays showing which facial regions the model attends to |
| `ablation_study.png` | Model comparison table with best model highlighted |

## Setup & Requirements

### Python Version
Python 3.9 or higher recommended.

### Install Dependencies

```bash
pip install torch torchvision
pip install opencv-python
pip install scikit-learn
pip install pandas numpy pillow
pip install matplotlib seaborn
```

Or install all at once:
```bash
pip install torch torchvision opencv-python scikit-learn pandas numpy pillow matplotlib seaborn
```

### Verify PyTorch

```python
import torch
print(torch.__version__)          # should be >= 1.13 for EfficientNet
print(torch.cuda.is_available())  # False is fine - CPU-only mode
```

### Windows Notes

- Set `num_workers = 0` in CFG (already done) - Windows does not support multiprocessing in DataLoader
- Use raw strings for paths: `r"C:\Users\..."` 
- If you see `RuntimeError: DataLoader worker` errors, confirm `num_workers=0`

## How to Run

### Step 1  Prepare the Dataset

Download FaceForensics++ videos using the official script:
```bash
python download.py /path/to/save -d original -c c23 -t videos -n 300
python download.py /path/to/save -d Deepfakes -c c23 -t videos -n 300
python download.py /path/to/save -d Face2Face -c c23 -t videos -n 300
python download.py /path/to/save -d FaceSwap -c c23 -t videos -n 300
python download.py /path/to/save -d NeuralTextures -c c23 -t videos -n 300
```

Or set `cfg.do_download = True` and the notebook will download automatically.

### Step 2:  Update the Path

In both notebooks, update `root_dir` in the CFG to point to your downloaded dataset:
```python
root_dir: str = r'C:\path\to\your\FaceForensics_300'
```

### Step 3  Run Notebook 1 (Binary)

Open `deepfake_detection_tuned.ipynb` and run all cells in order:
1. Imports and config
2. Load official splits
3. Extract frames (skip if already done: `skip_extract=True`)
4. Build index CSV
5. Train (Stage 1 → Stage 2)
6. Evaluate on the test set
7. Robustness testing
8. Ensemble experiments
9. Generate all visualizations

### Step 4  Run Notebook 2 (Multiclass)

Open `deepfake_multiclass.ipynb`. Frames from Notebook 1 are reused - set `skip_extract=True`. Run all cells in order.

### Step 5:  Run Inference on a New Image

**Binary (Real vs Fake):**
```python
predict_image(
    img_path = r"C:\path\to\image.jpg",
    model    = model,
    threshold = threshold   # tuned threshold from val set
)
# Output:
# Prediction: FAKE
# prob_real = 0.1823  |  prob_fake = 0.8177
```

**Multiclass (Manipulation Type):**
```python
predict_manipulation(
    img_path = r"C:\path\to\fake_frame.jpg",
    model    = model,
    top_k    = 3
)
# Output:
# Prediction: Deepfakes  (87.3%)
# Deepfakes          87.3%  ██████████████████████████
# Face2Face           9.1%  ██
# FaceSwap            3.6%  █
```

## Design Decisions

### Why CPU-only?

The notebooks are designed to run on any machine without a GPU. This was a deliberate constraint to make the project reproducible by anyone.

Trade-off: Training is slow (expect 2–4 hours for multiclass). Every design choice - model size, batch size, num_workers - was made with CPU in mind.

### Why `label_smoothing=0.10` in Multiclass?

Face2Face and FaceSwap both perform face region replacement with smooth blending. A model trained with hard labels (0/1) quickly becomes overconfident in the wrong class. Label smoothing of 0.10 means the target distribution is:
```
correct class: 0.90 + 0.10/3 ≈ 0.933
other classes: 0.10/3        ≈ 0.033
```
This slows down overconfidence and improves calibration.

### Why MixUp `alpha=0.3` for Multiclass and `alpha=0.2` for Binary?

Higher alpha = more aggressive interpolation = harder training task. The multiclass task needs stronger regularization because the three-class boundaries are tighter. Binary real/fake separation is coarser, so a lighter alpha (0.2) is sufficient.

### Why `WeightedRandomSampler`?

The dataset has 1 real video per 4 fake videos. Without correction, the model optimizes for the majority class. `WeightedRandomSampler` oversamples underrepresented classes at the batch level so each training batch sees a balanced distribution.

### Why Cosine LR Annealing over ReduceLROnPlateau?

`ReduceLROnPlateau` reacts to plateaus - on fine-grained texture tasks, these plateaus are often temporary, and the model recovers naturally. Stopping the LR prematurely cuts learning short. Cosine annealing decays smoothly regardless, giving the model more opportunity to explore. `CosineAnnealingWarmRestarts` in Stage 2 adds periodic restarts to escape local minima.

### Why Progressive Unfreezing?

Starting with a frozen backbone and training only the head prevents the randomly initialized head from sending large gradients into the pretrained backbone during early training. Once the head stabilizes (Stage 1), unfreezing the last blocks allows fine-tuning of high-level features with a much lower learning rate (10× reduction).

## Limitations

- **CPU-only speed:** Training a multiclass notebook takes 2–4 hours. Frame-by-frame robustness evaluation adds another 30–60 minutes.
- **300 videos per class:** The full FF++ dataset has 1000 videos per class. Results with 300 videos are indicative but not state-of-the-art.
- **No NeuralTextures in multiclass:** NeuralTextures was excluded to keep the 3-class problem tractable. Adding it would require more data and a harder decision boundary.
- **Frame-level vs video-level gap:** The model is evaluated on extracted frames. Deployment on raw video requires integrating frame extraction (Haar cascade) into the inference pipeline.
- **Compression sensitivity:** As shown in the robustness plots, performance degrades under heavy JPEG compression (q=20) and high Gaussian noise (σ=40). Real-world social media videos are typically at q=80+, which the model handles well.

## Citation

If you use this project or the FaceForensics++ dataset in your work, please cite the original paper:

```bibtex
@inproceedings{roessler2019faceforensicspp,
  title     = {FaceForensics++: Learning to Detect Manipulated Facial Images},
  author    = {Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Niessner, Matthias},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year      = {2019}
}
```

## License

This project is for research and educational purposes. The FaceForensics++ dataset requires signing an agreement with the original authors - see [their repository](https://github.com/ondyari/FaceForensics) for access details.
