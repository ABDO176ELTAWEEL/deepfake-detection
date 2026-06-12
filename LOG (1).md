# Development Log 
Deepfake Face Detection
**Project:** Deepfake Face Image Detection with Lightweight Convolutional Models  
**Team:** Abdelrahman Ibrahim Kamal · Aya Noah · Esraa Mohamed ( Group 3 )
**Course:** CISC 867 Deep Learning S26 

## Week 1 
Data & Preprocessing
**Dates:** Week 1  
**Owner:** Student A (Abdelrahman)

### Work Done
- Read the two core papers:
  - Rössler et al., *FaceForensics++* (ICCV 2019)
  - Qi et al., *DeepFake Detection by Analyzing Convolutional Traces* (IEEE T-IP 2020)
- Downloaded FaceForensics++ subset via the official download script
  - Compression: `c23`
  - Methods: `original`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`
  - Started with 50 videos/method → increased to 100 → final run used 300 videos/method
- Implemented Haar-cascade face extraction pipeline
  - 20 frames per video sampled at uniform intervals
  - 40% padding around detected face
  - Fallback: centre-crop when no face detected
  - Image size: 224 × 224 px
- Downloaded official FF++ pair-based split files: `train.json`, `val.json`, `test.json`

### Key Decisions
- Used **subject-level split** (first number of pair ID) after discovering that using the full video stem caused data leakage (Val AUC < 0.5)
- Kept each manipulation method in a **separate folder** (`Deepfakes/`, `Face2Face/`, etc.) instead of a single `fake/` folder, enables per-method evaluation and cleaner 4-class extension

### Issues Encountered
- Initial `split_group = video_id` (full stem `033_097`) caused same face to appear in both train and val → model could not generalise → fixed to `subject_id = "033"`
- `num_workers > 0` causes DataLoader crash on Windows → set to `0`

## Week 2 
Binary Baseline Training
**Dates:** Week 2  
**Owner:** Student B (Aya) and student A (Abdelrahman)

### Work Done
- Implemented **binary MobileNetV2 classifier** (`deepfake_detection_baseline_local_windows.ipynb`)
  - Backbone: MobileNetV2 pretrained on ImageNet
  - Classifier head: `Dropout(0.3) → Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→2)`
- Two-stage transfer learning:
  - **Stage 1** (5 epochs): freeze all features, train head only - LR = 1e-3
  - **Stage 2** (up to 25 epochs): unfreeze last 4 feature blocks - LR = 1e-5
- Applied `WeightedRandomSampler` to balance fake/real batches (4:1 imbalance)
- Added `CrossEntropyLoss(weight=[4.0, 1.0])` - upweights minority real class
- Training augmentations: `RandomHorizontalFlip`, `RandomRotation(10°)`, `RandomAffine`, `ColorJitter`
- Threshold tuning on validation set (39-point grid, 0.025 step)
- Per-video evaluation: mean-pooled frame scores → final prediction

### Iterative Fixes Applied
| Version | Bug Fixed |
|---------|-----------|
| v1 | `DEVICE = cpu` hardcoded → `cuda if available` |
| v2 | `split_group = video_id` → `subject_id` (root cause of AUC < 0.5) |
| v3 | Missing `class_weights` in loss → model predicted all fake |
| v4 | Unfreeze last 2 blocks → last 4 |
| v5 | `stage2_lr = 5e-5` → `1e-5` |
| v6 | `frames_per_video = 10` → `20` |
| v7 | `shuffle=True` → `WeightedRandomSampler` |

### Key Hyperparameters (Final)
```
batch_size       = 16 (local) / 32 (Colab)
stage1_lr        = 1e-3
stage2_lr        = 1e-5
weight_decay     = 1e-4
label_smoothing  = 0.05
class_weights    = (4.0, 1.0)   # real, fake
patience         = 4
frames_per_video = 20
```

### Midterm Results (Binary Baseline)
| Metric | Frame-level | Video-level |
|--------|-------------|-------------|
| Accuracy | reported in Table II | reported in Table II |
| Macro-F1 | reported in Table II | reported in Table II |
| AUC-ROC  | reported in Table II | reported in Table II |

*(Full numbers in midterm report Tables II–III)*

## Week 3 
Robustness Experiments & Multi-class Extension
**Dates:** Week 3  
**Owners:** Student C (Esraa) - robustness · Student B (Aya) - multi-class

### Robustness Work (deepfake_detection.ipynb)
- Evaluated binary model under two corruption types:
  - **JPEG recompression**: quality ∈ {90, 70, 50, 30, 10}
  - **Gaussian noise**: σ ∈ {0.01, 0.03, 0.05, 0.10, 0.20}
- Metrics per condition: Accuracy, Macro-F1, AUC-ROC
- Key finding: non-monotonic JPEG behaviour - model partially recovers at Q=50 due to blocking artefact patterns that resemble deepfake traces

### Multi-class Work (deepfake_multiclass.ipynb)
- Built **3-class fake-type classifier**: `Deepfakes`, `Face2Face`, `FaceSwap`
  - Rationale: removed `real` class to avoid 4:1 imbalance with 5-class setup
- Architecture: **EfficientNet-B0** spatial backbone + **Frequency Branch** (FFT log-magnitude)
  - Frequency branch: `2D FFT → log-magnitude → 3-layer Conv (~50K params) → 128-d`
  - Fusion: concatenate spatial + frequency features → final classifier
- Training: same two-stage protocol + `CosineAnnealingLR` instead of `ReduceLROnPlateau`
- Added **MixUp augmentation** with `SoftCrossEntropyLoss` for better generalisation
- Robustness evaluation also applied to multi-class model

### Key Decisions
- Used EfficientNet-B0 over MobileNetV2 for multi-class - better accuracy/size tradeoff
- Added frequency branch inspired by Qi et al. (2020) - convolutional traces visible in the frequency (FFT) domain
- Kept `NeuralTextures` out of 3-class due to high visual similarity causing confusion

## Week 4 - Evaluation, Figures & Report
**Dates:** Week 4  
**All members**

### Work Done
- Final evaluation on held-out test set (official FF++ split)
- Generated publication-quality figures:
  - **Fig. 1**: Training/validation curves (Accuracy, Macro-F1, AUC-ROC vs epoch)
  - **Fig. 2**: Test confusion matrix with per-class recall
- Threshold calibration analysis on validation set
- Error analysis: identified `NeuralTextures` as hardest method to detect
- Wrote and finalised midterm report (IEEE two-column format)
- Submitted code as voluntary supplement with midterm


## Week 5
Manipulation-Type & Unified Multi-class Models
**Dates:** Week 5
**Owners:** Student A (Abdelrahman) - training · Student B (Aya) - evaluation & 5-class

### Work Done
- Finalised the **3-class** EfficientNet-B0 + FFT model on the official split
  - Frame-level: 91.38% acc, 0.913 Macro-F1, 0.983 Macro-AUC
  - Video-level: 97.10% acc, 0.971 Macro-F1, 0.993 Macro-AUC
- Built the **unified 5-class classifier** (`deepfake_multiclass_v3_PC.ipynb`): one MobileNetV2 over **Real, Deepfakes, Face2Face, FaceSwap, NeuralTextures**
  - 3-stage training (head → partial unfreeze → full fine-tune) + MixUp (`alpha=0.3`) + label smoothing `0.10`
  - Test-Time Augmentation (7 views) at inference
  - Result: **78.95%** accuracy, **0.792** Macro-F1, **0.965** OvR Macro-AUC, **93.33%** Top-2
- Expanded robustness sweeps (JPEG + Gaussian) to the manipulation and unified models, at frame and group level
- Added **majority-class** and **frozen-backbone** baselines as reference floors

### Key Decisions
- Reintroduced `NeuralTextures` and real frames only in the unified 5-class model, accepting lower accuracy than the specialised 3-class model in exchange for full detect-and-classify coverage
- Used **TTA** as the default reported configuration for the 5-class model (~+5% accuracy over standard inference)

### Issues Encountered
- `NeuralTextures` and `Deepfakes` remain the hardest 5-class families (often confused with `Real`) → documented in error analysis
- Standard 5-class accuracy (73.70%) lifted to 78.95% only after TTA; verified every number directly against the notebook outputs

## Week 6
Final Report, Figures & Repository
**Dates:** Week 6
**Owners:** All members · Student B (Aya) - report & figures lead

### Work Done
- Wrote the **final IEEE report** (two-column, ~13 pages) covering binary detection, 3-class and unified 5-class attribution, baselines, ablation, robustness, error analysis, and discussion
- Produced final figures, including:
  - **Fig. 1**: pipeline overview (preprocessing → three classifiers → evaluation)
  - **Fig. 2**: dual-branch architecture (EfficientNet-B0 spatial + FFT frequency, fused)
  - per-model confusion matrices, Grad-CAM, robustness, and per-class plots (21 figures total)
- Defined the evaluation metrics formally and reported frame- and video-level results for all models
- Added the **GitHub repository** link as a reference `[13]` and documented run instructions in the README
- Addressed TA feedback points and locked all reported metrics to the notebook runs

### Key Decisions
- Placed the **pipeline overview first (Fig. 1)** and the **model architecture second (Fig. 2)** so the reader sees the system view before the detailed design
- Kept report and README numbers identical and traceable to the notebooks for reproducibility

### Issues Encountered
- Reconciled an earlier 5-class run (80.51%) against the final notebook (78.95%) so report and README agree
- Word equation objects (OMML) render correctly in Microsoft Word but blank in LibreOffice → noted as a cross-platform display caveat

## Repository Structure
```
deepfake-detection/
├── deepfake_detection_baseline_local_windows.ipynb  # Binary baseline (real vs fake)
├── deepfake_detection.ipynb                         # Binary + robustness evaluation
├── deepfake_multiclass.ipynb                        # 3-class fake-type classifier
├── deepfake_multiclass_v3_PC.ipynb                  # Unified 5-class (real + 4 manipulations)
├── plot_figures.ipynb                               # Figure generation
├── train.json                                       # Official FF++ train split
├── val.json                                         # Official FF++ val split
├── test.json                                        # Official FF++ test split
├── final_report_ieee.docx                           # Final IEEE report
├── README.md                                        # Project README
└── LOG.md                                           # This file
```

## Environment
| Item | Detail |
|------|--------|
| OS | Windows 10/11 (local) · Ubuntu via Google Colab |
| Python | 3.10 |
| PyTorch | 2.x (CPU local · GPU on Colab) |
| torchvision | 0.15+ |
| scikit-learn | 1.3+ |
| opencv-python | 4.8+ |
| Hardware (local) | CPU only |
| Hardware (Colab) | NVIDIA T4 / L4 |

AI was **not** used to:
- Generate experimental results or metric values
- Write methodology or literature review sections end-to-end
- Implement the core model architecture from scratch without understanding
