# Deepfake Detection

Binary classifier to detect AI-generated (deepfake) faces using CNN (MobileNetV2).

## Team
| Role | Responsibility |
|------|---------------|
| Student A | Data pipeline & preprocessing |
| Student B | Model building & training |
| Student C | Robustness testing & reports |

## Results
| Test | Accuracy |
|------|----------|
| Baseline | 71.25% |
| JPEG Quality 50 | 70.50% |
| JPEG Quality 20 | 65.75% |
| Noise level 10 | 56.00% |
| Noise level 25 | 58.25% |


## Dataset
FaceForensics++ — 2000 frames (1000 real, 1000 fake)

Source: [FaceForensics++ on Kaggle](https://www.kaggle.com/datasets/hungle3401/faceforensics)

> Note: Dataset is not included in this repo due to size. Download from the link above.

## Project Structure
deepfake-detection/
├── data/         # Dataset (not tracked)
├── models/       # Saved model weights
├── notebooks/    # Training notebook
├── reports/      # Midterm & final reports
├── scripts/      # Training & eval scripts
└── LOG.md        # Weekly progress log
## Setup
```bash
pip install torch torchvision kagglehub opencv-python pillow matplotlib
```

## How to Run
1. Open `notebooks/deepfake_detection.ipynb`
2. Run all cells in order
