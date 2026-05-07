# Deepfake Detection

Binary classifier to detect AI-generated (deepfake) faces using CNN (MobileNetV2).

## Team
| Role | Responsibility |
|------|---------------|
| Student A | Data pipeline & preprocessing |
| Student B | Model building & training |
| Student C | Robustness testing & reports |

## Results
| Model | Test Accuracy |
|-------|---------------|
| V1 (Baseline) | 71.25% |
| V2 (Improved) | 74.50% |

## Robustness Testing
| Condition | V1 | V2 |
|-----------|----|----|
| No degradation | 71.25% | 73.00% |
| JPEG Quality 50 | 70.50% | 65.75% |
| JPEG Quality 20 | 65.75% | 55.50% |
| Noise level 10 | 56.00% | 61.00% |
| Noise level 25 | 58.25% | 48.50% |

## Dataset
FaceForensics++ — 2000 frames (1000 real, 1000 fake)

Source: [FaceForensics++ on Kaggle](https://www.kaggle.com/datasets/hungle3401/faceforensics)

> Note: Dataset is not included in this repo due to size. Download from the link above.

## Project Structure
deepfake-detection/
├── data/         # Dataset (not tracked)
├── models/       # Saved model weights
├── notebooks/    # Training notebooks
├── reports/      # Midterm & final reports
├── scripts/      # Training & eval scripts
├── requirements.txt
└── LOG.md        # Weekly progress log

## Setup
```bash
pip install torch torchvision kagglehub opencv-python pillow matplotlib
```

## How to Run
1. Open `notebooks/deepfake_detection.ipynb`
2. Run all cells in order
3. ## Setup
```bash
pip install -r requirements.txt
```

## How to Run
1. Download dataset from Kaggle link above
2. Open `notebooks/training_v2.ipynb`
3. Run all cells in order

## Web Demo
Run the Gradio app for real-time detection:
```python
# Run last cell in training_v2.ipynb
app.launch(share=True)
```

   
