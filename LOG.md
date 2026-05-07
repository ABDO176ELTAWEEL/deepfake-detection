## Week 1

### Day 1 — 6/5/2026
- Read reference papers (XceptionNet, MesoNet)
- Created GitHub repository and project structure
- Assigned team roles:
  - Student A: Data pipeline
  - Student B: Model training
  - Student C: Testing & reports
- Set up development environment (Google Colab)

### Day 2 — 7/5/2026
- Downloaded FaceForensics++ dataset (2.73GB)
- Extracted 2000 frames from videos (1000 real, 1000 fake)
- Applied preprocessing: resize to 224x224, normalize
- Split data: 1600 train / 400 test
- Built and trained MobileNetV2 baseline model (5 epochs)
- Train Accuracy: 79.75% | Test Accuracy: 71.25%
- Robustness testing completed:
  - No degradation: 71.25%
  - JPEG Quality 50: 70.50%
  - JPEG Quality 20: 65.75%
  - Noise level 10: 56.00%
  - Noise level 25: 58.25%
- Saved model weights to models/

- ## Week 2

### Day 3 — 10/5/2026
- Created training_v2.ipynb with improved configuration
- Reduced learning rate from 0.001 to 0.0001
- Increased epochs from 5 to 10
- Train Accuracy improved to 98.19%
- Test Accuracy improved to 74.50%

### Day 4 — 11/5/2026
- Generated Confusion Matrix
- Ran classification report:
  - Fake: Precision 0.71, Recall 0.78, F1 0.74
  - Real: Precision 0.74, Recall 0.66, F1 0.70
- Saved model V2 weights to models/mobilenetv2_v2.pth

### Day 5 — 12/5/2026
- Ran robustness testing on V2 model
- Compared V1 vs V2 robustness results
- Observed overfitting issue — V2 less robust under JPEG compression
- Built Gradio web app for real-time deepfake detection
- Tested app with real-world images

### Day 6 — 13/5/2026
- Wrote final report (10-15 pages IEEE format)
- Prepared presentation slides on Canva
- Updated README.md with results
- Added requirements.txt
- Final GitHub cleanup and organization
