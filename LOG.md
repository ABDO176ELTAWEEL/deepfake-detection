## Week 1

### Day 1 — 5/6/2026
- Read reference papers (XceptionNet, MesoNet)
- Created GitHub repository and project structure
- Assigned team roles:
  - Student A: Data pipeline
  - Student B: Model training
  - Student C: Testing & reports
- Set up development environment (Google Colab)

### Day 2 — 7/6/2026
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
