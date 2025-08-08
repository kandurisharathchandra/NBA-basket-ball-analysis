NBA Basketball Analysis System 🏀
A computer vision–powered system for real-time basketball tracking and analysis, built using YOLOv5l6 and OpenCV.
The pipeline detects and tracks the ball and players, assigns team labels using jersey colors, and estimates ball possession with high accuracy.

✨ Features
Real-time Detection & Tracking

Fine-tuned YOLOv5l6 models for ball (250 epochs) and players (100 epochs).

Achieves 92% detection accuracy in real-game footage.

Ball Tracking Enhancement

Uses interpolation recovery to bridge missed frames, improving continuity by ~18%.

Possession Detection

Combines velocity heuristics and proximity filters to boost possession accuracy by ~21%.

Team Classification

Utilizes FashionCLIP (zero-shot Hugging Face model) to classify players based on jersey colors with 85%+ accuracy.

Modular Design

Easily extendable for event detection, statistics generation, and advanced analytics.

🛠 Tech Stack
Python, OpenCV, YOLOv5l6

NumPy, Hugging Face Transformers

FashionCLIP for jersey classification

📌 Applications
Sports analytics platforms

Automated highlight generation

Coaching tools for performance review
