# 📧 Email Spam Classifier

High-accuracy email spam/ham classifier using **TF-IDF + Logistic Regression** with a FastAPI inference server.

## Features
- TF-IDF bigrams with URL/money/number masking
- Confidence scores and per-class probabilities
- Top spam/ham indicator words endpoint
- Auto-trains on startup

## Quick Start
```bash
pip install -r requirements.txt
python spam_classifier.py --mode train
python spam_classifier.py --mode serve
```

## API
```bash
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d "{\"texts\": [\"You won a free iPhone!\", \"Meeting at 2pm\"]}"
```

## Docker
```bash
docker build -t email-spam-classifier . && docker run -p 8001:8001 email-spam-classifier
```

**Stack:** Python · scikit-learn · TF-IDF · FastAPI · Docker
