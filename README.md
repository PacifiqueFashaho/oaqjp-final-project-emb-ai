# Emotion Detection Application

An AI-based web application that detects anger, disgust, fear, joy, and sadness
in English text using the IBM Watson NLP emotion endpoint. The project packages
the detector as a Python module, includes unit tests and error handling, and
deploys a browser interface with Flask.

## Run locally

```bash
pip install flask requests
python server.py
```

Then open `http://localhost:5000`.

## Test and analyze

```bash
python -m unittest test_emotion_detection.py -v
pylint server.py EmotionDetection
```
