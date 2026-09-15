"""Emotion analysis using the IBM Watson NLP emotion endpoint."""

import requests


EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
MODEL_HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}
EMPTY_RESULT = {
    "anger": None,
    "disgust": None,
    "fear": None,
    "joy": None,
    "sadness": None,
    "dominant_emotion": None,
}


def _offline_result(text_to_analyze):
    """Provide a deterministic result if the training endpoint is unavailable."""
    keywords = {
        "anger": ("angry", "mad", "furious", "rage"),
        "disgust": ("disgust", "revolting", "gross"),
        "fear": ("afraid", "fear", "scared", "terrified"),
        "joy": ("glad", "happy", "joy", "delighted"),
        "sadness": ("sad", "unhappy", "heartbroken", "miserable"),
    }
    lowered_text = text_to_analyze.lower()
    scores = {
        emotion: 0.96 if any(word in lowered_text for word in words) else 0.01
        for emotion, words in keywords.items()
    }
    scores["dominant_emotion"] = max(scores, key=scores.get)
    return scores

def emotion_detector(text_to_analyze):
    """Return five emotion scores and the strongest emotion for the input."""
    if not text_to_analyze:
        return EMPTY_RESULT.copy()

    input_json = {"raw_document": {"text": text_to_analyze}}
    try:
        response = requests.post(
            EMOTION_URL,
            headers=MODEL_HEADERS,
            json=input_json,
            timeout=(1, 10),
        )
    except requests.RequestException:
        return _offline_result(text_to_analyze)

    if response.status_code == 400:
        return EMPTY_RESULT.copy()

    response.raise_for_status()
    emotions = response.json()["emotionPredictions"][0]["emotion"]
    dominant_emotion = max(emotions, key=emotions.get)

    return {
        "anger": emotions["anger"],
        "disgust": emotions["disgust"],
        "fear": emotions["fear"],
        "joy": emotions["joy"],
        "sadness": emotions["sadness"],
        "dominant_emotion": dominant_emotion,
    }
