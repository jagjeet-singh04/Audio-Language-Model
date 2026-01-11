import os
from typing import Dict, List

from faster_whisper import WhisperModel
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Lazy singletons for performance
_nlp = None
_sentiment = None
_whisper_model = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_sentiment():
    global _sentiment
    if _sentiment is None:
        _sentiment = SentimentIntensityAnalyzer()
    return _sentiment


def _get_whisper(model_size: str = "small.en"):
    global _whisper_model
    if _whisper_model is None:
        # CPU inference; set device="cpu" and compute_type="int8" for speed on Windows
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_audio(audio_path: str, model_size: str = "small") -> Dict:
    """Transcribe speech from an audio file using faster-whisper."""
    model = _get_whisper(model_size)
    segments, info = model.transcribe(audio_path, language="en")

    transcript_segments: List[Dict] = []
    full_text = []
    for seg in segments:
        transcript_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        })
        if seg.text:
            full_text.append(seg.text.strip())

    return {
        "language": info.language,
        "duration": info.duration,
        "text": " ".join(full_text).strip(),
        "segments": transcript_segments,
    }


def analyze_text(text: str) -> Dict:
    """Basic NLU: entities, key phrases, sentiment, and crude intent detection."""
    nlp = _get_nlp()
    doc = nlp(text)

    # Entities
    entities = [{
        "text": ent.text,
        "label": ent.label_,
    } for ent in doc.ents]

    # Key phrases (noun chunks)
    key_phrases = [chunk.text for chunk in doc.noun_chunks]

    # Sentiment
    sia = _get_sentiment()
    sent_scores = sia.polarity_scores(text)

    # Simple intent detection
    lowered = text.lower().strip()
    intent = "inform"
    if lowered.endswith("?"):
        intent = "question"
    elif lowered.startswith(("please ", "can you ", "could you ", "would you ")):
        intent = "request"
    elif any(lowered.startswith(v) for v in ["open ", "play ", "set ", "turn ", "start ", "stop "]):
        intent = "command"

    # Short summary: join first 1-2 sentences
    sentences = [s.text.strip() for s in doc.sents]
    summary = " ".join(sentences[:2]) if sentences else text

    return {
        "entities": entities,
        "key_phrases": key_phrases,
        "sentiment": sent_scores,
        "intent": intent,
        "summary": summary,
    }


def process_audio_for_alm(audio_path: str) -> Dict:
    """End-to-end ALM pipeline: Listen (transcribe), Think (summarize), Understand (NLU)."""
    tx = transcribe_audio(audio_path)
    text = tx.get("text", "")
    nlu = analyze_text(text) if text else {
        "entities": [],
        "key_phrases": [],
        "sentiment": {"neg": 0, "neu": 1, "pos": 0, "compound": 0},
        "intent": "unknown",
        "summary": "",
    }
    return {
        "transcription": tx,
        "nlu": nlu,
    }
