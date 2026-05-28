import os
import re
import time
import unicodedata

import numpy as np
from sentence_transformers import SentenceTransformer
from structlog import get_logger

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

logger = get_logger(__name__)


def normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------ Rule based classification for greetings and foul language ------------
GREETING_KEYWORDS = {
    "hi",
    "hello",
    "hey",
    "hii",
    "heyy",
    "good morning",
    "good afternoon",
    "good evening",
    "greetings",
    "how are you doing",
    "how are you",
}
GREETING_REGEX = [
    r"^(hi|hello|hey)\b",
    r"\bgood (morning|afternoon|evening)\b",
    r"\bhow are you\b",
    r"\bwhat s up\b",
    r"\bwhats up\b",
]


def is_greeting(text: str) -> bool:
    text = normalize(text)

    if text in GREETING_KEYWORDS and len(text.split()) < 4:
        return True

    for pattern in GREETING_REGEX:
        if re.search(pattern, text) and len(text.split()) < 4:
            return True

    return False


FOUL_WORDS = {
    "fuck",
    "shit",
    "bitch",
    "asshole",
    "bastard",
    "idiot",
    "moron",
    "dumb",
    "stupid",
    "madarchod",
    "madharchod",
    "benchod",
    "bhenchod",
    "gandu",
    "lavde",
    "chutiye",
    "chutya",
    "bhosdike",
}


def normalize_obfuscation(text: str) -> str:
    replacements = {
        "@": "a",
        "4": "a",
        "1": "i",
        "!": "i",
        "3": "e",
        "0": "o",
        "$": "s",
        "7": "t",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


def edit_distance_one(word, target):
    if abs(len(word) - len(target)) > 1:
        return False

    i = j = diff = 0

    while i < len(word) and j < len(target):
        if word[i] != target[j]:
            if diff == 1:
                return False
            diff += 1

            if len(word) > len(target):
                i += 1
            elif len(word) < len(target):
                j += 1
            else:
                i += 1
                j += 1
        else:
            i += 1
            j += 1

    return True


def is_foul(text: str) -> bool:
    text = normalize(text)
    text = normalize_obfuscation(text)

    tokens = text.split()

    for token in tokens:
        # Exact match
        if token in FOUL_WORDS:
            return True

        # Fuzzy match
        for foul in FOUL_WORDS:
            if edit_distance_one(token, foul):
                return True

    return False


def rule_based_classification(query: str):
    if is_greeting(query):
        logger.info("Rule based classification", query=query, result="greeting")
        return {"label": "greeting", "confidence": 1.0}

    if is_foul(query):
        logger.info("Rule based classification", query=query, result="foul_language")
        return {"label": "foul_language", "confidence": 1.0}

    logger.info("Rule based classification", query=query, result=None)

    return None


# ------------ Semantic classification for in scope and out of scope queries ------------
MODEL_PATH = settings.classifier_model_path
EMBEDDINGS_PATH = settings.classifier_embeddings_path

_model = None
_class_embeddings = None


def get_classifier_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = SentenceTransformer(MODEL_PATH)
            logger.info("Loading classifier model from", path=MODEL_PATH)
        else:
            logger.info("Classifier model not found")
            raise FileNotFoundError("Classifier model not found")
    return _model


def get_classifier_embeddings():
    global _class_embeddings
    if _class_embeddings is None:
        logger.info("Loading classifier embeddings from", path=EMBEDDINGS_PATH)
        if os.path.exists(EMBEDDINGS_PATH):
            _class_embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=True).item()
        else:
            logger.info("Classifier embeddings not found")
            raise FileNotFoundError("Classifier embeddings not found")
    return _class_embeddings


def predict(query: str):
    global _model, _class_embeddings
    model = _model or get_classifier_model()
    CLASS_EMBEDDINGS = _class_embeddings or get_classifier_embeddings()

    query_embdding = model.encode([query], normalize_embeddings=True)[0]

    best_label = None
    best_score = -1

    def cosine_similarity(a, b):
        return np.dot(a, b.T)

    for label, embeddings_matrix in CLASS_EMBEDDINGS.items():
        scores = cosine_similarity(query_embdding, embeddings_matrix)
        max_score = np.max(scores)

        if max_score > best_score:
            best_score = max_score
            best_label = label

    logger.info(
        "Semantic classification",
        query=query,
        label=best_label,
        confidence=float(best_score),
    )

    return {"label": best_label, "confidence": float(best_score)}


def classify(query: str) -> dict[str, str]:
    rule_result = rule_based_classification(query)
    if rule_result:
        return rule_result

    result = predict(query)

    if result["confidence"] > 0.6:
        return result

    return {"label": "out_of_scope", "confidence": result["confidence"]}


def _classification_decision(label: str) -> dict[str, str]:
    if label == "foul_language":
        return {"type": label, "message": "Please refrain from using foul language"}
    if label == "out_of_scope":
        return {"type": label, "message": "I can only help with complaints analytics"}
    if label == "greeting":
        return {"type": label, "message": "Hello, how can I help you?"}
    if label == "complaint":
        return {"type": label, "message": "Thank you for your question"}
    return {"type": "rephrase", "message": "Please rephrase your query"}


def classify_query(query: str) -> dict[str, str]:
    """
    Classifies the query and returns a dict:
    {
        "type": "complaint" | "out_of_scope" | "greeting" | "foul_language" | "rephrase",
        "message": "Message to be sent to the user"
    }
    """
    result = classify(query)
    return _classification_decision(result["label"])


if __name__ == "__main__":
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        start = time.monotonic()
        print("Bot:", classify_query(query))
        end = time.monotonic()
        print(f"Time taken: {end - start:.2f} seconds")
