from sentence_transformers import SentenceTransformer
import json
import numpy as np
from structlog import get_logger

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings

MODEL_PATH = settings.classifier_model_path
EMBEDDINGS_PATH = settings.classifier_embeddings_path
CLASS_EXAMPLES_PATH = settings.classifier_class_examples_path
COMPLAINTS_IN_SCOPE_PATH = settings.classifier_complaints_in_scope_path
COMPLAINTS_OUT_OF_SCOPE_PATH = settings.classifier_complaints_out_of_scope_path

logger = get_logger(__name__)

def build_model():
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading model from {MODEL_PATH}")
        return SentenceTransformer(MODEL_PATH)
    else:
        logger.info(f"Model not found, downloading from {settings.classifier_model_name}")
        model = SentenceTransformer(settings.classifier_model_name)
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        return model

def build_embeddings():
    CLASS_EXAMPLES = json.load(open(CLASS_EXAMPLES_PATH))

    with open(COMPLAINTS_IN_SCOPE_PATH, "r", encoding="utf-8") as f:
        CLASS_EXAMPLES["complaint"] = f.read().splitlines()

    with open(COMPLAINTS_OUT_OF_SCOPE_PATH, "r", encoding="utf-8") as f:
        CLASS_EXAMPLES["out_of_scope"] = f.read().splitlines()

    CLASS_EMBEDDINGS = {}
    model = build_model()

    for label, examples in CLASS_EXAMPLES.items():
        emb = model.encode(examples, normalize_embeddings=True)
        CLASS_EMBEDDINGS[label] = emb

    np.save(EMBEDDINGS_PATH, CLASS_EMBEDDINGS)
    _class_embeddings = CLASS_EMBEDDINGS

def build_classifier():
    build_model()
    build_embeddings()

if __name__ == "__main__":
    build_classifier()