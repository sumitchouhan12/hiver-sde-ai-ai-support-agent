import re
from pathlib import Path
import json

TAXONOMY = json.loads((Path(__file__).parent / "taxonomy.json").read_text())

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def weak_label(text: str) -> str:
    t = normalize(text)
    scores = {}
    for intent, phrases in TAXONOMY.items():
        if intent == "general_complaint_unclear":
            continue
        score = 0
        for p in phrases:
            if p in t:
                score += 1
        scores[intent] = score
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general_complaint_unclear"
