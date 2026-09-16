import argparse, json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .weak_label import weak_label

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    text_col = "customer_text" if "customer_text" in df else "text"
    df["weak_intent"] = df[text_col].fillna("").map(weak_label)
    # Remove a tiny minority class only if sklearn cannot fit due to one example.
    counts = df["weak_intent"].value_counts()
    keep = counts[counts >= 2].index
    df = df[df["weak_intent"].isin(keep)].copy()

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=50000)),
        ("lr", LogisticRegression(max_iter=1200, class_weight="balanced"))
    ])
    model.fit(df[text_col].fillna(""), df["weak_intent"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out/"tfidf_lr.joblib")

    golden = pd.read_csv(args.golden)
    golden["predicted_intent"] = model.predict(golden["customer_message"])
    golden[["annotation_id","customer_message","intent_proposed","predicted_intent"]].to_csv(
        out/"draft_predictions.csv", index=False
    )
    meta = {
        "training_rows": int(len(df)),
        "label_source": "heuristic weak labels; NOT hand-labelled",
        "golden_rows": int(len(golden)),
    }
    (out/"run_metadata.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
