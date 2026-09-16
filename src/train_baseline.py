import argparse, json
from pathlib import Path
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .weak_label import weak_label


def get_gold_label_column(golden: pd.DataFrame) -> str:
    """Prefer independently reviewed labels; support legacy draft labels."""
    if "human_intent" in golden.columns:
        return "human_intent"

    if "intent_proposed" in golden.columns:
        return "intent_proposed"

    raise ValueError(
        "Golden file must contain human_intent or intent_proposed"
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", required=True)

    args = ap.parse_args()

    df = pd.read_csv(args.input)

    text_col = (
        "customer_text"
        if "customer_text" in df
        else "text"
    )

    # Historical training data uses heuristic weak labels.
    df["weak_intent"] = (
        df[text_col]
        .fillna("")
        .map(weak_label)
    )

    counts = df["weak_intent"].value_counts()

    keep = counts[counts >= 2].index

    df = df[
        df["weak_intent"].isin(keep)
    ].copy()

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000
            )
        ),
        (
            "lr",
            LogisticRegression(
                max_iter=1200,
                class_weight="balanced"
            )
        )
    ])

    model.fit(
        df[text_col].fillna(""),
        df["weak_intent"]
    )

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        out / "tfidf_lr.joblib"
    )

    golden = pd.read_csv(
        args.golden
    )

    label_col = get_gold_label_column(
        golden
    )

    golden["predicted_intent"] = model.predict(
        golden["customer_message"]
    )

    prediction_cols = [
        "annotation_id",
        "customer_message",
        label_col,
        "predicted_intent"
    ]

    golden[prediction_cols].to_csv(
        out / "predictions.csv",
        index=False
    )

    label_status = (
        "USER-SUPPLIED HUMAN LABELS"
        if label_col == "human_intent"
        else "AI-assisted proposed labels; NOT hand-labelled"
    )

    metadata = {
        "training_rows": int(len(df)),
        "training_label_source": (
            "heuristic weak labels; NOT hand-labelled"
        ),
        "golden_rows": int(len(golden)),
        "gold_label_column": label_col,
        "gold_label_status": label_status
    }

    (
        out / "run_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
