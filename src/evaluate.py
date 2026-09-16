import argparse, json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def get_gold_label_column(df: pd.DataFrame) -> str:
    """Use reviewed labels for final evaluation; support legacy draft files."""
    if "human_intent" in df.columns:
        return "human_intent"
    if "intent_proposed" in df.columns:
        return "intent_proposed"
    raise ValueError("Golden file must contain human_intent or intent_proposed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.golden)
    model = joblib.load(args.model)

    label_col = get_gold_label_column(df)

    y_true = df[label_col]
    y_pred = model.predict(df["customer_message"])

    human_reviewed = label_col == "human_intent"

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "n_rows": int(len(df)),
        "gold_label_column": label_col,
        "label_status": (
            "USER-SUPPLIED HUMAN LABELS"
            if human_reviewed
            else "AI-assisted proposed labels; NOT hand-labelled"
        )
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics_name = (
        "final_metrics.json"
        if human_reviewed
        else "draft_intent_metrics.json"
    )

    (out / metrics_name).write_text(
        json.dumps(metrics, indent=2)
    )

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0,
        output_dict=True
    )

    per_intent_name = (
        "final_per_intent_metrics.csv"
        if human_reviewed
        else "draft_per_intent_metrics.csv"
    )

    pd.DataFrame(report).T.to_csv(
        out / per_intent_name
    )

    labels = sorted(y_true.unique())

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    cm_name = (
        "final_confusion_matrix.csv"
        if human_reviewed
        else "draft_confusion_matrix.csv"
    )

    pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    ).to_csv(out / cm_name)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
