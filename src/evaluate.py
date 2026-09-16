import argparse, json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.golden)
    model = joblib.load(args.model)
    y_true = df["intent_proposed"]  # draft only until human labels replace this
    y_pred = model.predict(df["customer_message"])

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "n_rows": int(len(df)),
        "label_status": "AI-assisted proposed labels; human review required"
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/"draft_intent_metrics.json").write_text(json.dumps(metrics, indent=2))
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(out/"draft_per_intent_metrics.csv")
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
    pd.DataFrame(cm, index=sorted(y_true.unique()), columns=sorted(y_true.unique())).to_csv(
        out/"draft_confusion_matrix.csv"
    )
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
