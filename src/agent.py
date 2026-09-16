import argparse, os
import pandas as pd
import joblib
from .weak_label import weak_label
from .retrieve import Retriever, clean_reply

ESCALATE_INTENTS = {
    "account_access_security",
    "billing_purchase_payment",
    "general_complaint_unclear"
}

def decide(intent, message, retrieval):
    m = message.lower()
    if intent in ESCALATE_INTENTS:
        return "ESCALATE", f"{intent} can require account-specific, financial, or ambiguous handling."
    if any(x in m for x in ["fire", "smoke", "swollen battery", "explosion", "injury", "danger"]):
        return "ESCALATE", "Potential safety-sensitive issue."
    if not retrieval or retrieval[0]["similarity"] < 0.18:
        return "ESCALATE", "Insufficient historical evidence for a grounded response."
    return "AUTO_HANDLE", "Clear intent with sufficiently similar historical resolution evidence."

def offline_reply(message, intent, evidence):
    if not evidence:
        return "I want to make sure we give you the right next step. A support specialist should review this."
    ev = clean_reply(evidence[0]["historical_reply"])
    return f"Based on a similar AppleSupport case, a relevant next step was: {ev} If that does not resolve the issue, this should be reviewed by a support specialist."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", required=True)
    ap.add_argument("--history", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    h = pd.read_csv(args.history)
    text_col = "customer_text" if "customer_text" in h else "text"
    reply_col = "apple_text" if "apple_text" in h else "text"
    h = h.dropna(subset=[text_col]).copy()
    # If no paired reply exists, use the historical text only as retrieval context.
    replies = h[reply_col].fillna("").astype(str).tolist()
    texts = h[text_col].fillna("").astype(str).tolist()

    retrieval = Retriever(texts, replies).search(args.message, k=3)
    if args.model:
        model = joblib.load(args.model)
        intent = model.predict([args.message])[0]
    else:
        intent = weak_label(args.message)

    decision, reason = decide(intent, args.message, retrieval)
    reply = offline_reply(args.message, intent, retrieval) if decision == "AUTO_HANDLE" else (
        "Thanks for reaching out. This needs a support specialist to review before we provide a definitive next step."
    )

    print("\nINTENT:", intent)
    print("DECISION:", decision)
    print("REASON:", reason)
    print("DRAFT REPLY:", reply)
    print("\nEVIDENCE:")
    for r in retrieval:
        print(f"- similarity={r['similarity']:.3f} | {clean_reply(r['historical_reply'])[:300]}")

if __name__ == "__main__":
    main()
