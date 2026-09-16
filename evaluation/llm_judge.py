import argparse, json, os
from pathlib import Path

RUBRIC = {
    "correctness": "Does the reply address the customer's actual issue?",
    "groundedness": "Is the response supported by the supplied historical evidence?",
    "helpfulness": "Does it provide a useful next step without overpromising?",
    "tone": "Is it concise, respectful, and support-appropriate?",
    "unsupported_claims": "Does it introduce claims not supported by the evidence?"
}

def build_prompt(customer, reply, evidence):
    return f"""You are evaluating an AI customer-support reply.
Score each dimension from 1 to 5. Return JSON only.

Customer:
{customer}

AI reply:
{reply}

Historical evidence:
{evidence}

Dimensions:
{json.dumps(RUBRIC, indent=2)}

For each dimension return score and one-sentence rationale.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # This harness intentionally does not call an API unless the user supplies
    # an API implementation. It documents the exact judge rubric and output shape.
    df = __import__("pandas").read_csv(args.input)
    schema = {
        "human_sample_size_target": 40,
        "dimensions": RUBRIC,
        "agreement_target": "Report Cohen's kappa for categorical labels or Spearman correlation for 1-5 ratings.",
        "note": "Human scores must be collected independently before comparing to judge scores."
    }
    Path(args.out).write_text(json.dumps(schema, indent=2))
    print("Wrote judge protocol:", args.out)

if __name__ == "__main__":
    main()
