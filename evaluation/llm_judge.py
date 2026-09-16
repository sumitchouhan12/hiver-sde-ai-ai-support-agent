import argparse
import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI


RUBRIC = {
    "correctness": "Does the reply address the customer's actual issue?",
    "groundedness": "Is the response supported by the supplied historical evidence?",
    "helpfulness": "Does it provide a useful next step without overpromising?",
    "tone": "Is it concise, respectful, and appropriate for customer support?",
    "unsupported_claims": "Does it introduce claims that are not supported by the evidence?"
}


def build_prompt(customer, reply, evidence):
    return f"""
You are evaluating an AI customer-support reply.

Score each dimension from 1 to 5.

Use only the customer message, AI reply and historical evidence provided below.

Return JSON only in this exact structure:

{{
  "correctness": {{"score": 1, "reason": "..."}},
  "groundedness": {{"score": 1, "reason": "..."}},
  "helpfulness": {{"score": 1, "reason": "..."}},
  "tone": {{"score": 1, "reason": "..."}},
  "unsupported_claims": {{"score": 1, "reason": "..."}}
}}

Scoring:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = very good

For unsupported_claims, a higher score means the reply contains fewer unsupported claims.

Customer:
{customer}

AI reply:
{reply}

Historical evidence:
{evidence}
"""


def judge_row(client, model, row):
    prompt = build_prompt(
        row.get("customer_message", ""),
        row.get("ai_reply", ""),
        row.get("evidence", "")
    )

    response = client.responses.create(
        model=model,
        input=prompt
    )

    text = response.output_text.strip()
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="CSV containing the reply evaluation examples"
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV"
    )

    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-5.6-mini")
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to run the LLM judge."
        )

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(args.input)

    required = {
        "customer_message",
        "ai_reply",
        "evidence"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    results = []

    for _, row in df.iterrows():

        try:
            result = judge_row(
                client,
                args.model,
                row
            )

            results.append({
                "judge_correctness": result["correctness"]["score"],
                "judge_groundedness": result["groundedness"]["score"],
                "judge_helpfulness": result["helpfulness"]["score"],
                "judge_tone": result["tone"]["score"],
                "judge_unsupported_claims": result["unsupported_claims"]["score"]
            })

        except Exception as exc:

            results.append({
                "judge_correctness": None,
                "judge_groundedness": None,
                "judge_helpfulness": None,
                "judge_tone": None,
                "judge_unsupported_claims": None,
                "judge_error": str(exc)
            })

    result_df = pd.DataFrame(results)

    output = pd.concat(
        [
            df.reset_index(drop=True),
            result_df
        ],
        axis=1
    )

    output.to_csv(
        args.out,
        index=False
    )

    print(
        json.dumps(
            {
                "rows": len(output),
                "model": args.model,
                "output": str(Path(args.out))
            },
            indent=2
        )
    )


if __name__ == "__main__":
    main()
