# Hiver SDE Intern Take-Home: AI Support Agent

## Scope
Brand selected: **AppleSupport** from Kaggle Customer Support on Twitter.

The prototype implements:
1. Intent classification over a small taxonomy.
2. Retrieval of historical AppleSupport resolutions.
3. Grounded reply drafting with an optional LLM.
4. Auto-handle vs human escalation.
5. Two intent baselines: majority class and TF-IDF + Logistic Regression.
6. Evaluation harness with intent metrics and optional LLM-as-judge.

> **Important:** `data/golden/golden_set_200_draft.csv` contains AI-assisted proposed annotations and is **not** the final hand-labelled golden set. Before submission, human-review every row and populate the final gold labels. The scripts intentionally distinguish draft evaluation from final evaluation.

## Dataset
The full TWCS file is large, so the runnable pipeline samples only the AppleSupport customer-side conversations. This is consistent with the assignment's instruction that subsampling is expected.

Expected raw file:
`data/raw/twcs.csv`

For a local run, place the Kaggle CSV at that path.

## Reproduce in under ~15 minutes
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
python -m src.prepare_data --input data/raw/twcs.csv --out data/sample/apple_customer.csv --max-rows 30000
python -m src.train_baseline --input data/sample/apple_customer.csv --golden data/golden/golden_set_200_draft.csv --out reports
python -m src.evaluate --golden data/golden/golden_set_200_draft.csv --model reports/tfidf_lr.joblib --out reports
```

If you only want a smoke test, use `--max-rows 5000`.

## Optional LLM mode
Set:
```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5.6-mini"
```
Then:
```bash
python -m src.agent --message "My battery is draining very quickly" --history data/sample/apple_customer.csv
```

Without an API key, the agent still runs in deterministic offline mode. The offline reply is deliberately conservative and grounded in retrieved historical text.

## Final evaluation protocol
The intended final split is:
- **Golden set:** 200 hand-labelled customer messages, held out from training.
- **Historical retrieval corpus:** separate AppleSupport customer/resolution pairs.
- **Training:** sampled historical conversations only.
- **Headline intent metric:** Macro-F1 on the final hand-labelled golden set.
- **Reply metrics:** human rubric + LLM-as-judge on a fixed subset, with judge-human agreement reported.
- **Decision metric:** false-auto rate is the primary safety metric for escalation.

## What is misleading about my headline number?
A high intent Macro-F1 does **not** prove the support agent resolves customers correctly. The golden set is small, the taxonomy compresses diverse real-world issues, and historical Twitter conversations are noisy and platform-specific. Retrieval quality, groundedness, escalation safety, and distribution shift can still fail even when intent classification looks strong.

## One-week next steps
- Finish human review of all 200 golden examples.
- Add adjudication for ambiguous labels.
- Add time-based and conversation-level leakage checks.
- Compare TF-IDF retrieval with a sentence-embedding retriever.
- Run human evaluation on 40-60 generated replies.
- Calibrate escalation thresholds against false-auto cost.
- Add targeted tests for billing, account access, safety-sensitive hardware, and ambiguous complaints.

## Repo structure
```text
src/
  prepare_data.py
  weak_label.py
  train_baseline.py
  evaluate.py
  agent.py
  retrieve.py
baselines/
evaluation/
data/golden/
data/sample/
reports/
decision_log.md
```
