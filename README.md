#  AI Support Agent

A small AI support-agent prototype built using the **AppleSupport** brand from the Kaggle Customer Support on Twitter dataset.

The main goal was to build something that can be evaluated and inspected easily rather than building a large production-style system.

## What it does

For an incoming customer message, the pipeline:

1. Predicts a support intent.
2. Retrieves similar historical AppleSupport conversations.
3. Uses the retrieved history as evidence for a reply.
4. Decides whether the case can be auto-handled or should go to a human.
5. Records the reason for the escalation decision.

The current implementation is intentionally simple so that the individual parts can be tested separately.

---

## Dataset

Dataset:

**Customer Support on Twitter (TWCS)**

Source: Kaggle `thoughtvector/customer-support-on-twitter`

Brand used for this assignment:

**AppleSupport**

The complete TWCS dataset is large, so the runnable version samples AppleSupport customer-side conversations.

The expected raw dataset location is:

```text
data/raw/twcs.csv
```

The raw dataset is not committed to the repository.

---

## Data used

From the original dataset:

* Total tweets: approximately 2.8M
* AppleSupport tweets: approximately 106K
* AppleSupport customer-side messages: approximately 98K
* Deduplicated customer messages: approximately 97K
* Distinct customers: approximately 71K

The historical data contains real Twitter support conversations, so the text is noisy and contains incomplete messages, spelling mistakes and short complaints. I kept this characteristic instead of trying to make the data look like a clean benchmark dataset.

---

## Intent taxonomy

I used a small 10-intent taxonomy:

```text
software_update_issue
battery_power_charging
connectivity_network
account_access_security
data_storage_backup
apps_media_services
device_hardware_issue
billing_purchase_payment
product_howto_information
general_complaint_unclear
```

The `general_complaint_unclear` class is intentional. Some Twitter messages do not contain enough information to assign a more specific intent.

The taxonomy and keyword definitions are available in:

```text
src/taxonomy.json
```

---

## System flow

```text
Customer message
       |
       v
Intent classification
       |
       v
Historical retrieval
       |
       v
Grounded reply draft
       |
       v
Escalation policy
       |
       +----> AUTO_HANDLE
       |
       +----> ESCALATE
```

The agent can run without an external LLM API. This makes the basic pipeline reproducible even when an API key is not available.

---

## Intent classification

The main transparent classifier is:

```text
TF-IDF
   +
Logistic Regression
```

The historical training data uses heuristic weak labels generated from the taxonomy.

These weak labels are used only to make the prototype runnable. They are **not treated as human ground truth**.

The final evaluation is performed against the separate 200-row human-reviewed golden set.

---

## Baselines

Two baselines are included.

### Baseline 1: Majority class

Always predicts the most common intent in the golden set.

This gives a simple lower reference point.

### Baseline 2: TF-IDF + Logistic Regression

Uses word and bigram TF-IDF features followed by Logistic Regression.

This was chosen because it is:

* fast to train
* easy to inspect
* deterministic
* easy to reproduce
* a useful reference before trying a larger model

---

## Final golden evaluation

The final golden set contains:

```text
200 customer messages
```

The file is:

```text
data/golden/golden_set_200_final.csv
```

The final intent labels are stored in:

```text
human_intent
```

The golden examples are kept separate from the historical training corpus.

The annotation process and label definitions are documented in:

```text
data/golden/ANNOTATION_GUIDE.md
```

An earlier AI-assisted draft is retained as:

```text
data/golden/golden_set_200_draft.csv
```

The draft labels are not used as the final ground truth.

---

## Final intent results

On the 200-example human-labelled evaluation set:

| Model                        | Accuracy | Macro-F1 |
| ---------------------------- | -------: | -------: |
| Majority class               |    31.0% |     4.7% |
| TF-IDF + Logistic Regression |    63.5% |    57.2% |

The detailed results are available under:

```text
reports/final/
```

Important files:

```text
final_metrics.json
final_per_intent_metrics.csv
final_confusion_matrix.csv
misclassified_examples.csv
top_failure_examples.csv
```

---

## Retrieval

Historical AppleSupport customer/resolution pairs are indexed using TF-IDF.

For a new message, the retriever returns the most similar historical examples.

The current implementation deliberately uses lexical retrieval rather than an embedding model.

This keeps the prototype small and makes the retrieval behaviour easier to inspect.

The retrieval implementation is in:

```text
src/retrieve.py
```

A future version would compare this against sentence embeddings and evaluate retrieval with metrics such as Recall@k.

---

## Reply generation

The default offline mode creates a conservative reply from retrieved historical evidence.

For example, instead of inventing a troubleshooting procedure, it uses a relevant historical resolution as the basis for the response.

An optional LLM integration point is also included.

The important constraint is that the LLM should use retrieved historical evidence rather than freely inventing a support answer.

---

## Escalation

The agent does not try to auto-handle every message.

It escalates cases involving:

* account access/security
* billing/payment
* unclear complaints
* safety-sensitive symptoms
* insufficient historical evidence

Examples of safety-sensitive signals include:

```text
fire
smoke
swollen battery
explosion
injury
danger
```

The primary safety metric I would use for this component is the **false-auto rate**: cases that should have gone to a human but were automatically handled.

The escalation logic is implemented in:

```text
src/agent.py
```

---

## Running the project

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare a sample AppleSupport dataset:

```bash
python -m src.prepare_data \
  --input data/raw/twcs.csv \
  --out data/sample/apple_customer.csv \
  --max-rows 30000
```

Train the baseline:

```bash
python -m src.train_baseline \
  --input data/sample/apple_customer.csv \
  --golden data/golden/golden_set_200_final.csv \
  --out reports
```

Run final intent evaluation:

```bash
python -m src.evaluate \
  --golden data/golden/golden_set_200_final.csv \
  --model reports/tfidf_lr.joblib \
  --out reports
```

For a quicker smoke test:

```bash
python -m src.prepare_data \
  --input data/raw/twcs.csv \
  --out data/sample/apple_customer.csv \
  --max-rows 5000
```

---

## Try the agent

The agent can be run without an API key:

```bash
python -m src.agent \
  --message "My battery is draining very quickly" \
  --history data/sample/apple_customer.csv
```

It prints:

```text
INTENT
DECISION
REASON
DRAFT REPLY
EVIDENCE
```

---

## Optional LLM mode

To use the optional LLM integration:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-5.6-mini"
```

The API key is not stored in the repository.

---

## Reply evaluation

Intent accuracy alone does not tell us whether the generated support reply is actually good.

For reply evaluation I defined five dimensions:

1. Correctness
2. Groundedness
3. Helpfulness
4. Tone
5. Unsupported claims

A fixed 40-example evaluation sheet is available at:

```text
evaluation/human_vs_llm_judge_40.csv
```

The LLM judge protocol is implemented in:

```text
evaluation/llm_judge.py
```

Human scores are intended to be collected independently before calculating agreement with the LLM judge.

I have intentionally not fabricated a judge-human agreement number.

For categorical decisions I would use Cohen's kappa. For 1-5 ratings, Spearman correlation can be used as a complementary measure.

---

## Failure analysis

The current baseline has 73 misclassified examples.

The main failure patterns I found are:

### 1. Vague complaints

Messages such as "this doesn't work" do not contain enough information for a reliable intent.

### 2. Update vs hardware overlap

A software update can cause symptoms that look like device or hardware problems.

### 3. Multi-issue messages

A single tweet can contain several different problems while the current classifier is single-label.

### 4. Service/app ambiguity

Short messages can mention an app or service without providing enough context to determine the actual problem.

### 5. Historical evidence mismatch

Lexical similarity does not always mean that two cases have the same underlying issue or resolution.

Examples are available in:

```text
reports/final/top_failure_examples.csv
```

---

## What is misleading about my headline number?

The **57.2% Macro-F1** is only an intent-classification result.

It does not mean that the complete support agent has 57.2% end-to-end support quality.

It does not measure:

* retrieval quality
* quality of the drafted reply
* whether the historical resolution actually applies
* escalation safety
* unsupported claims
* distribution shift

The golden set is also only 200 examples, and the source data consists of noisy Twitter conversations.

So the headline number should be treated as a classifier benchmark, not as an overall measure of the support agent.

---

## One-week next steps

If I continued the project for another week, I would focus on:

1. Adjudicating any remaining ambiguous golden examples.
2. Adding chronological and conversation-level leakage checks.
3. Comparing TF-IDF retrieval with sentence embeddings.
4. Measuring retrieval Recall@k.
5. Collecting independent human scores for 40-60 generated replies.
6. Running the LLM judge and measuring judge-human agreement.
7. Calibrating escalation thresholds against the cost of false-auto decisions.
8. Adding targeted tests for billing, account access, safety-sensitive hardware and ambiguous complaints.

---

## Repository structure

```text
hiver-sde-ai-ai-support-agent/
│
├── data/
│   ├── golden/
│   │   ├── ANNOTATION_GUIDE.md
│   │   ├── golden_set_200_draft.csv
│   │   └── golden_set_200_final.csv
│   │
│   └── sample/
│       └── apple_customer_history_10000.csv
│
├── evaluation/
│   ├── human_vs_llm_judge_40.csv
│   └── llm_judge.py
│
├── reports/
│   └── final/
│       ├── baseline_comparison.json
│       ├── final_metrics.json
│       ├── final_per_intent_metrics.csv
│       ├── final_confusion_matrix.csv
│       ├── misclassified_examples.csv
│       ├── top_failure_examples.csv
│       └── submission_report.md
│
├── src/
│   ├── agent.py
│   ├── evaluate.py
│   ├── prepare_data.py
│   ├── retrieve.py
│   ├── taxonomy.json
│   ├── train_baseline.py
│   └── weak_label.py
│
├── decision_log.md
├── requirements.txt
├── SUBMISSION_CHECKLIST.md
└── README.md
```

---

## Design decisions

The main non-obvious decisions are documented separately in:

```text
decision_log.md
```

Some important ones were:

* selecting one brand instead of mixing brands
* using customer-side messages as the input
* keeping an explicit unclear intent
* using weak labels for historical training only
* keeping the golden set separate
* using retrieval before reply generation
* using conservative escalation
* treating false-auto as a safety-sensitive metric
* keeping an offline fallback
* not treating intent Macro-F1 as end-to-end agent quality

---

## Limitations

This is a take-home prototype rather than a production support system.

The biggest limitations are:

* small final evaluation set
* noisy Twitter language
* heuristic training labels
* lexical retrieval
* single-label intent taxonomy
* limited human reply evaluation
* no production customer/account context

These limitations are intentionally documented because they affect how the reported numbers should be interpreted.
