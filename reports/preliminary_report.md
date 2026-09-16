# Preliminary Results and Evaluation Plan

## Scope
AppleSupport customer messages from the Customer Support on Twitter dataset.

## Baselines
**Baseline 1: Majority class**
Predict the most common intent in the training labels. This is the trivial floor.

**Baseline 2: TF-IDF + Logistic Regression**
A transparent supervised text classifier using word and bigram TF-IDF features.

## Preliminary metric
The current run uses **AI-assisted proposed labels** in the 200-row file, so the number below is a development diagnostic only.

- Draft Accuracy: nan
- Draft Macro-F1: nan
- Draft Weighted-F1: nan

**Do not submit this as the final headline number.** Replace the proposed labels with hand-labelled gold labels and rerun the harness.

## Top failure modes to inspect
1. **Vague complaints:** the same emotional language can map to several underlying issues.
2. **Update vs hardware overlap:** update-related symptoms can look like device failures.
3. **Multi-issue messages:** one tweet can contain account, network, and device problems.
4. **Product/service ambiguity:** App Store, Apple Music, iCloud, and device problems may share vocabulary.
5. **Historical resolution mismatch:** a semantically similar old case may have a different root cause or require account-specific handling.

## What is misleading about my headline number?
Intent Macro-F1 measures classification on a small, curated test set. It does not measure whether the generated response actually resolves the issue, whether retrieved evidence is correct, or whether escalation decisions are safe. Twitter language is noisy and historical support practices can change over time.

## One-week next steps
- Human-label and adjudicate all 200 examples.
- Add a second annotator on 40+ examples and report agreement.
- Add a time-based holdout and near-duplicate leakage test.
- Add retrieval Recall@k and evidence-groundedness checks.
- Run 40-60 human-scored replies and compare with the LLM judge.
- Tune escalation thresholds around false-auto rate.

