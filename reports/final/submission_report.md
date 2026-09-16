# Hiver SDE Intern Take-Home Report

## 1. Framing
**Brand:** AppleSupport  
**Golden evaluation set:** 200 user-supplied human-labelled examples.

Pipeline:
`customer message -> intent -> historical evidence retrieval -> grounded draft -> auto-handle / escalate`

The taxonomy contains 10 operational intents, including an explicit `general_complaint_unclear` class.

## 2. Baselines

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority class | 0.310 | 0.047 |
| TF-IDF + Logistic Regression | 0.635 | 0.572 |

## 3. Agent design
**Intent:** transparent TF-IDF + Logistic Regression baseline interface.

**Retrieval:** historical AppleSupport customer/resolution pairs indexed with TF-IDF.

**Reply:** deterministic grounded draft with an optional LLM integration point.

**Escalation:** conservative handling for account/access, billing/payment, ambiguous complaints, safety-sensitive symptoms, and weak evidence.

## 4. Evaluation
The final intent evaluation uses the 200-row user-supplied golden file. Per-intent results and the confusion matrix are in `reports/final/`.

There are **73** misclassified examples in the current baseline. See `reports/final/top_failure_examples.csv`.

Reply evaluation uses a fixed 40-example human-vs-LLM judge sheet. Independent human reply scores are required before calculating judge-human agreement. **No agreement statistic is fabricated.**

## 5. Top five failure modes
1. **Vague complaints:** insufficient technical context.
2. **Update vs hardware overlap:** an update can cause symptoms that resemble device failures.
3. **Multi-issue messages:** one tweet can contain multiple problems while the taxonomy is single-label.
4. **Service/app ambiguity:** generic “not working” language spans multiple services.
5. **Historical evidence mismatch:** lexical similarity does not guarantee resolution equivalence.

## 6. What is misleading about my headline number?
The **0.572 Macro-F1** measures intent classification only. It does not prove that retrieved evidence is correct, that a drafted reply resolves the issue, or that escalation is safe. The golden set is only 200 examples and Twitter support language is noisy.

## 7. One-week next steps
- Adjudicate ambiguous labels.
- Add embedding retrieval and Recall@k.
- Score 40-60 generated replies with humans.
- Run the LLM judge independently and report agreement.
- Calibrate escalation thresholds against false-auto rate.
- Add chronological and near-duplicate leakage tests.
- Re-run from a clean environment before submission.

## 8. Decision log
See `decision_log.md`.
