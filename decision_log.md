# Decision Log

1. **Brand = AppleSupport.** Selected one high-volume brand to keep the evaluation focused and reproducible.
2. **Customer-side messages only for classification.** Agent input should represent an incoming customer message, not an agent reply.
3. **Conversation-linked historical evidence.** Historical AppleSupport replies are used as resolution evidence rather than treating generic web knowledge as ground truth.
4. **Small taxonomy.** Ten intents were chosen to be operationally distinct without pretending every Twitter complaint has a precise intent.
5. **Explicit unclear class.** A catch-all intent prevents forced over-classification of vague complaints.
6. **Majority baseline.** Establishes the trivial floor required by the assignment.
7. **TF-IDF + Logistic Regression baseline.** Provides a transparent, fast supervised baseline.
8. **Weak labels are not gold labels.** Historical training labels are generated heuristically only to make the prototype runnable; the 200-row set must be hand-reviewed.
9. **Held-out golden set.** Golden examples are never used to fit the classifier in the final evaluation.
10. **Time-aware thinking.** Production evaluation should use chronological/conversation splits to reduce leakage from near-duplicate support threads.
11. **Retrieval before generation.** Replies should cite historical resolutions rather than rely only on model memory.
12. **Conservative escalation.** Account-specific, financial, safety-sensitive, or ambiguous cases should not be auto-handled.
13. **False-auto is safety-critical.** An unsafe auto-response is more damaging than an unnecessary escalation.
14. **Offline fallback.** The pipeline must remain runnable without an API key so reproduction does not depend on external service availability.
15. **Headline caveat.** Intent Macro-F1 is explicitly not treated as end-to-end support quality.
