# Model card: SoilMind extractive answer step

**What it does.** Answers a question by copying the best 1-2 sentences (or the whole short table) of the best
DOA/RRDI passage. No text is generated, so numbers come only from the cited passage. "My field" questions use
bilingual templates filled from Member 2/3 APIs instead.

**Uses our models only:** the fine-tuned retriever (passage ranking and sentence scores) and the intent classifier.
Passage ranking per language (validation hit@3): English `rrf`, Sinhala `dense`.

**Guardrail.** Refuse and escalate when the intent is `out_of_scope` or the best retriever score is below
**τ = 0.32** (chosen on validation macro-F1 of refuse/answer).

**Training data is NOT human-checked.** All questions here are AI-drafted, template-generated or machine-translated
(DATASET.md). The human-checked evaluation is notebook 04.

**Test results** (AI-drafted questions, test split):

| | Passage hit | Key-fact recall | ROUGE-L recall | Hallucinated-number rate | Mean words |
|---|---|---|---|---|---|
| Extractive (ours), English | 0.6228 | 0.6548 | 0.2318 | 0.0 | 76.5 |
| Extractive (ours), Sinhala | 0.4561 | 0.6657 | 0.1765 | 0.0 | 92.9 |
| Baseline (first 2 sentences), English | 0.6228 | 0.3016 | 0.1518 | 0.0 | 46.2 |
| Baseline (first 2 sentences), Sinhala | 0.4298 | 0.2045 | 0.0687 | 0.0 | 38.7 |

Refusal on the test split (intent + τ): precision 0.8261,
recall 0.76.

**Limits.** Copied text can be long or awkward (tables); a question answered only in the other language gets the
other-language text with a note; sentence ranking is weaker in Sinhala.
