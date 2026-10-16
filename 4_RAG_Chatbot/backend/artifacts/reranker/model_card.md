# Model card: SoilMind sentence reranker

**What it does.** After the retriever has found the top passages, scores every sentence / table row of them against
the question and picks the 1-3 that answer it (the chat then shows them with their table header and a
"Show full passage" link). Also nudges fertilizer questions towards documents matching the plot's zone, water regime
and age class (from Member 2's API; the question's own words win).

**Model.** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual MiniLM cross-encoder) fine-tuned with binary cross-entropy, 2 epochs max
(best epoch 1, by validation), lr 2e-05, max 256 tokens, seed 20261018. Trained in 148.0 min on cpu.

**Training data is auto-generated and NOT human-checked.** 24523 (question, sentence) pairs from the Phase 5
train split. A sentence is labelled relevant if it contains a key fact (number or variety code) of the expected
answer; sentences of look-alike passages are negatives. The questions themselves are AI-drafted, template-generated
or machine-translated (DATASET.md). Settings were chosen on the validation split; the 60-question evaluation set was
not used.

**Results** (share of answers showing *all* key facts / mean key-fact recall / mean words), all pickers with the
same short display and settings:

| Picker | Validation EN | Validation SI | Test EN | Test SI |
|---|---|---|---|---|
| retriever_cosine | 0.6224 / 0.739 / 39.0 | 0.5495 / 0.705 / 37.3 | 0.5238 / 0.6991 / 41.2 | 0.4205 / 0.5642 / 41.2 |
| pretrained_cross_encoder | 0.6429 / 0.7484 / 34.4 | 0.6264 / 0.7622 / 41.0 | 0.619 / 0.7521 / 38.2 | 0.5227 / 0.6386 / 43.2 |
| fine_tuned_reranker **(used)** | 0.6735 / 0.756 / 33.0 | 0.6154 / 0.7543 / 38.6 | 0.6548 / 0.8214 / 36.6 | 0.5341 / 0.6443 / 38.8 |
| current_selection_whole_tables (before) | 0.5714 / 0.6932 / 90.6 | 0.6703 / 0.7588 / 117.2 | 0.5357 / 0.6147 / 91.8 | 0.5682 / 0.6265 / 108.5 |

The chat uses **fine_tuned_reranker** (best validation share of answers with all key facts). Settings: 1 passage(s), up
to 3 units within 0.4 of the best; age bonus 0.3; field-profile boost 3 rank
positions. CPU cost: about 974 ms per question.

**Limits.** Labels only know numbers and variety codes, so questions answered in words are under-represented; the
"relevant" label can be noisy when a number appears in several rows.
