# Decision pipeline: input -> tokens -> DecisionHead

How one raw decision sample moves through `TokenSequenceBuilder` (`encoders/tokenizer.py`) and `DecisionHead` (`models/layers/decision_layers.py`). Real output from a live run below (toy hidden_dim=16, max_length=32 to show truncation).

## 1. Input

```python
sample = {
    "id": 0, "decisiontype": "choice",
    "instructions": "What should be done with this inventory item?",
    "options": ["reorder", "discount", "hold", "discontinue"],
    "state": {"sku": "SKU-4122", "item": "HDMI cable", "stock": 324, "weekly_sales": 1, "days_to_expiry": None},
    "answer": 3, "option": "discontinue", "record": {...},
}
```

`id`, `option`, `record` are ignored by the builder -- only `decisiontype`, `instructions`, `options`/`num_levels`, `state`, `answer` matter.

## 2. Tokenized (`TokenSequenceBuilder.build_decision_batch([sample], max_length=32)`)

Layout: `[CLS] <type> <instructions> [SEP] [MARK] opt0 [MARK] opt1 ... [SEP] <state> [SEP]` (special ids: `CLS=3, SEP=4, MARK=5, PAD=0`).

```
token_ids  (1, 32)  [3 731 342 510 760 621 363 913 729 454 917 4 5 865 199 5 869 466 5 55 5 555 280 4 177 174 131 174 593 174 729 4]
mask       (1, 32)  all True here (no padding needed at width 32); PAD elsewhere sets this False
marker_pos (1, 4)   [12 15 18 20]   -- position of each option's [MARK]
marker_end (1, 4)   [15 18 20 23]   -- exclusive end of each option's span (next [MARK], or the trailing [SEP] for the last option)
token_mask (1, 4)   [True True True True]
decisiontypes (1,)  [1]             -- DECISION_TYPES.CHOICE
answers    (1,)     [3]
```

State got truncated to fit `max_length=32` (compare to the untruncated single-row length of 64) -- `build_decision_sequence` only ever truncates the state, never the instructions or options.

`marker_end - marker_pos` gives each option's span length in tokens: `[3, 3, 2, 3]` here -- "reorder"/"discount"/"discontinue" tokenize to more than one piece, "hold" to one. This is exactly what `marker_pos`/`marker_end` exist to bound: DecisionHead pools the whole span, not just the `[MARK]` position, so a multi-token option isn't squeezed through a single vector.

## 3. DecisionHead.forward shapes

With `hidden_dim=16`, this batch of 1, 4 options:

| step | shape | notes |
| --- | --- | --- |
| `hidden_state` (input) | (1, 32, 16) | encoder output, `[CLS]` at position 0 |
| `span_mask` | (1, 4, 32) | `marker_pos <= t < marker_end`, per option |
| `span_lengths` | (1, 4, 1) | `[3, 3, 2, 3]` |
| `markers` (pooled) | (1, 4, 16) | mean of `hidden_state` over each option's span |
| `type_vector` | (1, 16) | learned embedding of `decisiontypes`, broadcast per option |
| `xs` = `trunk_norm(trunk_a(embedding_norm(markers) + type_vector))` | (1, 4, 16) | MoE trunk, masked to real options |
| `scores` = `scorer(xs)` | (1, 4) | one logit per option |
| `act_logits` = `act_head(hidden_state[:, 0] + type_vector)` | (1, 2) | `[defer, act]`, from the pooled `[CLS]` position |
| `output` (return value) | (1, 4) | `scores`, `MASKED_LOGIT` on padded option slots |

Example output logits for this row: `[0.977, 1.363, -0.126, 0.014]` (untrained weights).


## 4. The three outputs

`forward` returns only the option logits; the other two outputs are read from the head afterwards.

| output | where | shape | meaning |
| --- | --- | --- | --- |
| option logits | return value of `forward` | (batch, options) | one score per option, `MASKED_LOGIT` on padded slots |
| act probability | `head.act_probabilities` (from `head.act_logits`) | (batch,) | learned P(the decoded answer is right), read from `[CLS]` + type embedding |
| escalate flag | `head.escalate(threshold)` | (batch,) bool | `act_probabilities < threshold`; true means defer to system two |

**Option logits.** Trained by `DecisionLoss`. They are not an answer until `masked_softmax` and `decode_decisions` read them by question type: BINARY -> P(true), CHOICE -> argmax, SCORE -> expected level.

**Act branch.** `score_act(correct)` trains it with binary cross-entropy against per-row correctness and caches the gradient. The next `backward` applies that gradient to `act_head`, the `[CLS]` hidden state and the type embedding, then clears it. Skipping `score_act` leaves `act_head` untrained for that step. Call order: `forward` -> `score_act` -> `backward`. `act_weight` scales the act gradient.

**Escalate.** A threshold on P(act), with no weights of its own. Because P(act) tracks correctness, raising the threshold escalates more rows and trades system-one coverage for accuracy.

## 5. Helper functions and the layer

The helpers turn raw logits into decisions, labels for the act branch, and calibrated confidence. All of them take the same `token_mask` and `decisiontypes` the head's `forward` received.

| function | in | out | role |
| --- | --- | --- | --- |
| `masked_softmax` | logits, token_mask, optional temperature | (batch, options) probabilities | the one softmax used by the loss, decoding, calibration and confidence; padded slots are exactly 0 |
| `decode_decisions` | probabilities, decisiontypes | (batch,) | BINARY P(true), CHOICE argmax, SCORE expected level |
| `decision_correct` | decisions, answer, decisiontypes | (batch,) 0/1 | whether the decoded answer matches; the target for `score_act` |
| `decision_confidence` | probabilities, token_mask | (batch,) in [0, 1] | 1 - normalized entropy; how peaked the option distribution is |
| `fit_temperatures` | logits, answer, token_mask, decisiontypes, grid | `{bucket: temperature}` | grid-searches one temperature per (type, option count) bucket to minimize NLL |
| `calibrated_probabilities` | logits, token_mask, decisiontypes, temperatures | (batch, options) | `masked_softmax` with each row's bucket temperature (1.0 if unfitted) |

### Training step

```python
logits = net.forward(batch["token_ids"], **kwargs)
loss = loss_fn(logits, answers, token_mask, decisiontypes)

probabilities = masked_softmax(logits, token_mask)
decisions = decode_decisions(probabilities, decisiontypes)
correct = decision_correct(decisions, answers, decisiontypes)
act_loss = head.score_act(correct)

net.backward(loss_fn.backward())
```

`correct` is a label only: no gradient flows through the decode. The main loss and the act loss share one `backward`.

### Calibration (once, on held-out rows)

```python
net.eval()
val_logits = net.forward(val_ids, **val_kwargs)
temperatures = fit_temperatures(
    val_logits, val_answers, val_token_mask, val_decisiontypes, head.TEMPERATURE_GRID
)
```

### Inference

```python
net.eval()
logits = net.forward(token_ids, **kwargs)
probabilities = calibrated_probabilities(logits, token_mask, decisiontypes, temperatures)
decisions = decode_decisions(probabilities, decisiontypes)
confidence = decision_confidence(probabilities, token_mask)
escalate = head.escalate(threshold)
```

Rows where `escalate` is true go to system two; the rest keep `decisions`. `confidence` is an optional second signal, for example `escalate | (confidence < floor)`.

### Notes

- Fit temperatures on rows the model did not train on, using answer option indices; training rows give temperatures near 1 or below.
- Temperature never changes a row's argmax or a BINARY P(true) > 0.5 decision. It changes `decision_confidence` and SCORE expected levels, so decode and check correctness from the same probabilities.
- `fit_temperatures` buckets are named by `option_bucket`, e.g. `"choice:3-5"`; a bucket with no fit falls back to temperature 1.0.
- `escalate` and `act_probabilities` read the state left by the latest `forward`; call `net.eval()` first so MoE routing noise is off.
- The act branch is trained on training-row correctness, which approaches all ones as the model fits, so check the escalation rate on held-out rows before choosing a threshold.

### Where these steps run

`examples/NNet_system_one_decision_example.py` holds the shared steps (`split_rows`, `fit_calibration`, `evaluate`, `print_evaluation`). `examples/NNet_spectre_decision_example.py` and the Streamlit tab `app/system_one_tab.py` reuse them. Each splits rows into train, calibration and test sets, trains with the main loss plus `score_act`, fits temperatures on the calibration rows, and reports calibrated decoding, confidence, P(act) and the escalation split on the test rows.

# System-one decision head (Laya-style)

`src/polyergalio/models/layers/decision_layers.py` — encoder-only, single-pass decision head modelled on [receptron/laya](https://github.com/receptron/laya) (Jev `system_one` API). Added 2026-09-23; replaced the earlier mean-pool `SystemOneHead`.

## Sequence layout (one question per row)

`[CLS] <type> <instructions> [SEP] [MARK] opt0 [MARK] opt1 ... [SEP] <state> [SEP]`

- Only structural specials, no per-field tokens: typed state fields are serialized text (Laya: `json.dumps`), so JSON punctuation delimits them.
- Reserved ids in `generators/data_generators.py`: `PAD=0, BOS=1, EOS=2, CLS=3, SEP=4, MARK=5`, content from `TOKEN_OFFSET=6` (was 3 — shifted content ids for every sequence task).
- `build_decision_sequence(header, options, state, max_len)` builds a row and returns `(ids, marker_pos, marker_end)`; truncates only the state. `marker_end[i]` is the exclusive end of option `i`'s span — the next option's `[MARK]`, or the trailing `[SEP]` for the last one.

## Per-option span pooling (2026-09-27)

`DecisionHead` used to gather the encoder's hidden state at a single `[MARK]` position per option. It now mean-pools across each option's own span (`marker_pos` to `marker_end`), so a multi-token option's text isn't squeezed through one position — see `DECISION_HEAD_README.md` (repo root) for a worked example with real shapes and values.

- Every builder that lays out a decision sequence — `tokenizer.build_decision_sequence` / `build_binary_decision` / `build_choice_decision` / `build_score_decision` / `build_decision_row` / `build_decision_batch`, and the synthetic `generators/data_generators.build_decision_sequence` / `_decision` — now also returns/carries `marker_end` alongside `marker_pos`.
- `DecisionHead.forward` takes a new required `marker_end` kwarg (raises if missing, same style as `marker_pos`); there's no fallback to the old single-position gather, per the no-backwards-compat rule.
- Implementation: `span_mask[b,o,t] = marker_pos[b,o] <= t < marker_end[b,o]` (also masked by `token_mask`), `span_lengths = span_mask.sum(-1)` (floored at 1 to avoid dividing by zero on padded option slots, which are excluded downstream anyway via `token_mask`), `markers = einsum("bot,bth->boh", span_mask / span_lengths, hidden_state)`. `backward` distributes each option's incoming gradient back over its span with the same weights via `einsum("bot,boh->bth", ...)`, replacing the old `np.add.at` scatter onto a single position. `self.rows` (used only for the old gather/scatter) is gone; `span_mask` / `span_lengths` are cached on the head between forward and backward the same way `marker_pos` / `token_mask` are.
- Callers updated to pass `marker_end` alongside `marker_pos`: `examples/NNet_spectre_decision_example.py` (`make_decision_batch`), `examples/NNet_system_one_decision_example.py`, `app/system_one_tab.py`.
- **Not updated** (project rule: don't touch tests unless asked): `tests/test_decision_layers.py`, `tests/test_serialization.py`, `tests/test_network.py` all still call `DecisionHead.forward`/`build_decision_sequence` with the old two-value contract and will fail until `marker_end` is threaded into them too.

## Batching raw samples: TokenSequenceBuilder.build_decision_batch

`encoders/tokenizer.py` — the single entry point from a raw decision sample to everything the encoder and `DecisionHead` need. A sample is a plain dict, e.g.

```python
sample = {
    "id": 0, "decisiontype": "choice",
    "instructions": "What should be done with this inventory item?",
    "options": ["reorder", "discount", "hold", "discontinue"],
    "state": {"sku": "SKU-4122", "item": "HDMI cable", "stock": 324, "weekly_sales": 1, "days_to_expiry": None},
    "answer": 3, "option": "discontinue", "record": {...},
}
```

`TokenSequenceBuilder(tokenizer).build_decision_batch(samples, max_length=None)` reads `decisiontype` (a `DECISION_TYPES` member, its name in any case, or its value — `resolve_decisiontype`), `instructions`, `state`, `options` (CHOICE, or SCORE in place of `num_levels`; BINARY needs neither, fixed as `["false", "true"]`), and an optional `answer`; other keys (`id`, `option`, `record` above) are ignored. It tokenizes and lays out each row with `build_decision_row` (one sample in, `(ids, marker_pos, marker_end, decisiontype)` out — the per-row half of the same logic, usable alone), then pads to a common sequence length and option count and returns a dict:

- `token_ids` `(batch, sequence_length)`
- `mask` `(batch, sequence_length)` — the encoder's attention mask, `token_ids != PAD`
- `marker_pos`, `marker_end`, `token_mask` `(batch, options)` — `DecisionHead`'s own kwargs
- `decisiontypes` `(batch,)`
- `answers` `(batch,)`, included only when every sample carries an `answer`

This dict's keys are exactly `NeuralNetwork.forward`'s and `DecisionHead.forward`'s kwarg names, so a batch is used directly:

```python
batch = builder.build_decision_batch(samples, max_length=SEQUENCE_LENGTH)
logits = net.forward(batch["token_ids"], mask=batch["mask"], marker_pos=batch["marker_pos"],
                      marker_end=batch["marker_end"], token_mask=batch["token_mask"],
                      decisiontypes=batch["decisiontypes"])
loss = DecisionLoss(ordinal_weight=0.25)(logits, batch["answers"], batch["token_mask"], batch["decisiontypes"])
```

This formalizes what `examples/NNet_spectre_decision_example.py`'s `make_decision_sample` / `make_decision_batch` did by hand for its synthetic-instructions case; the library method takes the raw sample shape directly and covers all three `DECISION_TYPES`.

## Token embedding

`models/embedding/embedding.py`: `TextEmbedding(num_embeddings, embedding_dim, padding_idx=None)` — trainable lookup table like `torch.nn.Embedding`. N(0, 1) init; forward `weights[token_ids]` for any leading shape → `(..., embedding_dim)`; backward scatter-adds into `gradient_weights` (repeats summed) and returns zeros shaped like the ids; `padding_idx` row is zero and never trained. Rejects out-of-range or fractional ids. Replaced the old `FourierAttention` stub in that file (no more registry collision).

## Question types: the `DECISION_TYPES` Enum

`models/constants.py`: `class DECISION_TYPES(Enum): BINARY = 0, CHOICE = 1, SCORE = 2` (the Enum is the canonical framing; Laya's "noul" = BINARY).

- Batches carry numeric ids (`member.value`). `decision_type_ids(decisiontypes)` converts members, values or a mix to an int array; every check compares against `DECISION_TYPES.X.value`. `tokenizer.resolve_decisiontype` additionally accepts the member's name as a string, any case, for raw sample dicts.
- Bucket names use the lowercase member name: "binary:2", "choice:3-5", "score:3-5".

## Head, loss, decoding, calibration

- `DecisionHead(hidden_dim, head_hidden, num_types, activation_type="swish", act_weight=1.0, type_initialization="truncated_normal", type_initialization_kwargs=None, routed_scaling=1.0, num_groups=None, top_groups=None)`: `forward(hidden_state, marker_pos, marker_end, token_mask, decisiontypes)` (types default to CHOICE). Mean-pools hidden state across each option's `[marker_pos, marker_end)` span (see above) → `embedding_norm` → + type embedding → `trunk_a` (`MixtureOfExperts`: 4 shared + 32 routed experts, top-4, masked to real options) → `trunk_norm` → `scorer` (hidden→1) → `(batch, options)` logits (`MASKED_LOGIT` on padding). `activation_type` applies to the `trunk_a` experts and is stored as passed. No `named_sublayers`: `get_weights`/`get_gradients`/`update_weights` etc. name each layer explicitly.
- Act branch: `act_head` reads `hidden_state[:, 0]` ([CLS]) + type embedding → 2 logits [defer, act], left on `head.act_logits` (not returned by `forward`); `head.act_probabilities` = P(act). `score_act(correct)` is BCE against per-row correctness, caches its gradient (scaled by `act_weight`) for the next `backward` (call order `forward` → `score_act` → `backward`), which trains `act_head`, adds into the [CLS] hidden gradient and the type embedding. `escalate(threshold=0.5)` = `act_probabilities < threshold`, flagging rows for system two. Walkthrough: `DECISION_HEAD_README.md` sections 4–5.
- `decode_decisions` (per row, by its own type): BINARY → P(true) with options [false, true], CHOICE → argmax, SCORE → expected level Σ i·pᵢ.
- `decision_correct(decisions, answer, decisiontypes)`: per-row 0/1 against the answer index (SCORE rounds the expected level, BINARY thresholds P(true) at 0.5); the target for `score_act`.
- `DecisionLoss(ordinal_weight)` lives in `model_loss.py`: masked softmax CE, optional expected-squared-distance term on SCORE rows; `forward(prediction, targets, mask, decisiontypes)`.
- Calibration: `fit_temperatures(logits, answer, token_mask, decisiontypes, grid)` per `option_bucket` (fit on held-out rows; `head.TEMPERATURE_GRID` is the default grid); `calibrated_probabilities` applies them (1.0 for an unfitted bucket). `decision_confidence` = 1 − normalized entropy.

## Generator

`generate("decision", num_choices=4, num_levels=4, decision_types=tuple(DECISION_TYPES))` → `X` ids, `y` answer index, meta with the head's forward kwargs (`marker_pos, marker_end, token_mask, decisiontypes`) plus `attention_mask, words`. BINARY: query token present; CHOICE: which option token appears in the state; SCORE: count of query token (level).

## Train / eval mode (all layers)

- `Layer.training` (class default True), `Layer.train(mode=True)` / `Layer.eval()` switch the layer and every sublayer (found by `Layer.sublayers()`: Layer attributes, directly or in tuples/lists/dicts) and return self.
- Every `forward` with `training_now` (Dropout, Voting*, MixtureOfExperts, fft FourierAttention, SpectreAttention/Decoder, WaveletRefinementModule) now defaults to `None` = follow `self.training`. Explicit values still win; NeuralNetwork passes its own flag explicitly.
- `NeuralNetwork.train(mode)` / `eval()` propagate to every layer.
- The train/eval flag is not serialized: rebuilt layers and networks start in training mode.

## Serialization (all layers and networks)

- `Layer.serialize()` deep copies config and weights, so a serialized dict is a snapshot independent of further training.
- `tests/test_serialization.py`: every prediction is `model.eval().forward(...)` on both original and rebuilt.
  - Layers (every discovered package Layer, new ones need a `RECIPES` entry): eval/train reach sublayers; inference repeatable and weight-preserving; `training_now=None` follows the mode; config equals constructor args; construct → train (5 SGD steps) → serialize → pickle → deserialize: weights, inference, continued training (RNG reseeded on both, RNG state isn't serialized), no shared memory.
  - Networks (branching fan-out/merge, mixture-of-experts, encoder → DecisionHead with forward kwargs): graph, nodes/edges/consumers/shapes, forward + backward, continued training, no shared memory, serialize snapshot, save/load via file, node ordering and error cases.
- Known failure (2026-09-23): `fft_layers.FourierAttention` round trip (in progress on the user's side; `set_weights` currently receives a tuple).

## Open items

- No full transformer encoder block exists yet; decision tests use a hand-made contextual featurizer. Laya's `rl_common.py` internals are unpublished — those parts are inferred.
- `tests/test_decision_layers.py` / `test_serialization.py` / `test_network.py` need `marker_end` added wherever they call `DecisionHead.forward` or `build_decision_sequence` directly (see "Per-option span pooling" above) — left alone per project rule, not yet done.

Tests: `tests/test_decision_layers.py`, `tests/test_serialization.py`, `tests/test_embedding.py`; DecisionHead and TextEmbedding also registered in `test_network.py` LAYER_ARGUMENTS. `build_decision_batch` / `build_decision_row` and the span-pooling change are verified so far only by scratch script (see below), not by a committed test.

## Verified

- `build_decision_batch`: the user's exact CHOICE sample, plus a BINARY sample and a SCORE sample (via `num_levels`), batched together: correct `token_ids`/`mask`/`marker_pos`/`marker_end`/`token_mask` shapes, right-padded with PAD past each row's real length, `decisiontypes` in sample order, `answers` present. Dropping `answer` from one sample drops `answers` from the whole batch. `max_length` fixes the padded width instead of the batch's natural longest row. CHOICE without `options`, and SCORE without `num_levels` or `options`, raise.
- Span layout: for a batch of CHOICE/BINARY/SCORE rows, every real option's span is contiguous, non-overlapping with its neighbors, and at least 1 token long; multi-token option text (e.g. "discontinue") produces spans longer than 1, confirming `marker_end` isn't just `marker_pos + 1`.
- Span-pooling gradient check: `DecisionHead.eval()` (MixtureOfExperts' training-time routing noise breaks exact finite-difference checks, as expected — unrelated to this change) with a numeric central-difference check against `hidden_state` over 40 sampled (batch, position, hidden) entries: max |analytic − numeric| ≈ 2e-9.
- End to end with a real (tiny) `TextEmbedding -> SpectreAttention -> DecisionHead` network and a fake SentencePiece/scipy stand-in (neither installs in this sandbox): `build_decision_batch`'s dict fed straight into `net.forward(..., marker_pos=..., marker_end=..., token_mask=..., decisiontypes=...)`, `DecisionLoss`, `backward`, `SGD` steps (60), loss decreased, `decode_decisions` — no errors.
