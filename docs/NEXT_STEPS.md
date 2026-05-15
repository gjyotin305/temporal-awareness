# Next Steps: Agentic Self-Monitoring

Grounded in the evaluation of [AGENTIC_SELF_MONITORING_PROPOSAL.md](AGENTIC_SELF_MONITORING_PROPOSAL.md).

---

## Immediate blockers to fix before any analysis

### 1. The existing dataset is too small

`data/raw/two_hop_math_reasoning.json` has 12 questions. No probe, entropy statistic, or accuracy comparison is interpretable at that scale. Everything downstream is blocked on fixing this first.

### 2. The multi-turn framing does not test internal error propagation

Each hop in the current setup is a separate model call. The model can re-read the context and implicitly self-correct. The scientifically interesting question — does the residual stream at token T encode that something went wrong at token T-k — requires the error to propagate *within a single coherent generation*, not across separate calls.

### 3. No contrastive structure means error signal is confounded with task difficulty

Without minimal pairs (same problem, same prefix, only differing in whether an error was injected), rising entropy on incorrect traces cannot be separated from rising entropy on hard-but-correct traces. The two hypotheses the project wants to distinguish are currently indistinguishable in the data.

---

## Step 1: Build the contrastive dataset

**Target:** ~500 base problems × 3 variants × {2, 3, 4} hop depths = ~1500 traces minimum.

### Contrastive variant structure

For every base problem, produce three traces:

| Variant | Description |
|---|---|
| `clean` | Correct intermediate results fed at every hop |
| `error_at_1` | Wrong value injected at hop 1, chain continues from it |
| `error_at_2` | (3-hop+ only) Wrong value injected at hop 2 only |

**Use context injection, not spontaneous model errors.** Construct the assistant turn for the relevant hop with a deliberately wrong computation, then continue the chain. This gives an exact ground-truth injection point and makes pairs aligned at the token level.

Example:
```
# clean variant — hop 1 assistant turn
"5 × $2 = $10. The total cost is $10."

# error_at_1 variant — hop 1 assistant turn (injected)
"5 × $2 = $18. The total cost is $18."
```

Both variants then receive the identical hop-2 prompt. Activations and logits are captured at matched positions across both.

### Problem distribution

| Hop depth | Count | Categories |
|---|---|---|
| 2-hop | 250 | arithmetic, percentage, rate/distance, fractions, counting |
| 3-hop | 150 | percentage chains, compound rate, multi-stage geometry |
| 4-hop | 100 | multi-stage allocation, compound interest, nested fractions |

### Error types to include

- Off-by-one (boundary arithmetic)
- Wrong operator (× instead of +, etc.)
- Wrong intermediate unit (e.g., hours vs minutes)
- Magnitude error (10× or ÷10 off)
- Wrong percentage base

Stratify so each error type has roughly equal representation across hop depths.

---

## Step 2: Add token-level error alignment labels

Step-level labels (`step_correct`, `prior_error_exists`) are necessary but not sufficient. For mechanistic analysis you need to know *which token* in the generation is where the error crystallizes.

**Labels to capture per hop response:**

```python
{
  "hop_index": int,
  "is_correct": bool,
  "error_token_index": int | None,       # first token inconsistent with correct chain
  "propagates_prior_error": bool,        # uses a wrong intermediate from a prior hop
  "first_propagation_token_index": int | None  # where the bad value is first referenced
}
```

These are the positions to read activations from — not the last token of the step.

---

## Step 3: Collect verbalized confidence at each hop

After every hop response, append:

```
"On a scale of 1–5, how confident are you in that computation? Answer with just the number."
```

Record:
- The integer the model outputs (`verbalized_confidence`: 1–5)
- The full logit distribution over `{"1", "2", "3", "4", "5"}` tokens

This is the data needed to test the hypothesis that verbal confidence lags internal uncertainty signals. Without it, that hypothesis is untestable.

---

## Step 4: Standardize the trace artifact schema

Extend the `.pt` artifact format so every saved trace includes:

```python
{
  "id": str,
  "category": str,
  "hop_depth": int,
  "variant": str,                  # "clean" | "error_at_1" | "error_at_2"
  "injected_error": str | None,
  "hops": [
    {
      "hop_index": int,
      "prompt": str,
      "response_tokens": list[int],
      "response_text": str,
      "is_correct": bool,
      "error_token_index": int | None,
      "propagates_prior_error": bool,
      "verbalized_confidence": int | None,
      "verbalized_confidence_logits": dict
    }
  ],
  "final_answer": str,
  "final_answer_correct": bool
}
```

Activations and token-level logits are saved as companion `.pt` files keyed by `id` and `variant`, not embedded in the main record.

---

## Step 5: Token-level entropy and margin analysis (now tractable)

Once the contrastive dataset exists, the entropy/top-2 margin analysis from
[experiments/visualize_logits_across_tokens.py](../experiments/visualize_logits_across_tokens.py)
becomes a real measurement rather than a visualization exercise.

The key comparisons to run:

- `clean` vs `error_at_1` entropy trajectories at matched token positions
- Does entropy rise *before* the error token, at it, or only after?
- Does the confidence margin collapse at the propagation tokens in hop 2?
- Are these effects consistent across error types and hop depths?

---

## Step 6: Layer-wise probes on contrastive pairs

Train linear probes to predict `prior_error_exists` and `future_failure_risk` using activations from matched positions across `clean` vs `error_injected` pairs.

The comparison that matters: **does probe accuracy at layer L add predictive power beyond token-level entropy at that same position?** If not, the probe is restating the surface logit signal in a more expensive way.

Run probes at every layer and report the layer-depth profile of separability — not just overall AUC.

---

## Step 7: Activation patching to establish causality

Once pairs are aligned, patch residual stream activations from the `clean` trace into the `error_at_1` trace at the error token position. Measure whether the hop-2 answer recovers toward the correct value.

This is what turns the correlation study into a mechanistic claim:

> The activation at layer L, position T encodes the error state *causally* — patching it changes downstream behavior.

This step requires the token-index alignment from Step 2. It cannot be done with step-level labels alone.

---

## Step 8: Intervention baseline

With a working probe or entropy threshold, implement one intervention:

- If `prior_error_exists` probe score crosses threshold before hop 2, inject: *"Before continuing, verify your previous computation."*

Compare final accuracy on held-out chains:
- No intervention
- Entropy-threshold intervention
- Probe-score intervention
- Verbalized-confidence-threshold intervention

The question is whether the internal signal (probe or entropy) adds value over the model's own verbal confidence as a trigger.

---

## Files to create

| File | Purpose |
|---|---|
| `experiments/build_contrastive_dataset.py` | Generate base problems + inject error variants, save labeled JSON |
| `experiments/collect_traces.py` | Run model over dataset, capture activations + logits + verbalized confidence |
| `experiments/extract_token_diagnostics.py` | Compute entropy/margin trajectories aligned to token-error labels |
| `experiments/train_self_monitoring_probe.py` | Layer-wise linear probes on contrastive activation pairs |
| `experiments/evaluate_self_monitoring.py` | Compare probe vs entropy vs verbal confidence as failure predictors |
| `experiments/run_activation_patching.py` | Patch clean→error traces at error token positions, measure recovery |

---

## Success criteria (revised)

The first pass counts as successful when all of the following hold:

1. Contrastive dataset exists with ≥500 base problems, token-level error labels, and verbalized confidence.
2. Entropy and top-2 margin trajectories are compared across `clean` vs `error_injected` at aligned token positions, with a clear result on whether the signal precedes or follows the error token.
3. Layer-wise probes are trained and their accuracy profile across depth is reported.
4. At least one activation patching experiment establishes whether the error representation is causally upstream of downstream answer correctness.
5. A comparison table exists: entropy/margin vs probe vs verbalized confidence as intervention triggers, evaluated on held-out final accuracy.
