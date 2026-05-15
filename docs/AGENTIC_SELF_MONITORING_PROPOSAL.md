# Agentic Self-Monitoring Proposal

Status: draft proposal grounded in [GitHub issue #48](https://github.com/justinshenk/temporal-awareness/issues/48) and related [issue #27](https://github.com/justinshenk/temporal-awareness/issues/27).

## Summary

We should treat agentic self-monitoring as a new probe-based research track focused on whether a model's internal state contains a usable signal of accumulating error, uncertainty, or failure risk over multi-step reasoning.

The key move is to start with a narrow, instrumentable setting rather than jump straight to open-ended agents. For now, the proposal should stay entirely inside `experiments/`. The best Phase 0 is the existing two-hop math activation workflow in [experiments/two_hop_math_activations.py](/data/b22ai063/.mech_interp/temporal-awareness/experiments/two_hop_math_activations.py). That script already captures stepwise generations and hidden states across a sequential reasoning trace.

The important refinement is that the first self-monitoring signal should not be a standard linear probe alone. In the current setup, a linear probe would typically read the last-token activation for a generation step, which collapses the full within-generation trajectory into one representation. What we care about here is the token-by-token evolution of uncertainty as the answer is being generated. That is why [experiments/visualize_logits_across_tokens.py](/data/b22ai063/.mech_interp/temporal-awareness/experiments/visualize_logits_across_tokens.py) is so important: it computes next-token entropy and top-2 confidence margin for every generated token, which gives us a diagnostic signal over time rather than a single end-of-step readout.

## Research Question

Can we detect, from activations alone, when a model in a multi-step task has:

- already made an earlier mistake
- entered a higher-risk internal state for future mistakes
- become more uncertain than its verbal self-report suggests

This follows the framing in issue #48, but sharpens it into a tractable first claim:

> Before asking whether a model "knows it is uncertain" in a broad agentic sense, test whether internal activations encode accumulated error state and downstream failure risk in structured multi-step reasoning.

## Why This Fits The Current `experiments/` Scope

The current `experiments/` folder already gives us a minimal working surface:

- Sequential step collection in [experiments/two_hop_math_activations.py](/data/b22ai063/.mech_interp/temporal-awareness/experiments/two_hop_math_activations.py)
- Activation trajectory visualization in [experiments/visualize_activations_across_tokens.py](/data/b22ai063/.mech_interp/temporal-awareness/experiments/visualize_activations_across_tokens.py)
- Logit and entropy diagnostics in [experiments/visualize_logits_across_tokens.py](/data/b22ai063/.mech_interp/temporal-awareness/experiments/visualize_logits_across_tokens.py)
- Batch analysis over saved `.pt` traces in [experiments/run_visualize_logits_batch.py](/data/b22ai063/.mech_interp/temporal-awareness/experiments/run_visualize_logits_batch.py)

So even without pulling in the rest of the repo yet, we already have the pieces needed to:

- collect multi-step traces
- inspect hidden-state drift across steps
- inspect logit concentration and entropy across steps
- compare trajectories across examples

## Why Linear Probes Are Not The Main Diagnostic Here

Linear probes are still useful, but they are not sufficient as the primary monitoring tool for this experiment.

In the current workflow, probe-style analysis is naturally tied to hidden states taken from the last token of a generation step. That means a probe produces one score for the step after the model has already generated the sequence. This is useful for coarse labeling, but it misses the finer structure of self-monitoring:

- uncertainty can rise gradually across tokens within the same response
- confidence can collapse locally before the full step finishes
- error accumulation may appear as a trajectory pattern, not a single final-state signature

By contrast, [experiments/visualize_logits_across_tokens.py](/data/b22ai063/.mech_interp/temporal-awareness/experiments/visualize_logits_across_tokens.py) gives a token-level diagnostic signal by computing, at every generated token:

- next-token entropy
- top-2 probability margin

Those diagnostics are better aligned with the current question:

> Is the model's uncertainty accumulating as it generates the reasoning trace?

So for this proposal, linear probes should be treated as a secondary or downstream analysis layer, while token-level entropy and confidence margin are the primary monitoring signals.

## Proposed Scope

### What We Are Studying

Agentic self-monitoring should mean:

- tracking latent state over a reasoning trajectory
- estimating whether prior errors are accumulating
- predicting future failure before it is verbally acknowledged
- using that signal for intervention

### What We Are Not Studying First

We should not start with:

- unrestricted autonomous agents
- broad philosophical claims about self-awareness
- dynamic online probe training in the full sense of issue #27

Those are good long-term directions, but the first milestone should be a clean measurement problem with ground truth.

## Proposal: Three-Stage Research Plan

## Stage 0: Build A Minimal Self-Monitoring Benchmark

Use the existing two-hop math setup as a controlled multi-step environment.

For each example, record at every response step:

- prompt and prior conversation state
- model answer text
- final numeric answer if available
- ground-truth correctness for that step
- whether any previous step was already incorrect
- cumulative error count up to this point
- token-level or step-level confidence proxies from logits
- layer activations for the response step

This gives us the first supervised labels for self-monitoring:

- `step_correct`
- `prior_error_exists`
- `cumulative_error_count`
- `future_failure_risk`

Recommendation:

- start with 2-hop and 3-hop math chains
- keep tasks deterministic enough that correctness labels are automatic
- standardize prompts and reasoning format to reduce surface variance
- keep all outputs compatible with the current `.pt` artifacts used by the visualization scripts

## Stage 1: Establish Token-Level Monitoring Signals

Before training probes, we should quantify how uncertainty evolves within each response by using the logits saved at every generation step.

The primary diagnostics are:

1. Next-token entropy at each generated token
2. Top-2 probability margin at each generated token
3. Their trajectory over the full response, not just their final value

The goal is to test whether bad trajectories look different from good ones in ways we can see directly:

- does entropy drift upward before an incorrect hop completes?
- does the confidence margin collapse earlier on failing runs?
- do these signals change sharply near compounding mistakes?

This stage is especially important because it matches the current experiment implementation directly.

## Stage 2: Use Probes As A Secondary Readout

Once token-level diagnostics are characterized, linear probes can be added as a coarse summary of internal state.

Train linear probes layer-by-layer to predict:

1. Whether the model has already made an error in an earlier step
2. Whether the current step will lead to an incorrect downstream answer
3. Whether the model is in a high-risk trajectory state even if the current token distribution looks locally confident

Initial targets should prioritize observable failure signals over abstract uncertainty:

- primary target: `prior_error_exists`
- secondary target: `future_failure_risk`
- comparison baseline: token-level entropy and top-2 margin statistics

This keeps the result interpretable, while acknowledging the limitation of probes in this setup:

> Do last-token activations add predictive value beyond token-level entropy and confidence-margin diagnostics?

## Stage 3: Online Monitoring And Intervention

Once a useful probe exists, run it live over each step in a reasoning trajectory.

If the self-monitoring score crosses a threshold, trigger one of:

- a retry of the current step
- a "check your previous work" re-prompt
- a forced intermediate verification step
- a fresh-context restart with summarized state

The main intervention question is:

> Does probe-triggered monitoring improve final task success more than simple logprob thresholding?

That is the strongest bridge from measurement to practical agent reliability.

Within the current `experiments/` scope, this can stay lightweight:

- detect rising entropy or collapsing top-2 margin with [experiments/visualize_logits_across_tokens.py](/data/b22ai063/.mech_interp/temporal-awareness/experiments/visualize_logits_across_tokens.py)-style metrics
- detect activation drift or instability with [experiments/visualize_activations_across_tokens.py](/data/b22ai063/.mech_interp/temporal-awareness/experiments/visualize_activations_across_tokens.py)-style summaries
- flag steps where uncertainty diagnostics indicate the trajectory may be off track
- optionally check whether a probe score adds anything beyond those diagnostics

## Concrete MVP

The smallest compelling version of this project is:

1. Extend the two-hop math workflow into a dataset builder with per-step labels.
2. Compute token-level entropy and top-2 margin trajectories for each response.
3. Test whether those diagnostics predict future error or compounding mistakes.
4. Optionally train a linear probe to predict `prior_error_exists` and compare it against the token-level diagnostics.
5. Demonstrate one thresholded intervention that improves final accuracy on held-out math chains.

If we can do only one thing first, this is the right one.

## Recommended Deliverables

- A stepwise math-chain dataset with activations and correctness labels
- A token-level uncertainty report based on entropy and top-2 margin trajectories
- A layer-wise probe report only if it adds signal beyond the token-level diagnostics
- A baseline comparison table:
  token-level entropy/margin vs. probe score vs. verbalized confidence
- A small monitoring script or notebook in `experiments/`
- A short write-up tied back to issue #48

## Suggested `experiments/` Plan

The most natural implementation path, staying only inside `experiments/`, is:

- keep [experiments/two_hop_math_activations.py](/data/b22ai063/.mech_interp/temporal-awareness/experiments/two_hop_math_activations.py) as the prototype
- use [experiments/visualize_activations_across_tokens.py](/data/b22ai063/.mech_interp/temporal-awareness/experiments/visualize_activations_across_tokens.py) to inspect hidden-state changes across response steps
- use [experiments/visualize_logits_across_tokens.py](/data/b22ai063/.mech_interp/temporal-awareness/experiments/visualize_logits_across_tokens.py) to inspect entropy and confidence collapse
- use [experiments/run_visualize_logits_batch.py](/data/b22ai063/.mech_interp/temporal-awareness/experiments/run_visualize_logits_batch.py) for batch comparison across saved runs

Concretely, that could look like:

- extend `experiments/two_hop_math_activations.py` to emit correctness and cumulative-error labels
- add `experiments/extract_token_diagnostics.py`
- add `experiments/evaluate_self_monitoring.py`
- add `experiments/train_self_monitoring_probe.py`
- add `experiments/visualize_self_monitoring_scores.py`

## Success Criteria

The first pass should count as successful if we can show all of the following:

- a reproducible multi-step dataset with automatic labels
- evidence that token-level entropy or top-2 margin predicts future failure or compounding mistakes
- if probes are used, evidence that they improve over the token-level diagnostics rather than merely restating them
- at least one intervention that improves held-out final accuracy

## Main Hypotheses

- Internal error state will become more linearly separable as trajectories deepen.
- The most immediate warning signal will appear first in token-level entropy and confidence-margin trajectories.
- If linear probes help, they will likely work as a coarse summary layered on top of those token-level diagnostics.
- Verbal confidence will lag behind internal failure signals.
- Token-level monitoring will outperform verbalized confidence for deciding when to intervene.

## Open Design Choices

There are three design choices that matter most:

### 1. What Should The Probe Predict?

Recommendation:

- start with token-level entropy and top-2 margin as the main diagnostic
- use probes later to test whether last-token activations add extra predictive power
- predict prior error and future failure rather than generic uncertainty labels

This avoids collapsing the project into standard calibration work.

### 2. What Task Family Should We Start With?

Recommendation:

- start with math chains
- expand to multi-hop QA only after the math pipeline is stable

Math gives us clean labels and makes accumulation effects easier to measure.

### 3. How "Agentic" Does The First Version Need To Be?

Recommendation:

- define agentic minimally as multi-step stateful reasoning with dependence across steps
- do not require tool use or open-ended planning in the first benchmark

That lets us answer the scientific question before building a more complex harness.

## Relation To Issue #48

This proposal is aligned with issue #48 but makes three refinements:

1. It recommends a narrower Phase 0 centered on math-chain trajectories before broader task families.
2. It prioritizes token-level uncertainty diagnostics over last-token probe readouts as the first monitoring signal.
3. It keeps the current implementation surface restricted to `experiments/` before expanding outward.

## Relation To Issue #27

Issue #27 is still the longer-horizon vision: introspection during generation and eventual dynamic probe training.

This proposal is the bridge to that vision:

- first build reliable token-level diagnostics on controlled trajectories
- then test whether probe summaries add anything beyond those diagnostics
- then run them on saved experiment traces
- only later explore dynamic or self-trained probes

## Recommended Next Step

Turn the current two-hop activation experiment into a step-labeled dataset generator, then compute token-level entropy and top-2 margin trajectories as the first monitoring baseline.

That is the fastest path to a publishable result and the cleanest foundation for testing whether probes add anything beyond those token-level diagnostics.
