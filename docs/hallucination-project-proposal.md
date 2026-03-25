# Project Proposal: Adaptive Constraint Balancing for Evidence-Grounded QA

## Current Decision

This document contains both the long-term direction and the current working scope.

The **current decision** is to work only on a narrow Stage 1 version of the project:

- task: `question + one evidence passage -> short answer or ABSTAIN`
- dataset: `SQuAD 2.0` only
- model scope: one small baseline model, compared with the same model trained with explicit `ABSTAIN` supervision
- primary metric: unsupported-answer rate
- secondary metrics: answer `F1` and abstain `F1`
- current interpretation: this is the first concrete version of balancing answer quality against unsupported answering
- output policy: keep the external action binary for Stage 1, even if later models expose internal scores

### Stage 1 Decision Snapshot

Use this as the active project rule:

- input: one question and one evidence passage
- external output: `answer` or `ABSTAIN`
- optional internal signals: answerability, support, confidence, contradiction risk
- evaluation claim: improve the trade-off between answer quality and unsupported answers
- not allowed yet: custom external scoring frameworks

The concrete execution plan for this stage is in:

- `stages/01-grounded-abstention-baseline/notebooks/google-colab.ipynb`

Explicitly out of scope for now:

- `FEVER`
- `HotpotQA`
- `RAG`
- multi-hop reasoning
- custom architectures
- claim-level verification
- long-form generation
- solving hallucination in general

So the active goal is not:

> solve hallucination broadly

It is:

> show that explicit abstention training improves the trade-off between answer quality and unsupported answers on `SQuAD 2.0`

## 1. Problem Statement

This project does not try to "solve hallucination" in general.

That is too broad.

Instead, the project targets a narrower and more defensible problem:

> Given a question and a set of evidence passages, learn how to balance answer correctness, evidence support, and abstention so the system remains useful without producing too many unsupported answers.

This turns the problem into a constrained decision problem instead of a vague truthfulness claim.

In this setting, an unsupported answer is one failure mode, but not the only concern. The model also needs to remain useful, avoid excessive refusal, and handle uncertainty in a controlled way.

## 2. Exact Task To Solve

The task is **evidence-grounded question answering with constrained answer / abstain decisions**.

Formally:

- Input: a question `q` and evidence passages `E = {e1, e2, ..., ek}`
- Output: an answer `y` or `ABSTAIN`

Optional auxiliary task:

- Predict whether each generated claim or answer sentence is supported by the evidence

This is the right level of scope because it lets us define the competing constraints operationally:

- correct supported answer = useful behavior
- unsupported answer = grounding failure
- abstain when evidence is insufficient = correct cautious behavior
- excessive abstention on answerable questions = utility failure

So the research target is not one metric. It is the balance among:

- answer quality
- evidence support
- abstention behavior
- optionally confidence calibration

## 3. Input And Output Structures

### Input Structure

The model input should look like:

```json
{
  "question": "Who discovered penicillin?",
  "evidence": [
    "Alexander Fleming discovered penicillin in 1928.",
    "Penicillin became widely used in the 1940s."
  ]
}
```

### Output Structure

The simplest output is:

```json
{
  "decision": "answer",
  "answer": "Alexander Fleming",
  "confidence": 0.93
}
```

Or:

```json
{
  "decision": "abstain",
  "answer": null,
  "confidence": 0.18
}
```

If you want a stronger version, add support prediction:

```json
{
  "decision": "answer",
  "answer": "Alexander Fleming",
  "confidence": 0.93,
  "supported": true
}
```

Optional richer structure:

- answer text
- answer / abstain decision
- confidence score
- claim-level support labels

For a first project, the answer-plus-abstain format is enough.

### Why The Output Is Binary At First

Yes, this setup is strict.

That is intentional.

At Stage 1, the binary output keeps the project:

- measurable
- easy to debug
- hard to hide behind vague scoring rules

Right now, strictness is a strength, not a weakness.

### Better Compromise: Binary Action, Extra Internal Scores

A better longer-term design is:

- keep the final external action as `answer` or `ABSTAIN`
- allow the model to produce auxiliary internal scores

Useful auxiliary scores include:

- answerability score
- support score
- confidence score
- contradiction risk score

These can later help with:

- analysis
- threshold tuning
- adaptive balancing
- more informed answer / abstain decisions

But they should not replace the clean Stage 1 task.

So the current rule is:

> binary decision outside, richer scores inside

## 4. Training Data

Right now, this repo does not contain training data.

So the project needs external datasets.

For the broader research direction, a practical dataset setup is:

### Primary Datasets

- `SQuAD 2.0`
  - useful because it includes answerable and unanswerable questions
- `FEVER`
  - useful for evidence-based support vs refute style supervision
- `HotpotQA`
  - useful if you want multi-hop evidence grounding

### Synthetic Hard Negatives

This part is important.

You should generate unsupported examples by perturbing correct answers:

- swap names
- swap dates
- swap numbers
- swap locations
- insert plausible but unsupported causal claims

This creates training signals for unsupported specificity and fabricated detail.

### Current Stage 1 Dataset Decision

For the current implementation stage, use:

- `SQuAD 2.0` only

Reason:

- it directly supports answerable vs unanswerable behavior
- it is enough to build and test abstention behavior
- it keeps the first experiment small enough to finish

Everything else should wait until the first baseline comparison is done.

### Later-Stage Dataset Expansion

After Stage 1 is complete, a practical expansion is:

- `FEVER` for support supervision
- a small synthetic negative set for stronger grounding penalties
- `HotpotQA` later if multi-hop evidence becomes necessary

That is enough for a stronger second-stage project.

### Which Dataset Is Best?

The answer depends on what "best" means.

#### Best Single Dataset To Start

- `SQuAD 2.0`

Why:

- directly supports answerable vs unanswerable behavior
- easy to train a first abstention-capable baseline
- simpler than retrieval-heavy or multi-hop setups

Weakness:

- grounding is limited to a provided passage, so it is less realistic than open retrieval settings

#### Best Dataset For Hallucination Supervision

- `FEVER`

Why:

- directly models supported, refuted, and not-enough-info behavior
- much closer to claim grounding and verification
- useful for training support or hallucination detection heads

Weakness:

- it is framed more as claim verification than natural answer generation

#### Best Advanced Dataset For Reasoning And Explainability

- `HotpotQA`

Why:

- adds multi-hop reasoning
- includes supporting facts
- helps evaluate whether answers are tied to the right evidence

Weakness:

- harder to train and debug
- not the best first dataset if your pipeline is still unstable

#### Best Overall Recommendation

For this project, the best practical sequence is:

1. `SQuAD 2.0`
2. `FEVER`
3. synthetic hard negatives
4. `HotpotQA` later if needed

This gives you:

- abstention learning
- support supervision
- hallucination-style negative examples
- a path to stronger reasoning later

So the honest recommendation is:

> the best single starter dataset is `SQuAD 2.0`, and that is also the current Stage 1 decision; the best broader training mix later is `SQuAD 2.0 + FEVER + synthetic negatives`

## 5. Neural Architecture

Do not design a huge custom architecture first.

A modular architecture is the best choice:

### Option A: Retriever + Generator + Verifier

- Retriever
  - selects top-k evidence passages
- Generator
  - generates `answer` or `ABSTAIN`
- Verifier
  - predicts whether the generated answer is supported by evidence

This is the strongest research architecture, because it separates:

- evidence access
- answer generation
- support checking

### Option B: Generator + Classification Heads

If you want a simpler first implementation:

- one pretrained encoder-decoder or causal LM
- one generation head for answer text
- one classification head for answer vs abstain
- one classification head for supported vs unsupported

This is easier to build and train.

### Recommendation

For a student or early-stage project, start with **Option B**.

It is cheaper, simpler, and easier to debug.

If results are promising, then upgrade to a retriever-generator-verifier pipeline.

For the current framing, this model is not only generating answers. It is also learning how to trade off multiple constraints through its heads and losses.

Later, those heads can also expose auxiliary scores without changing the basic action format.

## 6. Constraint Objective

The training objective should not be only answer cross-entropy.

A better formulation is:

```text
L = w_answer * L_answer
  + w_abstain * L_abstain
  + w_support * L_support
  + w_calibration * L_calibration
```

Where:

- `L_answer`
  - standard sequence generation loss for correct answers
- `L_abstain`
  - classification loss for answer vs abstain
- `L_support`
  - binary classification loss for supported vs unsupported output
- `L_calibration`
  - penalty when confidence is high but support is weak

If you want an even harsher grounding objective, add:

```text
+ w_contradiction * L_contradiction
```

Where `L_contradiction` penalizes answers that conflict with the provided evidence.

### Stage 1 Decision

For Stage 1, use fixed weights.

That is enough to study whether explicit abstention changes the trade-off at all.

Do not introduce a complicated external scoring rule yet.

If needed, only log auxiliary scores internally.

### Later Research Direction

If Stage 1 works, the real research contribution can become:

> adapt the constraint weights so the model learns when to favor answering, when to favor abstention, and when to penalize unsupported confidence more strongly

This keeps the idea narrow enough to study and broad enough to be interesting.

### Key Design Principle

The system should balance:

- answer quality
- unsupported answers
- contradicted answers
- overconfident unsupported answers

And it should preserve:

- correct supported answers
- correct abstention when evidence is insufficient

That is much better than treating every wrong answer the same way.

The point is not "minimize hallucination at any cost."

The point is:

> find a better operating point between usefulness and grounding

That operating point can later be informed by internal scores, but the first experiment should still be judged through simple observable outcomes.

## 7. Evaluation Metrics

This project should not be evaluated by accuracy alone.

You need metrics for both usefulness and trustworthiness.

### Answer Quality

- `Exact Match (EM)`
- `F1`

### Hallucination / Grounding Quality

- unsupported answer rate
- contradiction rate
- supported-answer rate

### Abstention Quality

- abstain precision
- abstain recall
- abstain F1

### Calibration Quality

- `ECE` (Expected Calibration Error)
- `Brier Score`

### Suggested Core Metric Set

If you need a small metric set, use:

- `F1`
- unsupported answer rate
- abstain F1
- `ECE`

### Balance-Oriented Evaluation

Because this is a balancing project, do not report only a single best number.

Also report:

- the trade-off between answer `F1` and unsupported-answer rate
- the trade-off between abstain `F1` and answer `F1`
- whether your proposed method dominates the baseline on the same dev set

If useful, define a simple model-selection score such as:

```text
BalanceScore = F1 - alpha * UnsupportedRate + beta * AbstainF1
```

But do not hide the individual metrics behind one scalar.

That is enough to show whether the model is not only accurate, but also grounded.

### Stage 1 Evaluation Rule

For Stage 1, keep evaluation decision-level and simple:

- did the model answer or abstain
- was the answer supported
- what was the answer `F1`

Auxiliary scores may be logged, but they are not the main claim yet.

## 8. Baselines

A research project needs clear baselines.

Use at least these:

### Baseline 1: Standard QA Model

Train a normal QA model without any explicit constraint-balancing penalties.

This tells you what plain answer optimization achieves.

### Baseline 2: Fixed-Constraint QA Model With No-Answer Head

Train a model that can abstain, but without explicit support or calibration loss.

This shows whether explicit abstention already improves the trade-off.

### Baseline 3: Constraint-Balanced Model

Train the full model with:

- answer loss
- abstention loss
- support loss
- calibration or contradiction penalty

This is the first real balancing model.

### Baseline 4: Adaptive Constraint Model

If Stage 1 is successful, add:

- adaptive or scheduled weights across the constraint losses

This becomes the actual research contribution under the new framing.

## 9. Expected Contribution

The contribution should be stated modestly and clearly.

Not:

```text
we solve hallucination in language models
```

But:

```text
we study how to balance answer quality, abstention, and evidence support
in evidence-grounded QA, and test whether constraint-aware training reduces
unsupported answers without collapsing usefulness
```

That is believable.

## 10. Feasibility

Yes, this project is feasible if you keep the scope narrow.

Feasible version:

- one task
- one domain
- two or three datasets
- one base model
- one proposed loss design
- one strong baseline comparison

Not feasible version:

- all hallucination types
- all domains
- open-world truthfulness
- unrestricted long-form generation

So the honest answer is:

> this project can study a well-defined trade-off in evidence-grounded QA, not a universal balancing mechanism

That is still enough for a strong paper or thesis direction.

## 11. Final Recommended Project Definition

If you want one final sentence to anchor the whole project, use this:

> We propose an evidence-grounded question answering system that learns to balance answer correctness, abstention, and evidence support through constraint-aware training, with the goal of reducing unsupported answers without excessive refusal.

That is the version that is specific enough to build and strong enough to defend.

## 12. Discipline Rules

If the goal is to actually finish the project, use these rules and do not keep rewriting them every week.

### What Exact Task Am I Solving?

> Given a question and one evidence passage, generate a short answer only if the passage supports it; otherwise output `ABSTAIN`.

This is the first task.

Not:

- open-domain QA
- multi-hop retrieval
- long-form explanation
- general hallucination reduction across all LLM use cases

### What Exact Metric Am I Trying To Improve?

Primary metric:

- unsupported-answer rate

Secondary metrics:

- answer `F1`
- abstain `F1`

The goal is:

> improve the balance between unsupported-answer rate and answer quality without excessive abstention

Do not replace this with a custom composite scoring framework in Stage 1.

### What Single Experiment Am I Running This Week?

Run exactly this:

1. train a small baseline model on `SQuAD 2.0`
2. train the same model with explicit `ABSTAIN` supervision
3. compare unsupported-answer rate, answer `F1`, and abstain `F1` on the dev set

That is enough for one week.

Do not add a second experiment until this one is complete and written down.

### What Am I Refusing To Work On?

Refuse all of these for the first stage:

- `RAG`
- `FEVER`
- `HotpotQA`
- custom architectures
- claim-level verification
- long-form generation
- new domains
- multi-hop reasoning
- fancy calibration methods
- complicated external scoring frameworks
- trying to solve hallucination in general

These are not bad ideas.

They are just not allowed until the first baseline comparison is done.

### One-Sentence Weekly Compass

If you feel yourself drifting, come back to this:

> This week I am only trying to show that explicit abstention training improves the balance between answer quality and unsupported answers on `SQuAD 2.0`.

If an idea does not help that sentence, postpone it.

## 13. Constraint Roadmap

The full constraint list is:

- grounding
- retrieval
- verification
- calibrated uncertainty
- abstention
- penalty for unsupported confident claims

Trying to balance all of these in one first system is a bad idea.

It is not impossible, but it is too much for a single clean first-stage project.

### Honest Answer

If retrieval is included properly, this is realistically a **6-stage roadmap**.

If retrieval is excluded for now, the minimum clean program is **4 stages**.

### Recommended Stage Names

1. `Grounded Abstention Baseline`
2. `Evidence Support Verification`
3. `Confidence Calibration`
4. `Unsupported-Confidence Control`
5. `Retrieval-Grounded QA`
6. `Adaptive Constraint Balancing`

### Stage 1: Grounded Abstention Baseline

Task:

- input: one question and one evidence passage
- output: `answer` or `ABSTAIN`

Focus:

- basic grounding
- basic abstention

Why first:

- this is the smallest version that is still meaningful

Data:

- `SQuAD 2.0`

Model:

- one small baseline model
- one answer generation head
- one answer / abstain decision head if needed

Training objective:

- answer loss
- abstain loss

Main metrics:

- answer `F1`
- unsupported-answer rate
- abstain `F1`

What this stage proves:

- the model can use fixed evidence instead of free guessing
- explicit abstention can improve behavior on unanswerable cases

Main failure mode:

- the model becomes too cautious and hurts answer quality too much

Stop condition:

- you have a clean baseline comparison between no-abstain and abstain-trained models
- results are written down clearly

### Stage 2: Evidence Support Verification

Add:

- support / unsupported prediction

Focus:

- verify whether the produced answer is actually backed by the passage

Why now:

- you should not penalize unsupported answers until you can detect support more reliably

Data:

- start with `SQuAD 2.0`
- if needed later, extend with `FEVER` for stronger support supervision

Model change:

- add one verification head
- input can be question, passage, and predicted answer

Training objective:

- keep Stage 1 losses
- add support classification loss

Main metrics:

- support classification accuracy or F1
- supported-answer rate
- contradiction rate if defined

What this stage proves:

- the model can distinguish supported answers from unsupported ones instead of only guessing based on confidence

Main failure mode:

- the verifier learns shortcuts and does not really ground its decision in the passage

Stop condition:

- support predictions are stable enough to be used as a meaningful training or analysis signal

### Stage 3: Confidence Calibration

Add:

- confidence estimation
- calibration evaluation

Focus:

- confidence should reflect answer correctness and support strength

Why now:

- abstention and penalty mechanisms get much cleaner once confidence has meaning

Data:

- same Stage 1 or Stage 2 setup
- no new dataset is strictly required yet

Model change:

- expose a confidence score for answer / abstain decisions
- optionally expose confidence for support predictions too

Training objective:

- keep previous losses
- add calibration-aware loss if needed

Main metrics:

- `ECE`
- `Brier Score`
- relation between confidence and unsupported-answer rate

What this stage proves:

- the model's confidence is not just a raw score, but something that tracks reliability

Main failure mode:

- confidence values look smooth but do not actually correlate with correctness or support

Stop condition:

- higher confidence generally corresponds to higher support and correctness on the dev set

### Stage 4: Unsupported-Confidence Control

Add:

- explicit penalty for confident unsupported answers

Focus:

- reduce bluffing without collapsing usefulness

Why now:

- this is the first stage where "penalty for unsupported confident claims" is justified rather than guessed

Data:

- Stage 1 or Stage 2 data
- optionally add synthetic unsupported examples for harder negative cases

Model change:

- no major architecture change is required
- reuse support and confidence signals from earlier stages

Training objective:

- keep previous losses
- add a penalty when confidence is high and support is low or absent

Main metrics:

- unsupported-answer rate
- supported-answer rate
- answer `F1`
- abstain `F1`
- calibration metrics before vs after penalty

What this stage proves:

- the model can be pushed away from confident bluffing without becoming useless

Main failure mode:

- penalty works by forcing over-abstention instead of better grounding

Stop condition:

- unsupported confident answers decrease while answer quality remains acceptable

### Stage 5: Retrieval-Grounded QA

Change the task to:

- question -> retrieve evidence -> answer or abstain

Focus:

- realistic evidence access

Why late:

- retrieval changes the problem substantially
- if you add it too early, you will not know whether failures come from answering or from evidence selection

Data:

- retrieval-capable QA or verification datasets
- `HotpotQA` or later extensions become more relevant here

Model change:

- add a retriever
- pass top-k evidence to the answering system

Training objective:

- retrieval loss if trained
- keep answer, abstain, support, and calibration objectives downstream

Main metrics:

- retrieval recall at k
- answer `F1`
- unsupported-answer rate after retrieval
- end-to-end abstain quality

Role in the proof chain:

- this is not the proof of the core mechanism
- this is the realism test for whether the earlier gains survive noisy evidence access

What this stage proves:

- the Stage 1-4 behavior still holds up when evidence is not handed to the model directly
- the pipeline can distinguish retrieval failure from answer-generation failure well enough to debug honestly

What this stage does not prove:

- that the core grounding mechanism was correct in the first place
- that the system works outside retrieval-grounded QA settings

Main failure mode:

- the answer model is blamed for mistakes actually caused by missed or noisy retrieval

Stop condition:

- you can separate retrieval failures from answer-generation failures in evaluation
- end-to-end behavior preserves a meaningful share of the Stage 4 gains instead of collapsing under retrieval noise

### Stage 6: Adaptive Constraint Balancing

Add:

- dynamic weighting or decision control across the main constraints

Focus:

- answer quality
- evidence support
- abstention
- confidence
- unsupported-confidence penalties

Why last:

- real balancing only makes sense after the underlying signals are already working

Possible mechanisms:

- scheduled loss weights
- uncertainty-aware weighting
- learned gating over answer vs abstain decisions
- threshold tuning driven by dev-set trade-offs

Training objective:

- no longer fixed weights only
- weights or thresholds adapt based on support, confidence, or training dynamics

Main metrics:

- trade-off curve quality
- whether the adaptive method dominates fixed-weight baselines
- robustness across different operating points

Role in the proof chain:

- this is not proof that hallucination is solved
- this is the full-pipeline validation stage for the complete KeelNet design

What this stage proves:

- the full system can manage the answer quality / support / abstention trade-off better than fixed balancing baselines
- balancing is not just manual tuning, but a controlled mechanism with measurable benefit

Main failure mode:

- the mechanism becomes complicated but only reproduces what simple threshold tuning already does

What this stage does not prove:

- that the method generalizes to arbitrary open-ended generation
- that truthful behavior is solved outside the exact task, data, and evaluation setting

Stop condition:

- adaptive balancing beats fixed balancing in a reproducible and interpretable way
- the gain is not just threshold gaming or over-abstention hiding weak answers

### Final Recommendation

For the current project, stop at **Stage 4**.

That is enough to study:

- grounding
- abstention
- verification
- calibrated uncertainty
- penalty for unsupported confident claims

Do **not** include retrieval in the current stage.

If Stage 4 works, retrieval should become the next project, not just another extra module.
