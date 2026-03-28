# Deep Learning Formulation And Long-Term Path

## Core Claim

Reaching a true deep learning formulation does **not** require a hybrid system.

The key question is not whether the system mixes modular and learned parts.
The key question is whether the final answer-or-abstain decision is expressed
as a learned neural decision problem rather than added mostly through post-hoc
thresholds and fixed rules.

## What "Deep Learning Formulation" Means Here

In this project, a deep learning formulation means:

- the model receives structured inputs
- the model predicts decision-relevant outputs
- the final answer-or-abstain behavior is learned inside the model
- the losses shape the answer-versus-abstain trade-off directly

So the issue is not just "does the system contain a transformer?"
The issue is whether the final decision logic itself is learned.

## KeelNet In That Framing

- **Stage 4** is the strongest modular operating point in the repo, but it is
  not the clearest deep learning formulation of the final decision.
  Its behavior is still driven mainly by calibrated scores and fixed control
  rules.
- **Stage 5** moves closer to a learned formulation, but it still does not
  express the final decision as clearly as the later action-based stages.
- **Stage 7** is the strongest non-hybrid route toward a pure learned
  formulation.
  It turns candidate answers and `abstain` into one learned action space and
  predicts utility/risk-style signals inside the model.
- **Stage 8.2** is also a deep learning formulation, but it is hybrid because
  it injects calibrated Stage 4 support into the learned action layer.

## Why Hybrid Was Useful

Hybrid was useful in this project because the pure learned route was not yet
reliable enough.
The hybrid structure gave the learned action layer access to a stronger
calibrated support signal.

So:

- **best modular operating point today:** Stage 4
- **best pure learned direction:** Stage 7
- **best current learned endpoint in the repo:** Stage 8.2

This means the project does not need hybridization in principle.
It used hybridization because the current pure learned formulation still fails
to generalize its risk signal tightly enough beyond validation.

## Where The Project Is Now

The project is no longer in the early "does anything work?" phase.
It is in the transition from mechanism discovery to stronger learned
formulation.

The cleanest way to describe the current position is:

- **mechanism discovery:** done
- **failure diagnosis:** done
- **controlled proof strong enough for final claims:** not yet
- **better formulation:** in progress
- **robust comparison:** partially done
- **domain expansion:** not yet
- **deployment-grade design:** not yet

More concretely:

- The project has already shown that abstention matters, support is learnable,
  calibration improves the decision signals, and modular control can define a
  strong groundedness operating point.
- The main bottleneck is now much clearer than before:
  learned risk and control signals do not yet generalize tightly enough beyond
  validation-time selection.
- Stage 7 and Stage 8.2 are no longer random extensions.
  They are attempts to express the final answer-or-abstain decision more
  directly as a learned problem.
- The next step is not broader deployment or wider domains.
  The next step is to make the learned risk formulation generalize inside the
  controlled setting first.

This is a good research position.
The project is no longer searching blindly.
It is now asking the right next question:
how to make the learned risk signal generalize well enough to challenge the
modular anchor.

## Hidden Insight

The main failure is probably not ordinary QA quality.
It is **tail-risk decision making**.

The learned systems can already recover strong answerable-question quality.
Support is learnable, and calibration helps.
But the system still fails on the small set of borderline, high-cost cases
where it should abstain and does not.

That changes the interpretation of the project:

- a verifier with decent average metrics is not automatically useful for
  deployment control
- answerable $F_1$ gains do not mean the dangerous boundary cases are solved
- the real object is selective risk control, not just support-aware QA

This also helps explain why Stage 4 still looks strong.
Stage 4 is not better because it learns more.
It is better because it enforces a stable conservative rule on the edge cases
that dominate unsupported-answer risk.

So the deeper formulation is:

> KeelNet is not mainly failing at QA.
> It is failing at learning a robust abstention boundary under tail risk.

## Recommended Long-Term Path For KeelNet

The most realistic long-term path is staged rather than abrupt.

The roadmap is:

- **now:** strengthen the controlled proof
- **next:** make the pure learned risk formulation generalize
- **then:** compare robustness under shift
- **then:** expand task realism
- **finally:** move toward broader deployment-grade design

The best target sentence for the project is:

> Build a learned answer-or-abstain system that matches Stage 4's groundedness
> while retaining Stage 5/7 answer quality, then test whether that behavior
> survives domain shift.

## How To Execute That Roadmap

### 1. Now: Strengthen The Controlled Proof

This step is about making the narrow-setting claim hard to dismiss.
It does **not** mean inventing a new stage.
It means proving that the current conclusion survives fair comparison,
repetition, and simplification.

Highest-leverage priority:

- normalize **Stage 4** and **Stage 7** to the same clean protocol already used
  by **Stage 8.2**
- then compare Stage 4, Stage 7, and Stage 8.2 directly as:
  best modular anchor, best pure learned route, and best current bridge model

This is the most valuable next proof block because it tests the paper's central
structural question on matched conditions rather than diffusing effort across
every earlier stage first.

How to do it:

- rerun the strongest checkpoints on the same clean split protocol:
  Stage 4, Stage 7, and Stage 8.2 at minimum
- add simple baselines:
  raw confidence thresholding, calibrated thresholding, and a simple
  support-threshold controller
- run at least 3 seeds for the main compared systems
- add one ablation table for Stage 8.2:
  Stage 5 only, Stage 7-style action learner without calibrated support,
  Stage 8.2 with calibrated support, and optionally with versus without a hard
  support shield
- summarize the result in one clean table with:
  split, answerable $F_1$, overall $F_1$, unsupported-answer rate, abstain
  $F_1$, answer rate, and selected threshold

What "strong enough" looks like:

- the same conclusion survives across seeds
- the main systems are compared on the same protocol
- the learned system is better than naive threshold baselines, or honestly
  shown not to be
- the ablations support the claimed mechanism

### 2. Next: Make The Pure Learned Risk Formulation Generalize

This step is about the strongest non-hybrid target:
make Stage 7 strong enough that it no longer needs Stage 4 support injection to
look principled.

How to do it:

- treat Stage 7 as the main pure learned research path
- improve the candidate-risk signal rather than only tuning thresholds
- feed the learner stronger candidate-level evidence signals
  while keeping the final decision learned end-to-end
- improve hard negative construction so the model sees more borderline unsafe
  answer cases during training
- inspect dual-variable behavior, especially whether the learner is pushing
  harder on over-abstention than on unsupported-answer suppression
- calibrate or regularize the learned risk head directly if needed
- rerun clean-split Stage 7 variants until one of them approaches the Stage 4
  groundedness region without giving up all answerable quality

What success looks like:

- a pure Stage 7-style learner matches or beats Stage 4 on the same clean split
- the unsupported-answer gap shrinks for the right reason:
  better risk separation, not only lower answer rate

### 3. Then: Compare Robustness Under Shift

This step asks whether the learned abstain behavior survives outside the narrow
training distribution.

How to do it:

- keep the task structure the same:
  answer or abstain from supplied evidence
- change the distribution in one controlled way at a time:
  corpus style, question style, unsupported-answer pattern, or evidence
  structure
- start with no retraining:
  compare Stage 4 and the best learned model zero-shot on the shifted set
- only after that, try adaptation or fine-tuning
- report not just accuracy-style metrics but also:
  unsupported-answer rate, answer rate, supported-answer rate among answered
  cases, and abstain behavior

Good first domain-shift tests:

- train on SQuAD~2.0, test on policy or compliance document QA
- train on SQuAD~2.0, test on technical support or internal documentation QA
- train on clean benchmark questions, test on noisier paraphrased user-style
  questions

What success looks like:

- the learned system degrades gracefully under shift
- its abstain policy remains interpretable
- Stage 4 no longer has an overwhelming robustness advantage

### 4. Then: Expand Task Realism

Only after the controlled setting is strong enough should the task get more
realistic.

How to do it:

- move from single-passage evidence to retrieval-grounded evidence
- test longer passages and more distractor-heavy contexts
- test multiple plausible spans and noisier retrieval candidates
- keep the answer-or-abstain decision explicit even as the inputs get more
  realistic
- expand one realism dimension at a time so the failure source stays
  interpretable

The right order is:

- grounded extractive QA first
- then broader grounded QA
- then retrieval-grounded QA
- only later broader hallucination-control claims

What success looks like:

- the learned selective decision still works when the task gets harder
- the system does not collapse as soon as retrieval noise or longer evidence is
  introduced

### 5. Finally: Move Toward Broader Deployment-Grade Design

This is the last step, not the next step.
Deployment-grade design means the system is not only strong in a paper sense,
but also stable, inspectable, and usable.

How to do it:

- keep a modular safety anchor even while the learned system improves
- define deployment-facing budgets clearly:
  unsupported-answer rate, over-abstention rate, answer rate, and escalation
  behavior
- log calibration and abstention metrics continuously rather than only final
  headline scores
- build fallback behavior:
  when the learned system is uncertain, defer to abstention or to a safer
  modular policy
- test on realistic document sources before claiming broad usefulness
- introduce human review or audit paths for high-cost settings

What success looks like:

- the system fails in understandable ways
- calibration stays meaningful after deployment-like stress
- unsupported answering remains bounded under realistic inputs
- the learned model needs less modular backup over time, not more

## What Domain Shift Would Mean For KeelNet

For KeelNet, domain shift does not mean a completely different task.
It means the same answer-or-abstain grounded-QA idea is tested on a meaningfully
different data distribution.

Concrete examples:

1. **Different corpus style**
   Train on SQuAD~2.0 Wikipedia-style passages, then test on news articles,
   policy manuals, internal documentation, or technical support documents.

2. **Different question style**
   Train on short benchmark factoid questions, then test on noisier user-like
   questions, longer paraphrases, or more indirect phrasings.

3. **Different unsupported-answer pattern**
   Train where unsupported questions have one frequency and structure, then
   test where unsupported questions are more frequent, more subtle, or more
   distractor-heavy.

4. **Different evidence structure**
   Train on clean single-passage evidence, then test on longer passages, denser
   distractors, or passages with multiple plausible candidate spans.

The most realistic next-step shift test for this project would be:

- train on SQuAD~2.0
- then test on bounded document QA built from policy, compliance, or technical
  support documents

That would show whether the learned abstain and risk behavior survives outside
the narrow Wikipedia-style benchmark distribution.

## One Real Example Of The Full Path

One useful example is **AlphaFold**.

It is a different domain, but it followed the same broad research ladder:

- **controlled proof:** protein structure prediction was studied in narrow,
  benchmarked settings such as CASP
- **failure diagnosis:** earlier approaches revealed where template-based and
  coevolution-based methods were still insufficient
- **better formulation:** AlphaFold2 did not just scale up; it introduced a
  stronger structure-aware formulation for the prediction problem
- **robust comparison:** it was tested on strong blind benchmarks against
  serious external methods
- **domain expansion:** the work later expanded from core monomer structure
  prediction to broader settings such as complexes
- **deployment-grade design:** the system became broadly usable through
  downstream tools and the AlphaFold Protein Structure Database

The lesson is not that KeelNet should copy AlphaFold technically.
The lesson is that strong long-term systems usually do not begin as broad
deployment solutions.
They begin as narrow, benchmarked, diagnostic research programs that earn the
right to expand.
