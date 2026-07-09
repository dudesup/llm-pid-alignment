# PID-LoRA: Catastrophic Forgetting as a Closed-Loop Control Problem

> 🚧 **Work in progress.** Experimental design is complete (v4, see
> [`lora_pid_project_summary.md`](lora_pid_project_summary.md)); infrastructure is built
> and validated end-to-end (unit tests + a local smoke test on a tiny model, including a
> simulated disconnect/resume cycle). GPU access (Colab Pro) is now set up; training runs
> have not started yet. Results and figures will land here as they come in.

Can catastrophic forgetting during LLM fine-tuning be treated as a feedback control
problem — with the LoRA scaling coefficient α as the control signal and KL divergence
from the base model as the process variable?

Standard SFT is open-loop: you pick α once and hope the model doesn't drift too far.
This project closes the loop: a barrier PI controller measures KL(base ‖ current) on a
frozen control set every 25 steps and dynamically brakes α to keep drift within a bound
(KL ≤ setpoint), while training loss keeps improving.

## Research question

> Does closed-loop regulation of α via a KL-driven barrier PI controller yield a better
> learning/forgetting trade-off than (a) static α and (b) an open-loop threshold
> heuristic — and is KL divergence a sufficient process variable for the loop?

"The integral term turns out to do nothing" is an acceptable answer — the experiment is
designed to detect that outcome (see P-dominance protocol below), not to assume it away.

### The falsifiable prediction

The disturbance in this system is training itself: gradient descent pushes KL upward at
every step. Against a persistent disturbance:

- Proportional-only braking and the threshold heuristic must settle *above* the
  setpoint — they need a standing error to generate any braking signal at all
  (classic steady-state offset).
- The integral term accumulates the violation and keeps tightening α until KL
  returns to the setpoint.

**Prediction:** the PI branch converges to the setpoint; the heuristic oscillates above
it. If the KL trajectories do not show this separation, the PI machinery is not
justified for this system — and that gets reported as the headline result.

## Design highlights

- **Barrier, not setpoint tracking.** Error is asymmetric: `e = min(0, setpoint − KL)`.
  A symmetric controller would actively push the model away from base in early training
  (KL below setpoint → positive control signal). A safety controller should only brake.
  Consequence: `u(t) ≤ 0` always, so the reachable range is α ∈ [4, 16] with
  `α_max = α_base`.
- **No derivative term.** KL measured on 50 prompts is noisy; a D-term amplifies
  measurement noise. Light EMA (β = 0.5) on the measurement instead. "PID" in the title
  refers to the control-theoretic framing, not the implemented terms.
- **Two disjoint datasets.** The controller sees a 50-prompt control set (general-domain,
  not hh-rlhf — so the loop regulates base-capability drift, not in-domain style).
  Reported metrics come from a frozen 100-prompt held-out set the controller never sees —
  otherwise the controller directly optimizes the evidence (tautology).
- **Honest ablations.** Pareto front over four points: static α=8, static α=16, threshold
  heuristic, barrier PI — with the heuristic sharing the identical measurement pipeline
  (control set, EMA, cadence) so any difference is attributable to the control law alone.
- **P-dominance protocol.** After the runs, `|I(t)| / |Kp·e(t)|` is computed over all
  braking updates. If the integral contributes < 20% for most of the run, the system is
  reported as P-dominant — measured, not assumed.
- **Anti-windup, release decay, disambiguated failure flags** — see §8 of the
  [design doc](lora_pid_project_summary.md) for the full control law and gain-scaling
  rationale.

## Setup

- **Model:** Qwen2.5-3B-Instruct, 4-bit, LoRA r=8 — fits a single T4
- **Data:** Anthropic/hh-rlhf (chosen responses, prompt-masked loss)
- **KL:** forward KL(base ‖ current), frozen top-k reference log-probs (k=1000, fp16),
  identical truncated measurement across all branches
- **Budget:** 4 runs × 1000 steps ≈ 5 Colab T4 sessions

## Status

- [x] Experimental design (v4) — controller redesign, asymmetry consequences, gain scaling
- [x] Infrastructure — data/KL/checkpoint/logging pipeline, unit-tested and validated with
      a local smoke test (tiny model, simulated disconnect/resume cycle)
- [x] Baseline (α=16) and sweep (α=8) training code — implemented and tested; not yet
      run on real GPU
- [ ] Offline setpoint/gain-check scripts — not yet implemented in this repo
- [ ] Threshold heuristic controller — not implemented yet, deliberately deferred until
      real baseline/sweep data exists to inform it
- [ ] Barrier PI controller — not implemented yet, depends on the offline gain check above
- [ ] Analysis: Pareto front, P-dominance evaluation
- [ ] Stretch: LR-only modulation ablation (decomposing the dual role of α)

## Known limitations

Toy scale by design: n=1 per branch (results suggest, not prove), 3B model, and α
couples two mechanisms (adapter output scale and effective adapter learning rate) that
this iteration names but does not fully decompose. See §11 of the
[design doc](lora_pid_project_summary.md) for the complete table.

## Part of a broader framework

Companion project: [llm-control-alignment](https://github.com/dudesup/llm-control-alignment) —
runtime suppression of unsafe activations at inference time (H∞ control in SAE feature
space). Together they argue that alignment can be formalized as a control problem at two
levels: training-time regulation of distributional drift (this repo) and inference-time
regulation of activations (companion repo), sharing the same foundation of feedback
control and stability bounds.
