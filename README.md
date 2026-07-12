# PID-LoRA: Catastrophic Forgetting as a Closed-Loop Control Problem

> 🚧 **Work in progress.** Experimental design is complete (v7, see
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
> learning/forgetting trade-off than (a) static open-loop α and (b) a threshold
> heuristic — and is KL divergence a sufficient process variable for the loop?

"The integral term turns out to do nothing" is an acceptable answer — the experiment is
designed to detect that outcome (see P-dominance protocol below), not to assume it away.

### The falsifiable prediction

The disturbance in this system is training itself: gradient descent pushes KL upward at
every step. The threshold heuristic's `α *= 0.9` fires unconditionally on every
violating update, with no bound until the clip — that makes it a quantized-step
(bang-bang) integral controller, not a proportional brake. So **both** closed-loop
branches have integral action, and both are expected to bring KL down toward the
setpoint; neither should leave a persistent offset (a true proportional-only brake isn't
implemented by any branch here — see §3 of the design doc).

The prediction is about **how** each branch converges, not whether it does:

- The heuristic's fixed step sizes (0.9 down / 1.05 up) and hysteresis band should
  produce a **limit cycle** — bounded oscillation of roughly constant amplitude around
  the setpoint.
- The PI branch's continuous, magnitude-proportional correction should produce
  **smoother, decaying oscillation into a narrower band**, with lower KL variance in
  the second half of training.

**Prediction:** Figure 1 shows the heuristic oscillating at roughly constant amplitude
and the PI branch settling into a visibly tighter band around the setpoint. If the two
branches show no difference in variance/amplitude, the added PI machinery is not
justified for this system over the cheaper heuristic — and that gets reported as the
headline result.

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
  reported as P-dominant — measured, not assumed. P-dominance and Figure 1's convergence
  read aren't independent evidence: near true convergence `e(t) → 0` forces the integral
  to carry the braking load, so sustained P-dominance mechanically implies an offset, not
  convergence — if the data show both together, that's a pipeline bug to debug (most
  likely `I_history` logging or anti-windup), not a joint finding. See §10.
- **‖B·A‖ tug-of-war diagnostic.** Braking α also weakens the gradient signal reaching the
  adapter, but training keeps pushing — so `‖B·A‖` can grow to compensate for the
  shrinking `α/r` scale even while α is visibly suppressed. Logged every 25 steps
  (global scalar) and every 200 steps (per-layer breakdown, since the tug-of-war need not
  be uniform across layers) via the exact trace identity
  `‖BA‖²_F = tr((BᵀB)(AAᵀ))`. See §8/§9.
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

- [x] Experimental design (v7) — controller redesign, asymmetry consequences, gain
      scaling, control-law reclassification, ‖BA‖ tug-of-war diagnostic
- [x] Infrastructure — data/KL/checkpoint/logging pipeline, unit-tested and validated with
      a local smoke test (tiny model, simulated disconnect/resume cycle)
- [x] Baseline (α=16) and sweep (α=8) training code — implemented and tested; not yet
      run on real GPU
- [ ] ‖B·A‖ Frobenius norm logging (global + per-layer) — computation is unit-tested
      (7 tests) and wired into the training loop; not yet exercised by the smoke test or
      a real run
- [ ] Offline setpoint/gain-check scripts (`scripts/compute_setpoint.py`,
      `scripts/gain_check.py`) — written, no unit tests, never run against real metrics
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
