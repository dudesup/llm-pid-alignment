# Mini-Constitutional Alignment via LoRA + PID Control
## Research Design Document
**v7 — P-dominance protocol's inverted-separation case made explicit (v6 left a stale reading contradicting its own new consistency check); ‖BA‖ layer-aggregation decided; Figure 5 averaging window widened and unified across axes**

> **Terminological note:** The controller implemented is a **barrier PI controller** (proportional-integral, asymmetric error) — the derivative term is omitted due to measurement noise (see Section 8). "PID" in the project title refers to the general control-theoretic framing shared with the companion research note, not to the specific implementation.

> **Changelog v6 → v7:**
> - **Fixed a stale v4-era sentence in the P-dominance protocol (Section 10) that had gone from outdated to self-contradictory.** It claimed a P-dominant PI branch and the heuristic "differ only in proportionality of braking" — but the v6 internal-consistency-check paragraph directly beneath it establishes that P-dominance mechanically implies an offset, while the heuristic (a quantized-step integrator regardless of the PI branch's regime) is still expected to converge. Replaced with an explicit statement of the inverted case: **a P-dominant PI branch is a de facto proportional-only brake and should retain the classical steady-state offset, while the heuristic converges — the cheaper heuristic wins on convergence even though it likely still loses on smoothness/variance.** Section 10 now names all three P-dominance/Figure-1 combinations explicitly: convergence+I-dominance (headline), offset+P-dominance (inverted separation — legitimate), convergence+P-dominance (contradiction → pipeline bug, not physics).
> - **Decided the ‖B·A‖ layer-aggregation question, left open since v5.** Section 9: a global scalar `√(Σ_l ‖B_l·A_l‖²_F)` every 25 steps feeds Figure 4b and the `(α/r)·‖BA‖` product (both are global-α-scale arguments); a per-layer breakdown every 200 steps is added as a diagnostic supplement, since the tug-of-war need not be uniform across layers and a global sum can look flat purely from cross-layer cancellation. Free: both come from the same per-layer trace-identity pass (Section 8/9).
> - **Figure 5's Pareto-point averaging window widened from last-50-steps to last-200-steps, and unified across both axes.** At the 25-step KL cadence, 50 steps was only 2 noisy 50-prompt KL measurements anchoring a point the headline claim depends on; 200 steps gives 8, and — under the Section 15 plateau hypothesis — still sits inside the stabilized regime, so it doesn't reintroduce early-training bias. Loss (logged every step) is unaffected either way, but both axes now share the identical window so the Pareto point is internally consistent.

> **Changelog v5 → v6:**
> - **Removed the false claim that the static α=8/α=16 sweep "tests" a true P-only controller (Section 3, Section 8).** By definition a proportional controller's output changes with the measurement (`u = Kp·e(t)`); the static sweep reads no measurement at all — it is open-loop, its value fixed before training starts, and it neither transiently approaches nor settles to an equilibrium in response to the disturbance. The classical "P-only leaves a steady-state offset" argument stays in Section 8 as analytical motivation for including the integral term, but is now explicitly marked as **untested by any branch in this design** — a true P-only ablation would need its own (out-of-scope) fifth arm.
> - **Fixed a direct contradiction between Section 2 and Section 7.** Section 2's research question called the threshold heuristic "open-loop"; Section 7 calls it one of "the two closed-loop branches." The heuristic reads a measurement that its own past output influenced — that is a closed loop by definition, independent of how crude the control law is. Section 2 corrected; only the static branches (baseline, sweep) are open-loop.
> - **Added an internal consistency check tying the P-dominance protocol to Figure 1 (Section 10).** They are not independent tests: near true setpoint convergence, `e(t) → 0` forces `Kp·e(t) → 0`, so any braking still holding α down against the disturbance is carried almost entirely by `I(t)` — meaning sustained convergence mechanically implies I-dominance in that stretch, and sustained P-dominance mechanically implies an offset. If the data show P-dominance *and* setpoint convergence over the same stretch, that is not a joint finding but a signal of a pipeline bug (most likely `I_history` logging or the anti-windup freeze). Computed within the disturbance-active window (added in v5), not globally, since the quasi-static assumption behind this argument can legitimately break down across a regime change (e.g. the hypothesized natural KL plateau, Section 15).
> - **Code:** `lora_bA_frobenius_norm()` (`pidlora/model.py`) now uses the trace identity `‖BA‖²_F = tr((BᵀB)(AAᵀ))` — both factors are r×r — instead of materializing the full (out_features × in_features) delta matrix per layer. Exact, not approximate; existing tests (which compare against the explicit `B @ A` computation) now serve as a regression check on the identity itself.

> **Changelog v4 → v5:**
> - **Corrected the headline falsifiable prediction (Section 3).** The threshold heuristic's `α *= 0.9` while a violation persists is not proportional braking — it is unconditional accumulation with no bound until the clip, i.e. a quantized-step (bang-bang) integral controller. It therefore should *also* drive KL toward the setpoint, not leave a steady-state offset next to it — that part of the old v4 prediction was wrong and is corrected here before running the experiment, not after seeing Figure 1. The real predicted separation is **convergence character**: the heuristic should produce a limit cycle around the setpoint (amplitude set by the fixed 0.9/1.05 step sizes and the 0.8·setpoint hysteresis band), while PI should produce smoother convergence with proportional reaction to violation size and lower KL variance in the second half of training. Section 8 and Section 10 updated to match.
> - **Added ‖B·A‖ (Frobenius norm) logging every 25 steps** (Section 7, Section 9) to catch the suppression-vs-compensation tug-of-war live, not just at the final static sanity check: controller brakes α → gradient signal through the adapter weakens → training keeps pushing → ‖BA‖ grows to compensate → effective output scale recovers → KL rises again → controller brakes harder. New Figure 4b plots ‖BA‖ against α_history; a stable (α/r)·‖BA‖ product alongside anti-correlated α and ‖BA‖ trajectories is itself a result (controller in dynamic equilibrium with an adversarial optimizer), and is the live counterpart of the existing end-of-run α=16 re-evaluation sanity check (Section 10).
> - **Added a disturbance-active comparison window (Section 10).** Section 15 already hypothesizes the baseline KL trajectory naturally plateaus by ~step 500; if so, the "disturbance" (training systematically pushing KL upward) weakens in the second half of every run regardless of control law, and all branches could converge for reasons unrelated to the control law — collapsing the Figure 1 separation for a confounded reason. The control-law comparison (Figure 1 separation, AUC) is now reported both globally and restricted to the window where `KL_baseline > 1.2 · setpoint`, i.e. where the baseline is actively violating.
> - Limitation #7 clarified: top-k truncation affects not only the absolute KL scale but also the directional (forward-vs-reverse KL) motivation — the zero-avoiding property only operates within the preserved top-k support; tail-mass forgetting outside it is invisible to the controller either way.
> - Adam-transient reporting (Section 8) extended: in addition to logging grad norm in ±10-step windows around α changes, report whether transient magnitude correlates with the size of the α jump — cheap, and it is the concrete evidence for defending the 25-step update cadence.

> **Changelog v3 → v4:**
> - α_max = α_base = 16 (with asymmetric error, u(t) ≤ 0 always → α ∈ [4, 16]; the upper half of the old range was unreachable). HIGH_ALPHA flag removed as dead code; its diagnostic function ("setpoint too aggressive") migrates to a disambiguated LOW_ALPHA flag (Section 8).
> - Controller update frequency: every **25** steps (40 corrections per 1000-step run, up from 20). Ki rescaled to the update horizon: **Ki = 3 ≈ Kp/10**. Release decay rescaled to **0.9** per update (half-life ≈ 7 updates), consistent with the new accumulation rate. Light EMA (β = 0.5) on the KL measurement.
> - Gain tuning moved from "after the first 200 steps" (4 data points — not tunable) to **an offline gain check on the full baseline KL trajectory** before the PI branch runs.
> - P-dominance is now a **reporting protocol with a quantitative criterion** (Section 10), not a preemptive design concession. Falsifiable prediction added: P-only behavior and the threshold heuristic leave a steady-state error above setpoint; the integral term should eliminate it. *(Superseded by v5 — see above: this held for P-only but not for the heuristic.)*
> - Tail-handling decision for top-k reference log-probs made explicit (Section 5).
> - Stale v3 remnants fixed: Section 14 no longer references sweep α∈{8,32}; Section 15 note corrected; Section 11 row 3 aligned with the stretch-goal status of the LR-only run; compute budget updated to ~5 sessions.

---

## 1. Project Objective

To demonstrate that catastrophic forgetting during LLM fine-tuning can be framed as a closed-loop control problem — and that dynamic regulation of the LoRA scaling coefficient (`α`) via a KL-driven barrier PI controller yields a better learning/forgetting trade-off than a static `α`.

This project is part of a broader framework that treats alignment as a control-theoretic problem (see: "Modeling LLM Alignment Failures via Discrete-Time Control Theory"). The companion project (llm-control-alignment) addresses runtime suppression of unsafe activations during inference. This project addresses training-time control of distributional drift.

---

## 2. Research Question

> **Does closed-loop regulation of `α` via a KL-driven barrier PI controller yield a better learning/forgetting trade-off than static open-loop `α` and than a threshold heuristic — and is KL divergence a sufficient process variable for the controller?**

**Terminology (v6):** the threshold heuristic is **closed-loop**, not open-loop — it reads a measurement (KL_filt) that its own past output influenced, which is the definition of a feedback loop, regardless of how crude the control law is (Section 7 already calls it one of "the two closed-loop branches"; this line previously called it "open-loop," contradicting that). Only the static `α` branches (baseline, sweep) are open-loop — they never read KL_filt at all.

The second part is critical: the literature establishes that KL divergence correlates with forgetting (Shenfeld et al. 2025), but no prior work has examined whether it can serve as the process variable in a closed feedback loop. Note that "the integral term does not contribute despite correct gain scaling" is a legitimate answer to this question — the experiment is designed to detect it (Section 10, P-dominance protocol), not to assume it away.

---

## 3. Thesis

> SFT with dynamically regulated `α` (barrier PI controller, constraint: KL ≤ 0.5) maintains KL divergence within bounds without degrading target-task loss — achieving a better trade-off than static `α` and than a threshold heuristic under an equal training step budget. The setpoint is determined empirically from the step-500 checkpoint of the baseline run.

**Falsifiable secondary prediction (what distinguishes PI from the heuristic):** the two closed-loop branches are not "P-like vs I-like." The heuristic's `α *= 0.9` fires unconditionally on every update where a violation persists, with no proportionality to the violation's size and no bound until the clip — that is a quantized-step (bang-bang) integral controller, not a proportional brake. Both closed-loop branches therefore have integral action and both are expected to bring KL down toward the setpoint; neither should leave a persistent offset. The distinguishing prediction is **convergence character, not convergence vs offset**: the heuristic's fixed, unconditional step sizes (0.9 down / 1.05 up) combined with the 0.8·setpoint hysteresis band should produce a **limit cycle** — KL oscillating around the setpoint with amplitude set by those step sizes — while the PI controller's proportional reaction to violation *magnitude* (via Kp·e) plus the EMA-filtered, continuously-valued integral action should produce a **smoother trajectory with lower KL variance around the setpoint** in the second half of training. Figure 1 is read as: heuristic = bounded oscillation of roughly constant amplitude; PI = decaying oscillation settling into a narrower band. If Figure 1 shows no difference in variance/amplitude between the two branches, the added PI machinery is not justified for this system over the cheaper heuristic — and that is reported as such.

**Scope note (v6):** the classical "P-only leaves a steady-state offset" argument (Section 8) is analytical background for *why the integral term was included in the design*, not a claim tested by any branch in this experiment. A true proportional-only feedback controller — one whose brake reads KL_filt every update and scales with the current violation, with no accumulation — is not implemented here. The static α=8/α=16 sweep is **not** that controller: it is open-loop, reads no measurement at all, and its own value is fixed before training starts rather than settling to an equilibrium in response to the disturbance. It tests a different, unrelated hypothesis ("does a static low α match the trade-off without any loop" — Section 7), and provides no counterfactual for the P-only offset claim. If a true P-only ablation were wanted, it would need its own branch (a fifth arm: proportional brake, Ki=0) — out of scope for the current step/compute budget, and not required for the PI-vs-heuristic separation this design tests.

---

## 4. Literature Context

**Established findings:**
- KL divergence between base policy and fine-tuned policy correlates with catastrophic forgetting (Shenfeld et al. 2025)
- LoRA retains more source-domain knowledge than full fine-tuning but learns less on the target task (Biderman et al. 2024)
- AlignGuard-LoRA (2025) achieves ~50% forgetting reduction via Fisher-guided regularization
- AdaLoRA/DyLoRA dynamically adjust adapter rank/importance — but neither closes a PI loop on KL as a process variable
- To our knowledge, no prior work closes this loop using KL divergence as the process variable for a PID-style controller — this is an unconventional framing relative to mainstream alignment research (RLHF, Constitutional AI, scalable oversight), presented here as an independent engineering contribution rather than a claim of novelty over the full literature

**Novel contributions of this project:**
- Closed-loop regulation of `α` via barrier PI controller — `α` is treated as a control signal, not a hyperparameter
- KL divergence as process variable in a closed loop (not merely an evaluation metric)
- Control-theoretic framing: SFT = open-loop feedforward control; PID-SFT = closed-loop feedback control; the KL constraint is enforced as a barrier, not tracked as a setpoint

---

## 5. Assumptions

**Model:** Qwen2.5-3B-Instruct
- Fits on T4 (16 GB) in 4-bit without memory issues
- Large enough for forgetting to be measurable
- Smaller than Llama-3-8B → less time spent debugging before training even starts

**Dataset:** Anthropic/hh-rlhf
- Format: `chosen` / `rejected` pairs
- SFT trains on `chosen` responses only
- Loss computed on response tokens only (prompt is masked)

**KL divergence — Option A (frozen reference log-probs):**
- Before training: pass the control set through the model and save log-probs (not full logits) on CPU/disk — store top-k (k ~ 1000) in fp16; reduces memory by ~2 orders of magnitude vs full 152k vocab logits in fp32
- **Tail handling (explicit decision):** the forward KL sum Σ p_base·log(p_base/p_current) is truncated to the stored top-k support of p_base, with p_base renormalized over that support. This underestimates KL by the tail contribution; with k = 1000 the tail mass of p_base on natural text is typically < 1–2%, and — critically — the same truncated measurement is used identically across all branches, so the *comparison* between branches is unaffected. The bias affects only the absolute scale of the setpoint, which is itself derived from the same truncated measurement (baseline run), keeping the loop internally consistent.
- During training: every N steps, transfer reference log-probs batch-wise to GPU, compute KL, release memory
- Rationale: full Qwen2.5 vocab (~152k tokens) × 50 sequences × sequence length in fp16 = potentially several GB — keeping on GPU risks OOM alongside 4-bit model + LoRA + optimizer state on T4
- Direction: Forward KL(p_base ‖ p_current), averaged per token, then over the control set
- Rationale: single model in memory; VRAM-friendly on T4
- **Control set composition (explicit):** 50 general-domain prompts (NOT from hh-rlhf) — e.g. wikitext-2 sentences, factual questions, coding snippets (NOT multiple-choice — perplexity on MC is not interpretable). Rationale: if control set = hh-rlhf prompts, the controller regulates in-domain drift; if general, it regulates base capability drift. We choose general prompts so the PI controller preserves base capabilities, not merely in-domain style. This is a deliberate design decision with direct consequences for what the controller optimises.

**Formal KL definition:**
$$KL = \frac{1}{|S|} \sum_{s \in S} \frac{1}{|s|} \sum_{t=1}^{|s|} KL\bigl(p_{\text{base}}(\cdot \mid s_{<t}) \parallel p_{\text{current}}(\cdot \mid s_{<t})\bigr)$$

Where $S$ = control set, $s$ = sequence, $t$ = token position; distributions truncated and renormalized to the stored top-k support of $p_\text{base}$ (see tail handling above).

**Why Forward KL and not Reverse KL:**
Forward KL($p_\text{base}$ ‖ $p_\text{current}$) has the *zero-avoiding* property — it forces $p_\text{current}$ to cover the full support of $p_\text{base}$, preventing the model from "forgetting" regions of the base distribution. This is exactly the property needed to combat catastrophic forgetting.

By contrast, classical RLHF (PPO) uses Reverse KL($p_\text{current}$ ‖ $p_\text{base}$) as a regularizer — the goal there is to prevent the policy from straying too far during exploration, not to preserve full coverage of the base. Different objectives warrant different KL directions.

**Setpoint KL ≤ 0.5:**
- Not arbitrary — derived from the step-500 checkpoint of the baseline run (static α=16, 1000 steps)
- Setpoint = 50% of the typical KL measured at step 500 of that run
- Note: the baseline will naturally exceed the setpoint (it is open-loop). This is the core of the thesis — closed-loop vs open-loop control — not an unfair comparison.

---

## 6. Two Disjoint Datasets (Critical)

**Problem:** if the PI controller uses the same dataset as the forgetting evaluation → tautology (the controller directly optimizes the metric presented as evidence).

**Solution — two disjoint sets:**

| Dataset | Size | Role | Visible to PI controller? |
|---|---|---|---|
| Control set | 50 prompts | Error signal for PI controller; reference log-probs | YES |
| Held-out set | 100 prompts | Forgetting evaluation; reported results | NO — never |

The held-out set is frozen before the experiment begins and used exclusively for final reporting. It also serves as the source for preference margin evaluation: each hh-rlhf pair in the held-out split provides a (chosen, rejected) tuple for computing log-probability margin before and after fine-tuning. Composition: ~50 prompts from wikitext-2 (perplexity is interpretable on continuous text; MMLU is multiple-choice — perplexity on MC questions is not meaningful) + ~50 prompts from a held-out hh-rlhf split. The mixed composition allows measuring both in-domain degradation and loss of general base-model capabilities.

**Terminological note:** if the held-out set were drawn exclusively from hh-rlhf, perplexity would measure in-domain distributional drift, not catastrophic forgetting in the sense of lost general capabilities. The mixed held-out set is required for the claim "prevents catastrophic forgetting" to be methodologically valid.

---

## 7. Experiment Architecture

### Branch A — Baseline (static α)
```
Model (4-bit) + LoRA (r=8, α=16, static)
    ↓
Training loop on hh-rlhf chosen
    ↓
Every 25 steps : eval KL on control set + log loss + log grad norm + log ‖B·A‖ (Frobenius norm, adapter weights)
Every 200 steps: eval perplexity + preference margin on held-out set
```
KL is logged every 25 steps on the baseline as well (not only on closed-loop branches) so that Figure 1 curves have identical density and the offline gain check (Section 8) has 40 trajectory points to work with. ‖B·A‖ is logged on the baseline too, as the reference trajectory for the tug-of-war comparison described under Branch B.

### Branch B — Barrier-PI-controlled (dynamic α)
```
Model (4-bit) + LoRA (r=8, α=dynamic ∈ [4, 16])
    ↓
Training loop on hh-rlhf chosen
    ↓
Every 25 steps:
    1. Eval KL on control set
    2. KL_filt = 0.5·KL_raw + 0.5·KL_filt_prev   (EMA, β=0.5)
    3. e(t) = min(0, KL_setpoint − KL_filt)       (barrier error)
    4. Barrier PI update → new α  (see Section 8)
    5. base_model.scaling[adapter] = α_new / r
    ↓
Log: KL raw + filtered (control), loss, grad norm, α_history, I_history, ‖B·A‖ (Frobenius norm)
Every 200 steps: eval perplexity + preference margin on held-out set
```
40 controller corrections per 1000-step run. `I_history` is logged explicitly — it feeds the P-dominance diagnostic (Section 10). `‖B·A‖` is logged explicitly — it feeds the suppression-vs-compensation tug-of-war diagnostic (Section 8, Figure 4b): the controller brakes α while gradient descent keeps pushing, so the adapter's own norm can grow to compensate for the shrinking scale, partially undoing the brake at the weight level even while α is visibly suppressed.

### Threshold heuristic branch (replaces the old α=32 sweep point)
```
Every 25 steps (same frequency and same EMA-filtered KL signal as Branch B):
    if   KL_filt > setpoint:       α *= 0.9
    elif KL_filt < 0.8·setpoint:   α *= 1.05
    α = clip(α, 4, 16)

Every 200 steps: eval perplexity + preference margin on held-out set
```
The heuristic uses the identical measurement pipeline (control set, EMA, update cadence) as the PI branch — the *only* difference between the two closed-loop branches is the control law. Any observed difference is therefore attributable to the control law, not to measurement or timing.

Note on heuristic design: pure `if KL > setpoint: α *= 0.9` is a ratchet — α can only decrease, so measurement noise (50 prompts) would monotonically collapse it to α_min (degenerate solution). The recovery branch prevents this. If PI does not outperform this heuristic, the result is still valuable — and honestly reported (Section 10).

**Reclassification (v5):** the recovery branch does not make this a proportional controller. `α *= 0.9` fires on every update where `KL_filt > setpoint`, regardless of *how far* above setpoint KL is — the step size is fixed, not scaled by the violation. Applied repeatedly across consecutive violating updates, this is unconditional accumulation with no bound until the α=4 clip: functionally a quantized-step (bang-bang) integral controller, not a P controller. It is expected to bring KL down toward the setpoint just as the PI branch does (see Section 3) — the two closed-loop branches are not a P-vs-PI contrast, they are a *quantized-step integral* vs *continuous, magnitude-proportional PI* contrast. See Section 3 for the resulting prediction (limit cycle vs smoother convergence).

### Static sweep
- Static sweep: α ∈ {8} (α=16 already covered by the baseline run)

Rationale: α=32 would only confirm "more α → more drift" — already demonstrated by baseline α=16 exceeding the setpoint. The dangerous alternative hypothesis is "static low α achieves the same trade-off without any loop." α=8 tests this — and after the v4 range change it is even more clearly the right sweep point: **the PI controller operates exclusively within α ∈ [4, 16], i.e. exactly between the two static Pareto anchors α=8 and α=16.** Pareto front: {α=8, α=16, heuristic, PI} — four points, three of them in the low-drift region where the comparison is non-trivial.

---

## 8. Barrier Controller — PI-based (not PID)

**Why barrier control, not setpoint tracking:**
The thesis states KL ≤ setpoint as a constraint, not a tracking target. A symmetric PI controller would actively increase α when KL is below setpoint (positive error → positive u → α grows) — pushing the model away from base in early training before any drift has occurred. This is semantically wrong for a safety controller. The asymmetric error `e = min(0, setpoint − KL)` reframes this as barrier control: penalise violations, slowly release when compliant.

**Consequence of asymmetry — control range (v4):**
With `e(t) ≤ 0` always, and the release branch only decaying `I` toward zero without changing its sign, `u(t) = Kp·e(t) + I(t) ≤ 0` always. Therefore `α_new = clip(α_base + u, ·, ·)` can never exceed `α_base`. The controller's reachable range is **α ∈ [4, 16]**; we set `α_max = α_base = 16` and remove the upper saturation logic entirely. A barrier controller, by construction, only brakes.

**Why PI and not PID:**
The derivative term applied to a noisy signal (50 prompts, high KL variance) amplifies measurement noise. Alternative for v2 of the controller: derivative-on-measurement with EMA.

**Why the integral term has real work to do (and is not decoration):**
The disturbance in this system is training itself — gradient descent systematically pushes KL upward. A proportional-only brake can only hold α below base *in proportion to the current violation*, so a true P-only loop settles with KL persistently above the setpoint (steady-state error equal to the offset needed to generate the braking signal). The integral term accumulates the violation and keeps tightening α until KL returns to the setpoint.

**Important scope correction (v6):** this steady-state-offset argument is PI's advantage over a *true* P-only controller — a hypothetical branch not implemented in this design, and **not** the static α=8/α=16 sweep, which is open-loop (reads no measurement, settles nothing) and therefore cannot stand in for a proportional feedback controller. See Section 3's scope note. The threshold heuristic is **not** P-only either — its `α *= 0.9` on every violating update, unconditional and unbounded until the clip, is itself a form of integral action (quantized-step / bang-bang). So the heuristic is also expected to converge toward the setpoint rather than sit at an offset next to it. PI's advantage over the heuristic specifically is not "converges where the heuristic doesn't" but **how** it converges: continuous, magnitude-proportional correction (via Kp·e and a continuously-valued I) vs the heuristic's fixed step sizes and hysteresis band, which should manifest as a limit cycle. See the corrected secondary prediction in Section 3 and the P-dominance protocol in Section 10.

**Equations (all quantities updated every 25 training steps; 40 controller updates per 1000-step run):**

```
# Measurement filtering (light EMA — at Kp=30, unfiltered 50-prompt noise
# enters the loop at every update; one update of lag is an acceptable cost
# at 40 updates per run, which it was not at 20)
KL_filt(t) = 0.5 · KL_raw(t) + 0.5 · KL_filt(t−1)

# Barrier error — penalise violations only
e(t) = min(0, KL_setpoint − KL_filt(t))

if e(t) < 0:                               # KL exceeds setpoint — brake
    if not (α == α_min and e(t) < 0):     # anti-windup: freeze I when
        I(t) = I(t−1) + Ki · e(t)         # braking is already saturated
    else:
        I(t) = I(t−1)
else:                                      # KL under setpoint — release
    I(t) = I(t−1) · 0.9                   # half-life ≈ 7 updates —
                                           # commensurate with accumulation
                                           # rate (0.99 would decay only
                                           # ~33% over a full run: a de
                                           # facto ratchet)

u(t)   = Kp · e(t) + I(t)                 # u ≤ 0 always (barrier)
α_new  = clip(α_base + u(t), 4, 16)       # α_max = α_base; upper half
                                           # of old [4,32] was unreachable
```

**Initial parameters:**
- Kp = 30 (with asymmetric e: |e| ∈ [0, 0.5] → Kp = 1.0 is effectively dead; Kp ~ 20–50 is the realistic range)
- Ki = 3 ≈ Kp/10 — scaled so the integral can match the proportional term within ~8–10 updates of sustained violation. Sanity check: at |e| = 0.3, ten updates accumulate |I| ≈ 3·0.3·10 = 9 α-units — comparable to |Kp·e| = 9. (The v3 value Ki = 0.1 would have accumulated ≤ 1 α-unit over an entire run against a proportional term of ~15: a P controller with cosmetic bias.)
- Release decay = 0.9 per update
- EMA β = 0.5 on the KL measurement
- α_base = 16, α_min = 4, α_max = 16
- **Gain tuning: offline, from the baseline run — not online.** "Tune after the first 200 steps" would mean tuning from 4 noisy data points, once each 25-step cadence is accounted for at most 8. Instead: the full baseline KL trajectory (40 points) is used for an offline gain check — estimating the open-loop KL growth rate as a function of α from the two available static points (α=16 baseline, α=8 sweep). To be clear about the rigor involved: this is a linear interpolation between two operating points, not system identification in the formal sense — but it is still the right order of operations (characterise the open-loop response first, then tune the controller), and it is strictly more information than the 4–8 noisy points available from "tune after 200 steps".

**Dual role of α (important for interpretation):**
`α` simultaneously affects:
- (a) adapter output scale: `Δw = α/r · BA` — immediate effect on the forward pass
- (b) gradient scale: `∂L/∂A ∝ α/r` — analogous to changing the adapter's effective learning rate

These are two distinct physical mechanisms coupled into a single scalar. When interpreting results: "the controller slows distributional drift" ≠ "the controller reduces the effective learning rate" — which mechanism dominates should be verified via a dedicated LR-modulation run (stretch goal, Section 14).

**‖B·A‖ tug-of-war (v5, live counterpart of the mechanism split above):** Mechanism (b) means braking α also weakens the gradient signal reaching `A`, but training keeps pushing regardless — so the adapter's own weight norm `‖B·A‖` can grow to partially compensate for the shrinking `α/r` scale in mechanism (a), even while α is visibly suppressed. This is the plausible failure mode the end-of-run α=16 re-evaluation sanity check (Section 10) is designed to catch, but that check is a single static snapshot at the end of training — it does not show *when* or *how* the compensation happens. Logging `‖B·A‖` every 25 steps (Section 7, Section 9) makes the dynamic visible: if the tug-of-war is real, α_history and ‖BA‖_history should be visibly anti-correlated with the product `(α/r)·‖BA‖` roughly constant — the controller and the optimizer pulling against each other while overall output scale stays flat. Plotted as Figure 4b. A flat ‖BA‖ trajectory alongside falling α would instead indicate the brake is working without being fought at the weight level.

**Aggregation (v7, previously undecided):** the 25-step log is a single global scalar `√(Σ_l ‖B_l·A_l‖²_F)` across LoRA layers — the natural counterpart to global `α`, since `(α/r)·‖BA‖_global` is what Figure 4b's product argument needs. But the tug-of-war has no reason to be uniform across layers, and a global scalar can read "flat" purely because compensation in some layers cancels suppression read out of others in the sum — so a per-layer breakdown is logged every 200 steps (same cadence as the held-out eval) as a diagnostic supplement, not for Figure 4b itself. Both are computed from the same per-layer trace-identity pass over LoRA modules (Section 8/9), so the per-layer log costs nothing beyond what the global scalar already computes.

**Adam interaction (important):** Adam normalises gradients by the second moment — asymptotically, gradient rescaling via α has a much weaker effect than under SGD. Mechanism (a) (output scale) therefore likely dominates by construction under Adam. Additionally, sudden α changes render Adam's stale second moments temporarily inaccurate → transient instability after each controller correction. Mitigation: log grad norm in a ±10-step window around every α change to detect artefacts. Note that at a 25-step cadence these windows cover most of training — grad norm is logged every step anyway (Section 9); the window analysis is a post-hoc slicing of that log, not extra compute. **(v5)** In that same slicing, also report whether transient magnitude (peak grad-norm deviation within the window) correlates with the size of the α jump `|α_new − α_prev|` — cheap to compute from logs already collected, and it is the concrete evidence for defending the 25-step cadence against "why not correct more often" if reviewers ask.

**Defense against "why not just control LR via PID?":**
Changing `α` takes effect immediately — it modifies the adapter output scale within the same forward pass in which KL is measured on the control set. Changing LR only affects future steps (the gradient update occurs after the measurement). `α`-control therefore yields a tighter feedback loop: measure → correct → effect within the same step. LR-only modulation has an additional one-step delay in the loop.

**Sanity check — degenerate solution (LOW_ALPHA, disambiguated):**
HIGH_ALPHA is removed (unreachable — see control range above). Its former diagnostic function — "setpoint too aggressive" — migrates to the lower bound: an over-tight setpoint now manifests as the controller braking permanently. LOW_ALPHA therefore has **two distinct causes**, and the flag must record which one applies:

- `α_history < α_min + 1` for > 100 consecutive steps **AND KL_filt still above setpoint** → the setpoint is unreachable in this regime (the controller is working correctly against an impossible constraint) → flag `LOW_ALPHA_SETPOINT_UNREACHABLE`
- `α_history < α_min + 1` for > 100 consecutive steps **AND KL_filt below setpoint** → the integral is stuck / release too slow (controller pathology, not constraint pathology) → flag `LOW_ALPHA_STUCK_INTEGRAL`

The distinction costs one comparison in the logging callback and saves a debugging session when the flag fires.

---

## 9. Metrics

| Metric | Measures | Frequency | Dataset |
|---|---|---|---|
| Training loss | Task adaptation | Every step | hh-rlhf |
| KL divergence (raw + EMA-filtered) | Distributional drift (controller process variable) | Every 25 steps | Control set |
| Gradient norm | Training stability (incl. ±10-step windows around α changes) | Every step | — |
| α_history | Controller behavior | Every 25 steps | — |
| I_history (integral term) | P-dominance diagnostic (Section 10) | Every 25 steps | — |
| ‖B·A‖ — global scalar (√Σ per-layer ‖B_l·A_l‖²_F) | Suppression-vs-compensation tug-of-war, aggregate (Section 8, Figure 4b) | Every 25 steps | — |
| ‖B·A‖ — per-layer breakdown | Same diagnostic, layer resolution (v7): the tug-of-war need not be uniform across layers, and a global scalar can read "flat" if compensation in some layers cancels suppression in others in the aggregate | Every 200 steps | — |
| Perplexity | Catastrophic forgetting | Every 200 steps | **Held-out set** |
| Preference margin | Constitutional alignment retention | Every 200 steps | **Held-out set (hh-rlhf pairs)**; log-probs averaged per response token (length-normalized), not summed |

---

## 10. Expected Results

**Figure 1:** KL divergence over training steps — baseline vs threshold heuristic vs PI controller (control set, 40 points per curve). **Key prediction (revised, v5):** both closed-loop branches trend toward the setpoint (the heuristic has integral action too — see Section 3). The separation to look for is **convergence character**: the heuristic shows a bounded limit cycle around the setpoint (amplitude ≈ set by the 0.9/1.05 step sizes and the 0.8·setpoint hysteresis band); the PI branch shows smoother, decaying oscillation into a narrower band, with lower KL variance around the setpoint in the second half of training. This separation — or its absence — is the headline result.
**Figure 2:** Perplexity over training steps — baseline vs heuristic vs PI controller (held-out set; independent evidence)
**Figure 3:** Training loss — all branches
**Figure 4:** α_history and I_history for the PI controller (40 points) — does α oscillate within a sensible range or collapse to the boundary, and does the integral term visibly contribute?
**Figure 4b (v5):** α_history vs ‖B·A‖_history (PI branch, 40 points, dual axis) — the live suppression-vs-compensation tug-of-war (Section 8): tests whether falling α is met with rising ‖BA‖ (controller and optimizer pulling against each other, `(α/r)·‖BA‖` roughly flat) or whether ‖BA‖ stays flat alongside falling α (brake working unopposed). Same plot repeated for the heuristic branch as a comparison.
**Figure 5:** Pareto front: mean loss vs mean KL — 4 points: {α=8, α=16, heuristic, PI controller}. α=16 from baseline run; sweep contributes α=8 only; α=32 replaced by threshold heuristic. **Values averaged over the last 200 steps (v7, was 50)** — at the 25-step KL cadence, 50 steps is only 2 KL measurements on 50 noisy prompts each, too thin an anchor for a Pareto point the headline claim leans on; 200 steps gives 8 KL points, and under the Section 15 plateau hypothesis (stabilization by ~step 500) the last 200 steps sit inside the stationary regime, so widening the window doesn't pull in early-training transient bias. Loss is logged every step, so 50 vs 200 doesn't matter for that axis — but the **same window is used for both axes**, so the Pareto point stays internally consistent (loss and KL averaged over identical steps). Three of four points in the low-drift region — the non-trivial comparison.
**Figure 6:** Preference margin over training steps — baseline vs PI controller. Margin = mean(logP(chosen) − logP(rejected)) on held-out hh-rlhf pairs, where each logP is the response's log-probability averaged per token (length-normalized) rather than summed — a raw sum would penalize longer responses independent of their actual quality. A margin that stays positive and stable indicates the model retains constitutional alignment (helpful/harmless preferences) despite fine-tuning. A collapsing margin would indicate "constitutional forgetting" — distinct from the capability forgetting measured by perplexity.

**Evaluation α protocol:**
Applies to **both closed-loop branches** (PI controller and threshold heuristic — both have dynamic α and therefore the same potential confound). Final metrics (loss, perplexity, preference margin) evaluated at each branch's final α value. Sanity check: re-evaluate at α=16 with the same adapter weights (B, A unchanged) — the difference isolates how much "control" is pure adapter suppression vs genuine weight-level forgetting prevention. This is a single static snapshot at the end of training; Figure 4b (above) makes the same suppression-vs-compensation question observable across the whole run, not just at the endpoint.

**Disturbance-active comparison window (v5):** Section 15 hypothesizes the baseline KL trajectory naturally plateaus by ~step 500. If it does, the disturbance driving all three branches (gradient descent pushing KL upward) weakens in the second half of every run regardless of control law — and Figure 1's branches could converge for a reason unrelated to the control law, silently erasing the separation the experiment is designed to detect. To guard against this, the control-law comparison (Figure 1's visual separation, and the KL AUC below) is computed **both globally and restricted to the window where `KL_baseline(t) > 1.2 · KL_setpoint`**, i.e. where the baseline is actively violating the constraint. If the two AUC numbers diverge substantially, the globally-reported comparison is flagged as partly confounded by the natural plateau, and the restricted-window number is treated as the more meaningful one.

**P-dominance reporting protocol (check, then report — not assume):**
After gain tuning, compute the ratio |I(t)| / |Kp·e(t)| over all controller updates with e(t) < 0. **If |I(t)| < 0.2·|Kp·e(t)| for the majority of the run, the system is reported as P-dominant** — stated explicitly in the README. **The comparison then inverts (v7):** a P-dominant PI branch is functionally a near-pure proportional brake (braking scales with the current violation, negligible accumulation) — the same true-P-only regime whose steady-state offset was the original motivation for adding the integral term (Section 8). The heuristic, by contrast, is a quantized-step integrator regardless of what the PI branch's I-term is doing (Section 3, Section 7) — its accumulation doesn't turn off. So P-dominance does not mean "the two control laws differ only in proportionality of braking"; it means they differ *in kind*, in the direction that reverses the headline prediction: **the heuristic is expected to converge to the setpoint while the P-dominant PI branch retains an offset next to it.** This is a legitimate physical outcome — the cheaper heuristic would win Figure 1 on convergence, even if it still loses on smoothness/variance — and it is a different reading from the internal consistency check below: that check flags *P-dominance co-occurring with PI-branch convergence* as a bug; this paragraph describes *P-dominance co-occurring with a PI-branch offset* as consistent physics. Section 10 in full therefore covers three of the four convergence×dominance combinations: convergence + I-dominance (headline result), offset + P-dominance (inverted separation, heuristic wins on convergence — still a legitimate, reportable outcome), and convergence + P-dominance (contradiction — see below, debug before interpreting). The fourth combination, **offset + I-dominance**, is not a fourth open question — it is the saturated-actuator regime already named `LOW_ALPHA_SETPOINT_UNREACHABLE` (Section 8): the integral term has correctly driven the brake as hard as it can, but α has hit `α_min` and the setpoint sits outside the reachable range `[4, 16]`, so KL stays above setpoint despite (indeed, because of) sustained I-dominance. Anti-windup freezes `I` once α is pinned at the floor, so this regime has its own signature beyond the offset+I-dominance reading: `|I(t)|` stops changing over the freeze window, unlike the still-accumulating I of a genuinely converging run. All four combinations now have a named, physically distinct interpretation. This is a legitimate answer to the research question in Section 2 ("is KL sufficient as a process variable in a closed loop"), not a failure mode to be hidden. The difference between preemptively declaring P-dominance and *measuring* it is exactly the difference between abandoning the research question and answering it.

**Internal consistency check (v6) — P-dominance and Figure 1 convergence are not independent evidence:** near true convergence to the setpoint, `e(t) → 0`, which forces `Kp·e(t) → 0` — so whatever braking force `u(t)` still holds α down against the disturbance must be carried almost entirely by `I(t)`. Mechanically, therefore: **sustained convergence to the setpoint implies I-dominance in that same stretch of the run, and sustained P-dominance implies a non-negligible average `e(t)`, i.e. an offset.** These are two readings of one underlying regime, not two separately-falsifiable claims that happen to agree. Concretely: if the P-dominance criterion fires (system reported P-dominant) over the same stretch where Figure 1 shows the PI branch sitting on the setpoint, the two readings contradict each other — and the correct conclusion is **not** "an interesting joint result" but **a pipeline bug**, most likely in `I_history` logging or in the anti-windup freeze condition (Section 8) silently zeroing/miscounting `I` while α is nonetheless held down by some other path. Debug the logging/anti-windup before interpreting the physics.
This check is only valid quasi-statically — it assumes the disturbance (and hence the controller's operating regime) is roughly stationary over the window being checked. It is **computed within the disturbance-active window** (`KL_baseline(t) > 1.2 · KL_setpoint`, defined above) rather than globally: over the *whole* run, the baseline's hypothesized natural plateau (Section 15) could shift the PI branch from a P-dominant transient into an I-dominant, near-setpoint regime partway through without any contradiction — a global average would blur that legitimate regime change together with the bug it's meant to catch.

**Claim to be demonstrated:**
Area under the KL curve (control set) is smaller for the PI controller at comparable final loss, both globally and within the disturbance-active window, and — per the corrected secondary prediction — the PI branch shows a smoother, lower-variance approach to the setpoint while the heuristic shows a bounded limit cycle around it (n=1 per branch — results suggest, not prove). Perplexity on the held-out set indicates the result is not a controller artifact. Preference margin on held-out hh-rlhf pairs confirms the model retains helpful/harmless preferences — the metric that justifies the "constitutional alignment" framing. For a stronger claim: add a second seed for baseline and PI controller at the cost of one sweep point.

---

## 11. Known Limitations

| # | Limitation | How addressed |
|---|---|---|
| 1 | Data tautology | Two disjoint datasets (Section 6) |
| 2 | Degenerate solution (α → min) | Disambiguated LOW_ALPHA flags (Section 8) |
| 3 | Dual role of α | Named explicitly; LR-only comparison run is a **stretch goal** — if not executed, this remains an acknowledged limitation (Section 14) |
| 4 | KL measurement noise (50 prompts) | EMA (β=0.5) on measurement; PI instead of PID; noise-vs-lag trade-off recalibrated for the 25-step cadence |
| 5 | Integral windup | Conditional integration + release decay 0.9 (Section 8) |
| 6 | Integral term may be inert despite correct scaling | Measured, not assumed: P-dominance protocol with quantitative criterion (Section 10) |
| 7 | Top-k truncation of reference log-probs biases absolute KL downward — **and** confines the forward-KL zero-avoiding motivation to the preserved support only (v5) | Same measurement across all branches → comparisons unaffected; setpoint derived from the same truncated measurement (Section 5). The zero-avoiding property (Section 5, "Why Forward KL") only operates *within* the top-k support that is actually measured — mass the model shifts into the truncated tail of the base distribution is invisible to the controller and to this KL measurement either way, so this limitation applies to the directional choice of KL, not only to its absolute scale |
| 8 | n=1 per branch | Acknowledged limitation — toy experiment |
| 9 | Does not scale to frontier models | Stated explicitly in README: proof-of-concept |

---

## 12. Work Plan

**Week 1 — Non-PI branches (baseline, sweep, threshold heuristic):**
- Working training loop on hh-rlhf chosen
- KL evaluation every 25 steps (control set; raw + EMA)
- Perplexity + preference margin every 200 steps (held-out set; preference margin = mean(logP(chosen) − logP(rejected)) on hh-rlhf pairs)
- Baseline α=16, 1000 steps — setpoint derived from step-500 checkpoint metrics
- Sweep α ∈ {8} (1000 steps, end-of-run metrics only; α=16 sourced from baseline run)
- Threshold heuristic branch (1000 steps, same measurement pipeline as PI branch)
- **Offline gain check** (open-loop KL growth rate vs α, from baseline + α=8 trajectories) → Kp/Ki verification before Week 2

**Week 2 — Barrier PI Controller:**
- Barrier PI callback implementation (asymmetric error, EMA, anti-windup, release decay)
- Runtime dynamic α via `base_model.scaling`, range [4, 16]
- I_history logging + LOW_ALPHA disambiguation flags
- Comparison against baseline and heuristic

**Week 3 — Analysis and Documentation:**
- Pareto front plots; P-dominance protocol evaluation
- README with explicit reference to AdaLoRA/DyLoRA
- Connection to research note (control-theoretic framing)
- *(Stretch goal)* LR-only modulation run: replace α regulation with LR modulation using the same PI signal. If results are equivalent → mechanism (b) dominates. If different → mechanism (a) has a distinct contribution.

---

## 13. Connection to the Broader Framework

This project and the companion inference-time project (llm-control-alignment) together constitute a single argument:

> *Alignment can be formalized as a control problem at two levels: at inference time (runtime suppression via H∞ controller in SAE feature space) and at training time (closed-loop regulation of distributional drift via barrier PI controller on KL divergence). Both levels draw on the same theoretical foundation — feedback control and stability bounds — but operate in different spaces with different control signals.*

---

## 14. Open Items

### LR-only ablation
Section 8 identifies a comparison run where only the learning rate is modulated (rather than α). This run is not included in the base three-week budget. Decision: **stretch goal for Week 3** — to be executed if time permits after the Pareto front analysis. If not completed, this remains an explicitly acknowledged limitation: the dual nature of α is named but not empirically decomposed in this iteration.

### Compute budget (risk)
Full plan = baseline (α=16) + sweep α=8 + threshold heuristic + barrier PI controller = **4 full runs ≈ 5 T4 sessions** (the heuristic branch is a full 1000-step run with 25-step logging, not a "cheap" sweep — the old "~4 sessions" figure was calibrated to the pre-v4 plan). Session limits and disconnects on Colab are a real risk.

**Mitigation:** sweep restricted to α=8 (end-of-run metrics only); α=16 sourced from the baseline run; checkpointing every 250 steps on all branches so a disconnected session resumes rather than restarts.

Estimated total compute: **~5 T4 sessions of 2–3 hours each.**

---

## 15. Step Budget — All Branches

A consistent step count across all branches is required for a valid Pareto comparison.

| Branch | Steps | Metrics |
|---|---|---|
| Baseline α=16 | 1000 | Loss + KL every 25 steps + perplexity + preference margin every 200 steps; setpoint derived post-hoc from step-500 metrics |
| Sweep α=8 | 1000 | Loss + KL + preference margin at end of run only |
| Threshold heuristic | 1000 | Loss + KL every 25 steps + α_history + perplexity + preference margin every 200 steps |
| Barrier PI controller (Branch B) | 1000 | Loss + KL every 25 steps + α_history + I_history + perplexity + preference margin every 200 steps |

**Note:** Sweep = {8} only; the α=16 point on the Pareto front (Figure 5) is sourced from the full baseline run — it is not re-run as a separate sweep entry; α=32 is replaced by the threshold heuristic branch. Total: 4 runs ≈ 5 sessions.

**Why sweep runs also use 1000 steps (not 300):**
Truncating sweep runs to 300 steps and comparing them against baseline/PI controller at step 1000 would compare models at different points on their training trajectories — invalidating the Pareto front. Instead, we reduce logged metrics (end-of-run only), not compute.

**Assumption to verify from the baseline run:**
Loss and KL stabilize by approximately step 500 (hypothesis). If the baseline run shows otherwise (curves still descending at step 500), all branches will be extended to 2000 steps and the compute budget updated accordingly.

**Total estimated compute: ~5 T4 sessions of 2–3 hours each.**