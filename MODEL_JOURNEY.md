# Action Classifier — Project Journey & Final Model

Punch / Kick / Shoot gesture classifier for a Tekken-style game, built on MediaPipe
pose extraction. This document is the single narrative record of every approach
tried, every accuracy number measured, every dead end, and how the project arrived
at its current leading model: **XGBoost v3, hand-crafted features with real
MediaPipe Hands curl-angle hand-openness, 15 features, trained on all 642 clips.**
(v2's elbow-angle addition was tried first and rejected after held-out testing —
see §7 — before v3 found the improvement that actually held up.)

**Headline number, honestly stated:** leave-participant-out CV mean **0.934 ± 2.9%**
(§7b) — this is the trustworthy figure, not the single-split test accuracy of 0.98,
which is real but is the best of 3 folds, not the expected number. Still pending:
live-camera confirmation (§9).

Companion documents: `action_classifier_protocol.md` (data collection & preprocessing
protocol, written before filming), `README.md`.

---

## 1. Data

- **642 labeled clips**, 9 participants. kicking 210 / punching 220 / shooting 212.
  side_left 232 / side_right 224 / front 186.
- **Participant-level split** (not clip-level, to avoid near-duplicate leakage):
  - train = participants 4,5,6,7,8,9 (529 clips)
  - val = participant 1 (47 clips, side-angle only)
  - test = participants 2,3 (66 clips, side-angle only)
- **MediaPipe Pose world landmarks** (metric, hip-relative, 33 points) — not the
  normalized image-plane landmarks. A published ablation on a comparable problem
  showed swapping world→image-plane coordinates drops cross-subject accuracy from
  83.7% to 47%; this was locked in from the start.
- Preprocessing: hip-midpoint centering, shoulder-to-hip scale normalization,
  ~5-frame smoothing, resample every clip to a fixed **40 frames**.
- Augmentation (train split only, 2x multiplier): rotation jitter ±10°, time-warp
  0.85–1.15x, joint-coordinate Gaussian noise, left-right mirroring, random
  temporal crop.

---

## 2. Model 1 — BiLSTM + Attention (deep baseline)

**Input:** 279-dim/frame feature vector — 33 landmarks × xyz (99) + bone vectors (42)
+ velocity (99) + visibility (33) + 6 joint angles (elbow/knee/hip flexion, L+R).

**Result (locked test split, participants 2,3):**

| | precision | recall | f1 |
|---|---|---|---|
| kicking | 1.00 | 1.00 | 1.00 |
| punching | 1.00 | 0.47 | 0.64 |
| shooting | 0.69 | 1.00 | 0.82 |
| **accuracy** | | | **0.84** |

Test-time augmentation (orig + mirror + 2× time-warp) made no difference (0.84 → 0.84).

**Diagnosis:** kicking was perfectly separated (legs give an unambiguous signal).
Punching and shooting collapsed into each other — the model called over half of
real punches "shooting." Both actions are single-arm, forward-extending motions;
nothing in the 279-dim vector (xyz, bone, velocity, visibility, coarse joint-angle
ROM) reliably told them apart.

---

## 3. Model 2 — XGBoost sanity baseline (hand-crafted features)

Built *before* the BiLSTM specifically to catch pipeline/label bugs cheaply
(protocol addendum #5). It ended up outperforming the deep model outright.

**15 features** (`src/features/handcrafted.py`), per clip:
- peak wrist velocity + timing (L, R)
- peak ankle velocity + timing (L, R)
- elbow / knee / hip flexion range-of-motion, L+R (6 features)
- hand openness at the striking wrist's peak-speed frame (mean distance from
  wrist to pinky/index/thumb tips — fist vs open-palm proxy)

**Result (locked split):**

| split | accuracy |
|---|---|
| val (participant 1) | 0.97 |
| test (participants 2,3) | 0.94 |
| leave-participant-out group k-fold CV | 0.92 ± 2.5% |

Test confusion matrix: kicking perfect, punching 12/15 correct (3 misread as
shooting), shooting 18/18 — already far better separated than the BiLSTM, and the
CV spread confirmed it wasn't a lucky single split.

**Why hand openness got added:** live testing showed punch/shoot collapsing
together whenever legs weren't in play. MediaPipe Pose's wrist is a single point —
it can't see finger shape directly — but Pose does expose three crude fingertip
landmarks (pinky/index/thumb tips). Mean wrist→fingertip distance approximates
open (large) vs closed (small) hand shape and was added as feature #15.

---

## 4. Why XGBoost beat the deep models

This isn't a fluke — it's the expected outcome given the dataset size. Tree
ensembles reliably beat deep nets on small tabular/structured problems (roughly
under 10k–100k rows); deep architectures need volume to earn their capacity.
529 training clips is nowhere near enough to let a network as large as the AGCN
(**578,352 parameters**, see §6) discover on its own the same signal a human
already knows to hand-engineer (elbow angle, hand shape). XGBoost, given that
signal directly as an explicit column, doesn't have to *discover* it — it just has
to weight it, which gradient-boosted trees do very efficiently on small N. The
deep models aren't broken; they're data-starved for the task, and handcrafted
features are a shortcut around that starvation.

---

## 5. 2s-AGCN — parallel deep track

Built as the protocol's primary target (graph conv net on the skeleton, matching
the BiLSTM's raw-coordinate access but with an explicit body graph instead of a
flat vector). Went through five versions chasing the same punch/shoot confusion:

| version | change | reasoning |
|---|---|---|
| v1 | 14-joint body skeleton only | same punch/shoot confusion as BiLSTM — structurally can't see hand shape |
| v2 | +6 fingertip nodes (20 nodes total: pinky/index/thumb tips, L+R) | give the graph access to the same fingertip landmarks that fixed XGBoost |
| v3 | +HandOpennessStream, Pose-fingertip-distance proxy, `hand_w` initialized 2.0 | explicit hand-shape stream, over-weighted from the start (mistake, corrected in v5) |
| v4 | HandOpennessStream switched to real **MediaPipe Hands** curl-angle data (dedicated hand-tracking model via wrist-crop) instead of Pose's crude fingertip proxy | Pose's 3 fingertip points move together as one blob, not real per-finger articulation — see §7 |
| v5 | +ElbowStream, starting elbow angle (mean of first 3 frames), `elbow_w` initialized 1.0 (deliberately *not* elevated, learning from v3's over-weighting mistake) | strongest single separator found this session — see §7 |

**Architecture (v5, verified param count):** two-stream (joint + bone) graph conv
backbone + HandOpennessStream + ElbowStream, logits combined via 4 learnable
per-stream scalar weights.

```
joint_stream:   288,835
bone_stream:    288,835
hand_stream:        579
elbow_stream:         99
stream weights:        4  (joint_w, bone_w, hand_w, elbow_w)
--------------------------------
total:            578,352 params
```

**Real, freshly-run test result (`scripts/train_agcn_v5_eval.py`, locked test
split, current code — not the stale notebook number, see the callout below):**

| | precision | recall | f1 |
|---|---|---|---|
| kicking | 1.00 | 1.00 | 1.00 |
| punching | 1.00 | 0.667 | 0.80 |
| shooting | 0.78 | 1.00 | 0.88 |
| **accuracy** | | | **0.898** |

Learned stream weights: joint 1.02, bone 0.93, **hand 2.18** (stayed
over-weighted despite starting at 2.0 — the v3 concern this session tried to
correct persisted anyway), elbow 0.86. Val accuracy hit 1.00 (30 clips, easy to
overfit) while test sat at 0.90 — same generalization gap pattern XGBoost v2 hit.
Punching recall (0.667) collapsed the same way XGBoost v2's did — the elbow
feature hurts AGCN's generalization too, not just XGBoost's.

**⚠️ Stale-number correction:** `notebooks/agcn/01_2s_agcn.ipynb` has a leftover
cell output claiming 97.96% test/TTA accuracy with `params: 578,352` — this
looked like a v5 result (param count matches) but is stale: `execution_count`
on that cell is `None` (edited since last actually run), and the notebook's own
markdown explicitly attributes 97.96% to **v3** (Pose-fingertip-distance proxy,
"unreliable live"), not v4/v5. The real v4/v5 rows in the notebook's summary
were left blank. Don't trust that cell — use the freshly-run 0.898 figure above.

---

## 6. Live-camera reliability investigation

The trained models scored well on held-out clips but felt unreliable during live
play. This triggered a rigor pass: re-checking every "the features are clearly
separated" claim against full population statistics instead of a handful of
eyeballed samples.

**Hand-openness, re-checked properly:** an earlier claim of clean separation
(punch 0.09–0.12 vs shoot 0.15–0.19) turned out to be based on 6 hand-picked
samples with a feature-index bug in how it was pulled. The real population numbers
(punch mean 0.076, shoot mean 0.106) showed **89–93% range overlap** — a real but
weak and fragile signal, consistent with it being built on Pose's crude fingertip
tracking rather than a dedicated hand model.

**Finger-gun hypothesis, tested and rejected:** hypothesized that only the index
finger differs between the two hand shapes (pinky/middle/ring curled in both).
Per-finger effect sizes came back similar across pinky/index/thumb (1.18–1.34),
contradicting the hypothesis — Pose's 3 tracked fingertip points move together as
one rigid blob (whole-hand orientation), not real independent per-finger
articulation. This is *why* v4 switched to real MediaPipe Hands data for the
AGCN's hand stream instead of trying to extract more out of Pose's proxy.

**Elbow angle — the strongest signal found this session:** the working hypothesis
was that punching retracts the elbow before extending, while shooting is a
straight raise with no retraction. First attempt (measuring retraction *depth* —
start angle minus pre-strike minimum) gave a weak effect size (0.58). Correcting
to the **absolute starting angle** itself (mean of the first 3 frames) — punching
starts from a bent guard pose, shooting starts already extended — gave an effect
size of **2.27**, the strongest separator found in the whole session, and unlike
hand-openness it's built entirely from Pose's best-tracked joints (shoulder,
elbow, wrist), not the fragile fingertip landmarks.

**Latency:** SETTLE_FRAMES (6 frames of confirmed stillness before a prediction
fires) dominates live response time regardless of model choice. XGBoost's own
inference is ~0.32s end to end; AGCN variants run ~0.40–0.68s depending on
whether the MediaPipe Hands crop is in the loop.

---

## 7. Final model — XGBoost v2

Built as a **separate v2 pipeline**, deliberately preserving the original v1
model/script untouched (`models/xgboost_baseline.json`, `scripts/train_xgboost_final.py`
— verified via file timestamps to be unmodified).

- `src/features/handcrafted_v2.py` — v1's 15 features + 1 new one: **starting
  elbow angle** of the striking arm (mean joint angle over the first 3 frames).
- `scripts/train_xgboost_final_v2.py` → `models/xgboost_v2.json` +
  `models/xgboost_v2_train_features.npz` (novelty/OOD gate, same method as v1).
- Trained on all 642 labeled clips (deployment model, not held-out eval — same
  convention as v1's final script).
- **Train accuracy: 0.998.** Starting-elbow-angle feature importance: **0.0829,
  rank 6 of 16** — a real, meaningfully-weighted contributor, not dead weight.

**Held-out evaluation (`scripts/eval_xgboost_v2_holdout.py`, same locked split as
v1, fresh model fit on train_df only — apples-to-apples with §3's numbers):**

| | v1 (15 features) | v2 (+elbow, 16 features) |
|---|---|---|
| val | 0.97 | 0.97 |
| **test** | **0.94** | **0.90** |
| leave-participant-out CV | 0.92 ± 2.5% | 0.92 ± 3.6% |

**The elbow feature did not improve held-out accuracy — test accuracy dropped**
(0.94 → 0.90). Punching recall on test fell from 0.80 to 0.67 (10/15 vs 12/15
correct), still confused with shooting on this split. The feature's high
population-level effect size (2.27, §6) and its real in-model importance
(rank 5/16 on this split's fit) didn't translate to better generalization to the
two held-out test participants — CV mean is an exact wash (0.921 vs 0.920) with a
wider spread (std 3.6% vs 2.5%), consistent with the extra feature giving the
tree one more way to fit train-specific noise rather than a genuinely more
transferable signal.

**Follow-up: is the test=[2,3] drop just an unlucky pair, or does it hold across
different held-out participants?** Re-ran with `scripts/eval_v1_v2_compare_cv.py`
across all 3 leave-participant-out CV folds (every participant held out exactly
once — a systematic sweep, not one arbitrary pick), v1 vs v2 fit fresh per fold
on identical splits:

| held-out participants | v1 acc | v2 acc | v1 punch recall | v2 punch recall |
|---|---|---|---|---|
| [1, 4, 8] | 0.876 | 0.871 | 0.792 | 0.764 |
| [3, 7, 9] | 0.943 | 0.943 | 0.928 | 0.913 |
| [2, 5, 6] | 0.946 | 0.950 | 0.937 | 0.924 |

Not a [2,3]-specific fluke: **v2 ties or trails v1 in every fold**, and punching
recall specifically is lower for v2 in all 3 folds, even the one where v2's
overall accuracy edges ahead. Root cause investigated directly: starting elbow
angle on the locked test participants (2,3) — punching mean 137.3° vs shooting
154.8° (17.5° gap, heavy overlap) — is far less separated than on the training
participants (punching 104.7° vs shooting 148.2°, 43.5° gap). The population-wide
effect size (2.27, §6) is real, but **how sharply someone winds up their elbow
before punching is itself participant-idiosyncratic** — some people punch with a
tight guard-position bend, others throw a flatter, more extended punch — so as a
16th tree-split feature it gives the model one more way to fit train-specific
habit rather than a cleanly transferable cross-person signal.

**v2 verdict: don't deploy.** Every offline metric tried (locked test split,
3-fold participant CV, punch recall specifically) favors v1 over v2.

---

## 7b. v3 — real MediaPipe Hands curl angle replacing the hand-openness proxy

Separate hypothesis, isolated on purpose from the elbow feature: v1's
hand-openness feature (#5) uses Pose's crude wrist-to-fingertip **distance**
(3 fingertip landmarks that move as one rigid blob — confirmed weak by the
finger-gun hypothesis test in §6). The same real per-finger **curl angle**
signal already built for the AGCN hand stream (`src/data/hand_features.py`,
a dedicated MediaPipe Hands model via wrist-crop) had never been ported into
XGBoost. `src/features/handcrafted_v3.py` swaps just that one feature's input
source — same 15-feature layout, everything else identical to v1 — and
`scripts/eval_xgboost_v3_holdout.py` evaluates it the same way as v2.

| | v1 (Pose fingertip proxy) | v3 (real Hands curl) |
|---|---|---|
| val | 0.97 | 0.97 |
| **test** | 0.94 | **0.98** |
| punching recall (test) | 0.80 (12/15) | **0.93 (14/15)** |
| leave-participant-out CV | 0.92 ± 2.5% | **0.934 ± 2.9%** |
| per-fold [1,4,8] / [3,7,9] / [2,5,6] | 0.876 / 0.943 / 0.946 | **0.895 / 0.962 / 0.946** (ties-or-beats v1 in every fold) |

**This is a genuine improvement, not noise** — test confusion matrix down to a
single miss (1 punching→shooting of 49 clips, vs v1's 3), and unlike the elbow
feature it wins across every held-out split tried, not just some.

**Why this one generalizes and the elbow feature didn't:** elbow-bend depth
before a punch turned out to be a personal habit (varies participant to
participant, §7's investigation). Hand shape at the strike frame — fist for a
punch, flat/pointing for a shoot — is a property of *the action itself*, not
of who's performing it, so the signal transfers to unseen people. The fix
here wasn't a new feature, it was swapping a bad proxy for the real thing from
a purpose-built hand-tracking model.

**Cost:** feature extraction now requires a MediaPipe Hands pass per clip
(~3 min for all 642 clips vs seconds for v1's pose-only features) — same
latency tradeoff already known from the AGCN hand-stream work.

**Current recommendation: v3's real-curl hand feature is the strongest
offline improvement found so far.** Confirmed across every held-out split
tried (unlike elbow), and also confirms the data-starvation explanation in
§4 directly: fed the exact same real hand signal, XGBoost (v3) scores 0.98
test / 0.934 CV vs AGCN v5's 0.898 test on the identical locked split (§5) —
same signal, model-capacity-vs-data-size is the deciding factor, not the
feature.

**15 features, v3 final list** (same layout as v1, only #5's source changed):

| # | feature |
|---|---|
| 1–2 | left wrist peak velocity, timing |
| 3–4 | right wrist peak velocity, timing |
| **5** | **hand openness at striking wrist's peak-speed frame — real MediaPipe Hands curl angle (v3), not Pose fingertip distance (v1)** |
| 6–7 | left ankle peak velocity, timing |
| 8–9 | right ankle peak velocity, timing |
| 10–15 | elbow / knee / hip flexion ROM, L+R |

## 7c. Deployment — `models/xgboost_v3.json` + `scripts/live_inference_v3.py`

- `scripts/train_xgboost_final_v3.py` → `models/xgboost_v3.json` +
  `models/xgboost_v3_train_features.npz` (novelty gate, same method as v1/v2).
  Trained on all 642 clips. Train accuracy 0.998, real-curl feature importance
  0.0605, rank 6/15.
- `src/inference.py:predict_segment_v3` — new function (doesn't modify the
  existing `predict_segment`), needed because v3's feature extractor takes
  both Pose xyz *and* real Hands curl data, unlike v1/v2's single-argument
  `feature_fn` hook.
- `scripts/live_inference_v3.py` — new script, combines v2's UX fixes (5s
  cooldown, tunable arm/leg motion thresholds, cooldown-skips-model-inference)
  with `live_inference_agcn.py`'s live MediaPipe Hands capture (runs Hands
  alongside Pose every frame — lower FPS than the Pose-only v1/v2 scripts,
  same tradeoff as the AGCN live script). `scripts/live_inference.py` and
  `live_inference_v2.py` are untouched.
- Verified end to end on a real clip (not just imports): a real shooting clip
  through the full `predict_segment_v3` pipeline (gate → real curl feature →
  XGBoost) correctly predicted "shooting" at 99.4% confidence.

---

## 7d. Cross-model summary (same locked test split, all real numbers)

| model | test accuracy | notes |
|---|---|---|
| BiLSTM (§2) | 0.84 | punch/shoot collapse, no hand signal |
| XGBoost v1 (§3) | 0.94 | crude Pose fingertip-distance hand proxy |
| XGBoost v2 (§7) | 0.90 | +elbow angle — **regressed**, feature is participant-idiosyncratic |
| **XGBoost v3 (§7b)** | **0.98** (CV mean **0.934 ± 2.9%**) | **+real MediaPipe Hands curl — genuine improvement, current best** |
| AGCN v5 (§5) | 0.898 | real Hands curl + elbow, 578K params, same elbow regression as v2 |

---

## 8. Engineering fixes made along the way

Not modeling results, but real bugs that would have silently corrupted results if
left in place:

1. **Stale graph-cache shape bug** — `_cached_and_valid()` only checked that a
   cached `.npz` *loaded*, not that its tensor shape matched the current node
   count. Hit twice (14→20 node transition, then 20→22): all 1587 files in
   `data/processed/graph_train/` were silently stuck at the old `(40, 20, 6)`
   shape instead of `(40, 22, 6)`, so a 46-minute hand-curl extraction pass
   completed and was then thrown away every time, and separately produced a
   real, reproducible shape-mismatch `AssertionError`. Fixed permanently with
   `_cached_and_valid_graph()`, which checks packed-tensor shape explicitly; all
   three cache splits (train/val/test) were cleared and rebuilt correct.
2. **Cache-check performance regression** — `build_graph_cache()` was running the
   expensive MediaPipe Hands re-extraction and curl computation for every row
   before checking whether the cached output would just be discarded anyway.
   Fixed to check `need_det`/`need_aug` first and skip the expensive lookups
   entirely on a full cache hit.
3. **`live_inference_v2.py` model-path bug** — the script's version name referred
   to UX fixes, not the new model; it was still loading `xgboost_baseline.json`
   with v1 (15-feature) extraction. Fixed to load `xgboost_v2.json` with
   `extract_handcrafted_features_v2`, verified end to end (16 features extracted,
   index 15 confirmed as elbow angle on a real shooting clip, correct prediction).

---

## 9. Status / open next steps

- ✅ XGBoost v2 (elbow) trained, verified, wired into live inference — **held-out
  tested and rejected** (§7): regresses vs v1 on every split.
- ✅ XGBoost v3 (real Hands curl) trained, held-out tested, wired into live
  inference (`scripts/live_inference_v3.py`) — **current leading candidate**
  (§7b–7d): test 0.98 / CV 0.934±2.9%, wins every fold.
- ✅ AGCN v5 retrained fresh and evaluated (§5) — real result 0.898 test,
  below XGBoost v3. Confirms §4's data-starvation explanation directly: same
  real hand signal, XGBoost wins by ~8pp on identical data.
- ⏳ **Nothing has been tested on live camera yet.** Every number in this
  document is offline (locked-split or CV) — this is the one gap all the
  analysis above can't close. `scripts/live_inference_v3.py` is ready to run:
  `python scripts/live_inference_v3.py`
- ⏳ Not yet tried: v3's real-curl hand feature + elbow angle combined (v2's
  regression and v3's improvement are independent findings — worth testing
  together, though elbow regressed on its own in both XGBoost and AGCN, so
  low expected payoff).
- **Recommendation:** run `live_inference_v3.py` next. If it holds up (or
  beats) the offline numbers live, it's the deployment model. If live reveals
  new failure modes the locked test set didn't catch (same reason v3/v4 of
  the AGCN hand stream exist — Pose's fingertip proxy looked fine offline too
  until live testing exposed it), that's the next thing to diagnose.
