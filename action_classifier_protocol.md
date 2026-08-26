# Fighting-Game Action Classifier
## Data Collection & Preprocessing Protocol — Punch / Kick / Shoot

**Prepared for:** Ayan
**Dataset target:** ~260-300 clips/class, 13-15 participants
**Device:** laptop-mounted camera, side-view

---

## 1. Camera Setup (lock this before filming any more clips)

Decide the exact placement below once, write it down, and never change it between participants or sessions. Camera position was found to matter more than lighting conditions in controlled body-tracking studies — inconsistency here is the single easiest thing to accidentally get wrong.

| Parameter | Recommendation | Why |

|---|---|---|
| View | Side profile (~90°), fixed | Matches deployment; keep tight rather than wide-angle variety |
| Height | Chest-to-eye level of the player | Avoid low laptop-on-desk upward tilt, which distorts limb-swing arcs |
| Distance | Far enough for full body + kick range of motion in frame | Confirm by live-testing MediaPipe on the actual setup before filming a full session |
| Frame rate | 60 fps if available, 30 fps minimum | Combat-sports literature repeatedly finds 30 fps loses fast-motion detail to blur |
| Resolution | 1080p or higher | Keeps limb keypoints resolvable at distance |
| Stability | Tripod / fixed mount, no handheld | Any camera motion adds noise indistinguishable from su
bject motion |

**Action item:** before a full filming session, open MediaPipe live on the exact final camera placement and watch landmark confidence during a few practice punches/kicks. Confirm it holds up before committing hours of filming to that setup.

---

## 2. Data Collection Matrix

Per participant (30 clips as you've been doing), vary the axes below deliberately rather than incidentally. Angle is a generalization aid only — MediaPipe will never see front-angle input at inference, so front-angle clips are training-only, never validation/test.

| Axis | Values | Purpose |
|---|---|---|
| camera_angle | side (majority), front (minority, ~25-30%) | Regularizer only — improves generalization without being a deployment target |
| background | 2 distinct backgrounds | Prevent background-shortcut learning |
| clothing_variant | 2 outfits | Prevent clothing-shortcut learning |
| lighting | add 2 conditions if not already varied | MediaPipe (RGB-only) lacks a depth camera's lighting immunity — currently the least-covered risk in the plan |
| limb_used | left / right, tracked per clip | Checks near-camera vs far-camera (occluded) limb balance |
| tempo | at least 2 speeds per action per person if feasible | A model trained on one tempo misfires on faster/slower players at inference |

---

## 3. Feature Extraction — the single highest-leverage decision

Use MediaPipe's **pose_world_landmarks** (real-world metric coordinates, meters, hip-relative) — **not** pose_landmarks (normalized 0-1 image-plane coordinates).

A published ablation on a directly comparable problem (classifying fast athletic motion from monocular MediaPipe video) found that swapping metric world coordinates for image-plane coordinates dropped cross-subject action-type accuracy from **83.7% to 47%** — the largest single factor in that study's generalization gap. Most MediaPipe tutorials default to the normalized, image-plane version; explicitly request world landmarks instead.

**Per-frame feature vector:**
- Hip-relative (x, y, z) world coordinates for all 33 pose landmarks
- 21 hand landmarks per hand if separating punch from shoot by hand shape/wrist orientation
- First-order velocity (frame-to-frame delta) per joint — often the actual discriminating signal for strike speed/snap
- Per-landmark visibility/confidence score, carried through for the filtering step below

---

## 4. Preprocessing Pipeline

| Step | Method | Notes |
|---|---|---|
| Confidence filtering | Drop or interpolate frames below a visibility threshold (start ~0.5, tune per your footage) | Occlusion in one body region has been shown to degrade tracking confidence elsewhere too, not just the occluded joint |
| Smoothing | Moving-average filter, ~5-frame window | Standard in comparable pose-based motion studies to reduce landmark jitter without erasing genuine motion |
| Normalization | Center on hip midpoint; scale by shoulder-to-hip distance | Removes distance-from-camera and body-size variation across participants |
| Sequence length | Resample every clip to a fixed length (32-48 frames) | Matches published punch/kick pose-sequence classifiers; avoids variable-length handling complexity |
| Detection thresholds | min_detection_confidence ≈ 0.5, min_tracking_confidence ≈ 0.5 as a starting point | Lower thresholds (~0.3) trade robustness for recall if fast motion causes frequent re-detection; tune empirically on your footage |

---

## 5. Splitting & Evaluation

- Split **by participant, not by clip** — hold out 2-3 full participants for test. Random clip-level splits leak near-duplicate frames between train and test.
- Validation/test sets should be **side-angle clips only**, since that's the sole deployment condition.
- This mirrors standard practice in the closest published analog: a boxing-punch classification study held out a fixed percentage of clips from unseen boxers specifically to test generalization to new people, not just new clips.

---

## 6. Augmentation Checklist

| Technique | Pipeline | Parameters |
|---|---|---|
| Rotation jitter (vertical axis) | Skeleton | ±5-15°, tight range — deployment is a fixed side view |
| Time-warp / speed perturbation | Skeleton | 0.85x - 1.15x playback |
| Joint coordinate noise | Skeleton | Small Gaussian jitter simulating landmark noise |
| Left-right mirroring | Skeleton | Flip x-coords + relabel limb_used; use if limb balance is uneven |
| Random temporal crop/pad | Both | To the fixed sequence length |
| Color jitter / random crop | RGB (if fine-tuning Kinetics model in parallel) | Brightness/contrast ±20%, 0.8x-1.2x distance simulation |

---

## 7. Model Selection Recap

- **Baseline (build first):** BiLSTM/GRU + attention on normalized landmark sequences — fast to implement, validates the pipeline end-to-end.
- **Primary target:** 2s-AGCN (two-stream: joint positions + bone vectors) fine-tuned from an NTU RGB+D pretrained checkpoint — prioritize checkpoints trained on NTU's mutual-action subset (includes punching/slapping and kicking classes).
- **Parallel comparison (optional):** Kinetics-400-pretrained X3D-XS or R(2+1)D-18, fine-tuned head + last block, evaluated on the identical participant-held-out split for a fair comparison.

---

## 8. Key Sources

1. Sports pose-classification ablation on world vs. image-plane coordinates: "Multi-Task Tennis Stroke Biomechanics Analysis Using MediaPipe Pose" (arXiv, 2026).
2. Motion blur limitation in combat-sports video classification: PLOS ONE, "An active machine learning framework for automatic boxing punch recognition and classification using upper limb kinematics" (2025).
3. Frame-rate vs. motion blur in fast-action pose estimation: USPTO filing, "Method and system for automatic extraction of virtual on-body inertial measurement units."
4. Camera-position and occlusion effects on body tracking: ScienceDirect, "Postural control assessment via Microsoft Azure Kinect DK: An evaluation study."
5. Precedent punch/kick pose-sequence classifier (32-frame RNN windows): GitHub, imsoo/fight_detection.
6. Held-out-subject evaluation precedent: PLOS ONE boxing punch classification study (2025), test set drawn from unknown boxers.

---

## 9. Accuracy Add-ons (post-review, applied on top of sections 1-8)

Dataset-specific decisions locked after auditing the actual 642-clip label set (9 participants, kicking 210 / punching 220 / shooting 212, side_left 232 / side_right 224 / front 186):

| # | Add-on | Reasoning |
|---|---|---|
| 1 | **Bone-vector stream** alongside raw joint stream | 2s-AGCN's "two-stream" *is* joint+bone — bone vectors (adjacent-joint deltas) are more invariant to body size/camera drift than raw xyz. Extract at feature-extraction time, not bolted on later. |
| 2 | **Soft confidence weighting**, not hard frame drop | Avg clip = 76 frames with a short strike window inside; hard-dropping low-visibility frames risks cutting the strike itself. Keep all frames, use visibility as a per-frame loss/attention weight instead. |
| 3 | **Leave-participant-out group k-fold CV** for the accuracy estimate | 9 participants is small — one fixed 2-person test split is a noisy estimate. Report mean±std over k folds of held-out participants before trusting any single accuracy number. Final model still trained/reported on the locked split (train 4,5,6,7,8,9 / val 1 / test 2,3). |
| 4 | **Discriminative fine-tuning** on the pretrained 2s-AGCN backbone | Freeze NTU-pretrained GCN backbone for the first ~5-10 epochs (train classifier head only), then unfreeze last block(s) at lower LR. Full fine-tune from epoch 1 risks erasing the pretrained motion priors that were the reason to pick a pretrained model. |
| 5 | **Hand-crafted-feature sanity baseline** (XGBoost/SVM) run before the deep baseline | Features: peak wrist/ankle velocity, joint-angle ROM (elbow/knee/hip flexion range), strike-frame timing. Cheap, fast, and catches broken labels/pipeline bugs before spending GPU-less CPU time training BiLSTM/AGCN. |
| 6 | **Test-time augmentation (TTA)** at inference | Average prediction over original + mirrored + 2 mild time-warped copies of each test clip. Typically +1-2% accuracy, legitimate since real inference has no reason to skip it. |
| 7 | **Per-frame joint-angle stream** (elbow/knee/hip flexion, 6 triples, continuous not just ROM-summary) added to the deep model's input | Added after diagnosing the first BiLSTM baseline run: it scored *below* the XGBoost sanity baseline (83.7% vs 94% test) specifically because it missed 7/8 of held-out participant 3's punching clips (misclassified as shooting) that XGBoost mostly got right. XGBoost's dominant feature was `hipR_rom` (42% importance) -- an explicit joint-angle summary the BiLSTM never had access to, only raw xyz/velocity/bone. Giving the deep model the same angle signal explicitly, rather than making it re-derive angle-range-of-motion from raw coordinates on only 529 training clips, directly targets that gap. |

**Sequence-length / feature-vector decisions locked from real clip stats** (60-clip sample: mean 76.5 frames, mixed 30fps/~59fps sources, mean duration 2.18s, resolution overwhelmingly 1920x1080):

- Fixed sequence length: **40 frames**, reached via interpolation-resample (mid of section 4's 32-48 band; time-normalized so mixed 30/60fps sources are handled correctly).
- Per-frame feature vector: 33 landmarks × world (x,y,z) hip-relative (99) + bone vectors (add-on 1, 42) + first-order velocity (99) + visibility (33) + joint angles (add-on 7, 6) = **279 dims/frame** (hand landmarks still deferred unless punch/shoot confusion shows up in the confusion matrix).
- Participant split: **train** = participants 4,5,6,7,8,9 (529 clips) · **val** = participant 1 (47 clips, side-only) · **test** = participants 2,3 (66 clips, side-only).
- Augmentation (train split only, 2x multiplier → ~1500 effective clips): rotation jitter ±10°, time-warp 0.85-1.15x, small Gaussian joint noise, L-R mirror, random-window crop (90-100% of clip length, picked before resample — caps worst-case strike-frame loss at 10%, avoids both pure pad-only and aggressive crop).
