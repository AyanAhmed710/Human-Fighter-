"""
Live-webcam inference: a motion-triggered segmenter (not a continuously
reclassified rolling window) + the same preprocess -> handcrafted-feature
path used to train the XGBoost sanity baseline (src/features/handcrafted.py).

Why a state machine instead of "reclassify a rolling window every N frames":
every training clip is ONE bounded, hand-trimmed action -- rest, wind-up,
strike, return. A continuously-reclassified arbitrary sliding window doesn't
match that: resample_sequence() always stretches whatever's in the window to
a fixed length, so even a 2-frame natural fidget inside an otherwise-static
window gets stretched to fill the whole normalized sequence and can read as
a completed strike arc. The segmenter below only ever classifies once real,
sustained motion has actually started and settled -- much closer to what the
model was trained on, and idle/fidgeting genuinely never reaches the
classifier at all (not just filtered after the fact by a confidence check).
"""
import time
from collections import deque

import numpy as np

from src.config import (
    ACTIONS,
    ARM_LANDMARKS,
    COOLDOWN_SECONDS,
    LEG_LANDMARKS,
    MAX_SEGMENT_FRAMES,
    MIN_MOTION_FOR_ACTION,
    MIN_SEGMENT_FRAMES,
    MIN_VISIBILITY_FOR_ACTION,
    ONSET_FRAMES,
    PRE_ROLL_FRAMES,
    ROOT,
    SETTLE_FRAMES,
)
from src.data.preprocess import preprocess_clip
from src.features.handcrafted import (
    L_ANKLE,
    L_WRIST,
    R_ANKLE,
    R_WRIST,
    extract_handcrafted_features,
)

_MOTION_JOINTS = [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE]
_ARM_JOINTS = [L_WRIST, R_WRIST]
_LEG_JOINTS = [L_ANKLE, R_ANKLE]


def _instant_speed(xyz_frame: np.ndarray, prev_xyz_frame: np.ndarray) -> float:
    """Frame-to-frame displacement of the fastest of wrist/ankle L+R,
    normalized by this frame's own shoulder-hip distance -- same quantity
    and scale as the offline normalize()+velocity() pipeline, just computed
    incrementally one frame at a time instead of over a whole clip. Kept as
    the single-threshold version for backward compat (frame_quality_gate's
    whole-segment peak_speed check still uses the combined _MOTION_JOINTS
    list); ActionSegmenter itself uses _instant_speed_by_group below so arm
    and leg sensitivity can be tuned independently."""
    hip_mid = (xyz_frame[23] + xyz_frame[24]) / 2.0
    shoulder_mid = (xyz_frame[11] + xyz_frame[12]) / 2.0
    scale = max(float(np.linalg.norm(shoulder_mid - hip_mid)), 1e-6)
    deltas = [np.linalg.norm(xyz_frame[j] - prev_xyz_frame[j]) for j in _MOTION_JOINTS]
    return max(deltas) / scale


def _instant_speed_by_group(xyz_frame: np.ndarray, prev_xyz_frame: np.ndarray):
    """Same quantity as _instant_speed, split into (arm_speed, leg_speed) --
    lets ActionSegmenter give arms and legs separate motion thresholds
    instead of one shared bar. Torso/chest isn't a separate group here: the
    onset/settle detector was never gated on torso motion at all (only
    wrist/ankle limb-tip speed), so there's no existing chest threshold to
    split out -- "whole body" freedom is the combination of these two."""
    hip_mid = (xyz_frame[23] + xyz_frame[24]) / 2.0
    shoulder_mid = (xyz_frame[11] + xyz_frame[12]) / 2.0
    scale = max(float(np.linalg.norm(shoulder_mid - hip_mid)), 1e-6)
    arm = max(np.linalg.norm(xyz_frame[j] - prev_xyz_frame[j]) for j in _ARM_JOINTS) / scale
    leg = max(np.linalg.norm(xyz_frame[j] - prev_xyz_frame[j]) for j in _LEG_JOINTS) / scale
    return arm, leg


class ActionSegmenter:
    """Feed it one frame at a time. Returns a completed (xyz, vis) segment
    from push() only when a real motion event just finished; None the rest
    of the time (idle-watching, or still mid-action)."""

    def __init__(self, cooldown_seconds: float = COOLDOWN_SECONDS,
                 motion_threshold: float = MIN_MOTION_FOR_ACTION,
                 motion_threshold_arms: float = None,
                 motion_threshold_legs: float = None,
                 auto_cooldown: bool = True):
        """motion_threshold: shared onset/settle speed cutoff, kept as the
        single-value default for backward compat. motion_threshold_arms/
        _legs: independent overrides -- e.g. raise legs (walking/weight-
        shift tolerance) without touching arm sensitivity, or vice versa.
        Each defaults to `motion_threshold` when not given, so passing just
        `motion_threshold` still behaves exactly as before (one shared bar).

        auto_cooldown=True (default, backward compat): push() arms cooldown
        the instant a segment is captured, before the caller has even
        classified it -- what scripts/live_inference.py and
        live_inference_agcn.py already rely on, unchanged.
        auto_cooldown=False: push() does NOT arm cooldown on its own;
        the caller must call start_cooldown() explicitly, only when the
        classification result was a real recognized action, not a gate-
        rejected 'idle' -- so a rejected capture goes straight back to
        watching instead of eating the full cooldown for nothing (see
        scripts/live_inference_v2.py)."""
        self.cooldown_seconds = cooldown_seconds
        self.auto_cooldown = auto_cooldown
        self.motion_threshold_arms = motion_threshold_arms if motion_threshold_arms is not None else motion_threshold
        self.motion_threshold_legs = motion_threshold_legs if motion_threshold_legs is not None else motion_threshold
        self.pre_roll = deque(maxlen=PRE_ROLL_FRAMES)
        self.state = "idle"          # "idle" | "recording"
        self.segment_xyz = []
        self.segment_vis = []
        self.segment_aux = []
        self.above_count = 0
        self.below_count = 0
        self.prev_xyz = None
        self.cooldown_until = 0.0    # time.time() deadline; frames ignored entirely until then
        self._last_segment_aux = None

    def in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    def push(self, xyz_frame: np.ndarray, vis_frame: np.ndarray, aux_frame=None):
        """aux_frame: optional per-frame payload buffered in lockstep with
        xyz/vis (e.g. AGCN's raw hand landmarks) but NOT part of the return
        value -- keeps this method's return type/arity unchanged so existing
        2-value callers (scripts/live_inference.py) are unaffected. Retrieve
        a just-completed segment's aux frames via get_last_segment_aux()
        right after push() returns a non-None segment."""
        if self.in_cooldown():
            # ignore completely -- don't even track prev_xyz/pre_roll, so the
            # settle/retraction tail of the action that just finished can't
            # seed or contaminate the next capture
            self.prev_xyz = None
            return None

        has_person = not np.isnan(xyz_frame).any()
        arm_speed = leg_speed = 0.0
        if has_person and self.prev_xyz is not None:
            arm_speed, leg_speed = _instant_speed_by_group(xyz_frame, self.prev_xyz)
        self.prev_xyz = xyz_frame if has_person else None
        is_above = arm_speed > self.motion_threshold_arms or leg_speed > self.motion_threshold_legs
        is_below = arm_speed < self.motion_threshold_arms and leg_speed < self.motion_threshold_legs

        if self.state == "idle":
            self.pre_roll.append((xyz_frame, vis_frame, aux_frame))
            self.above_count = self.above_count + 1 if is_above else 0
            if self.above_count >= ONSET_FRAMES:
                # onset confirmed -- start recording, seeded with pre-roll
                # context so the wind-up before the trigger isn't lost
                self.state = "recording"
                self.segment_xyz = [f for f, _, _ in self.pre_roll]
                self.segment_vis = [v for _, v, _ in self.pre_roll]
                self.segment_aux = [a for _, _, a in self.pre_roll]
                self.above_count = 0
                self.below_count = 0
            return None

        # state == "recording"
        self.segment_xyz.append(xyz_frame)
        self.segment_vis.append(vis_frame)
        self.segment_aux.append(aux_frame)
        self.below_count = self.below_count + 1 if is_below else 0

        finished = self.below_count >= SETTLE_FRAMES or len(self.segment_xyz) >= MAX_SEGMENT_FRAMES
        if not finished:
            return None

        seg_xyz = np.stack(self.segment_xyz)
        seg_vis = np.stack(self.segment_vis)
        seg_aux = self.segment_aux
        self.state = "idle"
        self.segment_xyz, self.segment_vis, self.segment_aux = [], [], []
        self.pre_roll.clear()
        self.above_count = self.below_count = 0

        if len(seg_xyz) < MIN_SEGMENT_FRAMES:
            return None  # spurious blip, not a real action -- discard, stay idle, no cooldown needed

        # real segment captured. auto_cooldown=True (default): arm cooldown
        # right now, before the caller even classifies it, so the settle/
        # retraction tail happening *right now* can't leak into the next
        # capture. auto_cooldown=False: leave it to the caller via
        # start_cooldown() -- only once it knows the classification was a
        # real action, not a gate-rejected 'idle'.
        if self.auto_cooldown:
            self.cooldown_until = time.time() + self.cooldown_seconds
        self._last_segment_aux = seg_aux if seg_aux[0] is not None else None
        return seg_xyz, seg_vis

    def start_cooldown(self):
        """Explicitly arm cooldown -- for auto_cooldown=False callers, call
        this after push() returns a segment AND the classifier confirmed it
        was a real action (not gate-rejected 'idle'). No-op timing-wise if
        called outside that flow, but that's not the intended use."""
        self.cooldown_until = time.time() + self.cooldown_seconds

    def get_last_segment_aux(self):
        """aux_frame list (same length/order as the xyz/vis just returned by
        push()) for the most recently completed segment, or None if aux_frame
        was never passed to push() during that segment."""
        return self._last_segment_aux

    def is_recording(self) -> bool:
        return self.state == "recording"


def frame_quality_gate(vis: np.ndarray, normalized_xyz: np.ndarray, peak_speed: float):
    """Checks a captured segment actually looks like training data before
    trusting any prediction on it -- softmax confidence alone can't catch
    out-of-distribution input (a model can be confidently wrong on data
    unlike its training distribution). Returns (ok, reason)."""
    leg_vis = np.nanmean(vis[:, LEG_LANDMARKS])
    arm_vis = np.nanmean(vis[:, ARM_LANDMARKS])
    if leg_vis < MIN_VISIBILITY_FOR_ACTION or arm_vis < MIN_VISIBILITY_FOR_ACTION:
        return False, "step back -- full body not clearly visible"
    if peak_speed < MIN_MOTION_FOR_ACTION:
        return False, "no motion detected"
    return True, None


_novelty_cache = None  # lazy-loaded: (Xz_train, mean, std, threshold)


def _load_novelty_reference():
    global _novelty_cache
    if _novelty_cache is None:
        path = ROOT / "models" / "xgboost_train_features.npz"
        data = np.load(path)
        _novelty_cache = (data["X"], data["mean"], data["std"], float(data["novelty_threshold"]))
    return _novelty_cache


def novelty_gate(feat: np.ndarray):
    """Rejects inputs that don't statistically resemble ANY real training
    clip -- not just the 3 trained classes' softmax, but real data at all.
    This is the general-purpose "none of these" check: it doesn't need to
    have seen the specific non-action happening (waving, walking, anything)
    to reject it, unlike a 4th learned class which only recognizes the
    specific negatives it was shown. Distance is nearest-neighbor in
    standardized feature space vs the cached training set; threshold is the
    99th percentile of the training data's own natural nearest-neighbor
    spread (see scripts/train_xgboost_final.py). Returns (ok, reason)."""
    Xz_train, mean, std, threshold = _load_novelty_reference()
    z = (feat - mean) / std
    dist = float(np.min(np.linalg.norm(Xz_train - z, axis=1)))
    if dist > threshold:
        return False, "unfamiliar motion -- not a recognized action"
    return True, None


def predict_segment(model, xyz: np.ndarray, vis: np.ndarray, debug: bool = False,
                     feature_fn=extract_handcrafted_features):
    """Classifies one captured, bounded action segment (from
    ActionSegmenter.push()) -- same preprocess + handcrafted-feature path as
    scripts/train_xgboost_final.py. feature_fn defaults to the original
    15-feature extractor (backward compat, what scripts/live_inference.py
    and live_inference_v2.py already use) -- pass
    src.features.handcrafted_v2.extract_handcrafted_features_v2 to use the
    16-feature version (+ starting elbow angle) with models/xgboost_v2.json.
    Returns (label, probs[3], reason)."""
    sample = preprocess_clip(xyz, vis)
    peak_speed = float(np.max(np.linalg.norm(np.diff(sample["xyz"][:, _MOTION_JOINTS], axis=0), axis=-1)))

    ok, reason = frame_quality_gate(vis, sample["xyz"], peak_speed)
    if not ok:
        if debug:
            print(f"[gate] REJECT (visibility/motion): {reason}  peak_speed={peak_speed:.3f}")
        return "idle", np.array([1 / 3, 1 / 3, 1 / 3]), reason

    feat = feature_fn(sample["xyz"])

    # novelty_gate() is defined above but deliberately not called here --
    # reverted per user request to the state before it was added; it's
    # available to re-enable later if wanted (see novelty_gate() docstring)
    if debug:
        print(f"[gate] pass  peak_speed={peak_speed:.3f}")
        print(f"[feat] {np.round(feat, 3).tolist()}")

    probs = model.predict_proba(feat.reshape(1, -1))[0]
    label = ACTIONS[int(np.argmax(probs))]
    if debug:
        print(f"[predict] {label}  probs={np.round(probs, 3).tolist()}")
    return label, probs, None


def predict_segment_v3(model, xyz: np.ndarray, vis: np.ndarray, hand_landmarks: np.ndarray,
                        debug: bool = False):
    """XGBoost v3 counterpart of predict_segment -- v3's hand-openness feature
    (src.features.handcrafted_v3) needs real MediaPipe Hands curl-angle data,
    not just Pose xyz, so it can't go through predict_segment's single-arg
    feature_fn hook. Same gate/segment contract; hand_landmarks: (T, 2, 21, 3)
    raw MediaPipe Hands output for the same segment, same frame count/timeline
    as xyz/vis -- from ActionSegmenter's aux_frame buffering, same pattern
    src.data.graph_dataset.predict_segment_agcn already uses for the AGCN
    hand stream. Returns (label, probs[3], reason)."""
    from src.data.hand_features import crop_and_resample, finger_curl_series
    from src.features.handcrafted_v3 import extract_handcrafted_features_v3

    sample = preprocess_clip(xyz, vis)
    peak_speed = float(np.max(np.linalg.norm(np.diff(sample["xyz"][:, _MOTION_JOINTS], axis=0), axis=-1)))

    ok, reason = frame_quality_gate(vis, sample["xyz"], peak_speed)
    if not ok:
        if debug:
            print(f"[gate] REJECT (visibility/motion): {reason}  peak_speed={peak_speed:.3f}")
        return "idle", np.array([1 / 3, 1 / 3, 1 / 3]), reason

    curl, _presence = finger_curl_series(hand_landmarks)
    hand_curl = crop_and_resample(curl, None, sample["xyz"].shape[0])
    feat = extract_handcrafted_features_v3(sample["xyz"], hand_curl)

    if debug:
        print(f"[gate] pass  peak_speed={peak_speed:.3f}")
        print(f"[feat] {np.round(feat, 3).tolist()}")

    probs = model.predict_proba(feat.reshape(1, -1))[0]
    label = ACTIONS[int(np.argmax(probs))]
    if debug:
        print(f"[predict] {label}  probs={np.round(probs, 3).tolist()}")
    return label, probs, None
