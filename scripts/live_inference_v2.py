"""
Live webcam test of the XGBoost model -- v2: this script name has two
unrelated meanings bundled together, worth being explicit about:
  1. UX fixes: longer default cooldown (5s vs config.COOLDOWN_SECONDS' 1.2s),
     Pose skipped entirely during cooldown (not just ignored), cooldown only
     armed on a real predicted class (not a gate-rejected capture), separate
     tunable arm/leg motion thresholds.
  2. Model: models/xgboost_v2.json (src/features/handcrafted_v2.py, 16
     features -- the original 15 + starting elbow angle, effect size 2.27,
     the strongest separator found this session) instead of the original
     models/xgboost_baseline.json (15 features).
scripts/live_inference.py is left untouched -- original model, original UX.

Watches for a real, bounded motion event (onset -> sustained -> settle),
captures just that segment, classifies it ONCE, holds the result on screen
briefly, then goes back to watching -- same idea as the hand-trimmed
training clips (one deliberate action each), not a continuously
reclassified arbitrary sliding window (see src/inference.py for why that
approach misfired on standing still / partial framing).

Usage:
    python scripts/live_inference_v2.py
    python scripts/live_inference_v2.py --camera 1        # if you have multiple cameras
    python scripts/live_inference_v2.py --cooldown 3       # override the 5s default

Controls: press 'q' or ESC to quit.

Run scripts/train_xgboost_final_v2.py first if models/xgboost_v2.json
doesn't exist yet.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
import numpy as np
import xgboost as xgb

from src.config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_MOTION_FOR_ACTION,
    MIN_TRACKING_CONFIDENCE,
    N_POSE_LANDMARKS,
    RESULT_HOLD_SECONDS,
    ROOT,
)
from src.features.handcrafted_v2 import extract_handcrafted_features_v2
from src.inference import ActionSegmenter, predict_segment

MODEL_PATH = ROOT / "models" / "xgboost_v2.json"

# v2: 5s instead of config.COOLDOWN_SECONDS' 1.2s -- give the segmenter more
# time after a recognized action before it starts watching for the next one,
# so a settle-frame blip or retraction tail can't immediately seed a new
# capture on stale/insufficient evidence. This is the ONE dead-time window --
# RESULT_HOLD_SECONDS (1.5s, how long the label stays on screen) is nested
# inside it, not stacked after it; the on-screen cooldown countdown below
# makes that visible instead of feeling like an extra unexplained wait.
DEFAULT_COOLDOWN_SECONDS = 5.0

# v2: a little higher than config.MIN_MOTION_FOR_ACTION (0.05) -- live
# testing showed plain weight-shifting/walking-scale motion crossing that
# threshold and triggering a capture (most visible as false "kicking",
# since ankles are MediaPipe's best-tracked landmarks -- not a landmark-
# noise problem, a threshold-too-sensitive one). --motion-threshold lets you
# tune this yourself instead of it being a fixed guess.
DEFAULT_MOTION_THRESHOLD = 0.08

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

COLORS = {
    "kicking": (60, 200, 255),
    "punching": (60, 255, 120),
    "shooting": (255, 120, 60),
    "idle": (140, 140, 140),
    "watching": (100, 100, 100),
}


def draw_overlay(frame, display_state, label, probs, actions, fps, reason=None, cooldown_remaining=0.0):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 120), (30, 30, 30), -1)

    if display_state == "no_person":
        cv2.putText(frame, "no person detected", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    elif display_state == "watching":
        cv2.putText(frame, "watching...", (16, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS["watching"], 2)
    elif display_state == "cooldown":
        # live countdown -- so the wait is visible/predictable instead of an
        # opaque "is it stuck?" guess (the actual dead time is exactly
        # `cooldown` seconds, one window, not stacked with the result-hold
        # display; this makes that legible on screen)
        cv2.putText(frame, f"cooldown... {cooldown_remaining:.1f}s", (16, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (90, 90, 90), 2)
    elif display_state == "recording":
        cv2.putText(frame, "MOTION DETECTED - capturing", (16, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    else:  # "result" -- label is a real classification (or idle w/ reason) just produced
        color = COLORS.get(label, (255, 255, 255))
        cv2.putText(frame, label.upper(), (16, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
        if label == "idle" and reason:
            cv2.putText(frame, f"({reason})", (16, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        bar_x = 260
        for i, action in enumerate(actions):
            p = probs[i]
            bar_w = int(p * 300)
            y = 20 + i * 30
            c = COLORS.get(action, (255, 255, 255))
            cv2.rectangle(frame, (bar_x, y), (bar_x + 300, y + 20), (70, 70, 70), 1)
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + 20), c, -1)
            cv2.putText(frame, f"{action} {p:.2f}", (bar_x + 305, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.putText(frame, f"fps: {fps:.1f}", (w - 120, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "q / ESC to quit", (16, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="webcam device index (default 0)")
    ap.add_argument("--cooldown", type=float, default=None,
                     help=f"seconds to ignore all motion after a captured action, before watching "
                          f"for the next one (default {DEFAULT_COOLDOWN_SECONDS}s -- longer than "
                          f"scripts/live_inference.py's 1.2s, to collect more evidence before the "
                          f"next decision)")
    ap.add_argument("--motion-threshold", type=float, default=DEFAULT_MOTION_THRESHOLD,
                     help=f"shared normalized-speed cutoff for both arms and legs, used as the "
                          f"default for whichever of --motion-threshold-arms/-legs isn't set "
                          f"(default {DEFAULT_MOTION_THRESHOLD}, up from config.MIN_MOTION_FOR_ACTION's "
                          f"{MIN_MOTION_FOR_ACTION})")
    ap.add_argument("--motion-threshold-arms", type=float, default=None,
                     help="wrist-only motion cutoff, overrides --motion-threshold for arms -- lower "
                          "this if real punches/shoots are getting missed, without touching leg "
                          "sensitivity")
    ap.add_argument("--motion-threshold-legs", type=float, default=None,
                     help="ankle-only motion cutoff, overrides --motion-threshold for legs -- raise "
                          "this if walking/weight-shifting still triggers a false kick, without "
                          "touching arm sensitivity")
    ap.add_argument("--debug", action="store_true",
                     help="print gate/novelty/prediction diagnostics to console for every captured segment")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        print(f"model not found at {MODEL_PATH}")
        print("run: python scripts/train_xgboost_final_v2.py")
        sys.exit(1)

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    actions = ["kicking", "punching", "shooting"]

    # CAP_DSHOW -- default MSMF backend is slow/flaky to open on Windows
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"could not open camera {args.camera}")
        sys.exit(1)

    cooldown = args.cooldown if args.cooldown is not None else DEFAULT_COOLDOWN_SECONDS
    segmenter = ActionSegmenter(
        cooldown_seconds=cooldown,
        motion_threshold=args.motion_threshold,
        motion_threshold_arms=args.motion_threshold_arms,
        motion_threshold_legs=args.motion_threshold_legs,
        auto_cooldown=False,  # only start_cooldown() on a real predicted class, not every capture
    )
    result_label, result_probs, result_reason = None, np.array([1 / 3, 1 / 3, 1 / 3]), None
    result_until = 0.0  # time.time() deadline -- show the last result until this, then resume watching

    t_prev = time.time()
    fps = 0.0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as pose:
        print(f"camera open -- cooldown {cooldown}s -- press q or ESC to quit")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                break

            frame = cv2.flip(frame, 1)  # mirror -- feels natural facing the camera

            if segmenter.in_cooldown():
                # literally skip Pose (and the classifier, which was never
                # reachable during cooldown anyway) for the whole window --
                # not just "ignore the result", don't run the model at all.
                # Still tick push() with a throwaway frame every loop so
                # ActionSegmenter's prev_xyz stays nulled (see push()'s own
                # in_cooldown branch) -- its content is unused whenever
                # in_cooldown() is True, only the call cadence matters.
                segmenter.push(np.full((N_POSE_LANDMARKS, 3), np.nan, dtype=np.float32),
                                np.zeros(N_POSE_LANDMARKS, dtype=np.float32))

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
                t_prev = now
                cooldown_remaining = max(0.0, segmenter.cooldown_until - now)
                display_state = "result" if now < result_until else "cooldown"
                draw_overlay(frame, display_state, result_label, result_probs, actions, fps,
                             result_reason, cooldown_remaining)
                cv2.imshow("Tekken action classifier -- live (XGBoost, v2 cooldown)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            person_visible = result.pose_world_landmarks is not None
            if person_visible:
                lms = result.pose_world_landmarks.landmark
                xyz = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                vis = np.array([lm.visibility for lm in lms], dtype=np.float32)
                if result.pose_landmarks is not None:
                    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                xyz = np.full((N_POSE_LANDMARKS, 3), np.nan, dtype=np.float32)
                vis = np.zeros(N_POSE_LANDMARKS, dtype=np.float32)

            segment = segmenter.push(xyz, vis)
            if segment is not None:
                seg_xyz, seg_vis = segment
                if args.debug:
                    print(f"\n--- segment captured, {len(seg_xyz)} frames ---")
                result_label, result_probs, result_reason = predict_segment(
                    model, seg_xyz, seg_vis, debug=args.debug,
                    feature_fn=extract_handcrafted_features_v2)
                result_until = time.time() + RESULT_HOLD_SECONDS
                if result_label != "idle":
                    # only a real recognized action costs the cooldown --
                    # a gate-rejected capture (bad visibility, too little
                    # motion) goes straight back to watching instead
                    segmenter.start_cooldown()
                elif args.debug:
                    print("[cooldown] skipped -- gate-rejected, not a real class")

            # pick what to display this frame -- "cooldown" itself is handled
            # by the early-continue branch above (Pose doesn't even run
            # then), so this path only ever sees result/no_person/recording/
            # watching
            if time.time() < result_until:
                display_state = "result"
            elif not person_visible:
                display_state = "no_person"
            elif segmenter.is_recording():
                display_state = "recording"
            else:
                display_state = "watching"

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            cooldown_remaining = max(0.0, segmenter.cooldown_until - now)
            draw_overlay(frame, display_state, result_label, result_probs, actions, fps,
                         result_reason, cooldown_remaining)
            cv2.imshow("Tekken action classifier -- live (XGBoost, v2 cooldown)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 27 = ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
