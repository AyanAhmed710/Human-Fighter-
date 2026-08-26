"""
Live webcam test of the XGBoost v3 model -- real MediaPipe Hands curl-angle
hand-openness feature (src/features/handcrafted_v3.py) replacing v1/v2's
crude Pose-fingertip-distance proxy. This is the strongest offline result in
the project so far (MODEL_JOURNEY.md section 7b): test 0.98 vs v1's 0.94,
wins every leave-participant-out CV fold (v2's elbow-angle addition did not).

Combines two things bundled from earlier scripts:
  1. v2's UX fixes: longer default cooldown (5s), separate tunable arm/leg
     motion thresholds, cooldown only armed on a real predicted class (not a
     gate-rejected capture), Pose+Hands both skipped entirely during
     cooldown (not just ignored).
  2. live_inference_agcn.py's live MediaPipe Hands capture: v3's hand signal
     needs real per-finger landmarks, not just Pose, so this runs Hands
     alongside Pose every frame like the AGCN script does -- expect a lower
     live FPS than scripts/live_inference.py or live_inference_v2.py (both
     Pose-only).

scripts/live_inference.py and live_inference_v2.py are left untouched --
this is a new, separate script, not a modification of either.

Usage:
    python scripts/live_inference_v3.py
    python scripts/live_inference_v3.py --camera 1        # if you have multiple cameras
    python scripts/live_inference_v3.py --cooldown 3       # override the 5s default

Controls: press 'q' or ESC to quit.

Run scripts/train_xgboost_final_v3.py first if models/xgboost_v3.json
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
from src.data.extract_hands import N_HAND_LANDMARKS
from src.inference import ActionSegmenter, predict_segment_v3

MODEL_PATH = ROOT / "models" / "xgboost_v3.json"

DEFAULT_COOLDOWN_SECONDS = 5.0       # same reasoning as live_inference_v2.py
DEFAULT_MOTION_THRESHOLD = 0.08      # same reasoning as live_inference_v2.py

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
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
                     help=f"seconds to ignore all motion after a captured action (default "
                          f"{DEFAULT_COOLDOWN_SECONDS}s)")
    ap.add_argument("--motion-threshold", type=float, default=DEFAULT_MOTION_THRESHOLD,
                     help=f"shared normalized-speed cutoff for both arms and legs (default "
                          f"{DEFAULT_MOTION_THRESHOLD}, up from config.MIN_MOTION_FOR_ACTION's "
                          f"{MIN_MOTION_FOR_ACTION})")
    ap.add_argument("--motion-threshold-arms", type=float, default=None,
                     help="wrist-only motion cutoff, overrides --motion-threshold for arms")
    ap.add_argument("--motion-threshold-legs", type=float, default=None,
                     help="ankle-only motion cutoff, overrides --motion-threshold for legs")
    ap.add_argument("--debug", action="store_true",
                     help="print gate/prediction diagnostics to console for every captured segment")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        print(f"model not found at {MODEL_PATH}")
        print("run: python scripts/train_xgboost_final_v3.py")
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
    result_until = 0.0

    t_prev = time.time()
    fps = 0.0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    ) as hands:
        print(f"camera open -- cooldown {cooldown}s -- press q or ESC to quit")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                break

            frame = cv2.flip(frame, 1)  # mirror -- feels natural facing the camera

            if segmenter.in_cooldown():
                # skip Pose AND Hands entirely for the whole window, same
                # reasoning as live_inference_v2.py's Pose-only version --
                # neither model's output is reachable during cooldown anyway
                segmenter.push(np.full((N_POSE_LANDMARKS, 3), np.nan, dtype=np.float32),
                                np.zeros(N_POSE_LANDMARKS, dtype=np.float32))

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
                t_prev = now
                cooldown_remaining = max(0.0, segmenter.cooldown_until - now)
                display_state = "result" if now < result_until else "cooldown"
                draw_overlay(frame, display_state, result_label, result_probs, actions, fps,
                             result_reason, cooldown_remaining)
                cv2.imshow("Tekken action classifier -- live (XGBoost v3, real-curl hand feature)", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            hand_result = hands.process(rgb)

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

            hand_frame = np.full((2, N_HAND_LANDMARKS, 3), np.nan, dtype=np.float32)
            if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                for lm_set, handedness in zip(hand_result.multi_hand_landmarks, hand_result.multi_handedness):
                    idx = 0 if handedness.classification[0].label == "Left" else 1
                    hand_frame[idx] = np.array([[p.x, p.y, p.z] for p in lm_set.landmark], dtype=np.float32)
                    mp_drawing.draw_landmarks(frame, lm_set, mp_hands.HAND_CONNECTIONS)

            segment = segmenter.push(xyz, vis, aux_frame=hand_frame)
            if segment is not None:
                seg_xyz, seg_vis = segment
                seg_hand = np.stack(segmenter.get_last_segment_aux())
                if args.debug:
                    print(f"\n--- segment captured, {len(seg_xyz)} frames ---")
                result_label, result_probs, result_reason = predict_segment_v3(
                    model, seg_xyz, seg_vis, seg_hand, debug=args.debug)
                result_until = time.time() + RESULT_HOLD_SECONDS
                if result_label != "idle":
                    segmenter.start_cooldown()
                elif args.debug:
                    print("[cooldown] skipped -- gate-rejected, not a real class")

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
            cv2.imshow("Tekken action classifier -- live (XGBoost v3, real-curl hand feature)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 27 = ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
