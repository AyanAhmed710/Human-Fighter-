"""
Live webcam test of the 2s-AGCN model (protocol section 7 primary target).

Same motion-triggered segmenter as scripts/live_inference.py (onset ->
sustained -> settle, classify once, hold, resume watching) -- only the
classifier swaps: handcrafted-feature + XGBoost -> joint/bone+hand-curl
graph tensor + TwoStreamAGCN. src/inference.py's ActionSegmenter and
frame_quality_gate are reused (ActionSegmenter's optional aux_frame param
added for this); scripts/live_inference.py itself is untouched.

v4: also runs MediaPipe Hands alongside Pose every frame -- Pose's own
fingertip landmarks proved too low-confidence live (v3 diagnosis, see
notebooks/agcn/01_2s_agcn.ipynb's v3 note), so shooting needs Hands' real
per-knuckle landmarks instead. This is a second model running per frame,
so expect a lower live FPS than the XGBoost script or v2/v3 AGCN.

Handedness note: the display frame is mirrored (cv2.flip) before either
model runs, same as scripts/live_inference.py already does for Pose --
Hands' own Left/Right classification is trusted as-is on that mirrored
frame for consistency with how Pose's L/R landmark indices are already
treated throughout this pipeline (recordings aren't mirrored, live is).

Usage:
    python scripts/live_inference_agcn.py
    python scripts/live_inference_agcn.py --camera 1   # if you have multiple cameras

Controls: press 'q' or ESC to quit.

Run the notebooks/agcn/01_2s_agcn.ipynb notebook first if
models/agcn_2stream.pt doesn't exist yet (or copy the run's best.pt there).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    N_POSE_LANDMARKS,
    RESULT_HOLD_SECONDS,
    ROOT,
)
from src.data.extract_hands import N_HAND_LANDMARKS
from src.data.graph_dataset import predict_segment_agcn
from src.inference import ActionSegmenter
from src.models.agcn import TwoStreamAGCN

MODEL_PATH = ROOT / "models" / "agcn_2stream.pt"

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


def draw_overlay(frame, display_state, label, probs, actions, fps, reason=None):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 120), (30, 30, 30), -1)

    if display_state == "no_person":
        cv2.putText(frame, "no person detected", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    elif display_state == "watching":
        cv2.putText(frame, "watching...", (16, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLORS["watching"], 2)
    elif display_state == "cooldown":
        cv2.putText(frame, "cooldown...", (16, 45),
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
                     help="seconds to ignore all motion after a captured action, before watching "
                          "for the next one (default from src/config.py, currently 1.2s)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                     help="torch device (default: cuda if available, else cpu)")
    ap.add_argument("--debug", action="store_true",
                     help="print gate/prediction diagnostics to console for every captured segment")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        print(f"model not found at {MODEL_PATH}")
        print("run notebooks/agcn/01_2s_agcn.ipynb first, or copy a run's best.pt to this path")
        sys.exit(1)

    model = TwoStreamAGCN(base_channels=32, num_classes=3, dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=args.device, weights_only=True))
    model.to(args.device)
    model.eval()
    actions = ["kicking", "punching", "shooting"]

    # CAP_DSHOW -- default MSMF backend is slow/flaky to open on Windows
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"could not open camera {args.camera}")
        sys.exit(1)

    from src.config import COOLDOWN_SECONDS
    segmenter = ActionSegmenter(cooldown_seconds=args.cooldown if args.cooldown is not None else COOLDOWN_SECONDS)
    result_label, result_probs, result_reason = None, np.array([1 / 3, 1 / 3, 1 / 3]), None
    result_until = 0.0  # time.time() deadline -- show the last result until this, then resume watching

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
        print("camera open -- press q or ESC to quit")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                break

            frame = cv2.flip(frame, 1)  # mirror -- feels natural facing the camera
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
                result_label, result_probs, result_reason = predict_segment_agcn(
                    model, seg_xyz, seg_vis, seg_hand, device=args.device, debug=args.debug)
                result_until = time.time() + RESULT_HOLD_SECONDS

            # pick what to display this frame
            if time.time() < result_until:
                display_state = "result"
            elif segmenter.in_cooldown():
                display_state = "cooldown"
            elif not person_visible:
                display_state = "no_person"
            elif segmenter.is_recording():
                display_state = "recording"
            else:
                display_state = "watching"

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now

            draw_overlay(frame, display_state, result_label, result_probs, actions, fps, result_reason)
            cv2.imshow("Tekken action classifier -- live (2s-AGCN)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 27 = ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
