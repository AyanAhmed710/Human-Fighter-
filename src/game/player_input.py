"""
Background camera+classifier thread for one player -- strips scripts/
live_inference_v3.py down to its capture/inference core (no cv2 window/
overlay; the game engine owns the display), wrapped so the game's main loop
only ever has to do a non-blocking queue poll per frame. Reuses the exact
same segmenter/model/gate pipeline already validated end to end
(predict_segment_v3, ActionSegmenter) -- no new ML code, just a threading
wrapper around it.

One instance per player/camera. Each owns its own cv2.VideoCapture and its
own mp_pose.Pose/mp_hands.Hands context (MediaPipe models aren't shared
across instances, so two players running simultaneously in separate threads
is safe -- no shared mutable model state, matching how live_inference_v3.py
and live_inference_agcn.py already each open their own `with` contexts).
"""
import queue
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import xgboost as xgb

from src.config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, N_POSE_LANDMARKS, ROOT
from src.data.extract_hands import N_HAND_LANDMARKS
from src.inference import ActionSegmenter, predict_segment_v3

MODEL_PATH = ROOT / "models" / "xgboost_v3.json"

DEFAULT_COOLDOWN_SECONDS = 3.0   # shorter than live_inference_v3.py's 5s solo-demo
                                  # default -- see MODEL_JOURNEY.md/game strategy
                                  # notes: 5s was tuned to avoid false triggers on
                                  # one ambiguous frame, not a hard floor. 3s is
                                  # the agreed in-game value: enough to stop one
                                  # swing's follow-through re-triggering itself,
                                  # short enough matches stay fast-paced.
DEFAULT_MOTION_THRESHOLD = 0.08

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # same skeleton/hand overlay used while
                                          # testing the model live (scripts/
                                          # live_inference_v3.py) -- drawn onto
                                          # latest_frame so the in-game webcam
                                          # preview shows it too, not just a
                                          # plain camera feed.


class PlayerCameraInput:
    """queue.get_nowait() from the main game thread each frame to drain
    recognized actions -- ("punch"|"kick"|"shoot", confidence) tuples only;
    "idle"/gate-rejected results never get queued at all (no dead events for
    the game loop to filter out)."""

    ACTION_TO_GAME = {"punching": "punch", "kicking": "kick", "shooting": "shoot"}

    def __init__(self, camera_index: int, cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
                 motion_threshold: float = DEFAULT_MOTION_THRESHOLD, debug: bool = False):
        self.camera_index = camera_index
        self.cooldown_seconds = cooldown_seconds
        self.motion_threshold = motion_threshold
        self.debug = debug

        self.queue: "queue.Queue[tuple[str, float]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self.connected = False   # camera opened OK -- game can show a warning if not
        self.last_person_visible = False  # for on-screen "no person detected" per player
        self.latest_frame = None  # most recent raw BGR frame (post-mirror-flip), for
                                   # the game's own on-screen webcam preview -- read by
                                   # the main thread, written by this thread; a plain
                                   # reference swap (no lock) is fine here since a numpy
                                   # array assignment is atomic under the GIL and a
                                   # dropped/duplicated preview frame is harmless

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_action_nowait(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def _run(self):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.connected = False
            return
        self.connected = True

        segmenter = ActionSegmenter(
            cooldown_seconds=self.cooldown_seconds,
            motion_threshold=self.motion_threshold,
            auto_cooldown=False,  # only cool down on a real recognized action
        )

        with mp_pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as pose, mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as hands:
            while not self._stop_event.is_set():
                if segmenter.in_cooldown():
                    segmenter.push(np.full((N_POSE_LANDMARKS, 3), np.nan, dtype=np.float32),
                                    np.zeros(N_POSE_LANDMARKS, dtype=np.float32))
                    # still read+mirror a frame so the on-screen preview stays
                    # live during the cooldown window instead of freezing on
                    # the last pre-cooldown frame for 3s after every action --
                    # Pose/Hands themselves stay skipped (the actual expensive
                    # part), same as before.
                    ok, frame = cap.read()
                    if ok:
                        frame = cv2.flip(frame, 1)
                        remaining = max(0.0, segmenter.cooldown_until - time.time())
                        cv2.putText(frame, f"cooldown {remaining:.1f}s", (16, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 90, 90), 2)
                        self.latest_frame = frame
                    time.sleep(0.01)
                    continue

                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = pose.process(rgb)
                hand_result = hands.process(rgb)

                person_visible = result.pose_world_landmarks is not None
                self.last_person_visible = person_visible
                if person_visible:
                    lms = result.pose_world_landmarks.landmark
                    xyz = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                    vis = np.array([lm.visibility for lm in lms], dtype=np.float32)
                    # same skeleton overlay drawn while testing the model live
                    # (scripts/live_inference_v3.py) -- draws onto `frame`
                    # in-place using image-space pose_landmarks, not the
                    # metric pose_world_landmarks used for the xyz feature
                    # above.
                    if result.pose_landmarks is not None:
                        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                else:
                    xyz = np.full((N_POSE_LANDMARKS, 3), np.nan, dtype=np.float32)
                    vis = np.zeros(N_POSE_LANDMARKS, dtype=np.float32)

                hand_frame = np.full((2, N_HAND_LANDMARKS, 3), np.nan, dtype=np.float32)
                if hand_result.multi_hand_landmarks and hand_result.multi_handedness:
                    for lm_set, handedness in zip(hand_result.multi_hand_landmarks,
                                                    hand_result.multi_handedness):
                        idx = 0 if handedness.classification[0].label == "Left" else 1
                        hand_frame[idx] = np.array([[p.x, p.y, p.z] for p in lm_set.landmark],
                                                     dtype=np.float32)
                        mp_drawing.draw_landmarks(frame, lm_set, mp_hands.HAND_CONNECTIONS)

                self.latest_frame = frame  # after drawing, so the preview shows the skeleton overlay

                segment = segmenter.push(xyz, vis, aux_frame=hand_frame)
                if segment is not None:
                    seg_xyz, seg_vis = segment
                    seg_hand = np.stack(segmenter.get_last_segment_aux())
                    label, probs, reason = predict_segment_v3(model, seg_xyz, seg_vis, seg_hand,
                                                                debug=self.debug)
                    if label != "idle":
                        segmenter.start_cooldown()
                        game_action = self.ACTION_TO_GAME[label]
                        confidence = float(np.max(probs))
                        self.queue.put((game_action, confidence))
                    elif self.debug:
                        print(f"[player {self.camera_index}] rejected: {reason}")

        cap.release()
