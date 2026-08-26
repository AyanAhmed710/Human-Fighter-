"""
Single source of truth for paths, split, and hyperparams.
Notebooks/scripts import from here -- never hardcode these values twice.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- raw data ---
LABELS_CSV = ROOT / "combined_labels.csv"
ACTION_FOLDERS = {
    "kicking": ROOT / "kicking",
    "punching": ROOT / "punching",
    "shooting": ROOT / "shooting",
}
ACTIONS = ["kicking", "punching", "shooting"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

# --- cached artifacts ---
INTERIM_DIR = ROOT / "data" / "interim"             # raw MediaPipe Pose landmarks per clip (.npz)
INTERIM_HANDS_DIR = ROOT / "data" / "interim_hands"  # raw MediaPipe Hands landmarks per clip (.npz)
PROCESSED_DIR = ROOT / "data" / "processed"         # preprocessed+resampled feature arrays (.npz)
RUNS_DIR = ROOT / "runs"                            # per-experiment checkpoints + metrics

# --- MediaPipe ---
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
VISIBILITY_THRESHOLD = 0.5  # used as soft-weight floor, not a hard drop (protocol addendum #2)

# --- sequence / features ---
SEQ_LEN = 40
N_POSE_LANDMARKS = 33
SMOOTH_WINDOW = 5

# adjacent-joint pairs for bone-vector stream (MediaPipe pose topology, trimmed to
# limbs + torso -- face landmarks contribute nothing to punch/kick/shoot signal)
BONE_PAIRS = [
    (11, 13), (13, 15),   # left shoulder-elbow-wrist
    (12, 14), (14, 16),   # right shoulder-elbow-wrist
    (23, 25), (25, 27),   # left hip-knee-ankle
    (24, 26), (26, 28),   # right hip-knee-ankle
    (11, 12),             # shoulder-shoulder
    (23, 24),             # hip-hip
    (11, 23), (12, 24),   # shoulder-hip (torso sides)
    (27, 31), (28, 32),   # ankle-foot_index
]

# joint-angle triples (vertex is the middle index) -- same signal that made
# the XGBoost sanity baseline work (hipR_rom carried 42% of its feature
# importance). Fed to the deep model as a continuous per-frame stream instead
# of a single ROM summary, so it doesn't have to rediscover this from raw xyz
# on only 529 training clips (root cause of the punching->shooting gap on
# held-out participant 3, see notebook diagnosis).
ANGLE_TRIPLES = [
    (11, 13, 15),   # left elbow: shoulder-elbow-wrist
    (12, 14, 16),   # right elbow
    (23, 25, 27),   # left knee: hip-knee-ankle
    (24, 26, 28),   # right knee
    (11, 23, 25),   # left hip: shoulder-hip-knee
    (12, 24, 26),   # right hip
]

# --- participant split (locked, section 5 + addendum) ---
TRAIN_PARTICIPANTS = [4, 5, 6, 7, 8, 9]
VAL_PARTICIPANTS = [1]
TEST_PARTICIPANTS = [2, 3]

# --- augmentation (train split only) ---
AUG_MULTIPLIER = 2
ROTATION_JITTER_DEG = 10.0
TIME_WARP_RANGE = (0.85, 1.15)
JOINT_NOISE_STD = 0.01          # meters, small gaussian jitter on world coords
CROP_WINDOW_RANGE = (0.90, 1.00)  # fraction of clip length kept before resample

# --- live inference gating (src/inference.py) ---
# Every training clip was filmed full-body, far enough back to capture full
# kick range of motion (protocol section 1). A webcam showing only
# torso/arms is out-of-distribution input the model never saw -- MediaPipe
# still emits *something* for off-frame leg landmarks (low-confidence,
# jittery guesses), and the leg-angle features (the model's single dominant
# signal -- hipR_rom carried 42% of feature importance) read that jitter as
# a large kick-like motion. Softmax confidence doesn't catch this because a
# model can be confidently wrong on data unlike its training distribution --
# these gates check the input itself before trusting any prediction.
LEG_LANDMARKS = [23, 24, 25, 26, 27, 28]   # hips, knees, ankles
ARM_LANDMARKS = [11, 12, 13, 14, 15, 16]   # shoulders, elbows, wrists
MIN_VISIBILITY_FOR_ACTION = 0.5   # real training clips: min 0.74, mean 0.90 (both limb groups)
MIN_MOTION_FOR_ACTION = 0.05      # real training clips: min peak wrist/ankle speed ~0.02-0.035

# --- motion-triggered segmentation (replaces continuous rolling-window
# reclassification -- that design reclassified an arbitrary sliding window
# every few frames, so a resample_sequence() stretch of a 2-frame natural
# fidget across the whole fixed length made tiny noise look like a
# completed strike arc. Real training clips are each ONE bounded, hand-
# trimmed action; live inference should only ever classify an equivalently
# bounded segment, captured only when real motion actually starts and ends.
ONSET_FRAMES = 2          # consecutive above-threshold frames to confirm a real onset (not 1-frame noise)
SETTLE_FRAMES = 6         # consecutive below-threshold frames = motion has stopped, segment is done
PRE_ROLL_FRAMES = 10      # frames of context kept before onset, so the wind-up isn't lost
MIN_SEGMENT_FRAMES = 10   # shorter than this -> spurious blip, discard without classifying
MAX_SEGMENT_FRAMES = 90   # safety cap
RESULT_HOLD_SECONDS = 1.5 # how long a classification result stays on screen before returning to idle-watch
COOLDOWN_SECONDS = 1.2    # after a segment completes, ignore all motion for this long before watching
                          # for the next onset -- without this, the settle/retraction tail of one action
                          # can immediately seed the next capture and contaminate it (chained actions
                          # misclassifying each other)

# --- training ---
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 60
EARLY_STOP_PATIENCE = 10
SEED = 42
