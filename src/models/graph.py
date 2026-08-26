"""
Skeleton graph definition for the 2s-AGCN model (protocol section 7, primary
target).

v2: extended with fingertip nodes (17-22) after the first pass (elbow/knee/
hip skeleton only, 14 joints) landed on the same punching<->shooting
confusion as the BiLSTM -- both models were missing the one signal that
actually separates punch from shoot (fist vs open-palm shape), the same
signal that had to be bolted onto the XGBoost baseline as a hand-openness
feature to fix its identical confusion. A body-skeleton-only graph structurally
cannot represent that; fingertip landmarks can.

20 nodes: the original 14-joint set (config.BONE_PAIRS / ANGLE_TRIPLES node
set -- shoulders/elbows/wrists/hips/knees/ankles/foot-indices) + 6 fingertip
tips (L/R pinky, index, thumb). Face and other unused MediaPipe landmarks are
still excluded -- they carry no punch/kick/shoot signal.

Edges are defined here directly (not reused from config.BONE_PAIRS) since
BONE_PAIRS doesn't include the new fingertip connections and bone vectors are
computed straight from raw (T, 33, 3) xyz rather than the precomputed 14-edge
`bones` stream -- simpler and correct regardless of BONE_PAIRS' own ordering.
"""
import numpy as np
import torch

L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22

GCN_JOINTS = [
    11, 12, 13, 14, 15, 16,          # shoulders, elbows, wrists
    L_PINKY, R_PINKY, L_INDEX, R_INDEX, L_THUMB, R_THUMB,  # fingertips
    23, 24, 25, 26, 27, 28,          # hips, knees, ankles
    31, 32,                          # foot indices
]
_JOINT_POS = {j: i for i, j in enumerate(GCN_JOINTS)}
NUM_NODES = len(GCN_JOINTS)  # 20

ROOT_JOINT = 11

# Physical adjacency graph (used for spatial GCN conv -- general graph, not
# required to be a tree): the original 14-joint skeleton chain + a fingertip
# edge from each tip to its wrist.
GCN_EDGES = [
    (11, 13), (13, 15),   # left shoulder-elbow-wrist
    (12, 14), (14, 16),   # right shoulder-elbow-wrist
    (23, 25), (25, 27),   # left hip-knee-ankle
    (24, 26), (26, 28),   # right hip-knee-ankle
    (11, 12),              # shoulder-shoulder
    (23, 24),              # hip-hip
    (11, 23), (12, 24),   # shoulder-hip (torso sides)
    (27, 31), (28, 32),   # ankle-foot_index
    (L_WRIST, L_PINKY), (L_WRIST, L_INDEX), (L_WRIST, L_THUMB),
    (R_WRIST, R_PINKY), (R_WRIST, R_INDEX), (R_WRIST, R_THUMB),
]

# Strict spanning tree for the bone stream (one parent per non-root node) --
# same as GCN_EDGES but drops the redundant torso diagonal (12, 24), root =
# ROOT_JOINT (left shoulder, bone vector zero). 19 edges over 20 nodes.
_BONE_TREE_EDGES = [e for e in GCN_EDGES if e != (12, 24)]


# Node-array positions for src.models.agcn.ElbowStream -- shoulder/elbow/
# wrist per side, all physical (non-pseudo) nodes already in the 20-node
# graph, no new nodes needed. Session's strongest single separator (effect
# size 2.27 vs hand-openness's 1.2-1.3): starting elbow angle differs
# sharply between a bent guard pose (punch) and an already-extended arm
# (shoot) -- and unlike hand-openness, it's built entirely from Pose's
# best-tracked joints, not the fragile fingertip landmarks.
ELBOW_NODE_IDX = {
    "L": {"shoulder": _JOINT_POS[11], "elbow": _JOINT_POS[13], "wrist": _JOINT_POS[15]},
    "R": {"shoulder": _JOINT_POS[12], "elbow": _JOINT_POS[14], "wrist": _JOINT_POS[16]},
}

# v4: 2 extra "pseudo-nodes" appended after the 20 physical skeleton nodes --
# NOT part of the physical adjacency (build_adjacency() below stays sized
# NUM_NODES x NUM_NODES, graph conv never sees these), just a convenient way
# to piggyback the src.data.hand_features curl-angle signal onto the same
# packed tensor without a second dataset/model-input plumbing path. Index
# NUM_NODES = Left-hand curl node, NUM_NODES+1 = Right-hand curl node.
# src.models.agcn.HandOpennessStream reads only these two; joint_stream/
# bone_stream read only the first NUM_NODES and never see them.
HAND_PSEUDO_NODES = 2
TOTAL_PACKED_NODES = NUM_NODES + HAND_PSEUDO_NODES  # 22


def build_adjacency() -> torch.Tensor:
    """Symmetric, self-loop, degree-normalized adjacency (NUM_NODES,
    NUM_NODES) from GCN_EDGES -- the fixed physical part of the AGCN's
    adaptive adjacency (Ak in the paper)."""
    A = np.eye(NUM_NODES, dtype=np.float32)
    for a, b in GCN_EDGES:
        i, j = _JOINT_POS[a], _JOINT_POS[b]
        A[i, j] = 1.0
        A[j, i] = 1.0
    deg = A.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
    A_norm = d_inv_sqrt @ A @ d_inv_sqrt
    return torch.from_numpy(A_norm.astype(np.float32))


def joint_bone_from_sample(sample: dict, hand_curl: np.ndarray = None,
                            hand_presence: np.ndarray = None) -> np.ndarray:
    """sample: src.data.preprocess.preprocess_clip() output. Returns
    (SEQ_LEN, NUM_NODES, 6) -- xyz (channels 0:3) + bone-to-parent vector
    (channels 3:6) per node, restricted to GCN_JOINTS. Root joint's bone
    channels are zero. Bone vectors computed directly from sample["xyz"]
    (full 33-landmark stream), not from the precomputed BONE_PAIRS-only
    `bones` array -- needed since fingertip edges aren't in BONE_PAIRS.

    hand_curl (SEQ_LEN, 2, 5) + hand_presence (SEQ_LEN, 2), both from
    src.data.hand_features.finger_curl_series (already cropped/resampled to
    SEQ_LEN by the caller) -- when given, appends 2 pseudo-nodes (see
    HAND_PSEUDO_NODES) so the result is (SEQ_LEN, TOTAL_PACKED_NODES, 6):
    5 curl channels + 1 presence channel per hand, exactly filling the same
    6-channel width as the physical nodes' xyz+bone. When omitted, returns
    the plain (SEQ_LEN, NUM_NODES, 6) physical-skeleton-only tensor (backward
    compatible with any caller not using the hand signal)."""
    xyz = sample["xyz"]       # (T, 33, 3)
    T = xyz.shape[0]

    joint = xyz[:, GCN_JOINTS, :]                       # (T, NUM_NODES, 3)
    bone = np.zeros((T, NUM_NODES, 3), dtype=np.float32)
    for parent, child in _BONE_TREE_EDGES:
        bone[:, _JOINT_POS[child]] = xyz[:, child] - xyz[:, parent]

    base = np.concatenate([joint, bone], axis=-1).astype(np.float32)  # (T, NUM_NODES, 6)
    if hand_curl is None:
        return base

    assert hand_curl.shape[0] == T, f"hand_curl T={hand_curl.shape[0]} != sample T={T}"
    hand_nodes = np.concatenate(
        [hand_curl, hand_presence[..., None]], axis=-1
    ).astype(np.float32)  # (T, 2, 6)
    return np.concatenate([base, hand_nodes], axis=1)  # (T, TOTAL_PACKED_NODES, 6)
