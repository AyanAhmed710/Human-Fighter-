"""
2s-AGCN (protocol section 7, primary target): two-stream adaptive graph
convolutional network -- joint-position stream + bone-vector stream, fused
by summing logits (paper's score-level fusion). Trained here as one joint
run (both streams' logits supervised together) rather than the paper's
separately-trained-then-fused streams -- simpler single training loop,
standard practice at this scale.

Simplifications vs. the original paper (Shi et al., 2s-AGCN, CVPR 2019),
made deliberately for a 3-class, 20-joint, ~600-clip problem instead of
NTU-RGB+D's 60/120-class, 25-joint, 56k-clip scale:
  - single adaptive-adjacency subset (Ks=1) instead of the paper's 3-way
    inward/outward/identity partition -- one learnable+attention graph on
    top of the fixed physical one is enough capacity here and keeps the
    param count controlled for the small dataset.
  - trained from scratch, not fine-tuned from an NTU-RGB+D pretrained
    checkpoint -- the checkpoint's 25-joint layout doesn't line up with our
    20-joint MediaPipe subset without a lossy remap, and pulls in
    mmcv/mmaction2 (notoriously painful to build on Windows). The graph
    inductive bias itself (fixed skeleton adjacency instead of learning
    topology from raw xyz, as the BiLSTM had to) is the main data-efficiency
    win being tested here.

Each AdaptiveGraphConv combines:
  A -- fixed, normalized physical skeleton adjacency (src.models.graph)
  B -- learnable global offset (same for every clip)
  C -- per-sample data-dependent attention (softmax over a learned
       embedding-similarity of every joint pair, computed fresh per input)
matching the paper's Ck = Ak + Bk + Ck decomposition.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.graph import ELBOW_NODE_IDX, NUM_NODES, build_adjacency


class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A)                      # (V, V), fixed
        self.B = nn.Parameter(torch.zeros_like(A))          # (V, V), learnable
        reduced = max(out_channels // 4, 4)
        self.theta = nn.Conv2d(in_channels, reduced, 1)
        self.phi = nn.Conv2d(in_channels, reduced, 1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.res = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        # x: (B, C, T, V)
        theta = self.theta(x).mean(dim=2)   # (B, C', V) -- pool time before attention
        phi = self.phi(x).mean(dim=2)       # (B, C', V)
        attn = torch.einsum("bcv,bcw->bvw", theta, phi) / (theta.shape[1] ** 0.5)
        attn = F.softmax(attn, dim=-1)      # (B, V, V), data-dependent (Ck)

        adj = self.A.unsqueeze(0) + self.B.unsqueeze(0) + attn   # (B, V, V)
        out = torch.einsum("bctv,bvw->bctw", x, adj)             # graph mix over joints
        out = self.bn(self.conv(out))                            # channel mix
        return F.relu(out + self.res(x))


class TemporalConv(nn.Module):
    def __init__(self, channels: int, stride: int = 1, kernel: int = 9):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size=(kernel, 1),
                               stride=(stride, 1), padding=(pad, 0))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class AGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, stride: int = 1):
        super().__init__()
        self.gcn = AdaptiveGraphConv(in_channels, out_channels, A)
        self.tcn = TemporalConv(out_channels, stride=stride)
        if in_channels == out_channels and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        res = self.res(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return F.relu(x + res)


class AGCNStream(nn.Module):
    """One stream (joint or bone), input (B, T, V, in_channels)."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 num_classes: int = 3, A: torch.Tensor = None, dropout: float = 0.3):
        super().__init__()
        if A is None:
            A = build_adjacency()
        self.blocks = nn.ModuleList([
            AGCNBlock(in_channels, base_channels, A),
            AGCNBlock(base_channels, base_channels, A),
            AGCNBlock(base_channels, base_channels * 2, A, stride=2),
            AGCNBlock(base_channels * 2, base_channels * 2, A),
            AGCNBlock(base_channels * 2, base_channels * 4, A, stride=2),
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        # x: (B, T, V, C) -> (B, C, T, V)
        x = x.permute(0, 3, 1, 2).contiguous()
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=[2, 3])          # global average pool over time + joints
        x = self.dropout(x)
        return self.fc(x)


class HandOpennessStream(nn.Module):
    """v4: reads the 2 hand-curl pseudo-nodes (src.models.graph.
    HAND_PSEUDO_NODES) appended after the 20 physical skeleton nodes -- real
    per-finger curl angles from a dedicated MediaPipe Hands model
    (src.data.hand_features), not Pose's crude wrist-to-fingertip-landmark
    distance (v3). Dropped v3's approach because Pose's own fingertip
    landmarks run ~0.71-0.78 mean visibility and dip as low as 0.06 even on
    the curated recording setup -- too noisy to build the model's most-
    trusted signal on, which is exactly what made shooting unreliable live
    despite strong held-out test numbers. Hands crops in on the hand region
    with its own model instead of guessing at full-body scale.

    Presence-weighted mean+std per finger per hand (2 hands x 5 fingers x 2
    stats = 20-dim) -> small FC head, still skipping the graph conv entirely
    so this signal reaches the loss undiluted, same rationale as v3."""

    N_FINGERS = 5  # thumb/index/middle/ring/pinky -- src.data.hand_features.FINGER_ORDER

    def __init__(self, num_classes: int = 3, hidden: int = 24, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * self.N_FINGERS * 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    @classmethod
    def _curl_stats(cls, hand_nodes: torch.Tensor) -> torch.Tensor:
        # hand_nodes: (B, T, 2, 6) -- channels 0:5 = per-finger curl (0=fully
        # curled, 1=fully extended), channel 5 = presence (soft-confidence,
        # 0 where that hand wasn't detected that frame)
        curl = hand_nodes[..., :cls.N_FINGERS]         # (B, T, 2, 5)
        presence = hand_nodes[..., cls.N_FINGERS:cls.N_FINGERS + 1]  # (B, T, 2, 1)

        w = presence.clamp(min=1e-3)
        w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-3)             # (B, 1, 2, 1)
        mean = (curl * w).sum(dim=1, keepdim=True) / w_sum             # (B, 1, 2, 5)
        var = ((curl - mean) ** 2 * w).sum(dim=1, keepdim=True) / w_sum
        std = var.clamp(min=1e-8).sqrt()

        stats = torch.cat([mean.squeeze(1), std.squeeze(1)], dim=-1)   # (B, 2, 10)
        return stats.reshape(stats.shape[0], -1)                        # (B, 20)

    def forward(self, hand_nodes: torch.Tensor) -> torch.Tensor:
        return self.fc(self._curl_stats(hand_nodes))


class ElbowStream(nn.Module):
    """v5 addition: explicit hint stream for starting elbow angle -- the
    session's strongest single separator found so far (effect size 2.27 on
    the full 642-clip dataset, vs hand-openness's 1.2-1.3): punching starts
    from a bent guard elbow (~105 deg, std 30), shooting starts from an
    already near-extended arm (~149 deg, std 8), measured as the mean angle
    over the first few frames -- before any wind-up/extension has happened.

    Unlike HandOpennessStream, this is built entirely from shoulder/elbow/
    wrist -- Pose's best-tracked joints, not the fragile fingertip landmarks
    -- so it should be more robust live, not just a bigger number on the
    locked test set. Computed directly from the existing physical-node xyz
    channels (src.models.graph.ELBOW_NODE_IDX) -- no new pseudo-nodes, no
    cache/extraction changes needed, unlike the v4 hand-curl addition."""

    def __init__(self, num_classes: int = 3, hidden: int = 16, dropout: float = 0.3,
                 start_frames: int = 3):
        super().__init__()
        self.start_frames = start_frames
        self.fc = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    @staticmethod
    def _elbow_angle(shoulder: torch.Tensor, elbow: torch.Tensor, wrist: torch.Tensor) -> torch.Tensor:
        # each: (B, T, 3)
        v1 = shoulder - elbow
        v2 = wrist - elbow
        cos_ang = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-8)
        return torch.acos(cos_ang.clamp(-1.0, 1.0))  # (B, T) radians

    def forward(self, body_xyz: torch.Tensor) -> torch.Tensor:
        # body_xyz: (B, T, NUM_NODES, 3) -- first NUM_NODES physical nodes only
        feats = []
        for side in ("L", "R"):
            idx = ELBOW_NODE_IDX[side]
            shoulder = body_xyz[:, :, idx["shoulder"], :]
            elbow = body_xyz[:, :, idx["elbow"], :]
            wrist = body_xyz[:, :, idx["wrist"], :]
            angle = self._elbow_angle(shoulder, elbow, wrist)            # (B, T)
            feats.append(angle[:, :self.start_frames].mean(dim=1))       # (B,)
        x = torch.stack(feats, dim=-1)  # (B, 2) -- [L_start_angle, R_start_angle], radians
        return self.fc(x)


class TwoStreamAGCN(nn.Module):
    """Input: (B, SEQ_LEN, TOTAL_PACKED_NODES, 6) from
    src.models.graph.joint_bone_from_sample(sample, hand_curl, hand_presence)
    -- first NUM_NODES=20 rows are the physical skeleton (channels 0:3 joint
    xyz, 3:6 bone-to-parent vector), last HAND_PSEUDO_NODES=2 rows are the
    per-hand curl signal (channels 0:5 per-finger curl, 5 presence) -- see
    src.models.graph module docstring. Returns (logits, aux) to match
    src.train.run_epoch's `logits, _ = model(feat)` contract (same as
    BiLSTMBaseline), so src/train.py's train_baseline works unmodified.

    Four-way vote (joint + bone + hand + elbow), summed with learnable
    per-stream weights. v3 initialized hand_w elevated (2.0) to force the
    model to trust it -- in hindsight a risky move: it worked on the locked
    test set (97.96%) but that same concentrated trust in a fragile sensor
    (fingertip landmarks) is exactly what made it unreliable live. v5
    deliberately does NOT repeat that: elbow_w starts at the same 1.0 as
    joint_w/bone_w despite ElbowStream's feature having the strongest raw
    effect size found this session -- let the optimizer earn the weighting
    from validation performance instead of presetting another concentrated
    bet, now that we've seen where that pattern leads.

    v4: HandOpennessStream now reads real MediaPipe Hands curl angles
    (src.data.hand_features) instead of v3's Pose-fingertip-distance proxy.
    v5: added ElbowStream. NOTE both changed the state_dict -- v3/v4
    checkpoints are NOT compatible with this version; retrain."""

    def __init__(self, base_channels: int = 32, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        A = build_adjacency()
        self.joint_stream = AGCNStream(3, base_channels, num_classes, A, dropout)
        self.bone_stream = AGCNStream(3, base_channels, num_classes, A, dropout)
        self.hand_stream = HandOpennessStream(num_classes, dropout=dropout)
        self.elbow_stream = ElbowStream(num_classes, dropout=dropout)
        self.joint_w = nn.Parameter(torch.tensor(1.0))
        self.bone_w = nn.Parameter(torch.tensor(1.0))
        self.hand_w = nn.Parameter(torch.tensor(2.0))
        self.elbow_w = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        body, hand = x[:, :, :NUM_NODES, :], x[:, :, NUM_NODES:, :]
        joint_logits = self.joint_stream(body[..., :3])
        bone_logits = self.bone_stream(body[..., 3:])
        hand_logits = self.hand_stream(hand)
        elbow_logits = self.elbow_stream(body[..., :3])
        logits = (self.joint_w * joint_logits + self.bone_w * bone_logits
                  + self.hand_w * hand_logits + self.elbow_w * elbow_logits)
        return logits, {
            "joint_logits": joint_logits,
            "bone_logits": bone_logits,
            "hand_logits": hand_logits,
            "elbow_logits": elbow_logits,
            "stream_weights": (self.joint_w.item(), self.bone_w.item(),
                                self.hand_w.item(), self.elbow_w.item()),
        }
