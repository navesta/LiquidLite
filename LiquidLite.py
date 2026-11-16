#!/usr/bin/python3 -Xfrozen_modules=off

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function

import glob
import logging
import os
import signal
import sys
import multiprocessing
import queue
import threading
import time
import warnings
import argparse

warnings.filterwarnings("ignore", message=".*pipelines sequentially.*")

# Line-buffer stdout/stderr so progress logs flush promptly when the script is run
# in the background or piped (where Python otherwise block-buffers, hiding logs).
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:   # pre-3.7 streams without reconfigure()
    pass

# Must be set before torch is imported
_compile_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_compile")
os.makedirs(_compile_cache, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _compile_cache)
os.environ.setdefault("MIOPEN_USER_DB_PATH", "/tmp/.miopen")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
multiprocessing.set_start_method('fork')

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import pipeline
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Default output FPS used when a clip doesn't report one (software decode only).
DEFAULT_FPS = 24.0

ML_TASK            = "object-detection"
ML_MODEL           = "hustvl/yolos-tiny"
ML_BATCH_SIZE      = 8
OPT_COMPILE        = False
OPT_INFERENCE_MODE = True
# Pipeline queue depths
PIPELINE_QUEUE_DEPTH = 16

#Check if OpenCV is installed and if OpenCL is available for UMat (GPU) acceleration. (Used for YUV <-> RGB conversion)
try:
    import cv2
    cv2.ocl.setUseOpenCL(True)
    _CV2_UMAT = cv2.ocl.haveOpenCL() and cv2.ocl.useOpenCL()
    if _CV2_UMAT:
        _cv2_device = cv2.ocl.Device.getDefault()
        print(f"OpenCV UMat (GPU): enabled — {_cv2_device.name()} "
              f"({'GPU' if _cv2_device.type() == cv2.ocl.Device_TYPE_GPU else 'other'})")
    else:
        print(f"OpenCV UMat (GPU): disabled "
              f"(haveOpenCL={cv2.ocl.haveOpenCL()}, useOpenCL={cv2.ocl.useOpenCL()})")
except ImportError:
    cv2 = None
    _CV2_UMAT = False
    print("OpenCV UMat (GPU): disabled (cv2 not installed)")

# ---------------------------------------------------------------------------
# Liquid NPC tracker configuration
# ---------------------------------------------------------------------------
LIQUID_UNITS = 512   # total CfC neurons
LIQUID_MOTOR = 256   # AutoNCP motor (output) neurons; must be < LIQUID_UNITS
PROJ_DIM     = 128   # feature projection width feeding the CfC

LIQUID_UNITS = 128   # total CfC neurons
LIQUID_MOTOR = 64   # AutoNCP motor (output) neurons; must be < LIQUID_UNITS
PROJ_DIM     = 64   # feature projection width feeding the CfC
# Per-detection feature vector: [x1, y1, x2, y2, score].
LIQUID_INPUT_DIM = 5
# Scalar weight on the confidence loss term relative to the box term, in raw MSE
# magnitude.
CONF_WEIGHT = 0.1
# Extra dropped-frame loss slope by offset: dropped offset k (1..drop_n)
# gets factor (1 + OFFSET_WEIGHT * (k-1)). Set 0.0 to disable.
OFFSET_WEIGHT = 0.2
# Raw per-detection score cutoff handed to the HF pipeline.
YOLO_CONF_THRESHOLD  = 0.2
# Gate on the box/confidence that actually gets drawn.
CONFIDENCE_THRESHOLD = 0.01
MAX_TRACKED_OBJECTS  = 1

# --- IoU track association (shared by training and inference) ---
IOU_MATCH_GATE   = 0.01   # min IoU to link a detection to an existing track
TRACK_MAX_MISSES = 48   # retire a track after this many frames unseen

# --- Training ---
CHECKPOINT_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "liquid_tracker.pt")
TRAIN_EPOCHS     = 100
TRAIN_LR         = 1e-3
TRAIN_SEQ_LEN    = 32
TRAIN_BATCH      = 64
drop_n_DEFAULT = 5
keep_m_DEFAULT = 1
TRAIN_MIN_TRACK  = 8     # ignore tracks shorter than this many frames
CKPT_SAVE_EVERY  = 5     # overwrite the checkpoint every N epochs during training
# Scheduled sampling for the autoregressive rollout: probability of feeding the
# model's OWN prediction (vs. the real feature) on a dropped frame. Annealed from
# START to END over the cumulative training epochs — early epochs ~teacher-forced,
# late epochs ~free-running (matching inference, where only predictions are fed).
SS_START_PROB = 0.0
SS_END_PROB   = 1.0
SS_END_PROB   = 0.7 #Lower values means less accumulated errors.

# ---------------------------------------------------------------------------
# 1. DEFINE THE HYBRID LIQUID GENERAL DETECTOR
# ---------------------------------------------------------------------------
class LiquidGeneralDetector(nn.Module):
    def __init__(self):
        super(LiquidGeneralDetector, self).__init__()

        print(f"Loading ML model: {ML_MODEL}")
        self.spatial_detector = pipeline(
            task=ML_TASK,
            model=ML_MODEL,
            dtype=torch.float16,
            use_fast=True,
            batch_size=ML_BATCH_SIZE,
            device=0 if torch.cuda.is_available() else -1,
        )
        if OPT_COMPILE:
            self.spatial_detector.model = torch.compile(
                self.spatial_detector.model, backend="inductor", mode="reduce-overhead")

        self.classes = self.spatial_detector.model.config.id2label
        self.label2id = {name: cid for cid, name in self.classes.items()}

        # Per-detection feature vector: [x1, y1, x2, y2, score] (class_id excluded)
        self.input_dim = LIQUID_INPUT_DIM

        # Trainable liquid stack: project features → AutoNCP-wired CfC → heads.
        self.feature_projection = nn.Linear(self.input_dim, PROJ_DIM)
        # Adaptive Neural Circuit Policy wiring: LIQUID_UNITS total neurons with
        # LIQUID_MOTOR motor (output) neurons. The CfC output width == LIQUID_MOTOR.
        wiring = AutoNCP(LIQUID_UNITS, LIQUID_MOTOR)
        self.liquid_layer = CfC(PROJ_DIM, wiring, batch_first=True)

        self.bbox_head = nn.Linear(LIQUID_MOTOR, 4)
        self.conf_head = nn.Linear(LIQUID_MOTOR, 1)
        # Zero-init the box head so the initial delta is exactly 0: the model then
        # STARTS at the hold-last baseline (box == anchor) and training only has to
        # learn small corrections.
        nn.init.zeros_(self.bbox_head.weight)
        nn.init.zeros_(self.bbox_head.bias)

    def forward(self, current_detections, hx=None, anchor_box=None):
        # current_detections shape: (batch, seq_len, LIQUID_INPUT_DIM=5)
        # features = [x1, y1, x2, y2, score]
        # anchor_box: (batch, seq_len, 4) — the last-keyframe box the residual is
        # added to. On dropped frames the input box is zeroed (missing marker), so
        # the anchor is threaded separately rather than read from current_detections.
        proj = self.feature_projection(current_detections)
        # Pass projected features into the continuous-time ODE layer.
        liquid_out, hx = self.liquid_layer(proj, hx)

        # Residual box prediction: box = anchor + delta (delta=0 reproduces hold-last).
        delta = self.bbox_head(liquid_out)
        smoothed_box  = delta + anchor_box if anchor_box is not None else delta
        smoothed_conf = torch.sigmoid(self.conf_head(liquid_out))

        return smoothed_box, smoothed_conf, hx

    def liquid_parameters(self):
        """Trainable parameters of the liquid stack only (HF detector excluded)."""
        return [p for p in self.parameters() if p.requires_grad]

    def liquid_state_dict(self):
        """State dict of just the liquid submodules, for checkpointing."""
        sd = {}
        for name in ("feature_projection", "liquid_layer", "bbox_head", "conf_head"):
            for k, v in getattr(self, name).state_dict().items():
                sd[f"{name}.{k}"] = v
        return sd

    def load_liquid_state_dict(self, sd):
        """Load a checkpoint produced by liquid_state_dict()."""
        from collections import defaultdict
        buckets = defaultdict(dict)
        for full_k, v in sd.items():
            mod, _, sub = full_k.partition(".")
            buckets[mod][sub] = v
        for name, params in buckets.items():
            getattr(self, name).load_state_dict(params)


# ---------------------------------------------------------------------------
# IoU association (shared by training track-building and inference)
# ---------------------------------------------------------------------------
def _iou(a, b):
    """IoU of two [x1,y1,x2,y2] boxes (any consistent coordinate space)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0

def _greedy_match(prevBoxes, detBoxes, gate=IOU_MATCH_GATE):
    """To allow Liquid layer to follow a particular object in a scene. Greedily match detBoxes to prevBoxes by descending IoU.

    prevBoxes / detBoxes: lists of [x1,y1,x2,y2].
    Returns dict {det_index: prev_index} for matches above `gate`.
    """
    pairs = []
    for di, db in enumerate(detBoxes):
        for pi, pb in enumerate(prevBoxes):
            iou = _iou(db, pb)
            if iou >= gate:
                pairs.append((iou, di, pi))
    pairs.sort(reverse=True)
    matched_det, matched_prev, out = set(), set(), {}
    for iou, di, pi in pairs:
        if di in matched_det or pi in matched_prev:
            continue
        matched_det.add(di)
        matched_prev.add(pi)
        out[di] = pi
    return out

# ---------------------------------------------------------------------------
# Training (liquid layer only; HF detector frozen)
# ---------------------------------------------------------------------------
# Training decodes in SOFTWARE (cv2.VideoCapture),
def decode_frames_sw(ClipPath):
    """Yield BGR frames from a clip using OpenCV's software decoder."""
    cap = cv2.VideoCapture(ClipPath)
    if not cap.isOpened():
        print(f"  WARNING: cv2 could not open {ClipPath}")
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()

def collect_detections(detector, ClipPath):
    """Run the frozen HF detector over every training frame of one clip.
    """
    frames_dets = []
    for frame in decode_frames_sw(ClipPath):
        orig_h, orig_w = frame.shape[:2]
        img_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            preds = detector.spatial_detector(img_rgb, threshold=YOLO_CONF_THRESHOLD)
        dets = []
        for p in (preds or []):
            b = p["box"]
            dets.append({
                'box': [b["xmin"] / orig_w, b["ymin"] / orig_h,
                        b["xmax"] / orig_w, b["ymax"] / orig_h],
                'score': float(p["score"]),
                'class_id': detector.label2id.get(p["label"], 0),
            })
        frames_dets.append(dets)
        if len(frames_dets) % 100 == 0:
            print(f"  collected detections for {len(frames_dets)} frames")
    print(f"Collected detections for {len(frames_dets)} frames total")
    return frames_dets

def build_tracks(frames_dets):
    """Link per-frame detections into per-object temporal sequences via IoU.

    Returns a list of tracks; each track is a list of 5-float feature vectors
    [x1,y1,x2,y2,score] in temporal order.
    """
    active = []       # list of {'feat':[...], 'seq':[...], 'misses':int}
    completed = []

    def _retire(tr):
        if len(tr['seq']) >= TRAIN_MIN_TRACK:
            completed.append(tr['seq'])

    for dets in frames_dets:
        prev_boxes = [tr['feat'][:4] for tr in active]
        det_boxes  = [d['box'] for d in dets]
        match = _greedy_match(prev_boxes, det_boxes) if prev_boxes else {}

        matched_prev = set()
        new_tracks = []
        for di, d in enumerate(dets):
            feat = d['box'] + [d['score']]   # [x1,y1,x2,y2,score]; class_id excluded
            if di in match:
                tr = active[match[di]]
                tr['feat'] = feat
                tr['seq'].append(feat)
                tr['misses'] = 0
                matched_prev.add(match[di])
            else:
                new_tracks.append({'feat': feat, 'seq': [feat], 'misses': 0})

        # Age unmatched existing tracks; retire those unseen too long.
        survivors = []
        for pi, tr in enumerate(active):
            if pi in matched_prev:
                survivors.append(tr)
                continue
            tr['misses'] += 1
            if tr['misses'] > TRACK_MAX_MISSES:
                _retire(tr)
            else:
                survivors.append(tr)
        active = survivors + new_tracks

    for tr in active:
        _retire(tr)
    print(f"Built {len(completed)} tracks (>= {TRAIN_MIN_TRACK} frames)")
    return completed

def decimate_tracks(tracks, drop_n, keep_m=keep_m_DEFAULT):
    """Split tracks into windows using a keep_m + drop_n segment pattern.
    """
    if keep_m < 1:
        raise ValueError(f"keep_m must be >= 1 (got {keep_m})")
    seg_len = keep_m + drop_n
    if drop_n < 1:
        raise ValueError(f"drop_n must be >= 1 (got {drop_n})")

    inputs, targets, masks, dropped = [], [], [], 0
    for seq in tracks:
        usable = (len(seq) // seg_len) * seg_len
        if usable < TRAIN_MIN_TRACK:
            continue
        seq = seq[:usable]
        for start in range(0, len(seq), TRAIN_SEQ_LEN):
            window = seq[start:start + TRAIN_SEQ_LEN]
            if len(window) < TRAIN_MIN_TRACK:
                continue
            orig_len = len(window)
            if len(window) < TRAIN_SEQ_LEN:
                pad = TRAIN_SEQ_LEN - len(window)
                tail = window[-1:]
                window = list(window) + [tail[0] for _ in range(pad)]
            tgt = np.array(window, dtype=np.float32)
            inp = tgt.copy()
            msk = np.zeros((TRAIN_SEQ_LEN, 2), dtype=np.float32)
            for i in range(TRAIN_SEQ_LEN):
                # Padded tail is invalid for both loss and metrics.
                if i >= orig_len:
                    msk[i, 0] = 0.0
                    msk[i, 1] = 0.0
                    continue
                seg_pos = i % seg_len
                msk[i, 0] = 1.0
                if seg_pos >= keep_m:
                    inp[i, 0:5] = 0.0
                    msk[i, 1] = 1.0
                    dropped += 1
            inputs.append(inp)
            targets.append(tgt)
            masks.append(msk)
    if not inputs:
        return (np.empty((0, TRAIN_SEQ_LEN, LIQUID_INPUT_DIM), np.float32),
                np.empty((0, TRAIN_SEQ_LEN, LIQUID_INPUT_DIM), np.float32),
                np.empty((0, TRAIN_SEQ_LEN, 2), np.float32), 0)
    print(f"Prepared {len(inputs)} training windows; {dropped} steps decimated "
          f"(keep_m={keep_m}, drop_n={drop_n})")
    return np.stack(inputs), np.stack(targets), np.stack(masks), dropped

def _iou_batch(a, b):
    """Vectorized IoU for two arrays of [x1,y1,x2,y2] boxes, broadcasting on (...,4)."""
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    iw = np.clip(ix2 - ix1, 0.0, None)
    ih = np.clip(iy2 - iy1, 0.0, None)
    inter = iw * ih
    area_a = np.clip(a[..., 2] - a[..., 0], 0.0, None) * np.clip(a[..., 3] - a[..., 1], 0.0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0.0, None) * np.clip(b[..., 3] - b[..., 1], 0.0, None)
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-9), 0.0)

def rollout(detector, inp, keep, use_pred_prob, drop_n, keep_m=keep_m_DEFAULT):
    """Autoregressive rollout of the liquid layer over a window batch.

    inp      : (B, T, 5) feature windows [x1,y1,x2,y2,score]. Keyframe columns hold
               the real detection; dropped columns are ignored past t=0 (their inputs
               come from the model's own predictions instead).
    keep     : (B, T) float/bool — 1 where the frame is a keyframe (teacher-forced),
               0 where it is a dropped frame the liquid layer must fill.
    drop_n : number of dropped frames per segment.
    keep_m : number of kept detector frames per segment.
    use_pred_prob : probability of feeding the model's OWN prediction (vs. the real
               feature) on a *dropped* frame — scheduled sampling. Keyframes are
               always teacher-forced regardless. 1.0 = fully free-running.

    The CfC hidden state is threaded step-by-step and the fed-back prediction is NOT
    detached, so gradients flow through the rollout (the point of the change).
    Returns (pred_box (B,T,4), pred_conf (B,T,1)).
    """
    B, T, _ = inp.shape
    hx = None
    cur = inp[:, 0:1, :]                       # (B,1,5): step 0 = real feat[n]
    # Anchor for each step = box at its most-recent keyframe (never zeroed).
    # Segment pattern is [keep_m keyframes][drop_n dropped].
    seg_len = keep_m + drop_n
    t_idx = torch.arange(T, device=inp.device)
    seg_start = (t_idx // seg_len) * seg_len
    seg_pos = t_idx % seg_len
    kf_idx = seg_start + torch.minimum(seg_pos, torch.full_like(seg_pos, keep_m - 1))
    anchor = inp[:, kf_idx, 0:4]               # (B, T, 4)
    preds_box, preds_conf = [], []
    for t in range(T):
        box_t, conf_t, hx = detector(cur, hx, anchor_box=anchor[:, t:t + 1, :])
        preds_box.append(box_t)
        preds_conf.append(conf_t)
        if t + 1 < T:
            real_next = inp[:, t + 1:t + 2, :]            # (B,1,5)
            # Keep score channel teacher/fixed (do not feed predicted confidence).
            pred_feat = torch.cat([box_t, cur[:, :, 4:5]], dim=2)  # (B,1,5) [box,score]
            kf = keep[:, t + 1].view(B, 1, 1)             # 1 => keyframe
            if use_pred_prob <= 0.0:
                use_pred = torch.zeros(B, 1, 1, device=inp.device)
            elif use_pred_prob >= 1.0:
                use_pred = torch.ones(B, 1, 1, device=inp.device)
            else:
                use_pred = (torch.rand(B, 1, 1, device=inp.device)
                            < use_pred_prob).to(inp.dtype)
            # On keyframes always feed the real detection; on dropped frames feed the
            # prediction with probability use_pred_prob, else the real feature.
            feed_pred = (1.0 - kf) * use_pred              # (B,1,1)
            cur = feed_pred * pred_feat + (1.0 - feed_pred) * real_next
    pred_box  = torch.cat(preds_box, dim=1)    # (B,T,4)
    pred_conf = torch.cat(preds_conf, dim=1)   # (B,T,1)
    return pred_box, pred_conf

@torch.no_grad()
def compute_metrics(detector, inputs, targets, masks, drop_n, device,
                    keep_m=keep_m_DEFAULT):
    """Evaluate the liquid layer on dropped (liquid-filled) frames only.

    All boxes are in normalized [0,1] coordinates. Scores are reported over the set
    of dropped, non-padded frames (mask valid & dropped). Returns a dict:
      recovery@0.5 / recovery@0.75 : fraction of dropped frames with IoU >= thresh
      mean_iou_liquid              : mean IoU of liquid prediction vs detector box
      mean_iou_holdlast            : mean IoU of the hold-last baseline (reuse the
                                     most recent keyframe box) vs detector box
      center_err / size_err        : mean L2 of box-center / (w,h) error
      iou_by_offset                : mean liquid IoU indexed by gap offset 1..drop_n
    """
    was_training = detector.training
    detector.eval()
    T = inputs.shape[1]

    # Liquid predictions via free-running autoregressive rollout (use_pred_prob=1.0)
    # — the same way inference feeds back predictions — so the reported IoU reflects
    # real deployment behavior rather than a teacher-forced best case. Batched to
    # bound memory. keep = keyframe mask (valid AND not dropped).
    inp_t  = torch.from_numpy(inputs).to(device)
    keep_t = torch.from_numpy(
        (masks[:, :, 0] * (1.0 - masks[:, :, 1])).astype(np.float32)).to(device)
    pred_chunks = []
    for b in range(0, inp_t.shape[0], TRAIN_BATCH):
        pbox, _ = rollout(detector, inp_t[b:b + TRAIN_BATCH],
                          keep_t[b:b + TRAIN_BATCH], use_pred_prob=1.0,
                          drop_n=drop_n, keep_m=keep_m)
        pred_chunks.append(pbox.cpu().numpy())
    pred_box = np.concatenate(pred_chunks, axis=0) if pred_chunks \
        else np.empty((0, T, 4), np.float32)          # (N, T, 4)

    tgt_box = targets[:, :, 0:4]                        # (N, T, 4)

    # Hold-last baseline: frame i reuses the most-recent keyframe in each
    # [keep_m kept][drop_n dropped] segment.
    seg_len = keep_m + drop_n
    t_idx = np.arange(T)
    seg_start = (t_idx // seg_len) * seg_len
    seg_pos = t_idx % seg_len
    kf_idx = seg_start + np.minimum(seg_pos, keep_m - 1)
    hold_box = tgt_box[:, kf_idx, :]                    # (N, T, 4)
    anchor_box = inputs[:, kf_idx, 0:4]                 # (N, T, 4)

    # Restrict to dropped AND valid (non-pad) frames.
    sel = (masks[:, :, 0] * masks[:, :, 1]).astype(bool)  # (N, T)
    if not sel.any():
        if was_training:
            detector.train()
        return {'recovery@0.5': 0.0, 'recovery@0.75': 0.0,
                'mean_iou_liquid': 0.0, 'mean_iou_holdlast': 0.0,
                'mean_iou_teacher': 0.0,
                'center_err': 0.0, 'size_err': 0.0,
                'iou_by_offset': np.zeros(max(1, drop_n), np.float32),
                'iou_by_offset_hold': np.zeros(max(1, drop_n), np.float32),
                'iou_by_offset_teacher': np.zeros(max(1, drop_n), np.float32)}

    iou_liquid  = _iou_batch(pred_box, tgt_box)[sel]     # (M,)
    iou_hold    = _iou_batch(hold_box, tgt_box)[sel]     # (M,)
    iou_teacher = _iou_batch(tgt_box, tgt_box)[sel]      # (M,)

    p, t = pred_box[sel], tgt_box[sel]                  # (M, 4)
    p_cx, p_cy = (p[:, 0] + p[:, 2]) / 2, (p[:, 1] + p[:, 3]) / 2
    t_cx, t_cy = (t[:, 0] + t[:, 2]) / 2, (t[:, 1] + t[:, 3]) / 2
    center_err = np.sqrt((p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2)
    p_w, p_h = p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]
    t_w, t_h = t[:, 2] - t[:, 0], t[:, 3] - t[:, 1]
    size_err = np.sqrt((p_w - t_w) ** 2 + (p_h - t_h) ** 2)

    # Gap offset per selected frame: distance from its last keyframe (1..drop_n),
    # using the [keep_m kept][drop_n dropped] segment pattern.
    seg_len = keep_m + drop_n
    seg_pos = np.arange(T) % seg_len
    offset_grid = (seg_pos - keep_m + 1)[None, :].repeat(sel.shape[0], axis=0)
    offsets = offset_grid[sel]
    iou_by_offset         = np.zeros(max(1, drop_n), np.float32)
    iou_by_offset_hold    = np.zeros(max(1, drop_n), np.float32)
    iou_by_offset_teacher = np.zeros(max(1, drop_n), np.float32)
    cnt_by_offset         = np.zeros(max(1, drop_n), np.int64)
    for off in range(1, drop_n + 1):
        m = offsets == off
        if m.any():
            cnt_by_offset[off - 1] = int(m.sum())
            iou_by_offset[off - 1]         = float(iou_liquid[m].mean())
            iou_by_offset_hold[off - 1]    = float(iou_hold[m].mean())
            iou_by_offset_teacher[off - 1] = float(iou_teacher[m].mean())

    if os.environ.get("LIQUID_DEBUG") == "1":
        # Sanity check: rollout anchors should equal hold-last keyframe boxes.
        anchor_diff = np.abs(anchor_box - hold_box)
        anchor_max = float(anchor_diff.max()) if anchor_diff.size else 0.0
        print(f"[metrics] holdlast-anchor max|inp_kf-tgt_kf|={anchor_max:.3e}")

        rows = []
        for off in range(1, drop_n + 1):
            i = off - 1
            rows.append(
                f"o{off}:n={cnt_by_offset[i]} T={iou_by_offset_teacher[i]:.3f} L={iou_by_offset[i]:.3f} H={iou_by_offset_hold[i]:.3f}")
        print("[metrics] per-offset " + " | ".join(rows))

    if os.environ.get("LIQUID_STRICT_HOLDLAST") == "1":
        if not np.allclose(anchor_box, hold_box, atol=1e-7, rtol=0.0):
            raise RuntimeError("hold-last consistency check failed: anchors differ from keyframe boxes")

    if was_training:
        detector.train()
    return {
        'recovery@0.5':  float((iou_liquid >= 0.5).mean()),
        'recovery@0.75': float((iou_liquid >= 0.75).mean()),
        'mean_iou_liquid':   float(iou_liquid.mean()),
        'mean_iou_holdlast': float(iou_hold.mean()),
        'mean_iou_teacher':  float(iou_teacher.mean()),
        'center_err': float(center_err.mean()),
        'size_err':   float(size_err.mean()),
        'iou_by_offset': iou_by_offset,
        'iou_by_offset_hold': iou_by_offset_hold,
        'iou_by_offset_teacher': iou_by_offset_teacher,
    }

def save_checkpoint(detector, drop_n, keep_m, history, optim_state, ckpt_file):
    """Write the training checkpoint atomically (tmp file + os.replace), so an
    interrupted write (e.g. SIGTERM mid-save) can't corrupt the existing file."""
    tmp = ckpt_file + ".tmp"
    if keep_m is None:
        keep_m = keep_m_DEFAULT
    torch.save({'state_dict': detector.liquid_state_dict(),
                'meta': {'drop_n': drop_n, 'keep_m': keep_m,
                         'units': LIQUID_UNITS, 'motor': LIQUID_MOTOR,
                         'proj': PROJ_DIM},
                'history': history,
                'optim_state': optim_state},
               tmp)
    os.replace(tmp, ckpt_file)

# compatibility: older checkpoints without keep_m metadata default to 1

def _checkpoint_keep_m(meta, default=keep_m_DEFAULT):
    keep_m = meta.get('keep_m', default) if isinstance(meta, dict) else default
    try:
        return max(1, int(keep_m))
    except (TypeError, ValueError):
        return default


def train_liquid(detector, train_set, drop_n, device, epochs=TRAIN_EPOCHS,
                 history=None, optim_state=None, ckpt_file=None,
                 keep_m=keep_m_DEFAULT):
    """Train the liquid stack to recover full boxes from decimated input.

    train_set is an (inputs, targets, masks) tuple from decimate_tracks. After each
    epoch, quality metrics are evaluated on the train set.

    When continuing a prior run, pass its accumulated `history` (appended to in
    place, so epoch numbering is cumulative) and `optim_state` (Adam momentum, so
    the optimizer resumes warm). Returns (history, optimizer_state_dict).
    """
    if keep_m < 1:
        raise ValueError(f"--keep-m must be >= 1 (got {keep_m})")

    inputs, targets, masks = train_set
    detector.train()
    optimizer = torch.optim.Adam(detector.liquid_parameters(), lr=TRAIN_LR)
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)

    inp_t = torch.from_numpy(inputs).to(device)
    tgt_t = torch.from_numpy(targets).to(device)
    msk_t = torch.from_numpy(masks).to(device)
    tgt_box  = tgt_t[:, :, 0:4]
    tgt_conf = tgt_t[:, :, 4:5]
    # Loss is scored on DROPPED frames only.
    # box/conf channels.
    valid   = msk_t[:, :, 0:1]
    dropped = msk_t[:, :, 1:2]
    seg_len = keep_m + drop_n
    t_idx = torch.arange(inp_t.shape[1], device=device)
    seg_pos = t_idx % seg_len
    offset = torch.clamp(seg_pos - keep_m + 1, min=1, max=drop_n).to(inp_t.dtype)
    offset_scale = (1.0 + OFFSET_WEIGHT * (offset - 1.0)).view(1, -1, 1)
    weight  = valid * dropped * offset_scale                       # (N, T, 1)
    # Keyframe mask for the autoregressive rollout: valid AND not dropped. (N, T)
    keep_t  = (msk_t[:, :, 0] * (1.0 - msk_t[:, :, 1]))
    n = inp_t.shape[0]

    if os.environ.get("LIQUID_DEBUG") == "1":
        total_steps = masks.shape[0] * masks.shape[1]
        valid_steps = int(masks[:, :, 0].sum())
        dropped_steps = int(masks[:, :, 1].sum())
        padded_steps = max(0, total_steps - valid_steps)
        print(
            "[train] mask-stats "
            f"total={total_steps} valid={valid_steps} dropped={dropped_steps} "
            f"padded={padded_steps} padded_frac={padded_steps / max(1, total_steps):.4f}")

    def _weighted_mse(pred, tgt, w):
        # Mean of w * (pred-tgt)^2 over all elements, normalized by total weight.
        se = ((pred - tgt) ** 2) * w
        return se.sum() / w.sum().clamp_min(1e-9) / pred.shape[-1]

    def _weighted_ciou(pred, tgt, w):
        # Complete-IoU box loss = 1 - CIoU, weighted per frame by w and normalized by
        # total weight.
        eps = 1e-7
        px1, py1, px2, py2 = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
        tx1, ty1, tx2, ty2 = tgt[..., 0],  tgt[..., 1],  tgt[..., 2],  tgt[..., 3]
        # Guard against inverted predicted boxes (x2<x1) from the unbounded head.
        pxa, pxb = torch.minimum(px1, px2), torch.maximum(px1, px2)
        pya, pyb = torch.minimum(py1, py2), torch.maximum(py1, py2)
        pw, ph = (pxb - pxa).clamp_min(0), (pyb - pya).clamp_min(0)
        tw, th = (tx2 - tx1).clamp_min(0), (ty2 - ty1).clamp_min(0)

        inter_w = (torch.minimum(pxb, tx2) - torch.maximum(pxa, tx1)).clamp_min(0)
        inter_h = (torch.minimum(pyb, ty2) - torch.maximum(pya, ty1)).clamp_min(0)
        inter = inter_w * inter_h
        union = pw * ph + tw * th - inter + eps
        iou = inter / union

        # Smallest enclosing box diagonal^2 and center-distance^2.
        cw = torch.maximum(pxb, tx2) - torch.minimum(pxa, tx1)
        ch = torch.maximum(pyb, ty2) - torch.minimum(pya, ty1)
        c2 = cw * cw + ch * ch + eps
        p_cx, p_cy = (pxa + pxb) / 2, (pya + pyb) / 2
        t_cx, t_cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        rho2 = (p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2

        # Aspect-ratio consistency term (CIoU extra over DIoU).
        v = (4.0 / (torch.pi ** 2)) * (torch.atan(tw / (th + eps))
                                       - torch.atan(pw / (ph + eps))) ** 2
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + eps)
        ciou = iou - rho2 / c2 - alpha * v           # (...)
        wflat = w[..., 0]                            # drop the trailing dim -> (...)
        return ((1.0 - ciou) * wflat).sum() / wflat.sum().clamp_min(1e-9)

    if history is None:
        history = {'loss': [], 'train': []}
    done = len(history['train'])   # epochs already completed in prior runs
    total = done + epochs

    # On SIGTERM (e.g. the launcher's kill -9 is actually a TERM, or a scheduler
    # stopping the job), persist the latest weights/history/optimizer before exiting
    # so progress is never lost. Closes over the live detector/history/optimizer;
    # saves only if a checkpoint path was provided.
    def _on_sigterm(signum, frame):
        if ckpt_file is not None:
            print(f"\nSIGTERM received — saving checkpoint to {ckpt_file} "
                  f"({len(history['train'])} epochs) before exit")
            try:
                save_checkpoint(detector, drop_n, keep_m, history,
                                optimizer.state_dict(), ckpt_file)
            except Exception as e:
                print(f"  checkpoint save on SIGTERM failed: {e}")
        sys.exit(0)

    prev_sigterm = signal.signal(signal.SIGTERM, _on_sigterm)

    def _fmt(m):
        return (f"IoU={m['mean_iou_liquid']:.3f} (teacher {m['mean_iou_teacher']:.3f} hold-last {m['mean_iou_holdlast']:.3f})  "
                f"rec@.5={m['recovery@0.5']:.3f} rec@.75={m['recovery@0.75']:.3f}  "
                f"cErr={m['center_err']:.4f} sErr={m['size_err']:.4f}")

    for epoch in range(epochs):
        # Scheduled sampling: anneal the prob of feeding the model's own prediction
        # on dropped frames from SS_START_PROB to SS_END_PROB over the cumulative
        # epochs (so it keeps progressing across --reset 0 resumes). frac in [0,1].
        frac = ((done + epoch) / (total - 1)) if total > 1 else 1.0
        use_pred_prob = SS_START_PROB + (SS_END_PROB - SS_START_PROB) * frac

        perm = torch.randperm(n, device=device)
        epoch_loss, batches = 0.0, 0
        for b in range(0, n, TRAIN_BATCH):
            idx = perm[b:b + TRAIN_BATCH]
            w = weight[idx]
            pred_box, pred_conf = rollout(detector, inp_t[idx], keep_t[idx],
                                          use_pred_prob, drop_n, keep_m=keep_m)
            # Box term: CIoU (1 - Complete-IoU) — optimizes the reported IoU metric
            # directly and penalizes the box-size error that plain coordinate MSE
            # ignores.
            loss = (_weighted_ciou(pred_box, tgt_box[idx], w)
                    + CONF_WEIGHT * _weighted_mse(pred_conf, tgt_conf[idx], w))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        mean_loss = epoch_loss / max(1, batches)

        tr_m = compute_metrics(detector, inputs, targets, train_set[2], drop_n, device,
                               keep_m=keep_m)
        history['loss'].append(mean_loss)
        history['train'].append(tr_m)
        line = (f"epoch {done + epoch + 1:3d}/{total}  loss={mean_loss:.6f}  "
                f"ss={use_pred_prob:.2f}  train[{_fmt(tr_m)}]")
        print(line)

        # Overwrite the checkpoint every CKPT_SAVE_EVERY epochs (cumulative count),
        # so a crash/kill loses at most that many epochs of progress.
        cum_epoch = done + epoch + 1
        if ckpt_file is not None and cum_epoch % CKPT_SAVE_EVERY == 0:
            save_checkpoint(detector, drop_n, keep_m, history, optimizer.state_dict(), ckpt_file)
            print(f"  checkpoint saved to {ckpt_file} (epoch {cum_epoch})")

    detector.eval()
    signal.signal(signal.SIGTERM, prev_sigterm)   # restore prior handler
    return history, optimizer.state_dict()

def plot_metrics(history, drop_n, out_path):
    """Plot only IoU vs gap offset (final epoch) to a PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # headless: no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping metrics plot")
        return

    tr = history['train']
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    # IoU vs gap offset (final epoch) — liquid vs hold-last and teacher.
    offs = list(range(1, drop_n + 1))
    if offs and tr:
        ax.plot(offs, tr[-1]['iou_by_offset'], 'b-o', label='liquid', ms=4)
        ax.plot(offs, tr[-1]['iou_by_offset_hold'], 'k--s', label='hold-last', ms=4, alpha=0.7)
        if 'iou_by_offset_teacher' in tr[-1]:
            ax.plot(offs, tr[-1]['iou_by_offset_teacher'], 'g-.^', label='teacher', ms=4, alpha=0.7)
        ax.set_xticks(offs)
    ax.set_title(f'IoU vs gap offset (final epoch, drop_n={drop_n})')
    ax.set_xlabel('frames since last keyframe')
    ax.set_ylabel('IoU')
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved metrics plot to {out_path}")

def resolve_clips(spec):
    """Expand a clip spec into a sorted list of clip paths.
    """
    if os.path.isdir(spec):
        clips = sorted(glob.glob(os.path.join(spec, "*.mp4")))
    elif any(ch in spec for ch in "*?[]"):
        clips = sorted(glob.glob(spec))
    else:
        clips = [spec] if os.path.exists(spec) else []
    if not clips:
        raise RuntimeError(f"No clips matched '{spec}'")
    return clips

def build_windows(detector, spec, drop_n, label, keep_m=keep_m_DEFAULT):
    """Decode + detect + track + decimate every clip under `spec`, pool windows.
    """
    clips = resolve_clips(spec)
    print(f"{label}: {len(clips)} clip(s)")
    all_inp, all_tgt, all_msk, total_dropped = [], [], [], 0
    for clip in clips:
        print(f"[{label} clip] {clip}")
        frames_dets = collect_detections(detector, clip)
        tracks = build_tracks(frames_dets)
        inputs, targets, masks, dropped = decimate_tracks(tracks, drop_n, keep_m=keep_m)
        if inputs.shape[0] == 0:
            print(f"  (no windows from {clip} — skipped)")
            continue
        all_inp.append(inputs)
        all_tgt.append(targets)
        all_msk.append(masks)
        total_dropped += dropped
    if not all_inp:
        return None
    inp = np.concatenate(all_inp, axis=0)
    tgt = np.concatenate(all_tgt, axis=0)
    msk = np.concatenate(all_msk, axis=0)
    print(f"Pooled {inp.shape[0]} {label} windows across {len(clips)} clip(s); "
          f"{total_dropped} steps decimated")
    return inp, tgt, msk

def train(TrainSpec, CkptFile, drop_n, reset=False, epochs=TRAIN_EPOCHS,
          plot_prefix=None, keep_m=keep_m_DEFAULT, train_steps=0):
    # No IF_IP / gRPC server needed: training decodes clips in software.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keep_m = max(1, int(keep_m))
    print(f"Training liquid tracker (device={device}, drop_n={drop_n}, keep_m={keep_m}, "
          f"epochs={epochs}, reset={reset}, train_steps={train_steps})")
    detector = LiquidGeneralDetector().to(device)

    # Curriculum phases: 1, 1+step, ... and always force-include target drop_n.
    if train_steps and train_steps > 0:
        phase_drops = list(range(1, drop_n + 1, int(train_steps)))
        if phase_drops[-1] != drop_n:
            phase_drops.append(drop_n)
    else:
        phase_drops = [drop_n]
    phase_epochs = epochs // len(phase_drops)
    if phase_epochs < 1:
        raise RuntimeError(f"epochs // drop_n must be >= 1 (got epochs={epochs}, drop_n={drop_n})")

    print(f"Curriculum phases: {phase_drops}  (phase_epochs={phase_epochs}, remainder_discarded={epochs - phase_epochs * drop_n})")

    prev_history, optim_state = None, None
    start_idx = 0
    if not reset and os.path.exists(CkptFile):
        ckpt = torch.load(CkptFile, map_location=device, weights_only=False)
        ckpt_drop = int(ckpt.get('meta', {}).get('drop_n', drop_n))
        ckpt_keep_m = _checkpoint_keep_m(ckpt.get('meta', {}), default=keep_m_DEFAULT)
        if ckpt_keep_m != keep_m:
            raise RuntimeError(
                f"Checkpoint {CkptFile} was trained with keep_m={ckpt_keep_m}, "
                f"but this run uses keep_m={keep_m}. Use matching values, a different "
                f"--chkp-path, or --reset 1 to start fresh.")
        if ckpt_drop not in phase_drops:
            raise RuntimeError(
                f"Checkpoint {CkptFile} has meta.drop_n={ckpt_drop}, but planned phases are {phase_drops}. "
                f"Use a compatible --drop-n/--train-steps pair, or --reset 1 to start fresh.")
        detector.load_liquid_state_dict(ckpt['state_dict'])
        prev_history = ckpt.get('history')
        optim_state = ckpt.get('optim_state')
        done = len(prev_history['train']) if prev_history else 0
        loaded_idx = phase_drops.index(ckpt_drop)
        start_idx = loaded_idx + 1 if len(phase_drops) > 1 else loaded_idx
        next_phase = (phase_drops[start_idx] if start_idx < len(phase_drops)
                      else "<none>")
        print(f"Continuing training from {CkptFile} (drop_n={ckpt_drop}, keep_m={ckpt_keep_m}, "
              f"history_epochs={done}, next_phase={next_phase}, "
              f"phase_progress={min(start_idx + 1, len(phase_drops))}/{len(phase_drops)})")
    elif not reset:
        print(f"No checkpoint at {CkptFile} — starting from scratch")
    else:
        print("--reset set — starting from scratch (ignoring any existing checkpoint)")

    phases_to_run = phase_drops[start_idx:] if not reset else phase_drops
    if not phases_to_run:
        print("No phases left to run (checkpoint already at final curriculum phase).")
        return

    history = prev_history
    phase_epochs_total = phase_epochs * len(phase_drops)
    for pi, phase_drop in enumerate(phases_to_run, start=start_idx + 1):
        done_before = len(history['train']) if history else 0
        print(f"\n[phase] {pi}/{len(phase_drops)} drop_n={phase_drop} phase_epochs={phase_epochs} "
              f"cumulative_before={done_before}/{phase_epochs_total}")

        train_set = build_windows(detector, TrainSpec, phase_drop, "train", keep_m=keep_m)
        if train_set is None:
            raise RuntimeError("No training windows produced from any clip — are the "
                               "clips long enough and does the detector find objects?")

        history, optim_state = train_liquid(
            detector, train_set, phase_drop, device, epochs=phase_epochs,
            history=history, optim_state=optim_state, ckpt_file=CkptFile,
            keep_m=keep_m)

        done_after = len(history['train']) if history else done_before
        print(f"[phase] complete drop_n={phase_drop} cumulative_after={done_after}/{phase_epochs_total}")

    # Always publish final checkpoint metadata at target drop_n.
    save_checkpoint(detector, drop_n, keep_m, history, optim_state, CkptFile)
    print(f"Saved trained liquid weights to {CkptFile} "
          f"({len(history['train'])} epochs total, final drop_n={drop_n})")

    if plot_prefix:
        plot_path = f"{plot_prefix}-plot-metrics.png"
    else:
        plot_path = os.path.splitext(CkptFile)[0] + "_metrics.png"
    plot_metrics(history, drop_n, plot_path)



def _draw_track_overlay(frame, box, color, label_text):
    """Draw center marker for a normalized [x1,y1,x2,y2] box."""
    h, w = frame.shape[:2]
    x1 = max(0, int(box[0] * w))
    y1 = max(0, int(box[1] * h))
    x2 = min(w, int(box[2] * w))
    y2 = min(h, int(box[3] * h))
    cx = max(0, min(w - 1, (x1 + x2) // 2))
    cy = max(0, min(h - 1, (y1 + y2) // 2))
    cv2.drawMarker(frame, (cx, cy), color,
                   markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
    cv2.putText(frame, label_text, (cx + 6, max(12, cy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def run(InFile, OutFile, CkptFile=CHECKPOINT_PATH, drop_n_override=None,
        keep_m=keep_m_DEFAULT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keep_m = max(1, int(keep_m))
    print(f"Loading Liquid detector (device={device}, keep_m={keep_m}, center=1)")
    detector = LiquidGeneralDetector().to(device)

    ckpt_loaded = bool(CkptFile and os.path.exists(CkptFile))
    if ckpt_loaded:
        ckpt = torch.load(CkptFile, map_location=device, weights_only=False)
        detector.load_liquid_state_dict(ckpt['state_dict'])
        drop_n = ckpt.get('meta', {}).get('drop_n', 1)
        print(f"Loaded liquid weights from {CkptFile} (drop_n={drop_n})")
    else:
        print("No liquid weights found — using untrained detector")
        drop_n = 1

    if drop_n_override is not None:
        drop_n = drop_n_override

    cap = cv2.VideoCapture(InFile)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open input clip: {InFile}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else DEFAULT_FPS
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if orig_w <= 0 or orig_h <= 0:
        raise RuntimeError(f"Could not determine frame size for {InFile}")
    out_path = OutFile
    if os.path.splitext(out_path)[1].lower() != ".mp4":
        out_path = os.path.splitext(out_path)[0] + ".mp4"
        print(f"Software encoder writes mp4 — output -> {out_path}")
    writer = cv2.VideoWriter(out_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (orig_w, orig_h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not open VideoWriter for {out_path}")

    detector.eval()
    tracks = []
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        orig_h, orig_w = frame.shape[:2]
        if drop_n <= 1:
            img_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with torch.inference_mode():
                preds = detector.spatial_detector(img_rgb, threshold=YOLO_CONF_THRESHOLD)
            for pred in (preds or []):
                b = pred["box"]
                box = [b["xmin"] / orig_w, b["ymin"] / orig_h,
                       b["xmax"] / orig_w, b["ymax"] / orig_h]
                label = pred["label"]
                conf = float(pred["score"])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                _draw_track_overlay(frame, box, (0, 255, 0),
                                    f"{label}: {conf * 100:.1f}%")
        else:
            is_keyframe = (frame_no % drop_n == 0)
            matched_tracks = set()
            if is_keyframe:
                img_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                with torch.inference_mode():
                    preds = detector.spatial_detector(img_rgb, threshold=YOLO_CONF_THRESHOLD)
                preds = preds or []
                dets = []
                for pred in preds:
                    b = pred["box"]
                    dets.append({
                        'box': [b["xmin"] / orig_w, b["ymin"] / orig_h,
                                b["xmax"] / orig_w, b["ymax"] / orig_h],
                        'score': float(pred["score"]),
                        'class_id': detector.label2id.get(pred["label"], 0),
                        'label': pred["label"],
                    })
                prev_boxes = [tr['box'] for tr in tracks]
                det_boxes = [d['box'] for d in dets]
                match = _greedy_match(prev_boxes, det_boxes) if prev_boxes else {}
                for di, d in enumerate(dets):
                    feat = d['box'] + [d['score']]
                    if di in match:
                        tr = tracks[match[di]]
                        tr['class_id'] = d['class_id']
                        tr['label'] = d['label']
                        matched_tracks.add(match[di])
                    else:
                        tr = {'box': d['box'], 'hx': None,
                              'class_id': d['class_id'], 'label': d['label'],
                              'misses': 0, 'holdlast': d['box'], 'holdlast_conf': d['score'],
                              'pred_feat': None}
                        tracks.append(tr)
                    tr['holdlast'] = d['box']
                    tr['holdlast_conf'] = d['score']
                    t_in = torch.from_numpy(np.asarray(feat, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                    a_in = torch.from_numpy(np.asarray(tr['holdlast'], dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                    with torch.no_grad():
                        box_t, conf_t, hx = detector(t_in, tr['hx'], anchor_box=a_in)
                    tr['hx'] = hx
                    tr['box'] = [float(v) for v in box_t.squeeze().cpu().numpy()]
                    tr['pred_feat'] = [float(v) for v in box_t.squeeze().cpu().numpy()] + [float(conf_t.squeeze().cpu().numpy())]
                    tr['misses'] = 0
                    _draw_track_overlay(
                        frame,
                        d['box'],
                        (0, 255, 0),
                        f"{d['label']}: {d['score'] * 100:.1f}%",
                    )
            for ti, tr in enumerate(tracks):
                if ti in matched_tracks:
                    continue
                prev_pred = tr.get('pred_feat')
                if prev_pred is None:
                    prev_pred = [0.0, 0.0, 0.0, 0.0, 0.0]
                anchor = tr.get('holdlast', tr['box'])
                feat = np.asarray(prev_pred, dtype=np.float32)
                t_in = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).to(device)
                a_in = torch.from_numpy(np.asarray(anchor, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    box_t, conf_t, hx = detector(t_in, tr['hx'], anchor_box=a_in)
                tr['hx'] = hx
                tr['box'] = [float(v) for v in box_t.squeeze().cpu().numpy()]
                tr['pred_feat'] = [float(v) for v in box_t.squeeze().cpu().numpy()] + [float(conf_t.squeeze().cpu().numpy())]
                tr['misses'] += 1
                if tr.get('holdlast') is not None and tr['misses'] < drop_n:
                    _draw_track_overlay(
                        frame,
                        tr['holdlast'],
                        (255, 255, 255),
                        "HL",
                    )
                _draw_track_overlay(
                    frame,
                    tr['box'],
                    (0, 255, 255),
                    "LQ",
                )
            tracks = [tr for tr in tracks if tr['misses'] <= TRACK_MAX_MISSES]
            if len(tracks) > MAX_TRACKED_OBJECTS:
                tracks = sorted(tracks, key=lambda t: t['misses'])[:MAX_TRACKED_OBJECTS]
        writer.write(frame)
        frame_no += 1

    cap.release()
    writer.release()
    print(f"Wrote annotated video to {out_path}")
    return

    # Fallback to the ncps.pipeline path if the optional dependency is installed.
    from ncps.pipeline import Pipeline
    p = Pipeline(detector, InFile, OutFile, drop_n=drop_n)
    try:
        p.run()
    except Exception:
        import traceback
        traceback.print_exc()


def _positive_int(name, value):
    """argparse type: accept only positive integers (>= 1)."""
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"{name} must be an integer (got '{value}')")
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"{name} must be >= 1 (got {ivalue})")
    return ivalue


def _nonnegative_int(name, value):
    """argparse type: accept only non-negative integers (>= 0)."""
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"{name} must be an integer (got '{value}')")
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{name} must be >= 0 (got {ivalue})")
    return ivalue


def build_arg_parser():
    """Build the CLI used by the script entrypoint."""
    parser = argparse.ArgumentParser(description="Liquid detector tracker")
    parser.add_argument("--train", action="store_true",
                        help="train the liquid tracker on the supplied training set")
    parser.add_argument("--infer", action="store_true",
                        help="run inference on the supplied input video")
    parser.add_argument("--train-folder", type=str, default="",
                        help="folder containing .mp4 clips for training")
    parser.add_argument("--in", type=str, default="",
                        help="input video path for inference")
    parser.add_argument("--out", type=str, default="",
                        help="output video path for inference")
    parser.add_argument("--chkp-path", type=str, default=CHECKPOINT_PATH,
                        help="checkpoint path for training/evaluation/inference")
    parser.add_argument("--drop-n", type=lambda v: _positive_int("--drop-n", v),
                        default=drop_n_DEFAULT,
                        help="number of dropped/predicted frames per kept segment")
    parser.add_argument("--keep-m", type=lambda v: _positive_int("--keep-m", v),
                        default=keep_m_DEFAULT,
                        help="number of real detector frames kept before each drop_n segment")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS,
                        help="training epochs")
    parser.add_argument("--train-steps", type=lambda v: _nonnegative_int("--train-steps", v), default=0,
                        help="curriculum step over drop_n (0 disables; >0 uses phases 1..drop_n in this step)")
    parser.add_argument("--reset", type=int, choices=(0, 1), default=0,
                        help="force a fresh run (ignore existing checkpoint)")
    parser.add_argument("--plot-prefix", type=str, default="",
                        help="prefix for generated metrics/eval plots")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    in_path = getattr(args, "in")
    if args.train:
        if not args.train_folder:
            parser.error("--train requires --train-folder")
        train(args.train_folder, args.chkp_path, args.drop_n,
              reset=bool(args.reset), epochs=args.epochs,
              plot_prefix=args.plot_prefix, keep_m=args.keep_m,
              train_steps=args.train_steps)
    elif args.infer or in_path:
        if not in_path:
            parser.error("--infer requires --in")
        out_path = args.out or (os.path.splitext(in_path)[0] + "_liquid_out.mp4")
        run(in_path, out_path, CkptFile=args.chkp_path,
            drop_n_override=args.drop_n, keep_m=args.keep_m)
    else:
        parser.print_help()
