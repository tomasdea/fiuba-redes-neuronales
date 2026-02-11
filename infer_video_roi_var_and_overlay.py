import argparse
import json
import os
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def compute_stride(input_fps: float, sample_fps: float) -> int:
    if sample_fps <= 0:
        return 1
    stride = int(round(input_fps / sample_fps))
    return max(1, stride)


def sample_clip_rgb(video_path: str, sample_fps: float, clip_seconds: float, resize_hw: Tuple[int, int]) -> Tuple[np.ndarray, dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir: {video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = 30.0

    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stride = compute_stride(input_fps, sample_fps)
    target_input_frames = int(round(clip_seconds * input_fps))

    frames: List[np.ndarray] = []
    input_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if input_idx >= target_input_frames:
            break

        if (input_idx % stride) == 0:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = resize_hw
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(rgb)

        input_idx += 1

    cap.release()

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [T,H,W,C]
    arr = np.transpose(arr, (3, 0, 1, 2))  # [C,T,H,W]
    meta = {"orig_w": W0, "orig_h": H0, "input_fps": float(input_fps), "T": int(arr.shape[1])}
    return arr, meta


class Simple3DROI(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_ch, 16, 3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, 3, stride=(1,2,2), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, 3, stride=(2,2,2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, 3, stride=(2,2,2), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d((1,1,1)),
        )
        self.head_p = nn.Linear(128, 1)
        self.head_r = nn.Linear(128, 3)

    def forward(self, x):
        z = self.features(x).flatten(1)
        p_logit = self.head_p(z)
        reg = torch.sigmoid(self.head_r(z))  # cx,cy,side_norm in 0..1
        return p_logit, reg


def overlay_roi(video_path: str, roi: Tuple[int,int,int,int], out_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    x1, y1, x2, y2 = roi
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if roi != (0,0,0,0):
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 3)
            cv2.putText(frame, "ROI", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True, help="video_roi_var.pt")
    ap.add_argument("--outdir", default="roi_video_pred")
    ap.add_argument("--sample_fps", type=float, default=2.0)
    ap.add_argument("--clip_seconds", type=float, default=5.0)
    ap.add_argument("--resize_h", type=int, default=256)
    ap.add_argument("--resize_w", type=int, default=512)
    ap.add_argument("--min_side", type=int, default=224)
    ap.add_argument("--present_thr", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    x, meta = sample_clip_rgb(args.video, args.sample_fps, args.clip_seconds, (args.resize_h, args.resize_w))
    x_t = torch.from_numpy(x).unsqueeze(0)  # [1,C,T,H,W]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Simple3DROI(in_ch=3).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    with torch.no_grad():
        p_logit, reg = model(x_t.to(device))
        p = torch.sigmoid(p_logit).item()
        cx, cy, s_norm = reg[0].cpu().tolist()

    W0 = int(meta["orig_w"])
    H0 = int(meta["orig_h"])
    min_frame_side = min(W0, H0)

    if p < args.present_thr:
        roi = (0,0,0,0)
        side_px = 0
    else:
        # side is variable but has minimum 224
        side_px = int(round(s_norm * min_frame_side))
        side_px = max(args.min_side, side_px)
        side_px = min(side_px, min_frame_side)

        x1 = int(round(cx * W0 - side_px / 2))
        y1 = int(round(cy * H0 - side_px / 2))
        x1 = clamp(x1, 0, W0 - side_px)
        y1 = clamp(y1, 0, H0 - side_px)
        roi = (x1, y1, x1 + side_px, y1 + side_px)

    payload = {
        "video": str(Path(args.video).resolve()),
        "present_prob": float(p),
        "roi_xyxy": list(roi),  # [x1,y1,x2,y2] or zeros
        "roi_side_px": int(side_px),
        "orig_w": W0,
        "orig_h": H0,
        "min_side": int(args.min_side),
    }

    json_path = os.path.join(args.outdir, "roi.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    overlay_path = os.path.join(args.outdir, "overlay.mp4")
    overlay_roi(args.video, roi, overlay_path)

    print("Done")
    print("roi.json:", json_path)
    print("overlay.mp4:", overlay_path)


if __name__ == "__main__":
    main()
