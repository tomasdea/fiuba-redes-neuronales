import argparse
import json
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO


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
    if len(frames) == 0:
        raise RuntimeError(f"No pude samplear frames de {video_path}")

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [T,H,W,C]
    arr = np.transpose(arr, (3, 0, 1, 2))  # [C,T,H,W]

    meta = {
        "orig_w": W0,
        "orig_h": H0,
        "input_fps": float(input_fps),
        "sample_fps": float(sample_fps),
        "stride": int(stride),
        "clip_seconds": float(clip_seconds),
        "T": int(arr.shape[1]),
        "resize_h": int(resize_hw[0]),
        "resize_w": int(resize_hw[1]),
    }
    return arr, meta


def make_heatmap_from_video_person(
    person_model: YOLO,
    video_path: str,
    sample_fps: float,
    clip_seconds: float,
    conf: float,
    iou: float,
) -> Tuple[np.ndarray, int, int, int]:
    """
    Heatmap occupancy (bbox fill) on ORIGINAL resolution.
    Returns (heat_norm[H,W] in 0..1, W0, H0, processed_frames)
    """
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

    heat = np.zeros((H0, W0), dtype=np.float32)
    processed = 0
    input_idx = 0
    any_det = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if input_idx >= target_input_frames:
            break

        if (input_idx % stride) == 0:
            res = person_model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                classes=[0],   # person
                verbose=False
            )
            if res and len(res) > 0 and res[0].boxes is not None and len(res[0].boxes) > 0:
                xyxy = res[0].boxes.xyxy
                for i in range(len(res[0].boxes)):
                    x1, y1, x2, y2 = map(int, xyxy[i].tolist())
                    x1 = clamp(x1, 0, W0)
                    x2 = clamp(x2, 0, W0)
                    y1 = clamp(y1, 0, H0)
                    y2 = clamp(y2, 0, H0)
                    if x2 > x1 and y2 > y1:
                        heat[y1:y2, x1:x2] += 1.0
                        any_det = True
            processed += 1

        input_idx += 1

    cap.release()

    if processed == 0 or not any_det:
        return np.zeros((H0, W0), dtype=np.float32), W0, H0, processed

    heat = heat / float(processed)
    heat = np.clip(heat, 0.0, 1.0)
    return heat, W0, H0, processed


def square_covering_bbox(
    x1: int, y1: int, x2: int, y2: int,
    W: int, H: int,
    min_side: int
) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    side = max(min_side, bw, bh)
    side = min(side, W, H)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    sx1 = int(round(cx - side / 2.0))
    sy1 = int(round(cy - side / 2.0))
    sx1 = clamp(sx1, 0, W - side)
    sy1 = clamp(sy1, 0, H - side)
    return (sx1, sy1, sx1 + side, sy1 + side)


def best_square_by_sum(heat: np.ndarray, side: int) -> Tuple[int, int, int, int]:
    """
    Finds square window (side x side) with max heat sum using integral image.
    """
    H, W = heat.shape
    side = min(side, W, H)
    ii = cv2.integral(heat.astype(np.float32))

    best_sum = -1.0
    best = (0, 0)
    step = 1 if max(W, H) < 1500 else 2

    for y in range(0, H - side + 1, step):
        y2 = y + side
        for x in range(0, W - side + 1, step):
            x2 = x + side
            s = ii[y2, x2] - ii[y2, x] - ii[y, x2] + ii[y, x]
            if s > best_sum:
                best_sum = float(s)
                best = (x, y)

    x1, y1 = best
    return (x1, y1, x1 + side, y1 + side)


def choose_variable_square_roi(
    heat: np.ndarray,
    min_presence_ratio: float,
    min_side: int,
) -> Tuple[int, int, int, int]:
    """
    Returns ROI square (variable side >= min_side) or (0,0,0,0) if no activity.
    Strategy:
      1) Threshold heat -> bbox of active pixels
      2) ROI = minimal square covering bbox with side=max(min_side, bbox_w, bbox_h) clamped
      3) If activity is super spread (bbox wants > min(frame)), fallback to best square of side=min(frame) (rare)
    """
    H, W = heat.shape
    min_side = min(min_side, W, H)

    mask = heat >= float(min_presence_ratio)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)

    x1_t, x2_t = int(xs.min()), int(xs.max() + 1)
    y1_t, y2_t = int(ys.min()), int(ys.max() + 1)

    bw = x2_t - x1_t
    bh = y2_t - y1_t
    desired = max(min_side, bw, bh)

    if desired <= min(W, H):
        return square_covering_bbox(x1_t, y1_t, x2_t, y2_t, W, H, min_side=min_side)

    # if it spreads too much, take the best possible square (full min side)
    return best_square_by_sum(heat, side=min(W, H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sample_fps", type=float, default=2.0)
    ap.add_argument("--clip_seconds", type=float, default=5.0)
    ap.add_argument("--resize_h", type=int, default=256)
    ap.add_argument("--resize_w", type=int, default=512)
    ap.add_argument("--min_side", type=int, default=224, help="Minimum ROI square side in pixels")
    ap.add_argument("--min_presence_ratio", type=float, default=0.02)
    ap.add_argument("--person_model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        (out_dir / f"clips/{split}").mkdir(parents=True, exist_ok=True)
        (out_dir / f"labels/{split}").mkdir(parents=True, exist_ok=True)

    videos = sorted([p for p in videos_dir.iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mkv", ".mov"]])
    if not videos:
        raise RuntimeError("No encontré videos en videos_dir")

    rng = np.random.default_rng(args.seed)
    idxs = np.arange(len(videos))
    rng.shuffle(idxs)
    val_count = int(round(len(videos) * args.val_split))
    val_set = set(idxs[:val_count].tolist())

    person_model = YOLO(args.person_model)

    for i, vp in enumerate(videos):
        split = "val" if i in val_set else "train"
        name = vp.stem

        # clip tensor (model input)
        x, clip_meta = sample_clip_rgb(
            video_path=str(vp),
            sample_fps=args.sample_fps,
            clip_seconds=args.clip_seconds,
            resize_hw=(args.resize_h, args.resize_w)
        )

        # pseudo-label ROI (original pixel coords)
        heat, W0, H0, processed = make_heatmap_from_video_person(
            person_model=person_model,
            video_path=str(vp),
            sample_fps=args.sample_fps,
            clip_seconds=args.clip_seconds,
            conf=args.conf,
            iou=args.iou
        )

        roi = choose_variable_square_roi(
            heat=heat,
            min_presence_ratio=args.min_presence_ratio,
            min_side=args.min_side
        )

        present = 0 if roi == (0, 0, 0, 0) else 1
        min_frame_side = min(W0, H0)

        if present:
            x1, y1, x2, y2 = roi
            side = x2 - x1
            cx = ((x1 + x2) / 2.0) / W0
            cy = ((y1 + y2) / 2.0) / H0
            s_norm = side / float(min_frame_side)  # 0..1
        else:
            cx, cy, s_norm = 0.0, 0.0, 0.0

        # save clip
        clip_path = out_dir / f"clips/{split}/{name}.npz"
        np.savez_compressed(clip_path, x=x, meta=clip_meta)

        # save label json
        label = {
            "video": str(vp.resolve()),
            "present": int(present),
            "roi_xyxy": list(roi),          # in pixels, or zeros
            "roi_center_norm": [float(cx), float(cy)],  # 0..1
            "roi_side_norm": float(s_norm),             # 0..1 (relative to min(W,H))
            "min_side": int(args.min_side),
            "orig_w": int(W0),
            "orig_h": int(H0),
            "min_frame_side": int(min_frame_side),
            "sample_fps": float(args.sample_fps),
            "clip_seconds": float(args.clip_seconds),
            "processed_frames": int(processed),
            "min_presence_ratio": float(args.min_presence_ratio),
        }
        (out_dir / f"labels/{split}/{name}.json").write_text(json.dumps(label, indent=2), encoding="utf-8")

        print(f"[{split}] {vp.name} -> present={present} roi={roi} side_norm={s_norm:.3f}")

    print("OK dataset:", out_dir)


if __name__ == "__main__":
    main()
