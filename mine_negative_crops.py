import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------------
# Utils video sampling
# -----------------------------
def compute_stride(input_fps: float, sample_fps: float) -> int:
    if input_fps <= 0:
        input_fps = 30.0
    if sample_fps <= 0:
        return 1
    return max(1, int(round(input_fps / sample_fps)))


def sample_frames(video_path: str, sample_fps: float, clip_seconds: float) -> Tuple[List[np.ndarray], dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stride = compute_stride(fps, sample_fps)
    max_frames = int(round(clip_seconds * fps))

    frames = []
    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if idx >= max_frames:
            break
        if (idx % stride) == 0:
            frames.append(frame_bgr)
        idx += 1

    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No pude samplear frames")
    meta = {"orig_w": W, "orig_h": H, "input_fps": float(fps), "T": len(frames)}
    return frames, meta


# -----------------------------
# YOLO box extraction
# -----------------------------
def extract_person_boxes(results) -> List[np.ndarray]:
    """
    Returns list per frame: ndarray [N,4] xyxy (float), only person class (COCO id 0)
    """
    out = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            out.append(np.zeros((0, 4), dtype=np.float32))
            continue
        cls = r.boxes.cls.detach().cpu().numpy().astype(int)
        xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        xyxy = xyxy[cls == 0]
        out.append(xyxy if xyxy.size else np.zeros((0, 4), dtype=np.float32))
    return out


def window_overlap_sum(win: Tuple[int, int, int, int], boxes_per_frame: List[np.ndarray]) -> float:
    wx1, wy1, wx2, wy2 = win
    total = 0.0
    for boxes in boxes_per_frame:
        if boxes.shape[0] == 0:
            continue
        x1 = np.maximum(wx1, boxes[:, 0])
        y1 = np.maximum(wy1, boxes[:, 1])
        x2 = np.minimum(wx2, boxes[:, 2])
        y2 = np.minimum(wy2, boxes[:, 3])
        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        total += float(np.sum(inter_w * inter_h))
    return total


def find_empty_square(
    W: int,
    H: int,
    boxes_per_frame: List[np.ndarray],
    side: int,
    tries: int,
    rng: random.Random,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Random search for a square window with ZERO overlap with person boxes across all frames.
    """
    side = min(side, W, H)
    if side < 1:
        return None
    if W - side < 0 or H - side < 0:
        return None

    for _ in range(tries):
        x1 = rng.randint(0, W - side)
        y1 = rng.randint(0, H - side)
        win = (x1, y1, x1 + side, y1 + side)
        ov = window_overlap_sum(win, boxes_per_frame)
        if ov == 0.0:
            return win
    return None


# -----------------------------
# Clip save
# -----------------------------
def crop_and_resize_clip(frames_bgr: List[np.ndarray], win: Tuple[int, int, int, int], out_h: int, out_w: int) -> np.ndarray:
    x1, y1, x2, y2 = win
    out_frames = []
    for fbgr in frames_bgr:
        crop = fbgr[y1:y2, x1:x2]  # square
        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        out_frames.append(rgb)

    arr = np.stack(out_frames, axis=0).astype(np.float32) / 255.0  # [T,H,W,C]
    arr = np.transpose(arr, (3, 0, 1, 2))  # [C,T,H,W]
    return arr


# -----------------------------
# Main logic: scan all videos, keep best per video, then pick top N by side
# -----------------------------
def make_side_schedule(max_side: int, min_side: int) -> List[int]:
    """
    Descending schedule. Starts at max_side, then tries a few "big" steps down to min_side.
    We avoid tiny step-by-step to keep it efficient.
    """
    # A good practical ladder
    ladder = [max_side]
    for s in [1024, 896, 768, 640, 576, 512, 448, 384, 320, 288, 256, 224]:
        if min_side <= s <= max_side and s not in ladder:
            ladder.append(s)
    ladder = sorted(set(ladder), reverse=True)
    # ensure min_side included
    if min_side not in ladder:
        ladder.append(min_side)
    # keep only >= min_side
    ladder = [s for s in ladder if s >= min_side]
    return ladder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", required=True, help="Carpeta con videos fuente")
    ap.add_argument("--out_data_dir", required=True, help="Root del dataset: clips/{split}, labels/{split}")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--count", type=int, default=300, help="Cuantos negativos guardar (top por side)")
    ap.add_argument("--yolo", default="yolov8n.pt", help="weights YOLO")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--sample_fps", type=float, default=2.0)
    ap.add_argument("--clip_seconds", type=float, default=5.0)

    ap.add_argument("--min_side", type=int, default=224, help="lado mínimo del recorte cuadrado (en px del frame original)")
    ap.add_argument("--max_side", type=int, default=0, help="0 = auto (min(W,H)) por video; si no, fuerza máximo global")
    ap.add_argument("--tries_per_side", type=int, default=200, help="cuantos candidatos probar por cada side")
    ap.add_argument("--max_videos", type=int, default=0, help="0=sin limite; si no, limita cuantos videos intenta")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--resize_h", type=int, default=256, help="alto del tensor guardado")
    ap.add_argument("--resize_w", type=int, default=512, help="ancho del tensor guardado")

    args = ap.parse_args()

    src = Path(args.videos_dir)
    out_root = Path(args.out_data_dir)
    clips_dir = out_root / f"clips/{args.split}"
    labels_dir = out_root / f"labels/{args.split}"
    clips_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    videos = sorted([p for p in src.iterdir() if p.is_file() and p.suffix.lower() in [".avi", ".mp4", ".mkv", ".mov", ".m4v"]])
    if not videos:
        raise RuntimeError(f"No encontré videos en {src}")

    if args.max_videos and args.max_videos > 0:
        videos = videos[:args.max_videos]

    model = YOLO(args.yolo)

    # Collect best candidate per video: (side, video_path, win, frames_meta, frames_bgr)
    candidates = []
    tried = 0
    usable = 0

    print(f"Scanning {len(videos)} videos and keeping best empty square per video...")

    for vp in videos:
        tried += 1
        try:
            frames_bgr, meta = sample_frames(str(vp), args.sample_fps, args.clip_seconds)
            H0, W0 = frames_bgr[0].shape[:2]

            # Run YOLO once per video (batch)
            results = model.predict(frames_bgr, conf=args.conf, iou=args.iou, verbose=False)
            boxes_per_frame = extract_person_boxes(results)

            # define max_side for this video
            max_side_vid = min(W0, H0)
            if args.max_side and args.max_side > 0:
                max_side_vid = min(max_side_vid, args.max_side)

            if max_side_vid < args.min_side:
                continue

            schedule = make_side_schedule(max_side_vid, args.min_side)

            best = None  # (side, win)
            for side in schedule:
                win = find_empty_square(
                    W=W0,
                    H=H0,
                    boxes_per_frame=boxes_per_frame,
                    side=side,
                    tries=args.tries_per_side,
                    rng=rng
                )
                if win is not None:
                    best = (side, win)
                    break  # found largest possible due to descending schedule

            if best is None:
                continue

            usable += 1
            side, win = best
            candidates.append({
                "side": int(side),
                "video": str(vp.resolve()),
                "win": tuple(int(x) for x in win),
                "meta": meta,
                # keep frames to avoid re-reading if you want, but to save RAM we won't store frames.
                # We'll re-read when saving selected top N.
                "stem": vp.stem,
                "W0": int(W0),
                "H0": int(H0),
            })

            if usable % 50 == 0:
                print(f"[scan] usable={usable} candidates={len(candidates)} last_side={side} last_video={vp.name}")

        except Exception:
            continue

    if not candidates:
        print("No candidates found. Try lowering --conf or lowering --min_side.")
        return

    # Sort by side desc and pick top N
    candidates.sort(key=lambda d: d["side"], reverse=True)
    picked = candidates[:args.count]

    print(f"\nFound {len(candidates)} candidate crops. Picking top {len(picked)} by largest side.")
    print("Top-5 sides:", [c["side"] for c in picked[:5]])

    # Save picked (re-read video frames to crop)
    saved = 0
    for idx, c in enumerate(picked):
        vp = c["video"]
        win = c["win"]
        side = c["side"]

        try:
            frames_bgr, meta = sample_frames(vp, args.sample_fps, args.clip_seconds)
            x = crop_and_resize_clip(frames_bgr, win, args.resize_h, args.resize_w)

            stem = f"negcrop_best_{side}px_{idx:05d}_{Path(vp).stem}"
            npz_path = clips_dir / f"{stem}.npz"
            json_path = labels_dir / f"{stem}.json"

            np.savez_compressed(npz_path, x=x, meta={"source_video": vp, **meta, "crop_win_xyxy": win, "crop_side": side})

            label = {
                "video": vp,
                "present": 0,
                "roi_xyxy": [0, 0, 0, 0],
                "roi_center_norm": [0.0, 0.0],
                "roi_side_norm": 0.0,
                "min_side": int(args.min_side),
                "orig_w": int(c["W0"]),
                "orig_h": int(c["H0"]),
                "sample_fps": float(args.sample_fps),
                "clip_seconds": float(args.clip_seconds),
                "negative_crop_win_xyxy": list(win),
                "negative_crop_side": int(side),
                "teacher_yolo": args.yolo,
                "teacher_conf": float(args.conf),
                "teacher_iou": float(args.iou),
            }
            json_path.write_text(json.dumps(label, indent=2), encoding="utf-8")

            saved += 1
            if saved % 25 == 0:
                print(f"[save] {saved}/{len(picked)} last={Path(vp).name} side={side} win={win}")

        except Exception:
            continue

    print("\nDone.")
    print("videos_tried:", tried)
    print("usable_candidates:", len(candidates))
    print("saved_negatives:", saved)
    print("saved_to:", out_root.resolve())


if __name__ == "__main__":
    main()
