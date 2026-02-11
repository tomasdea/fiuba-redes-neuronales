import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class PersonStats:
    track_id: int
    frames_present: int
    time_seconds: float
    first_frame: int
    last_frame: int


@dataclass
class Results:
    video_path: str
    fps: float
    total_frames: int
    frame_size: Tuple[int, int]  # (w, h)
    unique_person_ids: int
    max_simultaneous_persons: int
    total_time_with_any_person: float
    total_person_time_sum: float
    person_stats: List[PersonStats]
    roi_bbox_variable: Tuple[int, int, int, int]  # x1,y1,x2,y2
    roi_bbox_fixed: Tuple[int, int, int, int]     # x1,y1,x2,y2
    roi_fixed_size: Tuple[int, int]               # (w,h)
    roi_area_ratio_variable: float
    roi_area_ratio_fixed: float
    gating_threshold_seconds: float
    gating_decision_run_heavy_model: bool
    outputs: Dict[str, str]


def clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0


def compute_variable_roi_from_heatmap(
    heatmap: np.ndarray,
    min_presence_ratio: float,
) -> Tuple[int, int, int, int]:
    """
    heatmap: float32 [H,W] with accumulated counts; normalized outside if needed.
    min_presence_ratio: keep pixels where normalized heatmap >= this threshold.
    """
    H, W = heatmap.shape
    mask = heatmap >= min_presence_ratio
    ys, xs = np.where(mask)
    if len(xs) == 0:
        # no persons: return full frame ROI=empty-ish centered 0
        return (0, 0, W, H)
    x1, x2 = int(xs.min()), int(xs.max() + 1)
    y1, y2 = int(ys.min()), int(ys.max() + 1)
    return (x1, y1, x2, y2)


def compute_best_fixed_roi(
    heatmap: np.ndarray,
    roi_w: int,
    roi_h: int,
    stride: int = 16,
) -> Tuple[int, int, int, int]:
    """
    Finds the fixed-size ROI (roi_w x roi_h) with maximum summed heatmap.
    Uses a stride on a coarse scan for speed.
    """
    H, W = heatmap.shape
    roi_w = min(roi_w, W)
    roi_h = min(roi_h, H)

    best_sum = -1.0
    best = (0, 0, roi_w, roi_h)

    # coarse scan
    for y in range(0, H - roi_h + 1, stride):
        for x in range(0, W - roi_w + 1, stride):
            s = float(heatmap[y:y + roi_h, x:x + roi_w].sum())
            if s > best_sum:
                best_sum = s
                best = (x, y, x + roi_w, y + roi_h)

    # refine around best with smaller stride
    bx1, by1, bx2, by2 = best
    refine_stride = max(1, stride // 4)
    x_start = clamp(bx1 - stride, 0, W - roi_w)
    x_end = clamp(bx1 + stride, 0, W - roi_w)
    y_start = clamp(by1 - stride, 0, H - roi_h)
    y_end = clamp(by1 + stride, 0, H - roi_h)

    for y in range(y_start, y_end + 1, refine_stride):
        for x in range(x_start, x_end + 1, refine_stride):
            s = float(heatmap[y:y + roi_h, x:x + roi_w].sum())
            if s > best_sum:
                best_sum = s
                best = (x, y, x + roi_w, y + roi_h)

    return best


def draw_roi(frame: np.ndarray, roi: Tuple[int, int, int, int], label: str) -> None:
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        frame, label, (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--model", default="yolov8n.pt", help="Ultralytics model path or name")
    ap.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--gating_seconds", type=float, default=1.0, help="If total_time_with_any_person >= this -> run heavy model")
    ap.add_argument("--min_presence_ratio", type=float, default=0.05,
                    help="For variable ROI: pixel/cell must have person presence at least this fraction of frames (0..1)")
    ap.add_argument("--roi_fixed", type=int, nargs=2, default=[250, 250], metavar=("W", "H"),
                    help="Fixed ROI size (width height)")
    ap.add_argument("--grid", type=int, nargs=2, default=[80, 80], metavar=("GW", "GH"),
                    help="Heatmap grid size to reduce computation (e.g. 80 80). Larger=more precise, slower.")
    ap.add_argument("--stride", type=int, default=8, help="Stride for scanning fixed ROI over heatmap grid")
    ap.add_argument("--only_person", action="store_true", help="Filter detections to class=person (COCO id 0). Recommended.")
    ap.add_argument("--save_overlay", action="store_true", help="Save overlay video with boxes+IDs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Heatmap on a coarse grid for efficiency
    gw, gh = args.grid
    heat = np.zeros((gh, gw), dtype=np.float32)

    # Tracking stats
    frames_present: Dict[int, int] = {}
    first_frame: Dict[int, int] = {}
    last_frame: Dict[int, int] = {}

    max_simul = 0
    frames_with_any_person = 0

    # Load model
    model = YOLO(args.model)

    # Video writer (optional)
    overlay_path = os.path.join(args.outdir, "overlay.mp4")
    writer = None
    if args.save_overlay:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(overlay_path, fourcc, fps, (W, H))

    frame_idx = 0

    # We'll use Ultralytics tracker with ByteTrack
    # NOTE: persist=True keeps IDs across frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Track on this frame
        # classes=[0] filters to person on COCO models
        classes = [0] if args.only_person else None

        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=args.conf,
            iou=args.iou,
            classes=classes,
            verbose=False,
        )

        # Parse detections
        persons_this_frame = 0
        if results and len(results) > 0:
            r = results[0]
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                # If ids exist, tracking is active
                ids = boxes.id
                xyxy = boxes.xyxy
                confs = boxes.conf
                clss = boxes.cls

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, xyxy[i].tolist())
                    conf = float(confs[i].item()) if confs is not None else 0.0
                    cls = int(clss[i].item()) if clss is not None else -1
                    tid = int(ids[i].item()) if ids is not None and ids[i] is not None else -1

                    # If not filtering by model and only_person flag, you could double-check cls==0
                    if args.only_person and cls != 0:
                        continue

                    persons_this_frame += 1

                    # Update coarse heatmap: map bbox to grid coords
                    gx1 = clamp(int(x1 / W * gw), 0, gw - 1)
                    gx2 = clamp(int(np.ceil(x2 / W * gw)), 0, gw)
                    gy1 = clamp(int(y1 / H * gh), 0, gh - 1)
                    gy2 = clamp(int(np.ceil(y2 / H * gh)), 0, gh)

                    heat[gy1:gy2, gx1:gx2] += 1.0

                    # Update tracking stats if we have a valid id
                    if tid >= 0:
                        frames_present[tid] = frames_present.get(tid, 0) + 1
                        if tid not in first_frame:
                            first_frame[tid] = frame_idx
                        last_frame[tid] = frame_idx

                    # Draw overlay
                    if writer is not None:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        label = f"person {tid}" if tid >= 0 else "person"
                        cv2.putText(
                            frame, f"{label} {conf:.2f}",
                            (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA
                        )

        max_simul = max(max_simul, persons_this_frame)
        if persons_this_frame > 0:
            frames_with_any_person += 1

        if writer is not None:
            cv2.putText(
                frame,
                f"frame {frame_idx} | persons {persons_this_frame} | max {max_simul}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    processed_frames = frame_idx
    if processed_frames == 0:
        raise RuntimeError("No frames processed.")

    # Normalize heatmap by number of processed frames to get presence ratio (0..1)
    heat_norm = heat / float(processed_frames)

    # Upscale heatmap to full resolution for visualization & ROI variable in full res
    heat_full = cv2.resize(heat_norm, (W, H), interpolation=cv2.INTER_LINEAR)

    # Variable ROI: based on min_presence_ratio
    roi_var = compute_variable_roi_from_heatmap(heat_full, args.min_presence_ratio)

    # Fixed ROI: find best window on the GRID heatmap for speed, then map back to full res
    roi_fw, roi_fh = args.roi_fixed
    # Convert desired ROI to grid units
    roi_gw = max(1, int(round(roi_fw / W * gw)))
    roi_gh = max(1, int(round(roi_fh / H * gh)))
    best_grid = compute_best_fixed_roi(heat_norm, roi_gw, roi_gh, stride=args.stride)
    bgx1, bgy1, bgx2, bgy2 = best_grid
    # Map back to full resolution
    x1 = int(bgx1 / gw * W)
    y1 = int(bgy1 / gh * H)
    x2 = int(bgx2 / gw * W)
    y2 = int(bgy2 / gh * H)
    roi_fix = (x1, y1, x2, y2)

    # Save heatmap image
    heat_img = (heat_full * 255.0).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_img, cv2.COLORMAP_JET)
    heat_path = os.path.join(args.outdir, "roi_heatmap.png")
    cv2.imwrite(heat_path, heat_color)

    # Compute times
    person_stats: List[PersonStats] = []
    total_person_time_sum = 0.0
    for tid, fr in sorted(frames_present.items(), key=lambda kv: kv[0]):
        tsec = fr / fps
        total_person_time_sum += tsec
        person_stats.append(
            PersonStats(
                track_id=tid,
                frames_present=fr,
                time_seconds=tsec,
                first_frame=first_frame.get(tid, -1),
                last_frame=last_frame.get(tid, -1),
            )
        )

    total_time_with_any_person = frames_with_any_person / fps
    gating = total_time_with_any_person >= args.gating_seconds

    # ROI area ratios
    def area_ratio(roi: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = roi
        return float(max(0, x2 - x1) * max(0, y2 - y1)) / float(W * H)

    res = Results(
        video_path=args.video,
        fps=float(fps),
        total_frames=int(total_frames) if total_frames > 0 else processed_frames,
        frame_size=(W, H),
        unique_person_ids=len(frames_present),
        max_simultaneous_persons=int(max_simul),
        total_time_with_any_person=float(total_time_with_any_person),
        total_person_time_sum=float(total_person_time_sum),
        person_stats=person_stats,
        roi_bbox_variable=roi_var,
        roi_bbox_fixed=roi_fix,
        roi_fixed_size=(roi_fw, roi_fh),
        roi_area_ratio_variable=area_ratio(roi_var),
        roi_area_ratio_fixed=area_ratio(roi_fix),
        gating_threshold_seconds=float(args.gating_seconds),
        gating_decision_run_heavy_model=bool(gating),
        outputs={
            "overlay_video": overlay_path if args.save_overlay else "",
            "heatmap_png": heat_path,
            "results_json": os.path.join(args.outdir, "results.json"),
        },
    )

    # Write JSON
    def to_jsonable(obj):
        if isinstance(obj, (PersonStats, Results)):
            return asdict(obj)
        raise TypeError

    json_path = os.path.join(args.outdir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=to_jsonable)

    print("Done.")
    print(f"results.json: {json_path}")
    print(f"heatmap:      {heat_path}")
    if args.save_overlay:
        print(f"overlay:      {overlay_path}")
    print(f"gating_decision_run_heavy_model: {gating}")


if __name__ == "__main__":
    main()

