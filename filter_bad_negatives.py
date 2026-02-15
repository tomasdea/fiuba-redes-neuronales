import argparse
import json
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


def has_person_fast(video_path: str, sample_fps: float, clip_seconds: float, conf: float, iou: float) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    stride = max(1, int(round(fps / sample_fps)))
    max_frames = int(round(clip_seconds * fps))

    frames = []
    idx = 0
    grabbed = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= max_frames:
            break
        if idx % stride == 0:
            frames.append(frame)
            grabbed += 1
            # con pocos frames alcanza para decidir “hay persona?”
            if grabbed >= 12:
                break
        idx += 1

    cap.release()
    if not frames:
        return False

    # YOLO predict batch
    results = model.predict(frames, conf=conf, iou=iou, verbose=False)
    for r in results:
        if r.boxes is None:
            continue
        if len(r.boxes) > 0:
            return True
    return False


def move_sample(ds_root: Path, split: str, stem: str, quarantine_root: Path):
    clip_src = ds_root / f"clips/{split}/{stem}.npz"
    lab_src  = ds_root / f"labels/{split}/{stem}.json"

    clip_dst = quarantine_root / f"clips/{split}/{stem}.npz"
    lab_dst  = quarantine_root / f"labels/{split}/{stem}.json"
    clip_dst.parent.mkdir(parents=True, exist_ok=True)
    lab_dst.parent.mkdir(parents=True, exist_ok=True)

    shutil.move(str(clip_src), str(clip_dst))
    shutil.move(str(lab_src), str(lab_dst))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="video_roi_dataset")
    ap.add_argument("--weights", default="yolov8n.pt", help="YOLO weights for person detection")
    ap.add_argument("--sample_fps", type=float, default=4.0)
    ap.add_argument("--clip_seconds", type=float, default=5.0)
    ap.add_argument("--conf", type=float, default=0.12)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max_check", type=int, default=0, help="0 = all negatives, else limit count")
    ap.add_argument("--quarantine_dir", default="video_roi_dataset_quarantine")
    args = ap.parse_args()

    ds = Path(args.data_dir)
    quarantine = Path(args.quarantine_dir)
    quarantine.mkdir(parents=True, exist_ok=True)

    global model
    model = YOLO(args.weights)

    moved = 0
    checked = 0

    for split in ["train", "val"]:
        labels_dir = ds / f"labels/{split}"
        for jf in sorted(labels_dir.glob("*.json")):
            d = json.loads(jf.read_text(encoding="utf-8"))
            present = int(d.get("present", 0))
            roi = d.get("roi_xyxy", [0,0,0,0])

            if not (present == 0 or roi == [0,0,0,0]):
                continue

            video_path = d["video"]
            stem = jf.stem

            checked += 1
            if args.max_check and checked > args.max_check:
                break

            try:
                if has_person_fast(video_path, args.sample_fps, args.clip_seconds, args.conf, args.iou):
                    move_sample(ds, split, stem, quarantine)
                    moved += 1
                    print(f"[MOVE] {split}/{stem} -> quarantine (teacher FN likely)")
            except Exception as e:
                print(f"[ERR] {split}/{stem}: {e}")

        if args.max_check and checked > args.max_check:
            break

    print("checked_negatives:", checked)
    print("moved_to_quarantine:", moved)
    print("quarantine_dir:", quarantine.resolve())
