import argparse
import numpy as np
from pathlib import Path
import cv2

def save_npz_as_mp4(npz_path: Path, out_path: Path, fps: float):
    data = np.load(npz_path)
    x = data["x"]  # [C,T,H,W]
    C, T, H, W = x.shape
    frames = np.transpose(x, (1, 2, 3, 0))  # [T,H,W,C]
    frames = (frames * 255.0).clip(0, 255).astype(np.uint8)

    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(T):
        bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        out.write(bgr)
    out.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--pattern", default="negcrop_best_*.npz")
    ap.add_argument("--outdir", default="inspect_negatives_best")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--fps", type=float, default=2.0)
    args = ap.parse_args()

    ds = Path(args.data_dir)
    clips_dir = ds / f"clips/{args.split}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(clips_dir.glob(args.pattern))
    print("found:", len(files))
    files = files[:args.limit]

    for i, npz in enumerate(files, 1):
        out_mp4 = outdir / f"{npz.stem}.mp4"
        save_npz_as_mp4(npz, out_mp4, args.fps)
        if i % 25 == 0 or i == len(files):
            print(f"exported {i}/{len(files)}")

    print("done:", outdir)

if __name__ == "__main__":
    main()
