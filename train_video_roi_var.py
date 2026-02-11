import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class VideoROIDataset(Dataset):
    def __init__(self, root: Path, split: str):
        self.clips_dir = root / f"clips/{split}"
        self.labels_dir = root / f"labels/{split}"
        self.items = sorted([p for p in self.clips_dir.iterdir() if p.suffix == ".npz"])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        clip_path = self.items[idx]
        name = clip_path.stem
        label_path = self.labels_dir / f"{name}.json"

        data = np.load(clip_path, allow_pickle=True)
        x = torch.from_numpy(data["x"].astype(np.float32))  # [C,T,H,W]

        label = json.loads(label_path.read_text(encoding="utf-8"))
        present = torch.tensor([label["present"]], dtype=torch.float32)
        cx, cy = label["roi_center_norm"]
        s = label["roi_side_norm"]
        y = torch.tensor([cx, cy, s], dtype=torch.float32)  # 3 values

        aux = {
            "name": name,
            "min_side": int(label["min_side"]),
            "orig_w": int(label["orig_w"]),
            "orig_h": int(label["orig_h"]),
            "min_frame_side": int(label["min_frame_side"]),
        }
        return x, present, y, aux


class Simple3DROI(nn.Module):
    """
    Input: [B,C,T,H,W]
    Output:
      p_logit: [B,1]
      reg:     [B,3] -> (cx,cy,side_norm) in [0..1] via sigmoid
    """
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
        reg = torch.sigmoid(self.head_r(z))
        return p_logit, reg


def masked_smooth_l1(pred, target, present):
    """
    pred/target: [B,3], present: [B,1]
    only apply regression loss if present==1
    """
    loss = nn.SmoothL1Loss(reduction="none")(pred, target)  # [B,3]
    w = present  # [B,1]
    return (loss * w).mean()


def train_epoch(model, loader, optim, device):
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total = 0.0

    for x, present, y, _ in loader:
        x = x.to(device)
        present = present.to(device)
        y = y.to(device)

        p_logit, reg = model(x)
        loss_p = bce(p_logit, present)
        loss_r = masked_smooth_l1(reg, y, present)
        loss = loss_p + 3.0 * loss_r

        optim.zero_grad()
        loss.backward()
        optim.step()

        total += float(loss.item())

    return total / max(1, len(loader))


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    total = 0.0

    for x, present, y, _ in loader:
        x = x.to(device)
        present = present.to(device)
        y = y.to(device)

        p_logit, reg = model(x)
        loss_p = bce(p_logit, present)
        loss_r = masked_smooth_l1(reg, y, present)
        loss = loss_p + 3.0 * loss_r
        total += float(loss.item())

    return total / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="video_roi_var.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data_dir)

    ds_tr = VideoROIDataset(root, "train")
    ds_va = VideoROIDataset(root, "val")
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = Simple3DROI(in_ch=3).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 1e9
    for e in range(1, args.epochs + 1):
        tr = train_epoch(model, dl_tr, optim, device)
        va = eval_epoch(model, dl_va, device)
        print(f"epoch {e:03d} | train {tr:.4f} | val {va:.4f}")

        if va < best:
            best = va
            torch.save(model.state_dict(), args.out)
            print(f"  saved: {args.out}")

    print("done. best val:", best)


if __name__ == "__main__":
    main()
