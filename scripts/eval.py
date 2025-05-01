# scripts/eval.py

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from utils.dataset_loader import build_transforms, collate_fn, CrowdDataset
from model.modern_point_net import build_modern_point_net


def load_pts_from_npy(path):
    """
    Load .npy file and convert:
      - If it's a 2D uint mask: extract (x,y) coords where mask==1
      - Else assume it's already an [N x 2] array of points
    """
    arr = np.load(path)
    if arr.ndim == 2 and arr.dtype in (np.uint8, np.int8, np.int32, np.int64):
        ys, xs = np.nonzero(arr)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
    else:
        pts = arr.astype(np.float32)
    return pts


class CustomInferenceDataset(Dataset):
    """
    - If `annotation_file` is VIA-style (has 'region_shape_attributes'), parse CSV.
    - Else if it has 'image' & 'gt' columns, map images → .npy in gt_dir.
    - Else pair every image in images_dir with same‑basename .npy in gt_dir.
    """

    def __init__(self, images_dir, gt_dir=None, transform=None, annotation_file=None):
        self.transform = transform
        self.samples = []

        # 1) VIA‑style CSV?
        if annotation_file and os.path.isfile(annotation_file):
            df = pd.read_csv(annotation_file)

            if "region_shape_attributes" in df.columns:
                # parse VIA CSV into per-image point lists
                for filename, grp in df.groupby("filename"):
                    img_path = os.path.join(images_dir, filename)
                    if not os.path.isfile(img_path):
                        continue
                    pts = []
                    for _, row in grp.iterrows():
                        attr = json.loads(row["region_shape_attributes"])
                        if "cx" in attr and "cy" in attr:
                            pts.append([attr["cx"], attr["cy"]])
                    pts = np.array(pts, dtype=np.float32)
                    self.samples.append((img_path, pts))

                if not self.samples:
                    raise RuntimeError(
                        f"No VIA‑style samples found in {annotation_file}"
                    )

            # 2) simple mapping CSV?
            elif {"image", "gt"}.issubset(df.columns):
                if gt_dir is None:
                    raise RuntimeError("--gt_dir must be set when using image/gt CSV")
                for _, row in df.iterrows():
                    img_path = os.path.join(images_dir, row["image"])
                    gt_path = os.path.join(gt_dir, row["gt"])
                    if os.path.isfile(img_path) and os.path.isfile(gt_path):
                        pts = load_pts_from_npy(gt_path)
                        self.samples.append((img_path, pts))

                if not self.samples:
                    raise RuntimeError(f"No valid image/gt rows in {annotation_file}")

            else:
                raise RuntimeError(
                    "Unrecognized CSV format: expected VIA 'region_shape_attributes' or 'image,gt' columns."
                )

        # 3) fallback: pair images ↔ .npy masks
        else:
            for fn in sorted(os.listdir(images_dir)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    img_path = os.path.join(images_dir, fn)
                    base = os.path.splitext(fn)[0]
                    gt_path = os.path.join(gt_dir or "", base + ".npy")
                    if os.path.isfile(gt_path):
                        pts = load_pts_from_npy(gt_path)
                        self.samples.append((img_path, pts))

            if not self.samples:
                raise RuntimeError(
                    f"No image/.npy pairs found in {images_dir} & {gt_dir}"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pts = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # unify format: list of {"point":pts}
        return img, [{"point": pts}]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SHHB or do custom inference on arbitrary folders"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/test_dataset_small_crowd",
        help="Root directory for data",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run on --images_dir + (--gt_dir or CSV) instead of SHHB splits",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        help="(Custom mode) Where to find images; default: data_root/images",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        help="(Custom mode) Where to find .npy ground‑truth; default: data_root/gt",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        help="(Custom mode) Optional CSV with VIA or image,gt columns; default: data_root/annotation.csv",
    )
    parser.add_argument(
        "--dataset", type=str, default="SHHB", help="Dataset name (for non‑infer mode)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Confidence threshold for predictions",
    )
    return parser.parse_args()


def evaluate(model, data_loader, device, threshold):
    model.eval()
    mae_list, mse_list, mape_list = [], [], []
    gt_counts, pred_counts = [], []

    for images, targets in data_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)

        scores = torch.softmax(outputs["pred_logits"], dim=-1)[..., 1]
        for i in range(images.size(0)):
            gt_pts = targets[i][0]["point"]
            gt_count = len(gt_pts)
            pred_mask = scores[i] > threshold
            pred_count = int(pred_mask.sum().item())

            mae_list.append(abs(pred_count - gt_count))
            mse_list.append((pred_count - gt_count) ** 2)
            mape_list.append(100 * abs(pred_count - gt_count) / (gt_count + 1e-8))
            gt_counts.append(gt_count)
            pred_counts.append(pred_count)

    mae = np.mean(mae_list)
    rmse = np.sqrt(np.mean(mse_list))
    mape = np.mean(mape_list)

    gt_arr = np.array(gt_counts)
    pred_arr = np.array(pred_counts)
    ss_tot = np.sum((gt_arr - gt_arr.mean()) ** 2)
    ss_res = np.sum((gt_arr - pred_arr) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    return mae, rmse, mape, r2


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    tfm = build_transforms(augment=False)

    if args.infer:
        images_dir = args.images_dir or os.path.join(args.data_root, "images")
        gt_dir = args.gt_dir or os.path.join(args.data_root, "gt")
        ann_file = args.annotation_file or os.path.join(
            args.data_root, "annotation.csv"
        )

        dataset = CustomInferenceDataset(
            images_dir=images_dir,
            gt_dir=gt_dir,
            transform=tfm,
            annotation_file=ann_file,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        print(f"→ Running custom inference on {len(dataset)} samples from {images_dir}")

    else:
        train_ds = CrowdDataset(
            data_root=args.data_root,
            dataset=args.dataset,
            transform=tfm,
            train=True,
            patch_size=128,
            num_patches=0,
            flip=False,
            scale_range=(0.7, 1.3),
        )
        test_ds = CrowdDataset(
            data_root=args.data_root, dataset=args.dataset, transform=tfm, train=False
        )
        full_ds = ConcatDataset([train_ds, test_ds])
        loader = DataLoader(
            full_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        print(f"→ Running SHHB evaluation on {len(full_ds)} samples")

    # build & load model
    model, _ = build_modern_point_net(num_classes=1)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model = model.to(device)

    mae, rmse, mape, r2 = evaluate(model, loader, device, args.threshold)
    mode = "Custom Inference" if args.infer else f"Full {args.dataset} Eval"
    print(f"\n=== {mode} Results ===")
    print(f"MAE   = {mae:.2f}")
    print(f"RMSE  = {rmse:.2f}")
    print(f"MAPE  = {mape:.2f}%")
    print(f"R²    = {r2:.2f}")


if __name__ == "__main__":
    main()
