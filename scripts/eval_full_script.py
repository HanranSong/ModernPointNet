import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from utils.dataset_loader import (
    build_crowd_dataset,
    collate_fn,
    build_transforms,
    CrowdDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate full SHHB (train and test) for MAE, MSE, MAPE, and R2 score"
    )
    parser.add_argument(
        "--data_root", type=str, default="./data", help="Root directory for the dataset"
    )
    parser.add_argument(
        "--dataset", type=str, default="SHHB", help="Dataset name (SHHB)"
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
    mae_list = []
    mse_list = []
    mape_list = []
    gt_counts_list = []
    pred_counts_list = []

    for images, targets in data_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        # Get predicted confidence scores and points
        scores = torch.softmax(outputs["pred_logits"], dim=-1)[:, :, 1]
        pred_points = outputs["pred_points"]

        for i in range(images.shape[0]):
            # The target is a list containing one dictionary
            gt_count = len(targets[i][0]["point"])
            pred_mask = scores[i] > threshold
            pred_count = int(pred_mask.sum().item())

            mae = abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            epsilon = 1e-8
            mape = 100 * abs(pred_count - gt_count) / (gt_count + epsilon)

            mae_list.append(mae)
            mse_list.append(mse)
            mape_list.append(mape)
            gt_counts_list.append(gt_count)
            pred_counts_list.append(pred_count)

    mae_val = np.mean(mae_list)
    mse_val = np.sqrt(np.mean(mse_list))
    mape_val = np.mean(mape_list)

    # Compute R2 score
    gt_counts_array = np.array(gt_counts_list)
    pred_counts_array = np.array(pred_counts_list)
    mean_gt = np.mean(gt_counts_array)
    ss_tot = np.sum((gt_counts_array - mean_gt) ** 2)
    ss_res = np.sum((gt_counts_array - pred_counts_array) ** 2)
    r2_val = 1 - (ss_res / (ss_tot + 1e-8))

    return mae_val, mse_val, mape_val, r2_val


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # For evaluation, we want to load full images from both train_data and test_data.
    # For the train portion, override num_patches and flip so that no augmentation is applied.
    eval_transform = build_transforms(augment=False)

    # Create the train evaluation dataset by forcing num_patches=0 and flip=False
    train_eval_dataset = CrowdDataset(
        data_root=args.data_root,
        dataset=args.dataset,
        transform=eval_transform,
        train=True,  # Use train_data folder
        patch_size=128,
        num_patches=0,  # Do not crop patches
        flip=False,  # No flipping
        scale_range=(0.7, 1.3),  # This parameter is ignored when no scaling is done
    )

    # Create the test evaluation dataset (this one is set to train=False by default)
    test_eval_dataset = CrowdDataset(
        data_root=args.data_root,
        dataset=args.dataset,
        transform=eval_transform,
        train=False,  # Use test_data folder
    )

    # Combine both datasets
    from torch.utils.data import ConcatDataset

    full_dataset = ConcatDataset([train_eval_dataset, test_eval_dataset])
    data_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Build and load the model
    from model.modern_point_net import build_modern_point_net

    model, _ = build_modern_point_net(num_classes=1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    mae, mse, mape, r2 = evaluate(model, data_loader, device, args.threshold)
    print(f"Full {args.dataset} Evaluation:")
    print(f"MAE   = {mae:.2f}")
    print(f"MSE   = {mse:.2f}")
    print(f"MAPE  = {mape:.2f}%")
    print(f"R2    = {r2:.2f}")


if __name__ == "__main__":
    main()
