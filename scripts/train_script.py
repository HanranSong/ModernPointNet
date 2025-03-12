import argparse
import os
import random
import time
import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter

# Import our modernized modules
from model.modern_point_net import ModernPointNet, PointNetLoss, build_modern_point_net
from utils.dataset_loader import build_crowd_dataset, collate_fn

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train ModernPointNet for crowd counting')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for the dataset')
    parser.add_argument('--dataset', type=str, default='SHHA',
                        help='Dataset name: SHHA, SHHB, UCF_QNRF, etc.')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--eval_freq', type=int, default=5,
                        help='Frequency of evaluation during training')
    
    # Model parameters
    parser.add_argument('--row', type=int, default=2,
                        help='Number of anchor point rows per grid cell')
    parser.add_argument('--line', type=int, default=2,
                        help='Number of anchor point columns per grid cell')
    parser.add_argument('--point_loss_coef', type=float, default=0.0002,
                        help='Weight for point regression loss')
    
    # I/O parameters
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    
    return parser.parse_args()

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_loss_ce = 0.0
    running_loss_points = 0.0
    
    start_time = time.time()
    
    for i, (images, targets) in enumerate(data_loader):
        # Move data to device
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss_dict = criterion(outputs, targets)
        loss = loss_dict.sum()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update running losses
        running_loss += loss.item()
        
        # Print progress
        if i % 10 == 0:
            batch_time = time.time() - start_time
            print(f'Epoch: [{epoch}][{i}/{len(data_loader)}] '
                  f'Loss: {loss.item():.4f} '
                  f'Time: {batch_time:.2f}s')
            start_time = time.time()
    
    # Calculate average losses
    avg_loss = running_loss / len(data_loader)
    
    return {
        'loss': avg_loss,
    }

@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate the model on validation set"""
    model.eval()
    
    mae_list = []
    mse_list = []
    mape_list = []  # new list for MAPE
    
    for images, targets in data_loader:
        # Move data to device
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Get predictions
        scores = torch.softmax(outputs['pred_logits'], dim=-1)[:, :, 1]
        pred_points = outputs['pred_points']
        
        # Process each image in the batch
        for i in range(images.shape[0]):
            # Get ground truth count
            gt_count = len(targets[i][0]['point'])
            
            # Apply confidence threshold
            threshold = 0.5
            pred_mask = scores[i] > threshold
            pred_count = int(pred_mask.sum().item())
            
            # Calculate MAE and MSE
            mae = abs(pred_count - gt_count)
            mse = (pred_count - gt_count) ** 2
            
            # Calculate MAPE, using epsilon to avoid division by zero.
            epsilon = 1e-8
            mape = 100 * abs(pred_count - gt_count) / (gt_count + epsilon)
            
            mae_list.append(mae)
            mse_list.append(mse)
            mape_list.append(mape)
    
    # Calculate final metrics
    mae = np.mean(mae_list)
    mse = np.sqrt(np.mean(mse_list))
    mape = np.mean(mape_list)
    
    return mae, mse, mape


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize logger
    log_file = os.path.join(args.log_dir, 'train_log.txt')
    with open(log_file, 'w') as f:
        f.write(f'Training started at {datetime.datetime.now()}\n')
        f.write(f'Args: {args}\n\n')
    
    # Initialize tensorboard
    writer = SummaryWriter(args.log_dir)
    
    # Create datasets
    train_dataset = build_crowd_dataset(args.data_root, args.dataset, train=True)
    val_dataset = build_crowd_dataset(args.data_root, args.dataset, train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Build model and criterion
    weight_dict = {'loss_ce': 1.0, 'loss_points': args.point_loss_coef}
    model, criterion = build_modern_point_net(pretrained_weights=args.resume, num_classes=1)
    
    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Create optimizer and scheduler
    params = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": args.lr * 0.1},
    ]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs - 1, gamma=0.1)
    
    # Training loop
    best_mape = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if args.resume is not None and os.path.exists(args.resume):
        print(f'Loading checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            print(f'Resuming from epoch {start_epoch}')
    
    print('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_stats = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Log training statistics
        for k, v in train_stats.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        
        # Save only the latest checkpoint
        latest_path = os.path.join(args.save_dir, 'checkpoint_latest.pth')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args
        }, latest_path)
        
        # Evaluate periodically
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            mae, mse, mape = evaluate(model, val_loader, device)
            
            # Log evaluation metrics
            writer.add_scalar('val/mae', mae, epoch)
            writer.add_scalar('val/mse', mse, epoch)
            writer.add_scalar('val/mape', mape, epoch)
            
            with open(log_file, 'a') as f:
                f.write(f'Epoch {epoch}: MAE={mae:.2f}, MSE={mse:.2f}, MAPE={mape:.2f}%\n')
            
            print(f'Epoch {epoch} evaluation: MAE={mae:.2f}, MSE={mse:.2f}, MAPE={mape:.2f}%')
            
            # Save best model if current MAE improves
            if mape < best_mape:
                best_mape = mape
                best_path = os.path.join(args.save_dir, 'checkpoint_best.pth')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'mae': mae,
                    'mse': mse,
                    'mape': mape,
                    'args': args
                }, best_path)
                print(f'New best model saved with MAPE: {mape:.2f}')

    
    # Final evaluation
    final_mae, final_mse, final_mape = evaluate(model, val_loader, device)
    print(f'Final evaluation: MAE={final_mae:.2f}, MSE={final_mse:.2f}, MAPE={final_mape:.2f}')
    
    with open(log_file, 'a') as f:
        f.write(f'Training completed at {datetime.datetime.now()}\n')
        f.write(f'Final evaluation: MAE={final_mae:.2f}, MSE={final_mse:.2f}, MAPE={final_mape:.2f}\n')
        f.write(f'Best MAE: {best_mape:.2f}\n')
    
    writer.close()
    print('Training completed.')

if __name__ == '__main__':
    main()
