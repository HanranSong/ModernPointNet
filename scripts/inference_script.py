import argparse
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import scipy.io as io
from model.modern_point_net import ModernPointNet, build_modern_point_net

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run inference with ModernPointNet')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image or directory of images')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save output visualizations')
    parser.add_argument('--threshold', type=float, default=0.75,
                        help='Confidence threshold for predictions')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of predictions')
    
    return parser.parse_args()

def load_image(image_path):
    """Load and preprocess an image for inference"""
    # Read image
    img = Image.open(image_path).convert('RGB')
    
    # Get original size for visualization
    orig_size = img.size
    
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor, img, orig_size

def load_gt_count(image_path):
    """
    Load ground truth count from a .mat file corresponding to image_path.
    Assumes that the ground truth file is located in a directory named 'ground_truth'
    that is a sibling to the 'images' directory.
    E.g., for .../SHHB/train_data/images/IMG_1.jpg, expect .../SHHB/train_data/ground_truth/GT_IMG_1.mat
    """
    # Derive the directory that should contain the ground truth files
    image_dir = os.path.dirname(image_path)
    parent_dir = os.path.dirname(image_dir)
    base_name = os.path.basename(image_path)
    img_id = os.path.splitext(base_name)[0]
    gt_path = os.path.join(parent_dir, "ground_truth", f"GT_{img_id}.mat")
    try:
        mat = io.loadmat(gt_path)
        # Try typical structure (as in ShanghaiTech dataset)
        try:
            points = mat['image_info'][0, 0]['location'][0, 0]
        except:
            if 'annPoints' in mat:
                points = mat['annPoints']
            else:
                # Fallback: find the first array with shape (n, 2)
                points = []
                for key in mat.keys():
                    if isinstance(mat[key], np.ndarray) and len(mat[key].shape) == 2 and mat[key].shape[1] == 2:
                        points = mat[key]
                        break
        return len(np.array(points))
    except Exception as e:
        print(f"Could not load ground truth from {gt_path}: {e}")
        return None

def visualize_predictions(image, points, count, output_path, point_size=3, color=(0, 0, 255)):
    """Visualize prediction points on the image"""
    # Convert PIL image to OpenCV format (RGB to BGR)
    img_cv = np.array(image)
    img_cv = img_cv[:, :, ::-1].copy()
    
    # Draw points
    for point in points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(img_cv, (x, y), point_size, color, -1)
    
    # Add count information (only predicted count here)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Count: {count}"
    cv2.putText(img_cv, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Save visualization
    cv2.imwrite(output_path, img_cv)
    
    return img_cv

def generate_density_map(image_size, points, sigma=3):
    """Generate a density map visualization from points"""
    h, w = image_size
    density_map = np.zeros((h, w), dtype=np.float32)
    
    if len(points) == 0:
        return density_map
    
    # Generate gaussians for each point
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            x0 = np.arange(0, w, 1, float)
            y0 = np.arange(0, h, 1, float)[:, np.newaxis]
            
            x_range = slice(max(0, x - 3*sigma), min(w, x + 3*sigma + 1))
            y_range = slice(max(0, y - 3*sigma), min(h, y + 3*sigma + 1))
            
            y0_f = y0[y_range, 0]
            x0_f = x0[x_range]
            
            y_grid, x_grid = np.meshgrid(y0_f, x0_f, indexing='ij')
            dist_sq = (x_grid - x)**2 + (y_grid - y)**2
            exponent = dist_sq / (2.0 * sigma**2)
            gaussian = np.exp(-exponent)
            
            density_map[y_range, x_range] += gaussian
    
    return density_map

def run_inference(model, image_path, output_dir, threshold=0.75, visualize=True, device='cuda'):
    """Run inference on a single image and visualize results"""
    # Load and preprocess image
    img_tensor, original_img, orig_size = load_image(image_path)
    
    # Load ground truth count
    gt_count = load_gt_count(image_path)
    
    # Move to device
    img_tensor = img_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Get predictions
    scores = torch.softmax(outputs['pred_logits'], dim=-1)[0, :, 1]
    points = outputs['pred_points'][0]
    
    # Filter predictions by threshold
    mask = scores > threshold
    filtered_points = points[mask].cpu().numpy()
    pred_count = len(filtered_points)
    
    # Create output filename
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    
    # Print results, showing predicted and ground truth counts
    if gt_count is not None:
        print(f"Image: {base_name}")
        print(f"Predicted count: {pred_count} | Ground truth: {gt_count}")
    else:
        print(f"Image: {base_name}")
        print(f"Predicted count: {pred_count} (No ground truth available)")
    
    if visualize:
        os.makedirs(output_dir, exist_ok=True)
        
        # Visualize points on image
        vis_path = os.path.join(output_dir, f"{name}_points{ext}")
        vis_img = visualize_predictions(original_img, filtered_points, pred_count, vis_path)
        
        # Generate and save density map
        density_map = generate_density_map((orig_size[1], orig_size[0]), filtered_points)
        density_path = os.path.join(output_dir, f"{name}_density.jpg")
        
        plt.figure(figsize=(10, 8))
        plt.imshow(original_img)
        plt.imshow(density_map, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.colorbar(label='Density')
        if gt_count is not None:
            plt.title(f'Density Map - Predicted: {pred_count} | GT: {gt_count}')
        else:
            plt.title(f'Density Map - Predicted: {pred_count}')
        plt.savefig(density_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # Create combined visualization
        combined_path = os.path.join(output_dir, f"{name}_combined{ext}")
        density_colored = cv2.applyColorMap((density_map * 255 / (density_map.max() + 1e-10)).astype(np.uint8), cv2.COLORMAP_JET)
        img_cv = np.array(original_img)[:, :, ::-1].copy()
        combined = cv2.addWeighted(img_cv, 0.7, density_colored, 0.3, 0)
        
        # Add count text including ground truth if available
        font = cv2.FONT_HERSHEY_SIMPLEX
        if gt_count is not None:
            text = f"Pred: {pred_count} | GT: {gt_count}"
        else:
            text = f"Pred: {pred_count}"
        cv2.putText(combined, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imwrite(combined_path, combined)
    
    return pred_count, filtered_points

def process_directory(model, image_dir, output_dir, threshold=0.75, visualize=True, device='cuda'):
    """Process all images in a directory"""
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f.lower())[1] in supported_formats]
    
    if not image_files:
        print(f"No supported images found in {image_dir}")
        return
    
    results = {}
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        count, points = run_inference(model, image_path, output_dir, threshold, visualize, device)
        results[image_file] = {'count': count, 'points': points}
    
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Image\tPredicted Count\n")
        for image_file, result in results.items():
            f.write(f"{image_file}\t{result['count']}\n")
    
    print(f"Processed {len(results)} images. Summary saved to {summary_path}")

def main():
    args = parse_args()
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading model...")
    model, _ = build_modern_point_net(num_classes=1)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    if os.path.isdir(args.image_path):
        print(f"Processing directory: {args.image_path}")
        process_directory(model, args.image_path, args.output_dir, args.threshold, args.visualize, device)
    else:
        print(f"Processing image: {args.image_path}")
        run_inference(model, args.image_path, args.output_dir, args.threshold, args.visualize, device)
    
    print("Inference completed successfully")

if __name__ == '__main__':
    main()
