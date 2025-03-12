import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

class FeatureExtractor(nn.Module):
    """
    Feature extraction backbone based on VGG16 with batch normalization.
    Returns multi-scale features from different VGG blocks.
    """
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        # Load pretrained VGG16 with batch normalization
        vgg = torchvision.models.vgg16_bn(pretrained=pretrained)
        features = list(vgg.features.children())
        
        # Extract features at different scales
        self.block1 = nn.Sequential(*features[:13])    # 1/4 scale
        self.block2 = nn.Sequential(*features[13:23])  # 1/8 scale
        self.block3 = nn.Sequential(*features[23:33])  # 1/16 scale
        self.block4 = nn.Sequential(*features[33:43])  # 1/32 scale
        
    def forward(self, x):
        # Extract features at different scales
        feat1 = self.block1(x)         # 1/4 scale
        feat2 = self.block2(feat1)     # 1/8 scale
        feat3 = self.block3(feat2)     # 1/16 scale
        feat4 = self.block4(feat3)     # 1/32 scale
        
        return [feat1, feat2, feat3, feat4]

class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.
    Enhanced with residual connections for better gradient flow.
    """
    def __init__(self, feature_channels=[64, 128, 256, 512], feature_size=512):
        super(FeaturePyramidNetwork, self).__init__()
        
        # Project input features to the same dimension
        self.p5_conv = nn.Conv2d(feature_channels[3], feature_size, kernel_size=1)
        self.p4_conv = nn.Conv2d(feature_channels[2], feature_size, kernel_size=1)  # Changed from [2] to [3]
        self.p3_conv = nn.Conv2d(feature_channels[1], feature_size, kernel_size=1)
        
        # Smooth the upsampled features
        self.p5_smooth = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.p4_smooth = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.p3_smooth = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        
        # Residual blocks for better feature representation
        self.p5_residual = ResidualBlock(feature_size)
        self.p4_residual = ResidualBlock(feature_size)
        self.p3_residual = ResidualBlock(feature_size)
        
    def forward(self, features):
        c2, c3, c4, c5 = features
        
        # Top-down pathway with residual connections
        p5 = self.p5_conv(c5)  # Use p5_conv for c5
        p5 = self.p5_residual(p5)
        
        p4 = self.p4_conv(c4)  # FIXED: Use p4_conv for c4
        p4_up = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p4 = p4 + p4_up
        p4 = self.p4_smooth(p4)
        p4 = self.p4_residual(p4)
        
        p3 = self.p3_conv(c3)  # Use p3_conv for c3
        p3_up = F.interpolate(p4, size=p3.shape[2:], mode='nearest')
        p3 = p3 + p3_up
        p3 = self.p3_smooth(p3)
        p3 = self.p3_residual(p3)
        
        return [p3, p4, p5]

class ResidualBlock(nn.Module):
    """
    Simple residual block for improving feature representation.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class PointHead(nn.Module):
    """
    Head network for point prediction (coordinates).
    Simplified and optimized compared to original implementation.
    """
    def __init__(self, in_channels, point_count=4, feature_size=512):
        super(PointHead, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Output layer for point coordinates
        self.output = nn.Conv2d(feature_size, point_count * 2, kernel_size=1)
        
    def forward(self, x):
        features = self.conv_layers(x)
        output = self.output(features)
        
        # Reshape to (batch_size, H*W*point_count, 2)
        batch_size = output.size(0)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(batch_size, -1, 2)
        
        return output

class ClassificationHead(nn.Module):
    """
    Head network for classification (confidence scores).
    Simplified and optimized compared to original implementation.
    """
    def __init__(self, in_channels, point_count=4, num_classes=2, feature_size=512):
        super(ClassificationHead, self).__init__()
        
        self.num_classes = num_classes
        self.point_count = point_count
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, feature_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Output layer for classification
        self.output = nn.Conv2d(feature_size, point_count * num_classes, kernel_size=1)
        
    def forward(self, x):
        features = self.conv_layers(x)
        output = self.output(features)
        
        # Reshape to (batch_size, H*W*point_count, num_classes)
        batch_size = output.size(0)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(batch_size, -1, self.num_classes)
        
        return output

class AnchorPoints(nn.Module):
    """
    Generate grid-based anchor points for regression.
    Simplified with clearer parameters and improved memory efficiency.
    """
    def __init__(self, pyramid_levels=[3], strides=None, row=2, line=2):
        super(AnchorPoints, self).__init__()
        
        self.pyramid_levels = pyramid_levels
        self.strides = [2 ** x for x in pyramid_levels] if strides is None else strides
        self.row = row
        self.line = line
        
    def generate_points_grid(self, stride, row, line):
        """Generate anchor points within a single grid cell"""
        row_step = stride / row
        line_step = stride / line
        
        # Calculate point shifts within the cell
        shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
        shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
        
        # Create a grid of points
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        points = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
        
        return points
    
    def shift_points_to_image(self, image_shape, stride, anchor_points):
        """Shift anchor points to cover the entire image"""
        # Calculate grid centers across image
        shift_x = (np.arange(0, image_shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, image_shape[0]) + 0.5) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        
        # Create shift vectors for all positions
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
        
        # Add anchor points to each position
        point_count = anchor_points.shape[0]
        position_count = shifts.shape[0]
        
        # Reshape for broadcasting and perform addition
        all_points = (anchor_points.reshape((1, point_count, 2)) + 
                      shifts.reshape((position_count, 1, 2))).reshape((-1, 2))
        
        return all_points
        
    def forward(self, image):
        """Generate all anchor points for the image"""
        # Get image shape
        image_shape = image.shape[2:]
        image_shapes = [(np.array(image_shape) + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        
        # Generate all anchor points across pyramid levels
        all_points = []
        for idx, level in enumerate(self.pyramid_levels):
            # Generate base anchor points for a single grid cell
            base_points = self.generate_points_grid(2**level, self.row, self.line)
            # Shift these points across the entire image
            shifted_points = self.shift_points_to_image(image_shapes[idx], self.strides[idx], base_points)
            all_points.append(shifted_points)
            
        # Combine all points
        all_points = np.concatenate(all_points, axis=0)
        all_points = np.expand_dims(all_points, axis=0)  # Add batch dimension
        
        # Convert to tensor and move to correct device
        points_tensor = torch.from_numpy(all_points.astype(np.float32))
        if image.is_cuda:
            points_tensor = points_tensor.cuda()
            
        return points_tensor

class ModernPointNet(nn.Module):
    """
    Modern implementation of P2PNet for crowd counting.
    Simplified architecture with clear components and enhanced feature extraction.
    """
    def __init__(self, num_classes=1, row=2, line=2, feature_size=512):  # feature_size=256
        super(ModernPointNet, self).__init__()
        
        # Feature extraction and fusion
        self.backbone = FeatureExtractor(pretrained=True)
        # Changed channel configuration to match VGG16 feature maps
        self.fpn = FeaturePyramidNetwork([64, 256, 512, 512], feature_size)
        
        # Number of anchor points per grid cell
        self.point_count = row * line
        
        # Point prediction and classification heads
        self.point_head = PointHead(feature_size, self.point_count, feature_size)
        self.class_head = ClassificationHead(feature_size, self.point_count, num_classes + 1, feature_size)
        
        # Generate anchor points
        self.anchor_points = AnchorPoints(pyramid_levels=[3], row=row, line=line)
        
        # Optional attention mechanism for better feature focus
        self.attention = SpatialAttention(feature_size)
        
    def forward(self, x):
        """Forward pass through the network"""
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Build feature pyramid
        pyramid_features = self.fpn(features)
        
        # Apply attention to mid-level features
        enhanced_features = self.attention(pyramid_features[1])
        
        # Generate predictions
        point_offsets = self.point_head(enhanced_features) * 100  # Scale factor for better training
        class_scores = self.class_head(enhanced_features)
        
        # Get anchor points and repeat for batch
        batch_size = x.shape[0]
        anchors = self.anchor_points(x).repeat(batch_size, 1, 1)
        
        # Add offsets to anchors to get final point predictions
        predicted_points = point_offsets + anchors
        
        # Return results
        outputs = {
            'pred_logits': class_scores,  
            'pred_points': predicted_points
        }
        
        return outputs

class SpatialAttention(nn.Module):
    """
    A simple spatial attention mechanism to enhance feature representation.
    This is an innovation not present in the original P2PNet.
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        
    def forward(self, x):
        # Generate attention map
        attn = F.relu(self.conv1(x))
        attn = self.conv2(attn)
        attn = torch.sigmoid(attn)
        
        # Apply attention
        return x * attn + x  # Residual connection

class HungarianMatcher(nn.Module):
    """
    Performs bipartite matching between predictions and ground truth using Hungarian algorithm.
    Simplified implementation with clear cost calculation.
    """
    def __init__(self, cost_class=1.0, cost_point=1.0):
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """Match predictions to ground truth labels"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Compute classification cost
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]
        
        # Compute point distance cost
        out_points = outputs["pred_points"].flatten(0, 1)
        tgt_points = torch.cat([v["point"] for v in targets])
        cost_point = torch.cdist(out_points, tgt_points, p=2)
        
        # Combine costs
        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()
        
        # Split costs by target and run Hungarian algorithm
        sizes = [len(v["point"]) for v in targets]
        indices = []
        
        # For each image in the batch
        start_idx = 0
        for i, size in enumerate(sizes):
            if size > 0:
                # Extract cost matrix for this image
                cost_matrix = C[i, :, start_idx:start_idx + size]
                # Run Hungarian algorithm (linear_sum_assignment)
                from scipy.optimize import linear_sum_assignment
                pred_idx, gt_idx = linear_sum_assignment(cost_matrix.numpy())
                indices.append((torch.as_tensor(pred_idx, dtype=torch.int64), 
                               torch.as_tensor(gt_idx, dtype=torch.int64)))
            else:
                indices.append((torch.tensor([], dtype=torch.int64), 
                               torch.tensor([], dtype=torch.int64)))
            start_idx += size
            
        return indices

class PointNetLoss(nn.Module):
    """
    Combined loss function for the ModernPointNet.
    Includes classification loss and point regression loss.
    """
    def __init__(self, num_classes=1, weight_dict=None, eos_coef=0.1):
        super(PointNetLoss, self).__init__()
        
        # Default weight dictionary if none provided
        if weight_dict is None:
            weight_dict = {'loss_ce': 1.0, 'loss_points': 0.0002}
            
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_class=1.0, cost_point=0.05)
        self.weight_dict = weight_dict
        
        # Add higher weight to the background class
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[0] = eos_coef  # Background class
        self.register_buffer('empty_weight', empty_weight)
        
    def loss_labels(self, outputs, targets, indices):
        """Classification loss calculation"""
        # Extract predicted logits
        pred_logits = outputs['pred_logits']
        
        # Extract matched indices
        idx = self._get_src_permutation_idx(indices)
        
        # Get target classes
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(pred_logits.shape[:2], 0, 
                                   dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o
        
        # Calculate cross entropy loss with class weights
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        
        return {'loss_ce': loss_ce}
    
    def loss_points(self, outputs, targets, indices, num_points):
        """Point regression loss calculation"""
        # Extract predicted points
        pred_points = outputs['pred_points']
        
        # Extract matched indices
        idx = self._get_src_permutation_idx(indices)
        
        # Get source points and target points
        src_points = pred_points[idx]
        target_points = torch.cat([t['point'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        # Calculate MSE loss
        loss_point = F.mse_loss(src_points, target_points, reduction='none')
        
        # Normalize by the number of points
        return {'loss_point': loss_point.sum() / max(num_points, 1)}
    
    def _get_src_permutation_idx(self, indices):
        """Helper to get source permutation indices"""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def forward(self, outputs, targets):
        """Forward pass for loss calculation"""
        # Match predictions to targets
        indices = self.matcher(outputs, targets)
        
        # Calculate total number of points (for normalization)
        num_points = sum(len(t["labels"]) for t in targets)
        
        # Calculate losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_points(outputs, targets, indices, num_points))
        
        # Apply weights to losses
        weighted_losses = {k: self.weight_dict[k] * v for k, v in losses.items() if k in self.weight_dict}
        
        return sum(weighted_losses.values())

def build_modern_point_net(pretrained_weights=None, num_classes=1):
    """
    Build and initialize the ModernPointNet model.
    
    Args:
        pretrained_weights: Path to pretrained weights file
        num_classes: Number of object classes (default is 1 for person counting)
        
    Returns:
        model: The initialized model
        criterion: The loss function for training
    """
    # Create model
    model = ModernPointNet(num_classes=num_classes, row=2, line=2)
    
    # Load pretrained weights if provided
    if pretrained_weights is not None:
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    # Create loss function
    criterion = PointNetLoss(num_classes=num_classes)
    
    return model, criterion
