import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

from torchvision.models import VGG16_BN_Weights, ResNet101_Weights

# # VGG16
# class FeatureExtractor(nn.Module):
#     """
#     Feature extraction backbone based on VGG16 with batch normalization.
#     Returns multi-scale features from different VGG blocks.
#     """
#     def __init__(self, pretrained=True):
#         super(FeatureExtractor, self).__init__()
#         # Load pretrained VGG16 with batch normalization
#         # vgg = torchvision.models.vgg16_bn(pretrained=pretrained)
#         vgg = torchvision.models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
#         features = list(vgg.features.children())
        
#         # Extract features at different scales
#         self.block1 = nn.Sequential(*features[:13])    # 1/4 scale
#         self.block2 = nn.Sequential(*features[13:23])  # 1/8 scale
#         self.block3 = nn.Sequential(*features[23:33])  # 1/16 scale
#         self.block4 = nn.Sequential(*features[33:43])  # 1/32 scale
        
#     def forward(self, x):
#         # Extract features at different scales
#         feat1 = self.block1(x)         # 1/4 scale
#         feat2 = self.block2(feat1)     # 1/8 scale
#         feat3 = self.block3(feat2)     # 1/16 scale
#         feat4 = self.block4(feat3)     # 1/32 scale
        
#         return [feat1, feat2, feat3, feat4]

# ResNet50
class FeatureExtractor(nn.Module):
    """
    Feature extraction backbone based on ResNet-50.
    Extracts multi-scale features from different ResNet layers.
    """
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        # Load pretrained ResNet-50; if pretrained is False, no weights are loaded.
        resnet = torchvision.models.resnet101(weights=ResNet101_Weights.DEFAULT if pretrained else None)
        # The initial block (conv, bn, relu, maxpool)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        # Use the four main layers as feature extractors.
        self.layer1 = resnet.layer1  # Output channels: 256, 1/4 scale
        self.layer2 = resnet.layer2  # Output channels: 512, 1/8 scale
        self.layer3 = resnet.layer3  # Output channels: 1024, 1/16 scale
        self.layer4 = resnet.layer4  # Output channels: 2048, 1/32 scale

    def forward(self, x):
        x = self.layer0(x)
        feat1 = self.layer1(x)   # 1/4 scale
        feat2 = self.layer2(feat1)  # 1/8 scale
        feat3 = self.layer3(feat2)  # 1/16 scale
        feat4 = self.layer4(feat3)  # 1/32 scale
        return [feat1, feat2, feat3, feat4]

# # VGG16
# class FeaturePyramidNetwork(nn.Module):
#     """
#     Feature Pyramid Network for multi-scale feature fusion.
#     Enhanced with residual connections for better gradient flow.
#     """
#     def __init__(self, feature_channels=[64, 128, 256, 512], feature_size=512):
#         super(FeaturePyramidNetwork, self).__init__()
        
#         # Project input features to the same dimension
#         self.p5_conv = nn.Conv2d(feature_channels[3], feature_size, kernel_size=1)
#         self.p4_conv = nn.Conv2d(feature_channels[2], feature_size, kernel_size=1)  # Changed from [2] to [3]
#         self.p3_conv = nn.Conv2d(feature_channels[1], feature_size, kernel_size=1)
        
#         # Smooth the upsampled features
#         self.p5_smooth = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
#         self.p4_smooth = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
#         self.p3_smooth = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        
#         # Residual blocks for better feature representation
#         self.p5_residual = ResidualBlock(feature_size)
#         self.p4_residual = ResidualBlock(feature_size)
#         self.p3_residual = ResidualBlock(feature_size)
        
#     def forward(self, features):
#         c2, c3, c4, c5 = features
        
#         # Top-down pathway with residual connections
#         p5 = self.p5_conv(c5)  # Use p5_conv for c5
#         p5 = self.p5_residual(p5)
        
#         p4 = self.p4_conv(c4)  # FIXED: Use p4_conv for c4
#         p4_up = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
#         p4 = p4 + p4_up
#         p4 = self.p4_smooth(p4)
#         p4 = self.p4_residual(p4)
        
#         p3 = self.p3_conv(c3)  # Use p3_conv for c3
#         p3_up = F.interpolate(p4, size=p3.shape[2:], mode='nearest')
#         p3 = p3 + p3_up
#         p3 = self.p3_smooth(p3)
#         p3 = self.p3_residual(p3)
        
#         return [p3, p4, p5]

# ResNet50
class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature fusion.
    Enhanced with residual connections for better gradient flow.
    """
    def __init__(self, feature_channels=[256, 512, 1024, 2048], feature_size=512):
        super(FeaturePyramidNetwork, self).__init__()
        
        # Project input features to the same dimension
        self.p5_conv = nn.Conv2d(feature_channels[3], feature_size, kernel_size=1)
        self.p4_conv = nn.Conv2d(feature_channels[2], feature_size, kernel_size=1)
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
        p5 = self.p5_conv(c5)
        p5 = self.p5_residual(p5)
        
        p4 = self.p4_conv(c4)
        p4_up = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p4 = p4 + p4_up
        p4 = self.p4_smooth(p4)
        p4 = self.p4_residual(p4)
        
        p3 = self.p3_conv(c3)
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
    def __init__(self, pyramid_levels=[4], strides=None, row=2, line=2):  # 3 for VGG16
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
    def __init__(self, num_classes=1, row=2, line=2, feature_size=512):
        super(ModernPointNet, self).__init__()
        self.backbone = FeatureExtractor(pretrained=True)
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], feature_size)
        self.point_count = row * line
        self.point_head = PointHead(feature_size, self.point_count, feature_size)
        self.class_head = ClassificationHead(feature_size, self.point_count, num_classes + 1, feature_size)
        self.anchor_points = AnchorPoints(pyramid_levels=[4], row=row, line=line)
        
        # Instantiate your attention module (e.g. CBAM or SpatialAttention)
        self.attention = CBAMAttention(feature_size)
        
        # Instantiate the multi-scale fusion module.
        # Here we assume both scales output features with `feature_size` channels.
        self.multi_scale_fusion = MultiScaleAttentionFusion([feature_size, feature_size], feature_size)

    def forward(self, x):
        # Extract multi-scale features
        features = self.backbone(x)
        pyramid_features = self.fpn(features)
        
        # Apply attention to the chosen pyramid level (p4 in this case)
        enhanced_features = self.attention(pyramid_features[1])
        
        # Generate predictions from enhanced features
        point_offsets = self.point_head(enhanced_features) * 100  # Scale factor for training
        class_scores = self.class_head(enhanced_features)
        
        # Determine the spatial size of enhanced_features
        _, _, h, w = enhanced_features.shape
        # Create a dummy tensor with the expected input size that corresponds to these features.
        # For example, if the downsampling factor is 16, then the dummy image size is (h*16, w*16).
        dummy = torch.zeros(x.shape[0], 3, h * 16, w * 16, device=x.device)
        # Generate anchors from this dummy tensor.
        anchors = self.anchor_points(dummy).repeat(x.shape[0], 1, 1)
        
        # Now, the predicted point offsets and the anchors have matching numbers.
        predicted_points = point_offsets + anchors
        
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
    
class CBAMAttention(nn.Module):
    """
    An improved attention module integrating both channel and spatial attention.
    Inspired by CBAM, this module first applies channel attention to recalibrate
    the feature map and then applies spatial attention to emphasize informative regions.
    A residual connection is added to ease gradient flow.
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAMAttention, self).__init__()
        # Channel Attention Module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        
        # Spatial Attention Module
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention: aggregate spatial information via average and max pooling,
        # pass through shared MLP, then combine.
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_attn = self.sigmoid_channel(avg_out + max_out)
        x_channel = x * channel_attn

        # Spatial Attention: compute spatial attention by concatenating average and max along channel dim,
        # convolve, and then apply sigmoid.
        avg_out_spatial = torch.mean(x_channel, dim=1, keepdim=True)
        max_out_spatial, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_attn = torch.cat([avg_out_spatial, max_out_spatial], dim=1)
        spatial_attn = self.conv_spatial(spatial_attn)
        spatial_attn = self.sigmoid_spatial(spatial_attn)
        
        # Apply spatial attention and add a residual connection.
        out = x_channel * spatial_attn
        return out + x

class MultiScaleAttentionFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(MultiScaleAttentionFusion, self).__init__()
        # in_channels_list: list of channel dimensions for each attention output
        # We fuse by concatenation and then use a 1x1 conv to reduce dimensions.
        self.fuse_conv = nn.Conv2d(sum(in_channels_list), out_channels, kernel_size=1)

    def forward(self, attn_features):
        # Assume attn_features is a list of attention outputs (e.g., from p3 and p4)
        # Resize all features to the resolution of the first feature map.
        target_size = attn_features[0].shape[2:]
        resized = [feat if feat.shape[2:] == target_size 
                   else F.interpolate(feat, size=target_size, mode='nearest')
                   for feat in attn_features]
        # Concatenate along the channel dimension.
        fused = torch.cat(resized, dim=1)
        # Fuse to the desired number of channels.
        fused = self.fuse_conv(fused)
        return fused

class HungarianMatcher(nn.Module):
    """
    Performs bipartite matching between predictions and ground truth using the Hungarian algorithm.
    This revised implementation computes matching for each sample individually to avoid constructing
    a huge cost matrix across the entire batch.
    """
    def __init__(self, cost_class=1.0, cost_point=1.0):
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        
    @torch.no_grad()
    def forward(self, outputs, targets):
        """Match predictions to ground truth labels for each sample in the batch."""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []
        # Process each sample independently
        for i in range(bs):
            # Get predictions for sample i
            pred_logits = outputs["pred_logits"][i]  # (num_queries, num_classes)
            out_prob = pred_logits.softmax(-1)         # (num_queries, num_classes)
            out_points = outputs["pred_points"][i]       # (num_queries, 2)
            
            # Get target dictionary for the sample
            # If the target is wrapped in a list, extract the first element.
            tgt = targets[i]
            if isinstance(tgt, list):
                tgt = tgt[0]
            
            gt_labels = tgt["labels"]  # (num_points,)
            gt_points = tgt["point"]   # (num_points, 2)
            
            # Compute classification cost: negative probabilities for the target classes
            # Resulting shape: (num_queries, num_points)
            cost_class = -out_prob[:, gt_labels]
            
            # Compute point regression cost using Euclidean distance
            cost_point = torch.cdist(out_points, gt_points, p=2)
            
            # Combine the two costs
            C = self.cost_point * cost_point + self.cost_class * cost_class
            C_cpu = C.cpu().detach().numpy()
            
            # Compute optimal assignment for the current sample using Hungarian algorithm
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(C_cpu)
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64),
                            torch.as_tensor(col_ind, dtype=torch.int64)))
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
