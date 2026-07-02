import torch
from torch import nn
# import ipdb  # Optional debugger, not required for running
import math
import torch.nn.functional as F


class SimpleAdapter(nn.Module):
    def __init__(self, c_in, c_out=768):
        super(SimpleAdapter, self).__init__()
        self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())

    def forward(self, x):
        x = self.fc(x)
        return x


class SimpleProj(nn.Module):
    def __init__(self, c_in, c_out=768, relu=True):
        super(SimpleProj, self).__init__()
        if relu:
            self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())
        else:
            self.fc = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x


class DomainInvariantRPL(nn.Module):
    """
    Domain-Invariant RPL module with ASPP multi-scale convolution and residual connection.
    Processes patch features to enhance domain-invariant representation learning.
    
    Input: Patch features (B, N, 768) where N = H*W (e.g., 37*37 for 518x518 input)
    Output: Enhanced patch features (B, N, 768) with same shape
    """
    
    def __init__(self, in_channels=768, out_channels=768, aspp_channels=192, spatial_size=37):
        """
        Args:
            in_channels: Input feature channels (default: 768)
            out_channels: Output feature channels (default: 768)
            aspp_channels: Channels for each ASPP branch (default: 192)
            spatial_size: Expected spatial size (H=W, default: 37 for 518x518 input)
        """
        super(DomainInvariantRPL, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aspp_channels = aspp_channels
        self.spatial_size = spatial_size
        
        # ASPP branches: 1x1 conv + 3x3 dilated convs (dilation=6, 12, 18)
        self.aspp_1x1 = nn.Conv2d(in_channels, aspp_channels, kernel_size=1, bias=False)
        self.aspp_3x3_d6 = nn.Conv2d(
            in_channels, aspp_channels, kernel_size=3, 
            padding=6, dilation=6, bias=False
        )
        self.aspp_3x3_d12 = nn.Conv2d(
            in_channels, aspp_channels, kernel_size=3,
            padding=12, dilation=12, bias=False
        )
        self.aspp_3x3_d18 = nn.Conv2d(
            in_channels, aspp_channels, kernel_size=3,
            padding=18, dilation=18, bias=False
        )
        
        # Batch normalization for ASPP branches
        self.bn_1x1 = nn.BatchNorm2d(aspp_channels)
        self.bn_3x3_d6 = nn.BatchNorm2d(aspp_channels)
        self.bn_3x3_d12 = nn.BatchNorm2d(aspp_channels)
        self.bn_3x3_d18 = nn.BatchNorm2d(aspp_channels)
        
        # Concatenation projection: 4 * aspp_channels -> out_channels
        self.aspp_proj = nn.Conv2d(4 * aspp_channels, out_channels, kernel_size=1, bias=False)
        self.aspp_bn = nn.BatchNorm2d(out_channels)
        
        # Residual connection: 1x1 conv to align channels
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.residual_bn = nn.BatchNorm2d(out_channels)
        
        # Final activation and normalization
        self.activation = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(out_channels)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Patch features (B, N, C) where N = H*W, C = in_channels
        Returns:
            out: Enhanced patch features (B, N, C) where C = out_channels
        """
        B, N_orig, C = x.shape
        
        # Infer spatial dimensions from N
        # For 518x518 input with patch_size=14: grid_size = 37x37, N = 1369
        # But after removing CLS token or other processing, N might not be a perfect square
        spatial_dim = int(math.sqrt(N_orig))
        
        # Track if we need to pad/truncate and by how much
        needs_padding = False
        pad_size = 0
        N = N_orig
        
        if spatial_dim * spatial_dim == N_orig:
            # Perfect square case
            H = W = spatial_dim
        elif self.spatial_size * self.spatial_size == N_orig:
            # Matches configured spatial_size
            H = W = self.spatial_size
        else:
            # Not a perfect square - try to find best H and W
            # Try to find factors close to sqrt(N)
            best_h, best_w = None, None
            min_diff = float('inf')
            
            # Search around sqrt(N) and spatial_size
            candidates = [spatial_dim, spatial_dim + 1, self.spatial_size, self.spatial_size - 1]
            for h in candidates:
                if h <= 0:
                    continue
                w = (N_orig + h - 1) // h  # Ceiling division
                if h * w == N_orig:
                    # Exact match
                    H, W = h, w
                    break
                diff = abs(h * w - N_orig)
                if diff < min_diff:
                    min_diff = diff
                    best_h, best_w = h, w
            else:
                # Use best match or pad/truncate
                if best_h is not None and min_diff <= N_orig * 0.1:  # Allow up to 10% difference
                    H, W = best_h, best_w
                    N = H * W
                    # Pad or truncate to match H*W
                    if N > N_orig:
                        # Need to pad
                        needs_padding = True
                        pad_size = N - N_orig
                        padding = torch.zeros(B, pad_size, C, device=x.device, dtype=x.dtype)
                        x = torch.cat([x, padding], dim=1)
                    elif N < N_orig:
                        # Need to truncate
                        x = x[:, :N, :]
                else:
                    # Fallback: use spatial_size and pad/truncate
                    H = W = self.spatial_size
                    N = H * W
                    if N > N_orig:
                        # Pad
                        needs_padding = True
                        pad_size = N - N_orig
                        padding = torch.zeros(B, pad_size, C, device=x.device, dtype=x.dtype)
                        x = torch.cat([x, padding], dim=1)
                    elif N < N_orig:
                        # Truncate
                        x = x[:, :N, :]
        
        # Reshape to (B, C, H, W) for convolution operations
        x_2d = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # ASPP branches
        aspp_1x1_out = self.bn_1x1(self.aspp_1x1(x_2d))
        aspp_d6_out = self.bn_3x3_d6(self.aspp_3x3_d6(x_2d))
        aspp_d12_out = self.bn_3x3_d12(self.aspp_3x3_d12(x_2d))
        aspp_d18_out = self.bn_3x3_d18(self.aspp_3x3_d18(x_2d))
        
        # Concatenate all ASPP branches
        aspp_concat = torch.cat([aspp_1x1_out, aspp_d6_out, aspp_d12_out, aspp_d18_out], dim=1)
        
        # Project concatenated features
        aspp_out = self.aspp_proj(aspp_concat)
        aspp_out = self.aspp_bn(aspp_out)
        
        # Residual connection: align input channels and add
        residual = self.residual_conv(x_2d)
        residual = self.residual_bn(residual)
        
        # Add residual and apply activation
        out = self.activation(aspp_out + residual)
        
        # Reshape back to (B, N, C)
        out = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        out = out.reshape(B, N, self.out_channels)  # (B, N, C)
        
        # If we padded input, truncate output to original size
        if needs_padding and pad_size > 0:
            out = out[:, :N_orig, :]
        
        # Apply LayerNorm (normalize over feature dimension)
        out = self.layer_norm(out)
        
        return out