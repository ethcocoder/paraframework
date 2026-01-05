import math
from ..module import Module, Parameter
from ..tensor import Tensor
from ..ops.math_ops import MatMul, Add

class Linear(Module):
    """Applies a linear transformation to the incoming data: y = xA^T + b"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(Tensor.uniform(in_features, out_features, a=-1/math.sqrt(in_features), b=1/math.sqrt(in_features)))
        if bias:
            self.bias = Parameter(Tensor.uniform(out_features, a=-1/math.sqrt(in_features), b=1/math.sqrt(in_features)))
        else:
            self.bias = None

    def forward(self, input):
        output = MatMul.apply(input, self.weight)
        if self.bias is not None:
            output = Add.apply(output, self.bias)
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class PatchEmbedding(Module):
    """
    Splits image into patches and projects them.
    Equivalent to Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size).
    """
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, NumPatches, EmbedDim)
        """
        # x: (B, C, H, W)
        # Using .data for direct reshape manipulations with backend
        from ..device import device
        xp = device.backend
        
        B, C, H, W = x.shape
        P = self.patch_size
        
        # Check dimensions
        if H % P != 0 or W % P != 0:
             # In experimental usage, we assume standard sizing or resize before calling
             raise ValueError(f"Image dimensions ({H},{W}) must be divisible by patch size ({P})")
            
        num_patches_h = H // P
        num_patches_w = W // P
        
        # 1. Reshape to separate patches
        # (B, C, H, W) -> (B, C, Nh, P, Nw, P)
        x_reshaped = x.data.reshape(B, C, num_patches_h, P, num_patches_w, P)
        
        # 2. Transpose to group patches together
        # (B, C, Nh, P, Nw, P) -> (B, Nh, Nw, C, P, P)
        x_transposed = x_reshaped.transpose(0, 2, 4, 1, 3, 5)
        
        # 3. Flatten the patch content
        # (B, Nh, Nw, C, P, P) -> (B, Nh*Nw, C*P*P)
        patches = x_transposed.reshape(B, num_patches_h * num_patches_w, C * P * P)
        
        # 4. Project using Linear layer
        patches_tensor = Tensor(patches, device_type=x.device)
        return self.proj(patches_tensor)
