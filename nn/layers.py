import math
from ..module import Module, Parameter
from ..tensor import Tensor
from ..ops.math_ops import MatMul, Add, Mean, Var, Div, Sub, Mul, Sqrt


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

class LayerNorm(Module):
    """Layer Normalization: y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        
        # Trainable parameters
        self.weight = Parameter(Tensor.ones(*normalized_shape))
        self.bias = Parameter(Tensor.zeros(*normalized_shape))
        
    def forward(self, x):
        # Calculate mean and variance across last dimensions
        axis = tuple(range(-len(self.normalized_shape), 0))
        
        mean = x.mean(axis=axis, keepdims=True)
        var = x.var(axis=axis, keepdims=True)
        
        # x_norm = (x - mean) / sqrt(var + eps)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        
        return x_norm * self.weight + self.bias

class Conv2d(Module):
    """Simple functional Conv2d (for PatchEmbedding/Projection)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        
        # Weight shape for Conv2d: (out_channels, in_channels, kH, kW)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        limit = math.sqrt(6 / fan_in)
        self.weight = Parameter(Tensor.uniform(out_channels, in_channels, *self.kernel_size, a=-limit, b=limit))
        
        if bias:
            self.bias = Parameter(Tensor.zeros(out_channels))
        else:
            self.bias = None
            
    def forward(self, x):
        # For simplicity and speed in this custom framework, 
        # if stride == kernel_size (patch embedding case), we use a faster reshaping implementation
        if self.stride == self.kernel_size and self.padding == 0:
            B, C, H, W = x.shape
            KH, KW = self.kernel_size
            OH, OW = H // KH, W // KW
            
            # (B, C, H, W) -> (B, C, OH, KH, OW, KW) -> (B, OH, OW, C, KH, KW)
            # Reusing PatchEmbedding logic
            # Flatten to (B, OH*OW, C*KH*KW)
            # Then matmul with weight (out_channels, C*KH*KW)
            
            xp = x.data
            x_reshaped = xp.reshape(B, C, OH, KH, OW, KW)
            x_transposed = x_reshaped.transpose(0, 2, 4, 1, 3, 5)
            patches = x_transposed.reshape(B * OH * OW, C * KH * KW)
            
            patches_tensor = Tensor(patches, device_type=x.device, requires_grad=x.requires_grad)
            
            # Flatten weight for matmul
            w_flat = self.weight.reshape((self.out_channels, C * KH * KW)).T # (C*KH*KW, out_channels)
            
            out = patches_tensor @ w_flat # (B*OH*OW, out_channels)
            
            if self.bias is not None:
                out = out + self.bias
                
            # Reshape back to (B, out_channels, OH, OW)
            out = out.reshape((B, OH, OW, self.out_channels))
            out = out.transpose((0, 3, 1, 2))
            
            return out
        else:
            raise NotImplementedError("General Conv2d not implemented yet - use stride=kernel_size for patches")

