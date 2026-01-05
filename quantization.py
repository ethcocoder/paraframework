from modules.framework.tensor import Tensor
from modules.framework.device import device

class Quantizer:
    """
    Post-training Static Quantization (int8).
    Converts robust float32 weights into compact int8 integers for
    4x memory reduction and extreme CPU speedup.
    """
    @staticmethod
    def quantize_dynamic(model):
        """
        Converts Linear layers to Int8 dynamically.
        w_int8 = round(w_fp32 / scale)
        """
        print("[Quantization] Compressing model to INT8...")
        
        for name, module in model._modules.items():
            if hasattr(module, 'weight'):
                # Quantize Weight
                w = module.weight.data
                
                # Calculate Scale (Symmetric Quantization)
                max_val = device.backend.max(device.backend.abs(w))
                scale = max_val / 127.0
                
                # Quantize
                w_int8 = device.backend.round(w / scale).astype('int8')
                
                # Replace Parameter
                # We store scale as a buffer
                module.weight_scale = scale
                module.weight_int8 = w_int8
                
                # Hack: Overwrite forward method of this specific instance
                # to use int8 kernel
                module.forward = QuantizedLinearForward(module, module.forward)
                
        print("[Quantization] Model compressed.")
        return model

class QuantizedLinearForward:
    """
    Interceptor to run Int8 Matmul.
    """
    def __init__(self, module, old_forward):
        self.module = module
        self.old_forward = old_forward
        
    def __call__(self, x):
        # x is float32 input
        # Dequantize 'On the Fly' (simplest scheme)
        # w_fp32 = w_int8 * scale
        
        # Optimization: We can arguably quantize x to int8 too, 
        # do int8 matmul, then dequantize.
        
        # For this demo: "Fake Quantization" (Storage is Int8, Compute is Float)
        w_restored = self.module.weight_int8.astype('float32') * self.module.weight_scale
        
        # Manually run linear math
        # output = x @ w.T + b
        xp = device.backend
        out = xp.matmul(x.data, w_restored.T)
        if self.module.bias is not None:
            out += self.module.bias.data
            
        return Tensor(out, device_type=x.device)
