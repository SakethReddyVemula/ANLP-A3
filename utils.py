import torch
import torch.cuda
from torch.cuda import max_memory_allocated, reset_max_memory_allocated, reset_peak_memory_stats
from fvcore.nn import FlopCountAnalysis, flop_count_table

class FLOPsCounter:
    def __init__(self):
        self.total_flops = 0
        self.hooks = []
        
    def count_flops_layer(self, module, input, output):
        if hasattr(module, 'weight'):
            # For linear layers
            if isinstance(module, torch.nn.Linear):
                flops = 2 * input[0].size(0) * input[0].size(1) * output.size(1)  # multiply-adds
            # For conv layers
            elif isinstance(module, torch.nn.Conv2d):
                flops = 2 * input[0].size(0) * output.size(1) * output.size(2) * \
                       output.size(3) * module.weight.size(2) * module.weight.size(3) * input[0].size(1)
            else:
                flops = 0
            self.total_flops += flops
    
    def start_counting(self, model):
        # Register hooks for all layers
        for module in model.modules():
            if hasattr(module, 'weight'):
                hook = module.register_forward_hook(self.count_flops_layer)
                self.hooks.append(hook)
    
    def stop_counting(self):
        # Remove all hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def reset(self):
        self.total_flops = 0

def get_gpu_memory_usage():
    """Get current GPU memory usage in GB"""
    try:
        return torch.cuda.max_memory_allocated() / 1024**3
    except:
        return 0

def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_flops(model, sample_input, attention_mask):
    """Calculate FLOPs for a single forward pass"""
    flops = FlopCountAnalysis(model, (sample_input, attention_mask))
    return flops.total()


class RobustLossTracker:
    def __init__(self, window_size=100, plateau_threshold=0.01):
        self.window_size = window_size
        self.plateau_threshold = plateau_threshold
        self.losses = []
        self.window_mins = []
        
    def add_loss(self, loss):
        """Add a new loss value and return True if we're on a plateau"""
        self.losses.append(loss)
        
        # Keep only the most recent window_size losses
        if len(self.losses) > self.window_size:
            self.losses.pop(0)
            
        # Once we have enough losses, check for plateau
        if len(self.losses) == self.window_size:
            # Calculate minimum loss in current window
            current_min = min(self.losses)
            self.window_mins.append(current_min)
            
            # Keep last 3 window minimums for plateau detection
            if len(self.window_mins) > 3:
                self.window_mins.pop(0)
                
            # Check if we're on a plateau by comparing window minimums
            if len(self.window_mins) == 3:
                # Calculate relative changes between consecutive window minimums
                changes = [abs(self.window_mins[i] - self.window_mins[i-1]) / self.window_mins[i-1] 
                          for i in range(1, len(self.window_mins))]
                
                # If all changes are below threshold, we're on a plateau
                return all(change < self.plateau_threshold for change in changes)
        
        return False