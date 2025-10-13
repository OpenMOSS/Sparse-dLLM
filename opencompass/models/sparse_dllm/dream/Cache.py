import torch 
from typing import Optional
import torch.nn.functional as F

## Dynamic Cache Eviction
class CustomCache:
    def __init__(self, n_layers: int, device: torch.device, kernel_size: Optional[int] = None, keep_ratio: float = 0.7):
        self.cache = {}
        self.keep_ratios = [keep_ratio for i in range(n_layers)]
        self.pool_kernel_size = kernel_size
        
    def get_cache(self, layer_id: int):
        return self.cache.get(layer_id, {"k": None, "v": None})
        
    def update_cache(self, layer_id: int, k: torch.Tensor, v: torch.Tensor):
        ## k: [B, n_kv_heads, seq_len, head_dim]
        ## v: [B, n_kv_heads, seq_len, head_dim]
        self.cache[layer_id] = {
            "k": k.clone(),
            "v": v.clone()
        }
            
    def filter_cache(self, layer_id: int, q_block: torch.Tensor, bef_filtered_len: int, block_len: int):
        cached = self.get_cache(layer_id)
        cached_k = cached["k"]
        cached_v = cached["v"]

        # q_block: [B, n_heads, block_len, head_dim]
        # cached_k: [B, n_kv_heads, seq_len, head_dim]
        filtered_cached_k = torch.cat([cached_k[:, :, :bef_filtered_len, :], cached_k[:, :, bef_filtered_len + block_len:, :]], dim = 2)
        filtered_cached_v = torch.cat([cached_v[:, :, :bef_filtered_len, :], cached_v[:, :, bef_filtered_len + block_len:, :]], dim = 2)
        if q_block.size(1) != filtered_cached_k.size(1):
            filtered_cached_k_attn = filtered_cached_k.repeat_interleave(
                q_block.size(1) // filtered_cached_k.size(1), dim=1
            )
        else:
            filtered_cached_k_attn = filtered_cached_k

        avg_q = q_block.mean(dim=-2)
        scores = torch.matmul(avg_q.unsqueeze(-2), filtered_cached_k_attn.transpose(-2, -1)).squeeze(-2)
        importance = scores.mean(dim=1)

        if self.pool_kernel_size is not None:
            importance = F.max_pool1d(
                importance.unsqueeze(1),
                kernel_size=self.pool_kernel_size,
                stride=1,
                padding=self.pool_kernel_size // 2
            ).squeeze(1)
        
        keep_num = int(importance.size(-1) * self.keep_ratios[layer_id])
        _, keep_indices = torch.topk(importance, k=keep_num, dim=-1)
        keep_indices = keep_indices.squeeze(0)
        
        n_kv_heads = filtered_cached_k.size(1)
        filtered_cached_k = filtered_cached_k[:, torch.arange(n_kv_heads, device = filtered_cached_k.device)[:, None], keep_indices]
        filtered_cached_v = filtered_cached_v[:, torch.arange(n_kv_heads, device = filtered_cached_k.device)[:, None], keep_indices]
        self.cache[layer_id] = {
            "k": filtered_cached_k,
            "v": filtered_cached_v
        }
        
    def clear(self):
        self.cache.clear()