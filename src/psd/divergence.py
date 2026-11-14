"""Positional Semantic Divergence (PSD) Loss Module.

Computes the Kullback-Leibler (KL) divergence between the topological target
distribution P_target and the empirical attention distributions on selected
Transformer layers.
"""

from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PSDLoss(nn.Module):
    """Calculates layer-selective Positional Semantic Divergence loss."""

    def __init__(
        self,
        selected_layers: List[int] = [11, 23, 34],
        eps: float = 1e-9,
        reduction: str = "mean",
    ):
        super().__init__()
        # Python 0-indexed layers (11, 23, 34 correspond to 1-based 12, 24, 35)
        self.selected_layers = selected_layers
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        attentions: Tuple[torch.Tensor, ...],
        P_target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """Computes PSD KL divergence on selected layers.

        Args:
            attentions: Tuple of attention tensors from model forward pass, each [batch, heads, seq_len, seq_len].
            P_target: [batch, seq_len, seq_len] or [seq_len, seq_len] target distribution.
            valid_mask: [seq_len] or [batch, seq_len] boolean mask indicating structured token positions.

        Returns:
            total_psd_loss: Scalar PyTorch tensor with gradient graph attached.
            per_layer_psd: Dict mapping layer_idx -> float loss value for logging.
        """
        if P_target.dim() == 2:
            P_target = P_target.unsqueeze(0)  # [1, seq_len, seq_len]

        batch_size, seq_len, _ = P_target.shape
        device = P_target.device

        # Causal mask for valid key positions (j <= i)
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)).unsqueeze(0)

        # Valid source tokens mask (rows to include in loss)
        if valid_mask is not None:
            if valid_mask.dim() == 1:
                valid_mask = valid_mask.unsqueeze(0)  # [1, seq_len]
            # Must be valid AND non-empty sequence
            num_valid = valid_mask.sum().float().clamp(min=1.0)
        else:
            valid_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
            num_valid = float(batch_size * seq_len)

        per_layer_losses = []
        per_layer_dict = {}

        # Safe clamped target distribution
        p_target_safe = P_target.clamp(min=self.eps, max=1.0)

        for l_idx in self.selected_layers:
            if l_idx >= len(attentions):
                continue

            # [batch, heads, seq_len, seq_len]
            layer_attn = attentions[l_idx]
            if layer_attn is None:
                continue

            # Match device and dtype
            layer_attn = layer_attn.to(device)

            # 1. Average attention across all attention heads
            # [batch, seq_len, seq_len]
            A_bar = layer_attn.mean(dim=1)

            # Ensure causal masking and safe clamping for logarithm
            A_bar_safe = A_bar.clamp(min=self.eps, max=1.0)

            # Re-normalize causal slice (j <= i)
            A_bar_causal = A_bar_safe * causal_mask.float()
            A_bar_causal = A_bar_causal / A_bar_causal.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            A_bar_causal = A_bar_causal.clamp(min=self.eps, max=1.0)

            # 2. Compute KL Divergence: D_KL(P_target || A_bar) = sum_{j<=i} P_target * (log P_target - log A_bar)
            kl_div_elements = p_target_safe * (torch.log(p_target_safe) - torch.log(A_bar_causal))

            # Sum across keys j <= i
            # [batch, seq_len]
            kl_per_token = (kl_div_elements * causal_mask.float()).sum(dim=-1)

            # 3. Mask out invalid/unmapped source tokens
            kl_valid = kl_per_token * valid_mask.float()
            layer_loss = kl_valid.sum() / num_valid

            # Ensure loss is non-negative
            layer_loss = F.relu(layer_loss)

            per_layer_losses.append(layer_loss)
            per_layer_dict[l_idx] = float(layer_loss.detach().item())

        if len(per_layer_losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {}

        # Average across the 3 selected layers
        psd_loss = torch.stack(per_layer_losses).mean()
        return psd_loss, per_layer_dict
