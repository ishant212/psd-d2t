"""Target distribution builder for Phase 2 PSD.

Constructs P_target(j|i) from:
1. Semantic similarity S_ij
2. Normalized sequential/positional distance dP_norm(i, j)
3. Normalized topological graph distance dG_norm(i, j)
with causal masking and numerically stable normalization.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F


class TargetDistributionBuilder:
    """Builds topological-semantic target attention distributions P_target."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        tau: float = 1.0,
        unmapped_epsilon: float = 1e-4,
        enable_semantic: bool = True,
        enable_positional: bool = True,
        enable_graph: bool = True,
    ):
        self.alpha = alpha if enable_semantic else 0.0
        self.beta = beta if enable_positional else 0.0
        self.gamma = gamma if enable_graph else 0.0
        self.tau = tau
        self.unmapped_epsilon = unmapped_epsilon
        self.enable_semantic = enable_semantic
        self.enable_positional = enable_positional
        self.enable_graph = enable_graph

    def build_target_distribution(
        self,
        seq_len: int,
        token_to_node_map: List[Optional[str]],
        graph_dist_matrix: np.ndarray,
        node_to_idx: Dict[str, int],
        semantic_sim_matrix: Optional[torch.Tensor] = None,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Constructs P_target(j|i) and valid_tokens_mask.

        Args:
            seq_len: Length of input sequence N.
            token_to_node_map: List of length N with node_id or None.
            graph_dist_matrix: [num_nodes, num_nodes] pairwise shortest path distance matrix.
            node_to_idx: Dict mapping node_id -> row index in graph_dist_matrix.
            semantic_sim_matrix: Optional [N, N] tensor of pairwise semantic similarities S_ij in [0, 1].
            device: Target torch device.

        Returns:
            P_target: [seq_len, seq_len] probability distribution where P_target[i, j] is P(j|i), causal masked.
            valid_mask: [seq_len] boolean tensor indicating structured/mapped token positions.
        """
        if seq_len == 0:
            return torch.zeros((0, 0), device=device), torch.zeros(0, dtype=torch.bool, device=device)

        # 1. Compute normalized positional distance dP_norm(i, j) = |i - j| / dP_max
        dP_max = float(max(seq_len - 1, 1))
        pos_indices = torch.arange(seq_len, dtype=torch.float32)
        # [seq_len, seq_len]
        dP = torch.abs(pos_indices.unsqueeze(1) - pos_indices.unsqueeze(0))
        dP_norm = dP / dP_max

        # 2. Compute normalized graph distance dG_norm(i, j)
        dG_max = float(np.max(graph_dist_matrix)) if graph_dist_matrix.size > 0 and np.max(graph_dist_matrix) > 0 else 1.0

        dG_tensor = torch.zeros((seq_len, seq_len), dtype=torch.float32)
        valid_mask = torch.zeros(seq_len, dtype=torch.bool)

        for i in range(seq_len):
            nid_i = token_to_node_map[i] if i < len(token_to_node_map) else None
            if nid_i is not None and nid_i in node_to_idx:
                valid_mask[i] = True
                idx_i = node_to_idx[nid_i]
                for j in range(seq_len):
                    nid_j = token_to_node_map[j] if j < len(token_to_node_map) else None
                    if nid_j is not None and nid_j in node_to_idx:
                        idx_j = node_to_idx[nid_j]
                        dG_tensor[i, j] = float(graph_dist_matrix[idx_i, idx_j]) / dG_max
                    else:
                        # Unmapped target token: maximum distance
                        dG_tensor[i, j] = 1.0
            else:
                # Unmapped source token: uniform maximum distance
                dG_tensor[i, :] = 1.0

        # 3. Semantic similarity S_ij
        if semantic_sim_matrix is None:
            # Identity / baseline semantic similarity
            S_ij = torch.eye(seq_len, dtype=torch.float32)
        else:
            S_ij = semantic_sim_matrix.to(dtype=torch.float32)

        # 4. Compute structural-semantic affinity R_ij:
        # R_ij = alpha * S_ij - beta * dP_norm(i,j) - gamma * dG_norm(i,j)
        R_ij = self.alpha * S_ij - self.beta * dP_norm - self.gamma * dG_tensor

        # Move to target device
        R_ij = R_ij.to(device)
        valid_mask = valid_mask.to(device)

        # 5. Causal Masking: For causal decoder, position i can only attend to j <= i
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        
        # Scale by temperature tau
        scaled_logits = R_ij / self.tau

        # Mask future positions with large negative value (-1e9) for numerical stability
        masked_logits = scaled_logits.masked_fill(~causal_mask, -1e9)

        # Apply numerically stable softmax across columns (keys j <= i)
        P_target = F.softmax(masked_logits, dim=-1)

        # Zero-out explicitly future positions in case of floating point precision
        P_target = P_target * causal_mask.float()

        # Re-normalize valid causal rows to ensure exact sum = 1.0
        row_sums = P_target.sum(dim=-1, keepdim=True)
        P_target = torch.where(row_sums > 0, P_target / row_sums, P_target)

        return P_target, valid_mask
