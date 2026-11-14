"""Frozen semantic representation encoder for computing token-level semantic similarity (S_ij).

Ensures the PSD target distribution is completely independent and gradient-free.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class SemanticEncoder:
    """Base class for frozen semantic encoders."""

    def compute_similarity_matrix(
        self,
        token_strings: List[str],
        token_to_node_map: List[Optional[str]],
        record_nodes: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Computes pairwise semantic similarity S_ij matrix with shape [seq_len, seq_len].

        Values should be in [0, 1] or [-1, 1].
        """
        raise NotImplementedError


class FrozenContextualSemanticEncoder(SemanticEncoder):
    """Computes frozen semantic representations using a dedicated embedding model or word embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        self.device = device
        self._cache: Dict[str, torch.Tensor] = {}
        self.model = None
        self.tokenizer = None
        self.model_name = model_name

    def _lazy_init(self):
        if self.model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
            except Exception as e:
                # Fallback to pure token-identity / lexical semantic encoder
                print(f"[SemanticEncoder] Notice: Using lexical semantic encoder fallback ({e})")
                self.model = None

    def compute_similarity_matrix(
        self,
        token_strings: List[str],
        token_to_node_map: List[Optional[str]],
        record_nodes: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Computes semantic similarity matrix S_ij in [0.0, 1.0] without gradients."""
        seq_len = len(token_strings)
        if seq_len == 0:
            return torch.zeros((0, 0), dtype=torch.float32)

        self._lazy_init()

        # Build node text dictionary
        node_texts = {n["id"]: n.get("label", "") for n in record_nodes}

        if self.model is not None and self.tokenizer is not None:
            # Group unique texts and embed
            unique_nodes = list(set(nid for nid in token_to_node_map if nid is not None))
            node_embeddings = {}

            if unique_nodes:
                texts = [node_texts.get(nid, nid) for nid in unique_nodes]
                with torch.no_grad():
                    inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    # Mean pooling
                    emb = outputs.last_hidden_state.mean(dim=1)
                    emb = F.normalize(emb, p=2, dim=-1)

                    for nid, vec in zip(unique_nodes, emb):
                        node_embeddings[nid] = vec.cpu()

            # Construct token embeddings
            token_embs = []
            default_vec = torch.zeros(self.model.config.hidden_size)
            for idx in range(seq_len):
                nid = token_to_node_map[idx]
                if nid is not None and nid in node_embeddings:
                    token_embs.append(node_embeddings[nid])
                else:
                    token_embs.append(default_vec)

            token_tensor = torch.stack(token_embs)  # [seq_len, dim]
            sim_matrix = torch.mm(token_tensor, token_tensor.t())  # [seq_len, seq_len]
            # Rescale cosine similarity from [-1, 1] to [0, 1] for non-zero vectors
            sim_matrix = torch.clamp((sim_matrix + 1.0) / 2.0, min=0.0, max=1.0)
            return sim_matrix
        else:
            # High-performance lexical & node-identity semantic similarity
            sim_matrix = torch.zeros((seq_len, seq_len), dtype=torch.float32)
            for i in range(seq_len):
                nid_i = token_to_node_map[i]
                for j in range(seq_len):
                    if i == j:
                        sim_matrix[i, j] = 1.0
                    elif nid_i is not None and nid_i == token_to_node_map[j]:
                        sim_matrix[i, j] = 1.0
                    elif token_strings[i].strip() and token_strings[i].strip() == token_strings[j].strip():
                        sim_matrix[i, j] = 0.8
            return sim_matrix
