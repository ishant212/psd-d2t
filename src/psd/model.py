"""Joint PSD model wrapper combining Hugging Face base models with PEFT LoRA and layer-selective PSD loss.

Phase 2.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from src.data.unified_schema import UnifiedRecord
from src.preprocessing.serialize import Serializer
from src.psd.graph_distance import GraphDistanceCalculator
from src.psd.target_distribution import TargetDistributionBuilder
from src.psd.divergence import PSDLoss
from src.psd.semantic_encoder import SemanticEncoder, FrozenContextualSemanticEncoder


@dataclass
class PSDOutput:
    """Structured return type for PSD forward pass."""
    total_loss: torch.Tensor
    lm_loss: torch.Tensor
    psd_loss: torch.Tensor
    per_layer_psd: Dict[int, float]
    logits: torch.Tensor
    selected_attentions: Dict[int, torch.Tensor]


class PSDModelWrapper(nn.Module):
    """Integrates Hugging Face models in FP16/BF16 with LoRA adapter, target graph distribution, and PSD loss."""

    def __init__(
        self,
        model_name_or_path: str = "models/qwen3_4b_hf",
        selected_layers: List[int] = [11, 23, 34],
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: List[str] = ["q_proj", "v_proj"],
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        tau: float = 1.0,
        lambda_psd: float = 0.1,
        psd_enabled: bool = True,
        enable_semantic: bool = True,
        enable_positional: bool = True,
        enable_graph: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device_str = device if torch.cuda.is_available() else "cpu"
        self.selected_layers = selected_layers
        self.lambda_psd = lambda_psd if psd_enabled else 0.0
        self.psd_enabled = psd_enabled and (self.lambda_psd > 0.0)
        self.model_name_or_path = model_name_or_path

        # 1. Initialize Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. Load Base Model with FP16/FP32 precision and Eager Attention
        dtype = torch_dtype if torch_dtype is not None else (torch.float16 if self.device_str == "cuda" else torch.float32)
        print(f"[PSDModelWrapper] Loading Hugging Face checkpoint '{model_name_or_path}' with {dtype}...")

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto" if self.device_str == "cuda" else None,
            attn_implementation="eager",
        )

        # 3. Apply PEFT LoRA
        print(f"[PSDModelWrapper] Attaching PEFT LoRA (r={lora_r}, lora_alpha={lora_alpha}, targets={target_modules})...")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        # 4. Helper modules
        self.serializer = Serializer()
        self.dist_calculator = GraphDistanceCalculator()
        self.target_builder = TargetDistributionBuilder(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            tau=tau,
            enable_semantic=enable_semantic,
            enable_positional=enable_positional,
            enable_graph=enable_graph,
        )
        self.psd_loss_fn = PSDLoss(selected_layers=selected_layers)
        self.semantic_encoder = FrozenContextualSemanticEncoder(device="cpu")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        token_to_node_map: Optional[List[Optional[str]]] = None,
        record: Optional[UnifiedRecord] = None,
        max_seq_len: int = 256,
    ) -> PSDOutput:
        """Executes forward pass, computing joint loss: L_LM + lambda_PSD * L_PSD.

        Note: labels should have prompt tokens set to -100 so causal LM loss is
        evaluated strictly on target/reference tokens.
        """
        # Safety clamp on maximum sequence length
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_seq_len]
            if labels is not None:
                labels = labels[:, :max_seq_len]
            if token_to_node_map is not None:
                token_to_node_map = token_to_node_map[:max_seq_len]

        seq_len = input_ids.shape[1]
        device = input_ids.device

        # 1. Forward pass through LoRA model requesting attention tensors
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=self.psd_enabled,
            output_hidden_states=False,
        )

        lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)

        # 2. Compute Target Distribution P_target if PSD enabled and record/mapping provided
        psd_loss = torch.tensor(0.0, device=device)
        per_layer_psd: Dict[int, float] = {}
        selected_attns: Dict[int, torch.Tensor] = {}

        if self.psd_enabled and record is not None and token_to_node_map is not None and outputs.attentions is not None:
            # Compute topological distance matrix
            dist_matrix, node_to_idx = self.dist_calculator.compute_distance_matrix(record)

            # Extract token strings for semantic similarity
            tok_ids = input_ids[0].tolist()
            token_strings = [self.tokenizer.decode([tid]) for tid in tok_ids]

            # Compute frozen semantic similarity
            sem_sim = self.semantic_encoder.compute_similarity_matrix(
                token_strings=token_strings,
                token_to_node_map=token_to_node_map,
                record_nodes=record.nodes,
            )

            # Build P_target
            P_target, valid_mask = self.target_builder.build_target_distribution(
                seq_len=seq_len,
                token_to_node_map=token_to_node_map,
                graph_dist_matrix=dist_matrix,
                node_to_idx=node_to_idx,
                semantic_sim_matrix=sem_sim,
                device=device,
            )

            # Compute PSD loss on selected layers ONLY
            psd_loss, per_layer_psd = self.psd_loss_fn(
                attentions=outputs.attentions,
                P_target=P_target,
                valid_mask=valid_mask,
            )

            # Retain only selected layer attention references for diagnostics
            for l_idx in self.selected_layers:
                if l_idx < len(outputs.attentions) and outputs.attentions[l_idx] is not None:
                    selected_attns[l_idx] = outputs.attentions[l_idx].detach()

        # 3. Total Loss: L_total = L_LM + lambda_PSD * L_PSD
        total_loss = lm_loss + self.lambda_psd * psd_loss

        return PSDOutput(
            total_loss=total_loss,
            lm_loss=lm_loss,
            psd_loss=psd_loss,
            per_layer_psd=per_layer_psd,
            logits=outputs.logits,
            selected_attentions=selected_attns,
        )
