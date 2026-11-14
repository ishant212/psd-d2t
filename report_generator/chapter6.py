from report_generator.styles import (
    add_chapter_heading, add_section_heading, add_subsection_heading,
    add_paragraph, add_bullet
)

def build_chapter6(doc):
    add_chapter_heading(doc, "CONCLUSION AND FUTURE WORK", chapter_num=6)

    # 6.1 CONCLUSION
    add_section_heading(doc, "6.1 Conclusion")
    add_paragraph(
        doc,
        "This thesis set out to resolve one of the most critical structural challenges in modern Natural Language Generation: "
        "the loss of topological dependencies and the emergence of severe positional bias ('Lost in the Middle') when autoregressive "
        "Transformers generate natural language from serialized, long-context structured data. Standard sequence-to-sequence and "
        "decoder-only language models treat structured inputs as flat strings of tokens, inadvertently divorcing structurally adjacent "
        "entities across hundreds of linear context positions and inducing severe attention diffusion, factual omissions, and hallucinations."
    )

    add_paragraph(
        doc,
        "To overcome these limitations without resorting to computationally cumbersome graph neural networks or modifying foundational "
        "Transformer architectures, we designed, implemented, and empirically validated the Topology-Aware Positional Semantic "
        "Divergence (PSD) framework. Grounded in information-theoretic Kullback-Leibler divergence, PSD constructs a joint causal target "
        "probability distribution that harmonizes token-level dense semantic affinity, normalized sequential displacement, and exact "
        "geodesic graph shortest-path / Manhattan tabular grid distances. By supervising internal self-attention distributions across strategic "
        "intermediate and deep layers during FP16 parameter-efficient fine-tuning (LoRA), PSD directly aligns neural attention "
        "with symbolic data topology."
    )

    add_paragraph(
        doc,
        "Across an extensive experimental campaign evaluated across eight model configurations spanning six contemporary open-weights families "
        "(Gemma 3 4B, Gemma 2 9B, Llama 3.2 3B, Llama 3.1 8B, Qwen 2.5 3B, Qwen 2.5 7B, Mistral 7B, and Phi-3.5-mini 3.8B) and three sequence "
        "horizons (L in {256, 1024, 2048} tokens) on an enterprise NVIDIA H100 PCIe (80 GB VRAM) platform alongside local workstation validation, "
        "PSD delivered decisive performance breakthroughs. At L = 2048, Gemma 3 (4B) achieved our highest long-context fluency. "
        "PSD increased its BLEU-4 by +4.77 points (+21.0% relative gain) and ROUGE-L by +6.68 points, outperforming all other benchmarked configurations. "
        "Simultaneously, Qwen 2.5 (3B) broke two landmark barriers, reaching 90.31% factual precision and driving hallucinations below the 10% threshold "
        "to 9.69% (compared to 20.00% in the L = 1024 baseline and 32.98% at L = 256). In cross-architecture benchmarking at L = 256, Llama 3.1 (8B) "
        "achieved an extraordinary +148.6% leap in BLEU-4 (5.97 to 14.84) and a 58.0% relative reduction in hallucinations (41.48% down to 17.44%). "
        "Target distribution component ablations and layer-selective depth sweeps confirmed that intermediate and deep Transformer layers act as the "
        "critical locus for topological regularization, stabilizing attention distributions at zero inference latency overhead."
    )

    # 6.2 SUMMARY OF KEY TAKEAWAY POINTS
    add_section_heading(doc, "6.2 Summary of Key Takeaway Points")
    add_paragraph(doc, "The primary theoretical and empirical takeaways of this research are summarized below:")

    add_bullet(
        doc,
        "Flat Linearization Destroys Structural Locality: Converting non-linear knowledge graphs and 2D tables into flat token sequences "
        "separates topologically linked entities by hundreds of tokens, precipitating severe positional bias ('Lost in the Middle') and attention diffusion in standard decoders.",
        bold_prefix="1. Structural Disruption:"
    )

    add_bullet(
        doc,
        "Attention Maps Can Directly Absorb Topological Priors: Multi-head self-attention possesses sufficient representational capacity to absorb "
        "graph shortest-path and tabular grid constraints directly through auxiliary KL divergence loss, eliminating the need for expensive external GNNs.",
        bold_prefix="2. Direct Attention Guidance:"
    )

    add_bullet(
        doc,
        "Positional and Graph Priors are Highly Complementary: Positional distance penalties enforce local n-gram fluency, while topological graph "
        "distance penalties enforce long-range relational grounding. Full PSD achieves the highest overall semantic fluency (ROUGE-L 49.72, BERTScore 90.91%).",
        bold_prefix="3. Complementary Priors:"
    )

    add_bullet(
        doc,
        "Topological Attention Prevents Hallucinations: Grounding attention on graph paths directly suppresses attention diffusion onto ungrounded tokens, "
        "yielding up to a 58.0% relative reduction in hallucinated factual claims (Llama 3.1 8B) and driving hallucinations below 10% (9.69% on Qwen 2.5 at L = 2048).",
        bold_prefix="4. Factuality Grounding:"
    )

    add_bullet(
        doc,
        "Middle-to-Deep Layer Specialization: Relational graph and table binding takes place primarily in intermediate and deep Transformer layers "
        "(yielding up to 89.05% factual precision on deep layers of Llama 3.2 3B), whereas supervising early layers harms low-level syntactic tokenization.",
        bold_prefix="5. Layer Specialization:"
    )

    add_bullet(
        doc,
        "Parameter-Efficient Regularization Has Zero Inference Cost: PSD operates strictly during fine-tuning over less than 0.08% trainable LoRA parameters. "
        "Trained models deploy with standard, unmodified inference toolchains at zero added FLOPs and zero decoding latency overhead.",
        bold_prefix="6. Deployment Efficiency:"
    )

    # 6.3 FUTURE WORK DIRECTIONS
    add_section_heading(doc, "6.3 Future Work Directions")
    add_paragraph(
        doc,
        "While this thesis establishes the decisive efficacy of PSD for parameter-efficient fine-tuning on consumer hardware, several promising "
        "avenues remain for future scientific exploration:"
    )

    add_bullet(
        doc,
        "Custom FlashAttention-2 Backward Kernels for Ultra-Long Contexts (L > 2048): Currently, single-GPU training is bounded at L=512 due to "
        "eager attention autograd unrolling memory spikes at L=1024. Developing a fused CUDA / Triton kernel that computes the PSD KL divergence "
        "directly within the forward-backward attention pass would eliminate memory materialization, enabling single-GPU training at L=4096 and L=8192.",
        bold_prefix="1. Fused Attention Kernels:"
    )

    add_bullet(
        doc,
        "Foundation Model Pre-Training from Scratch: Extending PSD from parameter-efficient fine-tuning (LoRA) to large-scale pre-training of "
        "foundational language models from scratch on massive multi-billion-token synthetic knowledge graphs and database crawls.",
        bold_prefix="2. Foundation Pre-Training:"
    )

    add_bullet(
        doc,
        "Human Evaluation and LLM-as-a-Judge Alignment Audits: Conducting extensive double-blind human pairwise preference studies and GPT-4o "
        "evaluations to complement automated lexical and factual metrics on specialized industry domains (clinical trials, financial filings).",
        bold_prefix="3. Human Alignment Studies:"
    )

    add_bullet(
        doc,
        "Dynamic Temperature and Adaptive Layer Scheduling: Investigating learned, dynamically scheduled temperature parameters tau(t) and "
        "adaptive layer weighting lambda_l(t) that relax structural regularization as training converges to further maximize lexical diversity.",
        bold_prefix="4. Dynamic Scheduling:"
    )

    add_bullet(
        doc,
        "Extension to Multi-Modal Vision-Graph-to-Text: Extending topological distance formulations to spatial bounding box graphs in multimodal "
        "document understanding (e.g., PDF financial statements, chart understanding, visual question answering).",
        bold_prefix="5. Multi-Modal Vision Extensions:"
    )
