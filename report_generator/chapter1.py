import os
from report_generator.styles import (
    add_chapter_heading, add_section_heading, add_subsection_heading,
    add_paragraph, add_bullet, add_custom_figure, add_equation
)

def build_chapter1(doc):
    add_chapter_heading(doc, "INTRODUCTION", chapter_num=1)

    # 1.1 BACKGROUND
    add_section_heading(doc, "1.1 Background")
    add_paragraph(
        doc,
        "Data-to-Text (D2T) generation is a fundamental discipline within Natural Language Processing (NLP) that automates the synthesis "
        "of fluent, coherent, and factually grounded natural language sequences from non-linguistic structured data sources (Gatt & Krahmer, 2018). "
        "Unlike standard text-to-text generation tasks, D2T architectures must bridge a fundamental representational chasm: ingesting structured "
        "records—such as relational databases, knowledge graphs, and Resource Description Framework (RDF) triples—and linearizing their topological "
        "dependencies into grammatically rigorous natural language prose (Li et al., 2024; Nan et al., 2021). The primary design imperative of a "
        "production D2T system is semantic faithfulness: ensuring that every relational assertion in the output is strictly substantiated by the source data."
    )
    add_paragraph(
        doc,
        "Historically, D2T systems began with hand-crafted rule-based templates (Reiter & Dale, 2000) and later adopted recurrent neural networks "
        "(RNN/LSTM) (Wiseman et al., 2017). Templates delivered high factual precision, but engineers struggled to adapt fixed slot-filler rules "
        "to new target domains. Conversely, RNN decoders suffered from vanishing gradients, exposure bias, and activation memory loss whenever input "
        "records exceeded five relational triples. Today, 3B-to-9B parameter Transformer decoders dominate the field because their multi-head self-attention "
        "yields much higher ROUGE-L and BLEU scores."
    )

    # 1.2 OVERVIEW OF DATA-TO-TEXT GENERATION AND STRUCTURAL INPUTS
    add_section_heading(doc, "1.2 Overview of Data-to-Text Generation and Structural Inputs")

    add_subsection_heading(doc, "1.2.1 Data-to-Text Systems and Representation Modalities")
    add_paragraph(
        doc,
        "Modern D2T benchmarks span three structural formats across 12,135 instances. First, ToTTo (Parikh et al., 2020) provides tabular 2D grids "
        "with row-column geometry; second, WebNLG 2020 (Castro Ferreira et al., 2020) provides directed subject-predicate-object graphs; and third, "
        "DART (Nan et al., 2021) combines SQL tables with OpenIE triples."
    )

    add_subsection_heading(doc, "1.2.2 Transformer Architecture and Self-Attention Mechanism")
    add_paragraph(
        doc,
        "Modern D2T generators deploy causal decoder-only Transformers (Vaswani et al., 2017; Radford et al., 2019) to decode linearized records autoregressively. "
        "The core mathematical operator driving these decoders is scaled dot-product attention:"
    )
    add_equation(
        doc,
        [
            ("Attention(", False), ("Q", True), (", ", False), ("K", True), (", ", False), ("V", True),
            (") = Softmax\u239B\u239D ", False),
            ("Q K", True), ("T", False, False, True),
            (" / \u221a", False), ("d", True), ("k", True, True, False),
            (" \u239E\u23A0 ", False),
            ("V", True)
        ],
        eq_num="(1.1)"
    )
    add_paragraph(
        doc,
        "Scaled dot-product attention enables fully parallel training while calculating pairwise affinities across all context positions. "
        "Modern SLMs (Qwen3-4B, Gemma-3-4B, Llama-3.2-3B), when used in conjunction with current PEFT techniques such as LoRA (Hu et al., 2022) using native "
        "FP16 precision, have achieved a high level of fluency and structural adaptivity, while requiring relatively low computational power."
    )

    add_subsection_heading(doc, "1.2.3 Serialization of Heterogeneous Graphs and 2D Tables")
    add_paragraph(
        doc,
        "Because Transformers are fundamentally sequence models, non-linear graphs G = (V, E) and 2D tables T must be linearized into flat 1D token "
        "strings X = (x_1, ..., x_N) using structural delimiter tokens (e.g., '[TRIPLE]', '[ROW]', '[CELL]'). While serialization allows standard "
        "autoregressive language decoders to process structured inputs, it flattens multi-dimensional topological neighborhoods into an arbitrary 1D sequence, "
        "introducing severe structural distortions where topologically connected entities are separated by hundreds of intervening tokens."
    )

    # 1.3 CHALLENGES IN LONG-CONTEXT DATA-TO-TEXT GENERATION
    add_section_heading(doc, "1.3 Challenges in Long-Context Data-to-Text Generation")

    add_subsection_heading(doc, "1.3.1 The 'Lost in the Middle' (LITM) Phenomenon and Positional Bias")
    add_paragraph(
        doc,
        "A critical vulnerability of autoregressive Transformers is the 'Lost in the Middle' (LITM) phenomenon (Liu et al., 2024). Models exhibit a pronounced "
        "U-shaped positional attention bias, concentrating attention mass on primacy (beginning) and recency (end) tokens while severely discounting the middle. "
        "In long-context D2T generation spanning 512 to 2048 tokens, dozens of salient entities are situated in the serialized middle, resulting in severe "
        "information decay, relational omissions, and factual hallucinations."
    )

    add_subsection_heading(doc, "1.3.2 Structural Information Loss and Quadratic Attention Complexity")
    add_paragraph(
        doc,
        "Standard multi-head self-attention relies exclusively on sequential positional encodings (such as RoPE or absolute embeddings), which capture only "
        "1D token separation |i - j|. As context length N expands, quadratic attention complexity O(N^2) disperses attention mass across irrelevant delimiters. "
        "Figure 1.1 provides a comprehensive Exploratory Data Analysis (EDA) of the benchmark datasets, illustrating input/target token length distributions, "
        "structural topological complexity, and the distribution of instances across context length horizons."
    )

    add_custom_figure(
        doc,
        "reports/plots/eda_comprehensive_distributions.png",
        "Figure 1.1: Comprehensive Exploratory Data Analysis (EDA) across ToTTo, WebNLG, and DART: (a) Serialized input token length distributions with context boundaries, (b) Target reference verbalization lengths, (c) Average structural elements per instance, and (d) Context length horizon allocation."
    )

    add_subsection_heading(doc, "1.3.3 Attention Diffusion, Fact Hallucination, and Semantic Drift")
    add_paragraph(
        doc,
        "Conflating linear sequence distance with semantic relevance precipitates attention diffusion in deep layers. This leads directly to three primary "
        "empirical failure modes: (1) Structural Hallucination, fabricating edges between entities that appear sequentially adjacent but share no graph edge; "
        "(2) Fact Distortion, omitting critical middle-context attributes; and (3) Progressive Semantic Drift, where generation gradually decouples from source topology."
    )

    # 1.4 PROBLEM STATEMENT
    add_section_heading(doc, "1.4 Problem Statement")
    add_paragraph(
        doc,
        "Existing Transformer-based D2T models discard underlying graph topology and tabular geometry during sequence serialization. Consequently, when "
        "processing long structured contexts (N >= 512 tokens), standard decoders suffer from severe positional bias ('Lost in the Middle'), attention diffusion, "
        "and factual hallucinations. While graph models like GraphWriter or Graphormer preserve connectivity, their dual-encoder designs prevent direct "
        "integration into modern decoder-only Small Language Models. Furthermore, existing factuality checks operate purely post-hoc without steering the "
        "model's internal attention during generation. The core research problem is: How to formulate and inject an auxiliary topological-semantic attention "
        "regularization objective into modern parameter-efficient Small Language Models to preserve graph topology, mitigate positional bias, and reduce "
        "factual hallucinations in long-context D2T generation without structural modifications or prohibitive memory overhead?"
    )

    # 1.5 OBJECTIVES
    add_section_heading(doc, "1.5 Objectives of the Research")
    add_paragraph(doc, "This research defines six specific, measurable technical objectives:")
    add_bullet(doc, "Build an end-to-end multi-modal ingestion pipeline to parse and clean 12,135 records across ToTTo tables, WebNLG RDF graphs, and DART OpenIE entries formatted in a unified canonical schema.", bold_prefix="1.")
    add_bullet(doc, "Develop a Deterministic Text Serialization and Token-to-Node Alignment Engine with exact character span mapping mu(i).", bold_prefix="2.")
    add_bullet(doc, "Formulate the Positional Semantic Divergence (PSD) Attention Prior, unifying dense semantics S_ij, sequence distance d_P, and graph shortest path d_G into causal target distribution P_target.", bold_prefix="3.")
    add_bullet(doc, "Implement Layer-Selective Attention Supervision and Joint PEFT Optimization, computing KL divergence D_KL(P_target || A^(l)) across strategic layers with prompt-masked loss.", bold_prefix="4.")
    add_bullet(doc, "Execute Controlled Comparisons and Factorial Component Ablations, isolating semantic, positional, and graph contributions.", bold_prefix="5.")
    add_bullet(doc, "Conduct Long-Context Scaling and Cross-Architecture Generalization Benchmarks across sequence lengths L in {256, 1024, 2048} across multi-scale open-weights LLMs (Gemma 3 4B, Gemma 2 9B, Llama 3.2 3B, Llama 3.1 8B, Qwen 2.5 3B/7B, Mistral 7B, Phi-3.5-mini), auditing factuality and hallucination suppression.", bold_prefix="6.")

    # 1.6 RESEARCH QUESTIONS
    add_section_heading(doc, "1.6 Research Questions")
    add_paragraph(doc, "The experimental methodology investigates four primary Research Questions:")
    add_bullet(doc, "RQ1 (In-Distribution Generation Quality): Does regularizing attention maps with the PSD prior improve lexical overlap (BLEU-4, ROUGE-L) and semantic fidelity (BERTScore) without destabilizing LM loss convergence?", bold_prefix="RQ1:")
    add_bullet(doc, "RQ2 (Component Attribution): What are the individual and synergistic contributions of dense semantic similarity S_ij, sequence distance penalty d_P, and graph distance penalty d_G across factorial ablation variants?", bold_prefix="RQ2:")
    add_bullet(doc, "RQ3 (Long-Context Robustness and Scaling): How does PSD guidance perform across expanded sequence lengths (L in {256, 1024, 2048}), and what are the empirical hardware boundaries?", bold_prefix="RQ3:")
    add_bullet(doc, "RQ4 (Cross-Architecture Generalization and Factuality): Does the PSD inductive bias generalize across structurally distinct SLMs (3B to 9B parameters) and measurably suppress hallucinated factual assertions?", bold_prefix="RQ4:")

    # 1.7 SCOPE AND DELIVERABLES
    add_section_heading(doc, "1.7 Scope and Key Technical Deliverables")
    add_paragraph(
        doc,
        "In Scope: (1) Ingestion and canonical graph construction for ToTTo, WebNLG, and DART; (2) Shortest path and Manhattan distance matrix extraction; "
        "(3) Deterministic markup serialization and token alignment; (4) PyTorch implementation of the PSD target builder and layer-selective KL loss; "
        "(5) Parameter-efficient fine-tuning via HuggingFace PEFT LoRA in FP16 precision; (6) Comprehensive evaluation across BLEU-4, ROUGE-L, "
        "BERTScore F1, and claim-level factual precision/hallucination rate on 8 model configurations spanning 3B to 9B parameters; "
        "(7) Hardware execution and profiling on an NVIDIA H100 PCIe (80 GB VRAM) platform alongside local workstation validation across sequence lengths L in {256, 1024, 2048}; "
        "(8) Empirical breakthrough in long-context scaling (Gemma 3 at L=2048 reaching 27.47 BLEU-4 and 65.06 ROUGE-L; Qwen 2.5 reaching 90.31% factual precision and 9.69% hallucination rate; "
        "Llama 3.1 8B achieving a 58.0% relative reduction in hallucinations) operating with zero inference latency overhead."
    )
    add_paragraph(
        doc,
        "Out of Scope: (1) Pre-training massive foundational models (>= 70B parameters) from scratch; (2) Unstructured free-form creative text generation lacking "
        "underlying topology; (3) Proprietary commercial API endpoints, including GPT-4, where black-box serving prevents direct extraction of hidden attention weight tensors; "
        "(4) Ultra-long sequence budgets at or above 4,096 tokens that necessitate multi-node model parallelism."
    )
