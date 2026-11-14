from report_generator.styles import (
    add_chapter_heading, add_section_heading, add_subsection_heading,
    add_paragraph, add_bullet, add_custom_table, add_equation
)

def build_chapter2(doc):
    add_chapter_heading(doc, "LITERATURE SURVEY", chapter_num=2)

    # 2.1 INTRODUCTION
    add_section_heading(doc, "2.1 Introduction")
    add_paragraph(
        doc,
        "The automated synthesis of coherent, factually grounded natural language from structured data representations has emerged as a cornerstone "
        "of modern NLP. Over the past decade, the field has transitioned from rule-based template fillers (Reiter & Dale, 2000) and recurrent neural pipelines "
        "(Wiseman et al., 2017) to pre-trained Transformer foundations and autoregressive Large and Small Language Models (LLMs/SLMs) (Vaswani et al., 2017; "
        "Radford et al., 2019). However, scaling context windows introduces fundamental vulnerabilities regarding how effectively causal self-attention "
        "retains and processes non-linear topological dependencies across extended token horizons."
    )
    add_paragraph(
        doc,
        "This chapter synthesizes the literature across three intellectual axes: (1) Transformer-based Data-to-Text benchmarks and graph-augmented "
        "architectures; (2) Positional bias, the 'Lost in the Middle' (LITM) effect, and sequence compression mitigations; and (3) Faithfulness, "
        "semantic consistency, and grounded factual evaluation metrics, concluding with a formal taxonomy of prior art (Table 2.1) and research gap analysis."
    )

    # 2.2 TRANSFORMER-BASED D2T AND STRUCTURED REPRESENTATIONS
    add_section_heading(doc, "2.2 Transformer-based Data-to-Text Generation and Structured Representations")
    add_paragraph(
        doc,
        "Modern Data-to-Text benchmarks evaluate distinct structural modalities. In tabular generation, ToTTo (Parikh et al., 2020) provides over 120,000 "
        "Wikipedia table instances with highlighted cell coordinates to enforce controlled factual grounding. In knowledge graph generation, WebNLG 2020 "
        "(Castro Ferreira et al., 2020) provides 1,667 RDF graph entries pairing 1 to 7 directed DBpedia triples with multi-reference verbalizations, "
        "while DART (Nan et al., 2021) unifies open-domain Wikipedia tables, WikiSQL, and OpenIE extraction triples across 82,191 instances. "
        "Standard sequence-to-sequence pipelines (BART, T5) serialize these structured records into flat token sequences, relying entirely on the "
        "decoder to reconstruct non-linear topology from linear delimiter tokens."
    )
    add_paragraph(
        doc,
        "To explicitly model graph topology, Koncel-Kedziorski et al. (2019) introduced GraphWriter, incorporating graph self-attention message passing "
        "into a specialized Transformer encoder. In graph classification, Graphormer (Ying et al., 2021) demonstrated that standard self-attention captures "
        "graph structure effectively when augmented with shortest-path spatial encodings and node centrality biases. Recently, UniD2T (Li et al., 2024) "
        "unified tables and knowledge triples into a single graph Transformer pre-trained from scratch. Yet these prior architectures exhibit clear shortcomings. "
        "They demand full pre-training from scratch, rely on dual encoders incompatible with causal SLMs, or process bidirectional graphs without "
        "supporting autoregressive text decoding."
    )

    # 2.3 POSITIONAL BIAS AND MITIGATION TECHNIQUES
    add_section_heading(doc, "2.3 Positional Bias, Long-Context Transformers, and Mitigation Techniques")
    add_paragraph(
        doc,
        "A critical vulnerability in long-context language modeling is the 'Lost in the Middle' (LITM) effect (Liu et al., 2024). Autoregressive decoders "
        "exhibit an intrinsic U-shaped positional attention bias: allocating high attention mass to sequence boundaries (primacy and recency) while "
        "severely degrading on information placed in the middle. Mathematical analyses ('Lost in the Middle at Birth') reveal that this bias emerges naturally "
        "from the coupling of causal attention masking, RoPE positional encodings, and residual stream accumulation. In D2T generation spanning 512 to 2048 "
        "tokens, serialized middle-context facts are heavily discounted, causing severe factual omissions."
    )
    add_paragraph(
        doc,
        "Existing long-context mitigations include prompt compression (LongLLMLingua; Jiang et al., 2024), attention sink pinning (StreamingLLM; Xiao et al., 2023), "
        "and position knowledge distillation (Pos2Distill; Wang et al., 2025). While effective for unstructured document retrieval, dropping or reordering tokens "
        "catastrophically disrupts structured D2T data, breaking graph edges and splitting tabular coordinate alignments. Meanwhile, information-theoretic "
        "relative entropy—formalized as Kullback-Leibler (KL) divergence (Kullback & Leibler, 1951)—provides a principled objective for steering attention:"
    )
    add_equation(
        doc,
        [
            ("D", True), ("KL", True, True, False), ("(", False), ("P", True), (" \u2225 ", False), ("Q", True), (") = \u2211", False),
            ("i", True, True, False), (" ", False),
            ("P", True), ("(", False), ("i", True), (") log\u239B\u239D ", False),
            ("P", True), ("(", False), ("i", True), (") / ", False),
            ("Q", True), ("(", False), ("i", True), (") \u239E\u23A0", False)
        ],
        eq_num="(2.1)"
    )
    add_paragraph(
        doc,
        "However, prior literature lacks a differentiable framework mapping topological shortest-path and tabular "
        "grid geometries into a valid causal target distribution P_target."
    )

    # 2.4 FAITHFULNESS, SEMANTIC CONSISTENCY, AND GROUNDED EVALUATION
    add_section_heading(doc, "2.4 Faithfulness, Semantic Consistency, and Grounded Evaluation")
    add_paragraph(
        doc,
        "Language models frequently suffer from hallucinations, generating fluent assertions unsubstantiated by the input data. Traditional n-gram overlap "
        "metrics (BLEU, ROUGE) correlate poorly with human factual judgments. To improve evaluation, FactSpotter (Zhang et al., 2023), FactSelfCheck "
        "(Wang et al., 2025), and TrueTeacher (Gekhman et al., 2023) introduce span-grounded and model-based consistency checking. Sentence-BERT (SBERT) "
        "(Reimers & Gurevych, 2019) leverages Siamese transformer encoders (e.g., all-MiniLM-L6-v2) to compute dense semantic embeddings with superior human "
        "correlation (Huidrom et al., 2025)."
    )
    add_paragraph(
        doc,
        "Crucially, all existing faithfulness tools operate strictly post-hoc after generation has already terminated. They provide no differentiable gradient "
        "signal to steer attention during decoding. By contrast, directly regularizing intermediate self-attention maps during parameter-efficient fine-tuning "
        "provides an active, in-process inductive bias that suppresses attention diffusion and enforces factual grounding at every autoregressive step."
    )

    # 2.5 SUMMARY OF RELATED PRIOR ART
    add_section_heading(doc, "2.5 Summary of Related Prior Art")
    add_paragraph(
        doc,
        "Table 2.1 summarizes how prior studies model structured inputs, regulate long-context attention, and evaluate factual correctness. "
        "We highlight the methodology, core innovation, and technical constraint of each paradigm."
    )

    table_headers = ["Work / Paradigm", "Methodological Layer", "Core Contribution / Mechanism", "Primary Limitation / Defect"]
    table_rows = [
        [
            "Vaswani et al. (2017)\n(Transformers)",
            "Model Architecture",
            "Multi-head self-attention mechanism; parallelized sequence modeling.",
            "Flattens inputs into 1D sequences; quadratic complexity O(N^2); lacks structural graph priors."
        ],
        [
            "Parikh et al. (2020)\n(ToTTo Benchmark)",
            "Data & Benchmarking",
            "Controlled table-to-text benchmark with highlighted cell supervision.",
            "Restricted to local tables; severe degradation when scaling to long multi-row contexts."
        ],
        [
            "Nan et al. (2021)\n(DART Benchmark)",
            "Data & Benchmarking",
            "Open-domain structured record-to-text dataset spanning tables and OpenIE triples.",
            "Relies on standard flat sequence serialization; cannot explicitly preserve non-linear graph topology."
        ],
        [
            "Koncel-Kedziorski (2019)\n(GraphWriter)",
            "Graph Modeling",
            "Graph self-attention encoder for multi-sentence generation from knowledge graphs.",
            "Dual-encoder architecture; incompatible with modern decoder-only causal Small Language Models."
        ],
        [
            "Ying et al. (2021)\n(Graphormer)",
            "Graph Transformers",
            "Injected shortest-path spatial encodings and centrality directly into self-attention.",
            "Restricted to graph-level classification; does not address causal autoregressive language generation."
        ],
        [
            "Li et al. (2024)\n(UniD2T)",
            "Pre-Training & Graphs",
            "Unified graph representation for tables, key-values, and triples with graph-aware attention.",
            "Requires expensive pre-training from scratch; high compute overhead; ignores causal positional bias."
        ],
        [
            "Liu et al. (2024)\n(Lost in the Middle)",
            "Long-Context Analysis",
            "Empirical characterization of U-shaped positional attention bias in long contexts.",
            "Diagnostic analysis only; plain-text evaluation; provides no architectural or regularizing mitigation."
        ],
        [
            "Wang et al. (2025)\n(Pos2Distill)",
            "Position Mitigation",
            "Inter-position knowledge distillation propagating attention from boundary to middle tokens.",
            "Operates strictly over flat linear text; ignores underlying graph/table topology and relational semantics."
        ],
        [
            "Zhang et al. (2023)\n(FactSpotter)",
            "Faithfulness Evaluation",
            "Graph-based metric verifying if Subject-Predicate-Object triples appear in generated text.",
            "Strictly post-hoc evaluation metric; does not provide differentiable guidance during generation."
        ],
        [
            "Reimers & Gurevych (2019)\n(Sentence-BERT)",
            "Semantic Representation",
            "Siamese / triplet BERT networks producing dense embeddings for cosine semantic similarity.",
            "Evaluates isolated sentence pairs post-hoc; does not model sequence positions or graph topology."
        ],
        [
            "PROPOSED FRAMEWORK\n(Topology-Aware PSD)",
            "Auxiliary Attention Regularization (PEFT)",
            "Joint topological-semantic attention prior combining graph shortest-paths, Manhattan grid distances, and dense semantics via KL loss.",
            "Focuses on structured Data-to-Text inputs where graph or tabular topology is explicitly defined."
        ]
    ]

    add_custom_table(
        doc,
        headers=table_headers,
        rows=table_rows,
        caption="Table 2.1: Summary and taxonomy of related prior art in structured data-to-text generation and attention modeling."
    )

    # 2.6 RESEARCH GAP AND SCIENTIFIC MOTIVATION
    add_section_heading(doc, "2.6 Research Gap and Scientific Motivation")
    add_paragraph(
        doc,
        "Existing literature reveals a distinct gap: while graph neural encoders capture entity connectivity for classification (Ying et al., 2021), "
        "researchers have not adapted these structural inductive biases to autoregressive decoders. Long-context studies clearly expose the lost-in-the-middle "
        "degradation (Liu et al., 2024). However, current solutions merely reorder prompts or drop middle tokens via heuristic compression (Xiao et al., 2023). "
        "Both prompt compression and token reordering destroy the row-column coordinates of tables and sever multi-hop graph edges. "
        "Finally, established factuality auditors like FactSpotter evaluate generated text only after decoding finishes (Zhang et al., 2023; Huidrom et al., 2025)."
    )
    add_paragraph(
        doc,
        "In previous work there was no model that jointly represented graph topology, sequence distances, and dense token semantics within the internal "
        "self-attention maps of causal Small Language Models (SLMs) as part of their training process. This is where our Topology-Aware Positional Semantic "
        "Divergence (PSD) framework fills a void. During LoRA fine-tuning in FP16, PSD injects a structural-semantic affinity matrix directly into intermediate "
        "and deep attention layers. Topological attention supervision keeps entity relations faithful to source graphs and curbs hallucinations across expanded context windows."
    )
