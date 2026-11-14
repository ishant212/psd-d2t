from report_generator.styles import (
    add_chapter_heading, add_section_heading, add_subsection_heading,
    add_paragraph, add_bullet, add_custom_table, add_equation
)

def build_chapter4(doc):
    add_chapter_heading(doc, "IMPLEMENTATION", chapter_num=4)

    # 4.1 INTRODUCTION
    add_section_heading(doc, "4.1 Introduction")
    add_paragraph(
        doc,
        "We outline the software architecture and module implementation of Topology-aware Positional Semantic Divergence in this chapter. "
        "We implemented an end-to-end reproducible pipeline in Python 3.13 and PyTorch 2.11 with CUDA 12.8, organized into six modular stages (Table 4.1)."
    )

    tbl_stg_headers = ["Stage Identifier", "Module Directory", "Key Script / Class", "Primary Responsibility"]
    tbl_stg_rows = [
        ["Stage 0", "src/loaders/", "loader_factory.py", "Auto-detection and parsing of native ToTTo, WebNLG, and DART formats"],
        ["Stage A", "src/cleaning/", "cleaner.py", "Schema validation, Unicode normalization, zero-width stripping, anomaly logging"],
        ["Stage B", "src/eda/", "analyzer.py", "Token statistics (T5 tokenizer), Zipf distribution, cross-dataset comparative EDA"],
        ["Stage C", "src/normalization/", "converter.py", "Transforming cleaned records into shared canonical JSON schema; roundtrip validation"],
        ["Stage D", "src/graph_construction/", "builder.py", "Constructing NetworkX DiGraph objects, attaching metadata, serializing .gpickle files"],
        ["Stage 1.5", "src/preprocessing/", "serializer.py", "Deterministic text serialization and token-to-node character span alignment"],
        ["Core PSD", "src/psd/", "target_distribution.py", "Constructing composite structural-semantic target distribution P_target"],
        ["Core Loss", "src/psd/", "divergence.py", "Multi-layer eager attention hooking and KL divergence computation (L_PSD)"],
        ["Training", "src/psd/", "trainer.py", "Custom training loop with LoRA PEFT in FP16 precision, gradient accumulation"]
    ]
    add_custom_table(doc, tbl_stg_headers, tbl_stg_rows, caption="Table 4.1: Stage-level execution modules and engineering responsibilities.")

    # 4.2 MULTI-STAGE DATA INGESTION AND GRAPH PIPELINE
    add_section_heading(doc, "4.2 Multi-Stage Data Ingestion and Graph Construction Pipeline")
    add_paragraph(
        doc,
        "Stage 0 (src/loaders/loader_factory.py) automatically identifies the file format and ingests Wikipedia JSONL tables, WebNLG XML trees, "
        "and DART JSON records. Next, the cleaner module (Stage A) repairs corrupted UTF-8 byte sequences, removes invisible zero-width spaces (\u200b), "
        "and retains all 12,135 raw samples. The EDA module (Stage B) computes token length distributions, Zipf word frequencies, and graph structural complexity. "
        "The normalization engine (Stage C) maps all heterogeneous tables and graphs into a unified canonical JSON schema. "
        "The schema represents entities as attributed nodes and structural relationships as directed labeled edges. The conversions are verified "
        "using lossless round trip tests. Stage D instantiates networkx.DiGraph objects with coordinate metadata, persisted as .gpickle files under data/graphs/{dataset}/."
    )

    # 4.3 DETERMINISTIC TEXT SERIALIZATION AND TOKEN ALIGNMENT
    add_section_heading(doc, "4.3 Deterministic Text Serialization and Token-to-Node Alignment")
    add_paragraph(
        doc,
        "To establish an unambiguous mapping between subword tokens and graph nodes without heuristic substring matching, we implement a deterministic "
        "markup serialization protocol (Table 4.2). The serializer tracks character offsets [s_u, e_u) for each structural node u. When tokenized into "
        "subword tokens (x_1, ..., x_N) with spans [s'_i, e'_i), the alignment function \u03bc(i) maps token x_i to node u via interval intersection:"
    )

    add_equation(
        doc,
        [
            ("\u03bc(i) = { ", False),
            ("u,  if [s\u2032", False), ("i", True, True, False), (", e\u2032", False), ("i", True, True, False),
            (") \u2229 [s", False), ("u", True, True, False), (", e", False), ("u", True, True, False),
            (") \u2260 \u2205;   None,  otherwise }", False)
        ],
        eq_num="(4.1)"
    )

    add_paragraph(
        doc,
        "Syntax delimiters ('[ROW]', '[CELL]', '[TRIPLE]') receive no node mapping."
    )

    tbl_mk_headers = ["Data Modality", "Serialization Tag Syntax", "Structural Meaning and Node Association"]
    tbl_mk_rows = [
        ["Relational Tables (ToTTo)", "[PAGE] <title> [SECTION] <section>", "Global document context and hierarchical table placement"],
        ["Relational Tables (ToTTo)", "[HEADER] <h_1> | ... | <h_K>", "Column header titles; establishes semantic typing of matrix columns"],
        ["Relational Tables (ToTTo)", "[ROW r] [CELL] <c_{r,k}>", "Standard table cell at row r, column k; mapped to node r{r}_c{k}"],
        ["Relational Tables (ToTTo)", "[HCELL] <c_{r,k}>", "Highlighted target cell containing salient factual information"],
        ["Knowledge Graphs (WebNLG/DART)", "[TRIPLE] [SUBJ] <s> [REL] <r> [OBJ] <o>", "Directed knowledge triple; s and o mapped to entity nodes, r to edge"]
    ]
    add_custom_table(doc, tbl_mk_headers, tbl_mk_rows, caption="Table 4.2: Deterministic serialization markup syntax across tables and RDF knowledge graphs.")

    # 4.4 CORE PSD ENGINE IMPLEMENTATION
    add_section_heading(doc, "4.4 Positional Semantic Divergence Core Engine Implementation")
    add_paragraph(
        doc,
        "Two dedicated PyTorch modules implement the PSD loss: TargetDistributionBuilder (src/psd/target_distribution.py) and PSDLoss (src/psd/divergence.py). "
        "TargetDistributionBuilder constructs the causal probability distribution P_target from the composite affinity matrix "
        "R_ij = alpha S_ij - beta dP_norm(i, j) - gamma * dG_norm(i, j). To mask future key positions (j > i), a value of -1e9 is used. "
        "After masking, the rows of the resulting matrix are normalized using row-wise softmax to guarantee that each row has exactly a sum of 1.0. "
        "All details concerning this implementation can be found in Listing A2.7."
    )
    add_paragraph(
        doc,
        "Because modern language models execute fused attention kernels that discard intermediate attention tensors, the base model is configured with "
        "attn_implementation='eager' and output_attentions=True. The PSDModelWrapper hooks into the forward pass, extracting attention matrices across "
        "selected intermediate and deep layers L_selected = {12, 24, 35} (0-indexed [11, 23, 34]), averaging across attention heads to produce A_bar^(l). "
        "The PSD loss function computes the Kullback-Leibler divergence D_KL(P_target || A_bar^(l)) over valid structured query positions (Listing A2.8)."
    )

    # 4.5 TRAINING LOOP AND OPTIMIZATION WORKFLOW
    add_section_heading(doc, "4.5 Training Loop and Optimization Workflow")
    add_paragraph(
        doc,
        "Fine-tuning is driven by a custom PyTorch trainer (src/psd/trainer.py). The optimizer is AdamW (learning rate 2e-4, weight decay 0.01, linear warmup "
        "over 50 steps). Training with per-device batch size 1 and 4 gradient accumulation steps yields 4 samples per optimization update. "
        "In every step, cross-entropy language modeling loss is evaluated strictly on target tokens (prompt tokens set to label -100). "
        "Concurrently, the PSD objective regularizes prompt tokens across selected layers, yielding the joint loss L_total = L_LM + lambda_PSD * L_PSD "
        "with optimal weighting lambda_PSD = 0.10. We trained and evaluated all checkpoints under three distinct context budgets: L = 256, 1024, and 2048 tokens, "
        "utilizing the enterprise NVIDIA H100 PCIe cluster for extended long-context scaling and the local workstation for standard context validation."
    )

    # 4.6 EVALUATION PIPELINE
    add_section_heading(doc, "4.6 Comprehensive Evaluation Pipeline Implementation")
    add_paragraph(
        doc,
        "For deterministic verbalization, the evaluation pipeline (src/psd/evaluate.py) applies greedy decoding with temperature 0.0, max new tokens 128, "
        "and repetition penalty 1.05. It integrates HuggingFace evaluate to compute corpus-level BLEU-4, ROUGE-1/2/L, METEOR, and contextualized RoBERTa-large BERTScore F1. "
        "In addition, an automated claim-level factuality evaluator parses the generated text, extracts subject-relation-object (SPO) semantic tuples "
        "using dependency-guided syntactic extraction, and audits whether each claim is topologically and semantically entailed by the input table or graph."
    )

    add_equation(
        doc,
        [
            ("Factual Precision % = ", False),
            ("Count(Verified Grounded Claims) / Count(Total Generated Claims) \u00d7 100%", False)
        ],
        eq_num="(4.2)"
    )

    add_equation(
        doc,
        [
            ("Hallucination Rate % = 100% - Factual Precision %", False)
        ],
        eq_num="(4.3)"
    )

    add_paragraph(
        doc,
        "Statistical significance across all evaluations is established using 1,000 paired bootstrap resamples."
    )

    # 4.7 TECHNICAL CHALLENGES ENCOUNTERED AND RESOLVED
    add_section_heading(doc, "4.7 Technical Challenges Encountered and Resolved")
    add_paragraph(doc, "Five critical engineering challenges were systematically addressed during system development:")
    add_bullet(doc, "Autograd Attention Memory Spikes at Long Contexts (L in {1024, 2048}): Retaining multi-layer eager attention maps with dynamic autograd computation graphs under long contexts creates quadratic activation memory demands. Resolved by training natively with gradient accumulation and deploying the enterprise NVIDIA H100 PCIe (80 GB VRAM) platform for long-context scaling up to 2048 tokens.", bold_prefix="1. Memory Boundedness:")
    add_bullet(doc, "Tokenizer Byte-Pair Offset Alignment Discrepancies: Subword tokenizers produced character spans splitting multi-byte UTF-8 entities. Resolved by implementing a byte-to-character mapping pre-tokenization sanitizer.", bold_prefix="2. Offset Drift:")
    add_bullet(doc, "Non-Stationary Semantic Encoders and Gradient Loops: Allowing the semantic encoder to receive gradients caused representational collapse. Resolved by strictly freezing all-MiniLM-L6-v2 (grad_theta S_ij = 0).", bold_prefix="3. Gradient Loops:")
    add_bullet(doc, "Corrupted Unicode and Zero-Width Characters: Wikipedia tables contained zero-width spaces (\u200b) corrupting span offsets. Resolved via Stage A ftfy and regex normalization.", bold_prefix="4. Unicode Normalization:")
    add_bullet(doc, "Lossless Graph Round-Trip Integrity: Table conversions initially lost column relationships. Resolved by introducing explicit row_adjacent and column_adjacent edges, validated via automated test suites.", bold_prefix="5. Graph Integrity:")
