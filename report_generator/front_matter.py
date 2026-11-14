from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from report_generator.styles import (
    add_paragraph, add_custom_table, set_cell_shading, set_cell_border
)

def build_front_matter(doc):
    # =========================================================================
    # 1. COVER PAGE
    # =========================================================================
    p0 = doc.add_paragraph()
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p0.paragraph_format.space_before = Pt(36)
    p0.paragraph_format.space_after = Pt(14)
    r = p0.add_run("A project report on")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.italic = True

    p1 = doc.add_paragraph()
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1.paragraph_format.space_before = Pt(8)
    p1.paragraph_format.space_after = Pt(28)
    p1.paragraph_format.line_spacing = 1.2
    r = p1.add_run("TOPOLOGY-AWARE POSITIONAL SEMANTIC DIVERGENCE MODELING FOR LONG-CONTEXT DATA-TO-TEXT GENERATION")
    r.font.name = "Times New Roman"
    r.font.size = Pt(20)
    r.font.bold = True

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_before = Pt(14)
    p2.paragraph_format.space_after = Pt(14)
    r = p2.add_run("Submitted in partial fulfillment for the award of the degree of")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.italic = True

    p3 = doc.add_paragraph()
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p3.paragraph_format.space_before = Pt(10)
    p3.paragraph_format.space_after = Pt(24)
    p3.paragraph_format.line_spacing = 1.2
    r = p3.add_run("Bachelor of Technology in Computer Science and Engineering\n(Artificial Intelligence and Machine Learning)")
    r.font.name = "Times New Roman"
    r.font.size = Pt(20)
    r.font.bold = True

    p4 = doc.add_paragraph()
    p4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p4.paragraph_format.space_before = Pt(12)
    p4.paragraph_format.space_after = Pt(8)
    r = p4.add_run("by")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.italic = True

    p5 = doc.add_paragraph()
    p5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p5.paragraph_format.space_before = Pt(8)
    p5.paragraph_format.space_after = Pt(36)
    r = p5.add_run("ISHANT SHEKHAR EESHU (21BCE1981)")
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.font.bold = True

    p6 = doc.add_paragraph()
    p6.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p6.paragraph_format.space_before = Pt(24)
    p6.paragraph_format.space_after = Pt(6)
    p6.paragraph_format.line_spacing = 1.2
    r = p6.add_run("SCHOOL OF COMPUTER SCIENCE AND ENGINEERING\nVELLORE INSTITUTE OF TECHNOLOGY, CHENNAI")
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.font.bold = True

    p7 = doc.add_paragraph()
    p7.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p7.paragraph_format.space_before = Pt(8)
    p7.paragraph_format.space_after = Pt(0)
    r = p7.add_run("October, 2026")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)

    # =========================================================================
    # 2. TITLE PAGE (Identical formatting per guidelines)
    # =========================================================================
    doc.add_page_break()
    p0 = doc.add_paragraph()
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p0.paragraph_format.space_before = Pt(36)
    p0.paragraph_format.space_after = Pt(14)
    r = p0.add_run("A project report on")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.italic = True

    p1 = doc.add_paragraph()
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1.paragraph_format.space_before = Pt(8)
    p1.paragraph_format.space_after = Pt(28)
    p1.paragraph_format.line_spacing = 1.2
    r = p1.add_run("TOPOLOGY-AWARE POSITIONAL SEMANTIC DIVERGENCE MODELING FOR LONG-CONTEXT DATA-TO-TEXT GENERATION")
    r.font.name = "Times New Roman"
    r.font.size = Pt(20)
    r.font.bold = True

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_before = Pt(14)
    p2.paragraph_format.space_after = Pt(14)
    r = p2.add_run("Submitted in partial fulfillment for the award of the degree of")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.italic = True

    p3 = doc.add_paragraph()
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p3.paragraph_format.space_before = Pt(10)
    p3.paragraph_format.space_after = Pt(24)
    p3.paragraph_format.line_spacing = 1.2
    r = p3.add_run("Bachelor of Technology in Computer Science and Engineering\n(Artificial Intelligence and Machine Learning)")
    r.font.name = "Times New Roman"
    r.font.size = Pt(20)
    r.font.bold = True

    p4 = doc.add_paragraph()
    p4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p4.paragraph_format.space_before = Pt(12)
    p4.paragraph_format.space_after = Pt(8)
    r = p4.add_run("by")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.italic = True

    p5 = doc.add_paragraph()
    p5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p5.paragraph_format.space_before = Pt(8)
    p5.paragraph_format.space_after = Pt(36)
    r = p5.add_run("ISHANT SHEKHAR EESHU (21BCE1981)")
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.font.bold = True

    p6 = doc.add_paragraph()
    p6.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p6.paragraph_format.space_before = Pt(24)
    p6.paragraph_format.space_after = Pt(6)
    p6.paragraph_format.line_spacing = 1.2
    r = p6.add_run("SCHOOL OF COMPUTER SCIENCE AND ENGINEERING\nVELLORE INSTITUTE OF TECHNOLOGY, CHENNAI")
    r.font.name = "Times New Roman"
    r.font.size = Pt(16)
    r.font.bold = True

    p7 = doc.add_paragraph()
    p7.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p7.paragraph_format.space_before = Pt(8)
    p7.paragraph_format.space_after = Pt(0)
    r = p7.add_run("October, 2026")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)

    # =========================================================================
    # 3. DECLARATION
    # =========================================================================
    doc.add_page_break()
    p_dec_title = doc.add_paragraph()
    p_dec_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dec_title.paragraph_format.space_before = Pt(24)
    p_dec_title.paragraph_format.space_after = Pt(24)
    r = p_dec_title.add_run("DECLARATION")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.underline = True

    p_dec_body = doc.add_paragraph()
    p_dec_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_dec_body.paragraph_format.line_spacing = 1.5
    p_dec_body.paragraph_format.space_after = Pt(16)
    r = p_dec_body.add_run(
        "I hereby declare that the thesis entitled “TOPOLOGY-AWARE POSITIONAL SEMANTIC DIVERGENCE "
        "MODELING FOR LONG-CONTEXT DATA-TO-TEXT GENERATION” submitted by ISHANT SHEKHAR EESHU (21BCE1981), "
        "for the award of the degree of Bachelor of Technology in Computer Science and Engineering "
        "(Artificial Intelligence and Machine Learning), Vellore Institute of Technology, Chennai is a record of "
        "bonafide work carried out by me under the supervision of Dr. Suguna M., Associate Professor, School of "
        "Computer Science and Engineering, Vellore Institute of Technology, Chennai."
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)

    p_dec_body2 = doc.add_paragraph()
    p_dec_body2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_dec_body2.paragraph_format.line_spacing = 1.5
    p_dec_body2.paragraph_format.space_after = Pt(48)
    r = p_dec_body2.add_run(
        "I further declare that the work reported in this thesis has not been submitted and will not be "
        "submitted, either in part or in full, for the award of any other degree or diploma in this institute "
        "or any other institute or university."
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)

    p_sig = doc.add_paragraph()
    p_sig.paragraph_format.line_spacing = 1.3
    r = p_sig.add_run("Place: Chennai\nDate: 16 October 2026\t\t\t\t\tSignature of the Candidate")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True

    # =========================================================================
    # 4. CERTIFICATE
    # =========================================================================
    doc.add_page_break()
    p_cert_sch = doc.add_paragraph()
    p_cert_sch.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cert_sch.paragraph_format.space_before = Pt(20)
    p_cert_sch.paragraph_format.space_after = Pt(6)
    r = p_cert_sch.add_run("School of Computer Science and Engineering\nVELLORE INSTITUTE OF TECHNOLOGY, CHENNAI")
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)
    r.font.bold = True

    p_cert_title = doc.add_paragraph()
    p_cert_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cert_title.paragraph_format.space_before = Pt(8)
    p_cert_title.paragraph_format.space_after = Pt(20)
    r = p_cert_title.add_run("CERTIFICATE")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.underline = True

    p_cert_body = doc.add_paragraph()
    p_cert_body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_cert_body.paragraph_format.line_spacing = 1.5
    p_cert_body.paragraph_format.space_after = Pt(16)
    r = p_cert_body.add_run(
        "This is to certify that the report entitled “TOPOLOGY-AWARE POSITIONAL SEMANTIC DIVERGENCE "
        "MODELING FOR LONG-CONTEXT DATA-TO-TEXT GENERATION” is prepared and submitted by ISHANT SHEKHAR EESHU "
        "(21BCE1981) to Vellore Institute of Technology, Chennai, in partial fulfillment of the requirement "
        "for the award of the degree of Bachelor of Technology in Computer Science and Engineering "
        "(Artificial Intelligence and Machine Learning) is a bonafide record carried out under my guidance. "
        "The project fulfills the requirements as per the regulations of this University and in my opinion meets "
        "the necessary standards for submission. The contents of this report have not been submitted and will not "
        "be submitted either in part or in full, for the award of any other degree or diploma and the same is certified."
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)

    p_guide = doc.add_paragraph()
    p_guide.paragraph_format.space_before = Pt(20)
    p_guide.paragraph_format.space_after = Pt(20)
    p_guide.paragraph_format.line_spacing = 1.2
    r = p_guide.add_run("Signature of the Guide:\nName: Dr. Suguna M., Associate Professor\nDepartment of Artificial Intelligence and Machine Learning\nDate:")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)

    p_exam = doc.add_paragraph()
    p_exam.paragraph_format.space_before = Pt(10)
    p_exam.paragraph_format.space_after = Pt(20)
    p_exam.paragraph_format.line_spacing = 1.2
    r = p_exam.add_run("Signature of the Internal Examiner:\t\t\tSignature of the External Examiner:\nName:\t\t\t\t\t\tName:\nDate:\t\t\t\t\t\tDate:")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)

    p_hod = doc.add_paragraph()
    p_hod.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_hod.paragraph_format.space_before = Pt(16)
    p_hod.paragraph_format.line_spacing = 1.2
    r = p_hod.add_run(
        "Approved by the Head of Department,\n"
        "Computer Science and Engineering\n"
        "(Artificial Intelligence and Machine Learning)\n\n"
        "Name: Dr. Tulasi Prasad Sariki\n"
        "Date:\n"
        "(Seal of School of Computer Science and Engineering)"
    )
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)

    # =========================================================================
    # 5. ABSTRACT
    # =========================================================================
    doc.add_page_break()
    p_abs_title = doc.add_paragraph()
    p_abs_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_abs_title.paragraph_format.space_before = Pt(20)
    p_abs_title.paragraph_format.space_after = Pt(18)
    r = p_abs_title.add_run("ABSTRACT")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.underline = True

    add_paragraph(
        doc,
        "Data-to-Text (D2T) generation automatically converts structured inputs like relational tables, RDF graphs, and multi-relation records "
        "into fluent, grammatically sound, and factually faithful natural language text. "
        "While modern decoder-only Small Language Models (SLMs) generate fluent sentences, their factual precision degrades substantially "
        "when inputs expand past 512 tokens. "
        "Standard pipelines flatten 2D table grids and branching graph edges into a single linear sequence of subword tokens separated by markup tags. "
        "In doing so, entities that are structurally adjacent in the underlying topology become separated by hundreds or thousands of tokens. "
        "Under the well-documented 'Lost in the Middle' (LITM) effect, causal multi-head self-attention mechanisms succumb "
        "to U-shaped positional biases and attention diffusion, heavily discounting middle-sequence tokens and hallucinating "
        "unsupported relationships."
    )

    add_paragraph(
        doc,
        "To resolve this fundamental architectural deficiency without incurring the prohibitive computational overhead of external "
        "graph message-passing networks, this thesis introduces Topology-aware Positional Semantic Divergence (PSD) modeling. "
        "PSD is a mathematically formulated auxiliary attention regularization objective applied during parameter-efficient fine-tuning "
        "(parameter-efficient fine-tuning via LoRA). The framework constructs a joint structural-semantic affinity prior "
        "that unifies dense token semantic similarity (via a frozen contextual encoder), normalized sequence displacement, and exact "
        "topological geodesic shortest-path distances across multi-relational graphs or Manhattan grid offsets across tabular matrices. "
        "A causal target probability distribution is formed and directly injected into strategic intermediate and deep self-attention "
        "layers using a layer-selective Kullback-Leibler (KL) divergence objective."
    )

    add_paragraph(
        doc,
        "We evaluate the proposed framework across three heterogeneous benchmark datasets (ToTTo, WebNLG 2020, and DART) across "
        "three structurally diverse language model families: Qwen3-4B, Llama-3.2-3B-Instruct, and Gemma-3-4B-Instruct. "
        "Empirical findings demonstrate that PSD delivers extraordinary improvements in lexical alignment and semantic fidelity. "
        "In controlled 500-step training at sequence length L=512, PSD achieves 22.84 BLEU-4 versus 7.92 for the baseline (+188.4% relative gain) "
        "and elevates BERTScore F1 from 89.34% to 91.95%. When scaled to expanded evaluation contexts at L=1024, PSD surges to 32.15 BLEU-4 "
        "versus 18.78 (+71.2% gain), while decreasing internal topological attention divergence by 78.21%. Furthermore, on Gemma-3-4B, PSD yields a "
        "21.2% relative reduction in hallucinated claims (17.54% vs 22.26%) and a +20.8% gain in METEOR score. Comprehensive component "
        "ablations confirm that positional and graph distance priors act as complementary structural scaffolds, simultaneously mitigating "
        "attention dispersion and enforcing strict factual grounding over expanded sequence horizons."
    )

    p_kw = doc.add_paragraph()
    p_kw.paragraph_format.space_before = Pt(12)
    p_kw.paragraph_format.space_after = Pt(0)
    p_kw.paragraph_format.line_spacing = 1.3
    r_kw_bold = p_kw.add_run("Keywords: ")
    r_kw_bold.font.name = "Times New Roman"
    r_kw_bold.font.size = Pt(11)
    r_kw_bold.font.bold = True
    r_kw = p_kw.add_run(
        "Data-to-Text Generation, Long-Context Transformers, Lost-in-the-Middle, Graph Topology, "
        "Positional Semantic Divergence, Small Language Models, Parameter-Efficient Fine-Tuning, Factuality."
    )
    r_kw.font.name = "Times New Roman"
    r_kw.font.size = Pt(11)

    # =========================================================================
    # 6. ACKNOWLEDGEMENT
    # =========================================================================
    doc.add_page_break()
    p_ack_title = doc.add_paragraph()
    p_ack_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_ack_title.paragraph_format.space_before = Pt(20)
    p_ack_title.paragraph_format.space_after = Pt(18)
    r = p_ack_title.add_run("ACKNOWLEDGEMENT")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.underline = True

    add_paragraph(
        doc,
        "It is my pleasure to express with a deep sense of gratitude to Dr. Suguna M., Associate Professor, "
        "School of Computer Science and Engineering, Vellore Institute of Technology, Chennai, for her constant guidance, "
        "continual encouragement, and profound scholarly insight extended throughout this project. Above all, she taught me "
        "rigorous methodological discipline, academic integrity, and scientific patience. My association with her is not confined "
        "to academic requirements alone; it has been a privilege to work under an expert in advanced machine learning and artificial intelligence."
    )

    add_paragraph(
        doc,
        "It is with immense gratitude that I extend my sincere thanks to the visionary leadership of Dr. G. Viswanathan, "
        "our Honorable Chancellor, Mr. Sankar Viswanathan, Dr. Sekar Viswanathan, and Dr. G. V. Selvam, Vice Presidents, "
        "Dr. Sandhya Pentareddy, Executive Director, Ms. Kadhambari S. Viswanathan, Assistant Vice-President, "
        "Dr. V. S. Kanchana Bhaaskaran, Vice-Chancellor, Dr. T. Thyagarajan, Pro-Vice Chancellor, VIT Chennai, "
        "and Dr. P. K. Manoharan, Additional Registrar, for fostering an exceptional research infrastructure, providing advanced "
        "computational environments, and inspiring academic excellence throughout the tenure of this program."
    )

    add_paragraph(
        doc,
        "A special mention of gratitude is due to Dr. Viswanathan V., Dean, Dr. Nithyanandam P., Associate Dean, "
        "Dr. Suganya G., Associate Dean, and Dr. Sweetlin Hemalatha C., Associate Dean, School of Computer Science and Engineering, "
        "Vellore Institute of Technology, Chennai, for generously sharing their knowledge, evaluating milestone demonstrations, "
        "and facilitating seamless academic coordination."
    )

    add_paragraph(
        doc,
        "In jubilant appreciation, I express my whole-hearted thanks to Dr. Tulasi Prasad Sariki, Head of the Department, "
        "B.Tech. Computer Science and Engineering (Artificial Intelligence and Machine Learning), and the Project Coordinators "
        "for their administrative support, technical advice, and encouragement to execute this ambitious thesis."
    )

    add_paragraph(
        doc,
        "My heartfelt thanks to all the faculty members and technical staff of VIT Chennai who contributed to my foundational "
        "learning. Finally, I owe an eternal debt of gratitude to my parents for their unconditional sacrifices, moral backing, "
        "and unceasing faith in my aspirations, and to my fellow peers and friends who offered valuable discussions and constructive "
        "feedback throughout this journey."
    )

    p_ack_sig = doc.add_paragraph()
    p_ack_sig.paragraph_format.space_before = Pt(24)
    p_ack_sig.paragraph_format.line_spacing = 1.3
    r = p_ack_sig.add_run("Place: Chennai\nDate: 16 October 2026\t\t\t\t\tISHANT SHEKHAR EESHU")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True

    # =========================================================================
    # 7. TABLE OF CONTENTS
    # =========================================================================
    doc.add_page_break()
    p_toc_title = doc.add_paragraph()
    p_toc_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_toc_title.paragraph_format.space_before = Pt(20)
    p_toc_title.paragraph_format.space_after = Pt(18)
    r = p_toc_title.add_run("TABLE OF CONTENTS")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True

    toc_items = [
        ("DECLARATION", "iii"),
        ("CERTIFICATE", "iv"),
        ("ABSTRACT", "v"),
        ("ACKNOWLEDGEMENT", "vi"),
        ("TABLE OF CONTENTS", "viii"),
        ("LIST OF TABLES", "xi"),
        ("LIST OF FIGURES", "xii"),
        ("LIST OF SYMBOLS, ABBREVIATIONS AND NOMENCLATURE", "xiii"),
        ("CHAPTER 1  INTRODUCTION", "15"),
        ("    1.1 Background", "15"),
        ("    1.2 Overview of Data-to-Text Generation and Structural Inputs", "15"),
        ("        1.2.1 Data-to-Text Systems and Representation Modalities", "15"),
        ("        1.2.2 Transformer Architecture and Self-Attention Mechanism", "16"),
        ("        1.2.3 Serialization of Heterogeneous Graphs and 2D Tables", "16"),
        ("    1.3 Challenges in Long-Context Data-to-Text Generation", "16"),
        ("        1.3.1 The 'Lost in the Middle' (LITM) Phenomenon and Positional Bias", "16"),
        ("        1.3.2 Structural Information Loss and Quadratic Attention Complexity", "17"),
        ("        1.3.3 Attention Diffusion, Fact Hallucination, and Semantic Drift", "17"),
        ("    1.4 Problem Statement", "17"),
        ("    1.5 Objectives of the Research", "18"),
        ("    1.6 Research Questions", "18"),
        ("    1.7 Scope and Key Technical Deliverables", "19"),
        ("CHAPTER 2  LITERATURE SURVEY", "20"),
        ("    2.1 Introduction", "20"),
        ("    2.2 Transformer-based Data-to-Text Generation and Structured Representations", "20"),
        ("    2.3 Positional Bias, Long-Context Transformers, and Mitigation Techniques", "21"),
        ("    2.4 Faithfulness, Semantic Consistency, and Grounded Evaluation", "22"),
        ("    2.5 Summary of Related Prior Art", "22"),
        ("    2.6 Research Gap and Scientific Motivation", "23"),
        ("CHAPTER 3  SYSTEM DESIGN AND METHODOLOGY", "25"),
        ("    3.1 Introduction", "25"),
        ("    3.2 Cloud FIST H100 GPU Server — System Requirements and Environment", "25"),
        ("        3.2.1 Hardware and System Specifications", "26"),
        ("        3.2.2 Software Requirements for the PSD Project", "26"),
        ("        3.2.3 Research Model and Training Requirements", "27"),
        ("        3.2.4 Dataset and Storage Requirements", "28"),
        ("        3.2.5 GPU Suitability and Memory Considerations", "28"),
        ("        3.2.6 Environment Setup and Verification Status", "29"),
        ("        3.2.7 System Requirements Synthesis and Deployment Protocol", "29"),
        ("    3.3 Theoretical Framework: Positional Semantic Divergence (PSD)", "30"),
        ("        3.3.1 Task Formulation and Mathematical Representation", "30"),
        ("        3.3.2 Structural-Semantic Affinity Matrix Formulation (R_ij)", "31"),
        ("        3.3.3 Causal Target Attention Distribution (P_target)", "31"),
        ("    3.4 Multi-Modal Topological Distance Spaces", "32"),
        ("        3.4.1 Graph Geodesic Shortest-Path Metric (All-Pairs Shortest Path via BFS)", "32"),
        ("        3.4.2 Manhattan Coordinate Grid Metric (2D Relational Tables)", "32"),
        ("        3.4.3 Normalized Geometric Distance Formulations", "32"),
        ("    3.5 Frozen Contextual Semantic Representation", "33"),
        ("    3.6 Layer-Selective Attention Supervision Objective", "33"),
        ("        3.6.1 Attention Extraction and Head Averaging", "33"),
        ("        3.6.2 Layer-Selective Supervision Rationale", "34"),
        ("        3.6.3 Positional Semantic Divergence Loss (L_PSD)", "34"),
        ("        3.6.4 Joint Optimization Objective and Target-Only Language Model Masking", "35"),
        ("    3.7 Parameter-Efficient Fine-Tuning Strategy (PEFT LoRA)", "35"),
        ("    3.8 System Architecture and Pipeline Modules", "36"),
        ("    3.9 Benchmark Datasets and Data Partitioning Scheme", "36"),
        ("    3.10 Evaluation Methodology and Statistical Validation Protocols", "37"),
        ("CHAPTER 4  IMPLEMENTATION", "38"),
        ("    4.1 Introduction", "38"),
        ("    4.2 Multi-Stage Data Ingestion and Graph Construction Pipeline", "39"),
        ("    4.3 Deterministic Text Serialization and Token-to-Node Alignment", "39"),
        ("    4.4 Positional Semantic Divergence Core Engine Implementation", "40"),
        ("    4.5 Training Loop and Optimization Workflow", "40"),
        ("    4.6 Comprehensive Evaluation Pipeline Implementation", "41"),
        ("    4.7 Technical Challenges Encountered and Resolved", "41"),
        ("CHAPTER 5  RESULTS AND DISCUSSION", "43"),
        ("    5.1 Introduction and Benchmarking Protocol", "43"),
        ("    5.2 Long-Context Sequence Scaling (L = 1024 and L = 2048 Tokens)", "43"),
        ("    5.3 Hallucination Suppression and Claim-Level Factual Grounding", "46"),
        ("    5.4 Cross-Architecture Generalization Benchmark (L = 256 Tokens)", "47"),
        ("    5.5 Comprehensive Ablation Studies", "49"),
        ("        5.5.1 Target Distribution Component Decomposition Ablation", "49"),
        ("        5.5.2 Loss Weight Sensitivity and Layer Placement Analysis", "50"),
        ("    5.6 Qualitative Case Study: Factual Grounding vs. Baseline Prior Drift", "51"),
        ("    5.7 Summary of Key Technical Insights and Theoretical Synthesis", "52"),
        ("CHAPTER 6  CONCLUSION AND FUTURE WORK", "54"),
        ("    6.1 Conclusion", "54"),
        ("    6.2 Summary of Key Takeaway Points", "55"),
        ("    6.3 Future Work Directions", "56"),
        ("REFERENCES (APA FORMAT)", "57"),
        ("APPENDICES", "61"),
        ("    Appendix 1: Complete Configuration Schemas and Unified Data Models", "61"),
        ("    Appendix 2: Key Source Code Listings and Core Algorithms", "62"),
        ("    Appendix 3: Experimental Parameters, Hyperparameter Grids, and Reproduction Guide", "72")
    ]

    p_tbl = doc.add_table(rows=len(toc_items), cols=2)
    p_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (title, page) in enumerate(toc_items):
        cell_t = p_tbl.rows[idx].cells[0]
        cell_p = p_tbl.rows[idx].cells[1]
        cell_t.width = Inches(5.0)
        cell_p.width = Inches(0.8)
        cell_t.text = title
        cell_p.text = page
        p_t = cell_t.paragraphs[0]
        p_p = cell_p.paragraphs[0]
        p_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_t.paragraph_format.line_spacing = 1.0
        p_p.paragraph_format.line_spacing = 1.0
        p_t.paragraph_format.space_before = Pt(0.5)
        p_t.paragraph_format.space_after = Pt(0.5)
        p_p.paragraph_format.space_before = Pt(0.5)
        p_p.paragraph_format.space_after = Pt(0.5)
        is_major = title.startswith("CHAPTER") or title in ("DECLARATION", "CERTIFICATE", "ABSTRACT", "ACKNOWLEDGEMENT", "REFERENCES (APA FORMAT)", "APPENDICES")
        for run in p_t.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(9.5)
            run.font.bold = is_major
        for run in p_p.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(9.5)
            run.font.bold = is_major

    # =========================================================================
    # 9. LIST OF TABLES
    # =========================================================================
    doc.add_page_break()
    p_lot_title = doc.add_paragraph()
    p_lot_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_lot_title.paragraph_format.space_before = Pt(16)
    p_lot_title.paragraph_format.space_after = Pt(12)
    r = p_lot_title.add_run("LIST OF TABLES")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True

    tables_list = [
        ("Table 2.1", "Summary and taxonomy of related prior art in structured data-to-text generation and attention modeling", "22"),
        ("Table 3.1", "Observed hardware and system specifications of the Cloud FIST H100 GPU server environment", "26"),
        ("Table 3.2", "Project software requirements and dependencies for the PSD research pipeline", "26"),
        ("Table 3.3", "Model memory footprints and precision configurations for PSD training", "27"),
        ("Table 3.4", "Dataset partition sizes and persistent storage footprint requirements", "28"),
        ("Table 3.5", "Cloud FIST server environment verification audit and action checklist", "29"),
        ("Table 3.6", "Small Language Model (SLM) backbone architectural specifications across the 8-model benchmark suite", "35"),
        ("Table 3.7", "Dataset partition statistics across cleaned, normalized, and graph-serialized splits", "36"),
        ("Table 4.1", "Stage-level execution modules and engineering responsibilities", "38"),
        ("Table 4.2", "Deterministic serialization markup syntax across tables and RDF knowledge graphs", "39"),
        ("Table 5.1", "Long-Context D2T Performance Benchmark across Gemma 3 (4B), Llama 3.2 (3B), and Qwen 2.5 (3B) at L = 1024 and L = 2048 Tokens", "44"),
        ("Table 5.2", "Cross-Architecture Generalization Benchmark Suite across 8 Model Configurations at Standard Context Length (L = 256 Tokens)", "47"),
        ("Table 5.3", "Target Attention Formulation Component Decomposition Ablation across 7 Variants", "49"),
        ("Table 5.4", "Cross-Model Loss Weight and Layer Placement Ablation Suite across Gemma 3, Llama 3.2, and Qwen 2.5", "50"),
        ("Table 5.5", "Qualitative Error Case Study on Wikipedia Table-to-Text Benchmark comparing Baseline LoRA and PSD (Ours)", "51"),
        ("Table A3.1", "Complete experimental parameter grid for training, LoRA PEFT, and inference", "73")
    ]

    p_tbl_t = doc.add_table(rows=len(tables_list), cols=3)
    p_tbl_t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (t_num, t_cap, t_page) in enumerate(tables_list):
        c0 = p_tbl_t.rows[idx].cells[0]
        c1 = p_tbl_t.rows[idx].cells[1]
        c2 = p_tbl_t.rows[idx].cells[2]
        c0.width = Inches(1.2)
        c1.width = Inches(3.8)
        c2.width = Inches(0.8)
        c0.text = t_num
        c1.text = t_cap
        c2.text = t_page
        for c, p in [(c0, c0.paragraphs[0]), (c1, c1.paragraphs[0]), (c2, c2.paragraphs[0])]:
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(1.0)
            p.paragraph_format.space_after = Pt(1.0)
            for run in p.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(9.5)
        c2.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT

    # =========================================================================
    # 10. LIST OF FIGURES
    # =========================================================================
    doc.add_page_break()
    p_lof_title = doc.add_paragraph()
    p_lof_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_lof_title.paragraph_format.space_before = Pt(16)
    p_lof_title.paragraph_format.space_after = Pt(12)
    r = p_lof_title.add_run("LIST OF FIGURES")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True

    figures_list = [
        ("Figure 1.1", "Comprehensive Exploratory Data Analysis (EDA) of input sequence lengths, target verbalizations, and structural element distributions across ToTTo, WebNLG, and DART benchmarks", "17"),
        ("Figure 5.1", "Comparison of ROUGE-L and BLEU-4 Across Context Lengths (L = 1024 and L = 2048 Tokens) comparing Baseline LoRA and PSD (Ours)", "45"),
        ("Figure 5.2", "Three-Point Factual Precision Scaling Curve (L = 256 -> 1024 -> 2048 Tokens) across Gemma-4B, Llama-3B, and Qwen-3B, showing Qwen PSD reaching 90.31%", "46"),
        ("Figure 5.3", "Hallucination Rate Reduction Across Sequence Lengths and Model Scales comparing Baseline and PSD Pipeline (Lower is Better)", "47"),
        ("Figure 5.4", "Relative Percentage Improvement in BLEU-4 and ROUGE-L Over Baseline Fine-Tuning across Evaluated Language Model Architectures", "48"),
        ("Figure 5.5", "Ablation Analysis: (a) Loss Weight (lambda_PSD) Inverted U-Curve, and (b) Layer Placement Factual Precision across SLM Architectures", "51")
    ]

    p_tbl_f = doc.add_table(rows=len(figures_list), cols=3)
    p_tbl_f.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (f_num, f_cap, f_page) in enumerate(figures_list):
        c0 = p_tbl_f.rows[idx].cells[0]
        c1 = p_tbl_f.rows[idx].cells[1]
        c2 = p_tbl_f.rows[idx].cells[2]
        c0.width = Inches(1.2)
        c1.width = Inches(3.8)
        c2.width = Inches(0.8)
        c0.text = f_num
        c1.text = f_cap
        c2.text = f_page
        for c, p in [(c0, c0.paragraphs[0]), (c1, c1.paragraphs[0]), (c2, c2.paragraphs[0])]:
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(1.0)
            p.paragraph_format.space_after = Pt(1.0)
            for run in p.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(9.5)
        c2.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT

    # =========================================================================
    # 11. LIST OF SYMBOLS, ABBREVIATIONS AND NOMENCLATURE
    # =========================================================================
    doc.add_page_break()
    p_abbr_title = doc.add_paragraph()
    p_abbr_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_abbr_title.paragraph_format.space_before = Pt(16)
    p_abbr_title.paragraph_format.space_after = Pt(12)
    r = p_abbr_title.add_run("LIST OF SYMBOLS, ABBREVIATIONS AND NOMENCLATURE")
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True

    abbr_items = [
        ("APSP", "All-Pairs Shortest Path"),
        ("BART", "Bidirectional and Auto-Regressive Transformers"),
        ("BERT", "Bidirectional Encoder Representations from Transformers"),
        ("BFS", "Breadth-First Search"),
        ("BLEU", "Bilingual Evaluation Understudy"),
        ("BPE", "Byte-Pair Encoding"),
        ("D2T", "Data-to-Text"),
        ("DART", "Data Record to Text Benchmark Dataset"),
        ("EDA", "Exploratory Data Analysis"),
        ("GNN", "Graph Neural Network"),
        ("HF", "Hugging Face"),
        ("KL", "Kullback-Leibler (Relative Entropy)"),
        ("LITM", "Lost in the Middle"),
        ("LLM", "Large Language Model"),
        ("LoRA", "Low-Rank Adaptation"),
        ("METEOR", "Metric for Evaluation of Translation with Explicit ORdering"),
        ("NLP", "Natural Language Processing"),
        ("NLG", "Natural Language Generation"),
        ("NLL", "Negative Log-Likelihood"),
        ("OGB", "Open Graph Benchmark"),
        ("OOM", "Out of Memory"),
        ("PEFT", "Parameter-Efficient Fine-Tuning"),
        ("PSD", "Positional Semantic Divergence"),
        ("RDF", "Resource Description Framework"),
        ("ROUGE", "Recall-Oriented Understudy for Gisting Evaluation"),
        ("SBERT", "Sentence-BERT"),
        ("SLM", "Small Language Model"),
        ("SPO", "Subject-Predicate-Object (Knowledge Triple)"),
        ("T5", "Text-to-Text Transfer Transformer"),
        ("ToTTo", "Table-to-Text Benchmark Dataset"),
        ("VRAM", "Video Random-Access Memory"),
        ("WebNLG", "Web Natural Language Generation Benchmark")
    ]

    p_tbl_a = doc.add_table(rows=len(abbr_items), cols=2)
    p_tbl_a.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (abbr, desc) in enumerate(abbr_items):
        c0 = p_tbl_a.rows[idx].cells[0]
        c1 = p_tbl_a.rows[idx].cells[1]
        c0.width = Inches(1.8)
        c1.width = Inches(4.0)
        c0.text = abbr
        c1.text = desc
        for c, p in [(c0, c0.paragraphs[0]), (c1, c1.paragraphs[0])]:
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(0.5)
            p.paragraph_format.space_after = Pt(0.5)
            for run in p.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(9.5)
        c0.paragraphs[0].runs[0].font.bold = True

    # Mathematical Symbols
    p_sym_heading = doc.add_paragraph()
    p_sym_heading.paragraph_format.space_before = Pt(8)
    p_sym_heading.paragraph_format.space_after = Pt(4)
    r = p_sym_heading.add_run("Mathematical Symbols and Nomenclature")
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True

    sym_items = [
        ("D", "Structured input data instance (graph D=(V,E) or table D={c_r,k})"),
        ("X = (x_1, ..., x_N)", "Linear serialized token sequence of input prompt"),
        ("Y = (y_1, ..., y_M)", "Target natural language token sequence"),
        ("R_{i,j}", "Composite structural-semantic affinity score between token i and j"),
        ("S_{i,j}", "Cosine semantic similarity between token representations e_i and e_j"),
        ("d_P^{norm}(i, j)", "Normalized positional separation |i - j| / max(N-1, 1) in sequence"),
        ("d_G^{norm}(i, j)", "Normalized shortest-path topological graph distance between nodes"),
        ("d_T(u, v)", "Manhattan coordinate grid distance |r_u - r_v| + |c_u - c_v| for tables"),
        ("P_{target}(j | i)", "Causal target probability distribution over preceding key tokens j <= i"),
        ("A^{(l)}(j | i)", "Head-averaged self-attention probability matrix at layer l"),
        ("L_{selected}", "Set of monitored intermediate and deep Transformer layers"),
        ("D_{KL}(P || Q)", "Kullback-Leibler divergence from distribution Q to distribution P"),
        ("L_{LM}", "Cross-entropy autoregressive language modeling loss"),
        ("L_{PSD}", "Positional Semantic Divergence attention regularization loss"),
        ("L_{total}", "Joint multi-task training loss: L_{LM} + lambda_{PSD} * L_{PSD}"),
        ("lambda_{PSD}", "Regularization weighting hyperparameter for PSD loss (default: 0.1)"),
        ("alpha, beta, gamma", "Weighting coefficients for semantic, positional, and graph distance terms"),
        ("tau", "Softmax temperature parameter controlling target distribution sharpness"),
        ("mu(i)", "Alignment mapping function assigning token index i to discrete node u in V")
    ]

    p_tbl_s = doc.add_table(rows=len(sym_items), cols=2)
    p_tbl_s.alignment = WD_TABLE_ALIGNMENT.CENTER
    for idx, (sym, desc) in enumerate(sym_items):
        c0 = p_tbl_s.rows[idx].cells[0]
        c1 = p_tbl_s.rows[idx].cells[1]
        c0.width = Inches(1.8)
        c1.width = Inches(4.0)
        c0.text = sym
        c1.text = desc
        for c, p in [(c0, c0.paragraphs[0]), (c1, c1.paragraphs[0])]:
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(0.5)
            p.paragraph_format.space_after = Pt(0.5)
            for run in p.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(9.5)
        c0.paragraphs[0].runs[0].font.italic = True
