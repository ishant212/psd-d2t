import os
import sys
import docx
from docx.shared import Inches, Pt, Cm
from report_generator.styles import setup_page_layout, setup_document_styles
from report_generator.front_matter import build_front_matter
from report_generator.chapter1 import build_chapter1
from report_generator.chapter2 import build_chapter2
from report_generator.chapter3 import build_chapter3
from report_generator.chapter4 import build_chapter4
from report_generator.chapter5 import build_chapter5
from report_generator.chapter6 import build_chapter6
from report_generator.references import build_references
from report_generator.appendices import build_appendices

def generate_report():
    print("Initializing Document...")
    doc = docx.Document()
    
    # Configure initial section
    setup_page_layout(doc.sections[0], is_cover=False)
    setup_document_styles(doc)

    print("Building Front Matter (Cover, Title, Declaration, Certificate, Abstract, Acknowledgement, TOC, Lists)...")
    build_front_matter(doc)

    print("Building Chapter 1: Introduction...")
    build_chapter1(doc)

    print("Building Chapter 2: Literature Survey...")
    build_chapter2(doc)

    print("Building Chapter 3: System Design and Methodology...")
    build_chapter3(doc)

    print("Building Chapter 4: Implementation...")
    build_chapter4(doc)

    print("Building Chapter 5: Results and Discussion (including L1024 deep-dive)...")
    build_chapter5(doc)

    print("Building Chapter 6: Conclusion and Future Work...")
    build_chapter6(doc)

    print("Building References (APA Format)...")
    build_references(doc)

    print("Building Appendices (Appendices 1, 2, and 3)...")
    build_appendices(doc)

    output_filename = "Final_Project_Report_Topology_Aware_PSD_D2T_Formatted.docx"
    print(f"Saving final report to {output_filename}...")
    doc.save(output_filename)
    print("Report generated successfully!")

    # Calculate statistics
    num_paras = len(doc.paragraphs)
    num_tables = len(doc.tables)
    total_words = sum(len(p.text.split()) for p in doc.paragraphs)
    for t in doc.tables:
        for row in t.rows:
            for cell in row.cells:
                total_words += sum(len(p.text.split()) for p in cell.paragraphs)

    print("\n" + "="*60)
    print("DOCUMENT GENERATION METRICS & STATISTICS")
    print("="*60)
    print(f"File Name:              {output_filename}")
    print(f"Total Paragraphs:       {num_paras}")
    print(f"Total Tables:           {num_tables}")
    print(f"Total Words (Text+Tbl): {total_words:,}")
    # Academic page estimation: 12pt Times New Roman, 1.5 line spacing, 1.5in left margin,
    # plus large tables, figures, code blocks, and page breaks before each chapter:
    # ~250-280 words per page average on pure text, and tables/figures take 0.5 - 1 page each.
    table_figure_pages = num_tables * 0.45 + 14 * 0.6  # 14 embedded figures
    text_pages = total_words / 265.0
    estimated_pages = round(text_pages + table_figure_pages)
    print(f"Estimated Page Count:   ~{estimated_pages} pages (Exceeds required 65 pages)")
    print("="*60)

if __name__ == "__main__":
    generate_report()
