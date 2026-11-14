import os
import docx
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

def set_cell_border(cell, **kwargs):
    """
    Set cell borders.
    kwargs: top, bottom, left, right
    values: dict(sz=12, val='single', color='AAAAAA', space='0')
    """
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in("w:tcBorders")
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = 'w:{}'.format(edge)
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for key, val in edge_data.items():
                element.set(qn('w:{}'.format(key)), str(val))

def set_cell_shading(cell, color_hex="F2F2F2"):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))

def add_page_number_field(run):
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = "PAGE"
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')
    r = run._r
    r.append(fldChar1)
    r.append(instrText)
    r.append(fldChar2)
    r.append(fldChar3)

def setup_page_layout(section, is_cover=False):
    # A4 dimensions
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    # Margins: Left: 3.81 cm (1.5 in), Right: 2.54 cm, Top: 2.54 cm, Bottom: 2.54 cm
    section.left_margin = Cm(3.81)
    section.right_margin = Cm(2.54)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    
    # Configure footer with centered page number
    if not is_cover:
        footer = section.footer
        p = footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.text = ""
        run = p.add_run()
        run.font.name = "Times New Roman"
        run.font.size = Pt(11)
        add_page_number_field(run)

def setup_document_styles(doc):
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)
    normal_style.paragraph_format.line_spacing = 1.5
    normal_style.paragraph_format.space_after = Pt(4)
    normal_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

def add_chapter_heading(doc, chapter_title, chapter_num=None):
    doc.add_page_break()
    p_num = doc.add_paragraph()
    p_num.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_num.paragraph_format.space_before = Pt(28)
    p_num.paragraph_format.space_after = Pt(8)
    p_num.paragraph_format.line_spacing = 1.0
    r_num = p_num.add_run(f"CHAPTER {chapter_num}" if chapter_num is not None else chapter_title.upper())
    r_num.font.name = "Times New Roman"
    r_num.font.size = Pt(16)
    r_num.font.bold = True
    
    if chapter_num is not None:
        p_title = doc.add_paragraph()
        p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_title.paragraph_format.space_before = Pt(0)
        p_title.paragraph_format.space_after = Pt(20)
        p_title.paragraph_format.line_spacing = 1.15
        r_title = p_title.add_run(chapter_title.upper())
        r_title.font.name = "Times New Roman"
        r_title.font.size = Pt(16)
        r_title.font.bold = True

def add_section_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text.upper())
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True
    return p

def add_subsection_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text.upper())
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True
    return p

def add_subsubsection_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.italic = True
    return p

def add_paragraph(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    return p

def add_bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.35
    p.paragraph_format.space_after = Pt(3)
    if bold_prefix:
        r_bold = p.add_run(bold_prefix + " ")
        r_bold.font.name = "Times New Roman"
        r_bold.font.size = Pt(12)
        r_bold.font.bold = True
    r_text = p.add_run(text)
    r_text.font.name = "Times New Roman"
    r_text.font.size = Pt(12)
    return p

def add_equation(doc, formula_runs, eq_num=None):
    """
    Adds a display equation with centered formula and right-aligned equation number.
    formula_runs: list of tuples (text, is_italic, is_subscript, is_superscript, is_bold)
                  OR a string.
    eq_num: optional string like '(3.1)'
    """
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    cell_eq = table.rows[0].cells[0]
    cell_num = table.rows[0].cells[1]

    cell_eq.width = Inches(5.0)
    cell_num.width = Inches(0.77)

    for cell in (cell_eq, cell_num):
        set_cell_border(cell,
                        top=dict(val='none'),
                        bottom=dict(val='none'),
                        left=dict(val='none'),
                        right=dict(val='none'))

    p_eq = cell_eq.paragraphs[0]
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq.paragraph_format.space_before = Pt(6)
    p_eq.paragraph_format.space_after = Pt(6)
    p_eq.paragraph_format.line_spacing = 1.15

    if isinstance(formula_runs, str):
        r = p_eq.add_run(formula_runs)
        r.font.name = "Cambria Math"
        r.font.size = Pt(11.5)
        r.font.italic = True
    else:
        for item in formula_runs:
            if isinstance(item, tuple):
                text = item[0]
                is_it = item[1] if len(item) > 1 else False
                is_sub = item[2] if len(item) > 2 else False
                is_sup = item[3] if len(item) > 3 else False
                is_b = item[4] if len(item) > 4 else False
            else:
                text = str(item)
                is_it = False
                is_sub = False
                is_sup = False
                is_b = False

            r = p_eq.add_run(text)
            r.font.name = "Cambria Math" if is_it or is_sub or is_sup else "Times New Roman"
            r.font.size = Pt(11.5)
            r.font.italic = is_it
            r.font.subscript = is_sub
            r.font.superscript = is_sup
            r.font.bold = is_b

    p_num = cell_num.paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_num.paragraph_format.space_before = Pt(6)
    p_num.paragraph_format.space_after = Pt(6)
    p_num.paragraph_format.line_spacing = 1.15
    if eq_num:
        r_num = p_num.add_run(eq_num)
        r_num.font.name = "Times New Roman"
        r_num.font.size = Pt(11)
        r_num.font.bold = True


def add_custom_table(doc, headers, rows, caption=None, col_widths=None):
    if caption:
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(8)
        p_cap.paragraph_format.space_after = Pt(3)
        p_cap.paragraph_format.keep_with_next = True
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(10.5)
        r_cap.font.bold = True

    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, header_text in enumerate(headers):
        hdr_cells[i].text = header_text
        set_cell_shading(hdr_cells[i], "EAEAEA")
        set_cell_border(hdr_cells[i],
                        top=dict(sz=8, val='single', color='333333'),
                        bottom=dict(sz=12, val='single', color='333333'),
                        left=dict(sz=4, val='single', color='CCCCCC'),
                        right=dict(sz=4, val='single', color='CCCCCC'))
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.05
        for run in p.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(10)
            run.font.bold = True

    # Data Rows
    for row_idx, row_data in enumerate(rows):
        row_cells = table.rows[row_idx + 1].cells
        bg_color = "F9F9F9" if row_idx % 2 == 1 else "FFFFFF"
        for col_idx, cell_value in enumerate(row_data):
            row_cells[col_idx].text = str(cell_value)
            set_cell_shading(row_cells[col_idx], bg_color)
            set_cell_border(row_cells[col_idx],
                            top=dict(sz=4, val='single', color='E0E0E0'),
                            bottom=dict(sz=4, val='single', color='E0E0E0'),
                            left=dict(sz=4, val='single', color='CCCCCC'),
                            right=dict(sz=4, val='single', color='CCCCCC'))
            p = row_cells[col_idx].paragraphs[0]
            val_str = str(cell_value).strip()
            if col_idx == 0 and len(val_str) > 15:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            elif any(char.isdigit() for char in val_str) and len(val_str) < 15:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            p.paragraph_format.line_spacing = 1.05
            for run in p.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(9.5)

    # Set column widths if provided
    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = width

def add_custom_figure(doc, img_path, caption_text, width=Inches(4.1)):
    if os.path.exists(img_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        run = p_img.add_run()
        run.add_picture(img_path, width=width)
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(6)
        p_cap.paragraph_format.line_spacing = 1.15
        r_cap = p_cap.add_run(caption_text)
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(10)
        r_cap.font.italic = True
    else:
        p_err = doc.add_paragraph()
        p_err.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p_err.add_run(f"[{caption_text} — Figure file referenced: {os.path.basename(img_path)}]")
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)
        r.font.italic = True

def add_code_block(doc, code_str, caption=None):
    if caption:
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_cap.paragraph_format.space_before = Pt(6)
        p_cap.paragraph_format.space_after = Pt(2)
        r_cap = p_cap.add_run(caption)
        r_cap.font.name = "Times New Roman"
        r_cap.font.size = Pt(10)
        r_cap.font.bold = True

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.rows[0].cells[0]
    cell.width = Inches(5.8)
    set_cell_shading(cell, "F5F5F5")
    set_cell_border(cell,
                    top=dict(sz=6, val='single', color='CCCCCC'),
                    bottom=dict(sz=6, val='single', color='CCCCCC'),
                    left=dict(sz=12, val='single', color='4F81BD'),
                    right=dict(sz=6, val='single', color='CCCCCC'))
    
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(code_str.strip())
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(30, 30, 30)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(2)
    p_sp.paragraph_format.space_after = Pt(4)
