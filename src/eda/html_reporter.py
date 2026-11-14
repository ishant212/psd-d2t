"""
HTML Report Generator for Stage B EDA.
Generates reports/EDA_Report.html embedding plots and statistics side by side across ToTTo, WebNLG, and DART.
"""
import os
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Topology-Aware Pipeline — Stage B Exploratory Data Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            color: #212529;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 20px 0;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .dataset-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #e9ecef;
        }}
        .card h2 {{
            color: #1e3c72;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            margin-top: 15px;
            border: 1px solid #dee2e6;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 8px 12px;
            border: 1px solid #dee2e6;
            text-align: left;
        }}
        th {{
            background-color: #e9ecef;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Stage B — Exploratory Data Analysis Report</h1>
            <p>Comparative Topology & Statistical Overview: ToTTo, WebNLG 2020, and DART</p>
        </div>

        <div class="dataset-grid">
            <!-- TOTTO CARD -->
            <div class="card">
                <h2>ToTTo (Wikipedia Tables)</h2>
                {totto_stats_html}
                <h3>Histograms & Distributions</h3>
                <img src="plots/totto/sentence_and_token_histograms.png" alt="ToTTo Histograms">
                <h3>Vocabulary Growth & Zipf</h3>
                <img src="plots/totto/vocab_growth_zipf_plot.png" alt="ToTTo Zipf">
                <h3>Table Distributions</h3>
                <img src="plots/totto/totto_table_distributions.png" alt="ToTTo Table Stats">
                <h3>Correlation Heatmap</h3>
                <img src="plots/totto/correlation_heatmap.png" alt="ToTTo Correlation">
            </div>

            <!-- WEBNLG CARD -->
            <div class="card">
                <h2>WebNLG 2020 (RDF Triples)</h2>
                {webnlg_stats_html}
                <h3>Histograms & Distributions</h3>
                <img src="plots/webnlg/sentence_and_token_histograms.png" alt="WebNLG Histograms">
                <h3>Vocabulary Growth & Zipf</h3>
                <img src="plots/webnlg/vocab_growth_zipf_plot.png" alt="WebNLG Zipf">
                <h3>Entity & Relation Frequencies</h3>
                <img src="plots/webnlg/relation_entity_frequencies.png" alt="WebNLG Frequencies">
                <h3>Correlation Heatmap</h3>
                <img src="plots/webnlg/correlation_heatmap.png" alt="WebNLG Correlation">
            </div>

            <!-- DART CARD -->
            <div class="card">
                <h2>DART (Triples & Mixed)</h2>
                {dart_stats_html}
                <h3>Histograms & Distributions</h3>
                <img src="plots/dart/sentence_and_token_histograms.png" alt="DART Histograms">
                <h3>Vocabulary Growth & Zipf</h3>
                <img src="plots/dart/vocab_growth_zipf_plot.png" alt="DART Zipf">
                <h3>Entity & Relation Frequencies</h3>
                <img src="plots/dart/relation_entity_frequencies.png" alt="DART Frequencies">
                <h3>Correlation Heatmap</h3>
                <img src="plots/dart/correlation_heatmap.png" alt="DART Correlation">
            </div>
        </div>
    </div>
</body>
</html>
"""

def generate_html_report(reports_dir: str = "reports"):
    """Generates the EDA_Report.html file from computed CSV statistics."""
    def load_stats_html(ds_name: str) -> str:
        csv_path = os.path.join(reports_dir, f"{ds_name}_statistics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df.to_html(classes="table", index=False)
        return "<p>Statistics file not found.</p>"

    totto_html = load_stats_html("totto")
    webnlg_html = load_stats_html("webnlg")
    dart_html = load_stats_html("dart")

    html_content = HTML_TEMPLATE.format(
        totto_stats_html=totto_html,
        webnlg_stats_html=webnlg_html,
        dart_stats_html=dart_html
    )

    output_path = os.path.join(reports_dir, "EDA_Report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Generated unified EDA HTML report at {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_html_report()
