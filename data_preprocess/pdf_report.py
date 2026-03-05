import os
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, HRFlowable
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_pdf_report(df: pd.DataFrame, output_path: str = "data/report/data_understanding_report.pdf"):
    PDF_OUT = Path(output_path)
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    
    # Directory to store temporary images for the PDF
    img_dir = PDF_OUT.parent / "charts"
    img_dir.mkdir(exist_ok=True)

    # Copy data to avoid mutating original, then ensure numeric columns are correctly loaded
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                temp = df[col].astype(str).str.strip().str.replace(r'\s+', '', regex=True).str.replace(',', '.', regex=False)
                converted = pd.to_numeric(temp, errors='coerce')
                if converted.notna().sum() > 0 or df[col].isna().all():
                    df[col] = converted
            except Exception:
                pass

    shape = df.shape
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100

    # Custom styling
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=24,
        textColor=HexColor("#1A365D"),
        spaceAfter=30,
        fontName="Helvetica-Bold",
        alignment=TA_CENTER
    )
    
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=HexColor("#2B6CB0"),
        spaceBefore=20,
        spaceAfter=15,
        fontName="Helvetica-Bold",
        borderPadding=(0, 0, 4, 0),
        borderColor=HexColor("#E2E8F0"),
        borderWidth=1
    )
    
    subsection_style = ParagraphStyle(
        "SubSectionHeader",
        parent=styles["Heading3"],
        fontSize=14,
        textColor=HexColor("#4A5568"),
        spaceBefore=15,
        spaceAfter=10,
        fontName="Helvetica-Bold"
    )

    normal_style = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=11,
        textColor=HexColor("#2D3748"),
        leading=16, # line height
        spaceAfter=6,
        fontName="Helvetica"
    )
    
    bullet_style = ParagraphStyle(
        "CustomBullet",
        parent=normal_style,
        leftIndent=20,
        firstLineIndent=-10
    )

    doc = SimpleDocTemplate(
        str(PDF_OUT), 
        pagesize=A4, 
        rightMargin=50, 
        leftMargin=50, 
        topMargin=50, 
        bottomMargin=50
    )
    story = []

    # Title
    story.append(Paragraph("Data Analysis & Insights Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=HexColor("#CBD5E0"), spaceAfter=20, spaceBefore=0))

    # 1. Overview
    story.append(Paragraph("1. Dataset Overview", section_style))
    story.append(Paragraph(f"<b>• Total Rows:</b> {shape[0]:,}", bullet_style))
    story.append(Paragraph(f"<b>• Total Columns:</b> {shape[1]}", bullet_style))
    if numeric_cols:
        story.append(Paragraph(f"<b>• Numeric Features ({len(numeric_cols)}):</b> {', '.join(numeric_cols)}", bullet_style))
    if categorical_cols:
        story.append(Paragraph(f"<b>• Categorical Features ({len(categorical_cols)}):</b> {', '.join(categorical_cols)}", bullet_style))

    # 2. Missing Values Analysis
    story.append(Paragraph("2. Data Quality & Missing Values", section_style))
    has_missing = False
    for col in df.columns:
        if missing_values[col] > 0:
            has_missing = True
            story.append(Paragraph(
                f"<b>• {col}:</b> {missing_values[col]:,} missing entries (<font color='red'>{missing_percent[col]:.2f}%</font>)", 
                bullet_style
            ))
    if not has_missing:
        story.append(Paragraph("<i>✓ Excellent: No missing values found in the dataset.</i>", normal_style))

    # 3. Descriptive Statistics for Numeric Features
    story.append(Paragraph("3. Descriptive Statistics (Numeric)", section_style))
    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        for col in numeric_cols:
            stats_text = (
                f"<b>{col}:</b> "
                f"Mean = {desc.loc['mean', col]:,}, "
                f"Median = {desc.loc['50%', col]:,}, "
                f"Min = {desc.loc['min', col]:,}, "
                f"Max = {desc.loc['max', col]:,}, "
                f"StdDev = {desc.loc['std', col]:,}"
            )
            story.append(Paragraph(stats_text, bullet_style))
    else:
        story.append(Paragraph("<i>No numeric columns identified.</i>", normal_style))

    # 4. Categorical Features Summary
    if categorical_cols:
        story.append(Paragraph("4. Categorical Features Summary", section_style))
        for col in categorical_cols:
            unique_count = df[col].nunique()
            story.append(Paragraph(f"<b>• {col}</b> ({unique_count:,} unique values)", bullet_style))
            top_values = df[col].value_counts().head(3)
            examples = ", ".join([f"{k} [{v:,}]" for k, v in top_values.items()])
            story.append(Paragraph(f"<i>Top values:</i> {examples}", ParagraphStyle("SubBullet", parent=bullet_style, leftIndent=40)))
            story.append(Spacer(1, 4))

    # 5. Data Visualizations
    story.append(PageBreak())
    story.append(Paragraph("5. Data Visualizations", section_style))

    # 5a. Correlation Heatmap
    if len(numeric_cols) > 1:
        story.append(Paragraph("Correlation Heatmap", subsection_style))
        story.append(Paragraph("This matrix displays the linear correlation coefficients between numeric features. Values closer to 1 or -1 indicate strong relationships.", normal_style))
        story.append(Spacer(1, 10))
        
        plt.figure(figsize=(8, 6))
        # Ensure only numeric data is used for correlation, dropping full NaNs
        clean_numeric_df = df[numeric_cols].dropna(axis=1, how='all')
        if not clean_numeric_df.empty and clean_numeric_df.shape[1] > 1:
            corr_matrix = clean_numeric_df.corr()
            sns.heatmap(corr_matrix, annot=False, cmap="YlGnBu", fmt=".2f", linewidths=.5)
            plt.title("Feature Correlation Matrix", fontsize=14, pad=15)
            plt.tight_layout()
            
            heatmap_path = img_dir / "correlation_heatmap.png"
            plt.savefig(heatmap_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            story.append(RLImage(str(heatmap_path), width=6.5*inch, height=4.8*inch))
            story.append(Spacer(1, 15))
            
            # Text Correlation summary
            story.append(Paragraph("Top Correlations:", subsection_style))
            corr_matrix_abs = corr_matrix.abs()
            upper = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))
            top_corr = upper.stack().sort_values(ascending=False).dropna().head(5)
            
            if not top_corr.empty:
                for (f1, f2), val in top_corr.items():
                    actual_val = corr_matrix.loc[f1, f2]
                    direction = "Positive" if actual_val > 0 else "Negative"
                    story.append(Paragraph(f"<b>• {f1} & {f2}:</b> {actual_val:.3f} ({direction})", bullet_style))
            else:
                story.append(Paragraph("<i>Not enough variance for correlation.</i>", normal_style))
        else:
             story.append(Paragraph("<i>Not enough valid numeric data for correlation.</i>", normal_style))

    # 5b. Histograms for Numeric Features
    if numeric_cols:
        story.append(PageBreak())
        story.append(Paragraph("Numeric Feature Distributions", subsection_style))
        story.append(Paragraph("Histograms showing the underlying distribution and spread of numerical data.", normal_style))
        story.append(Spacer(1, 15))
        
        valid_cols = [c for c in numeric_cols if df[c].notna().sum() > 0]
        for idx, col in enumerate(valid_cols[:6]): # Limit to first 6
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col].dropna(), kde=True, bins=40, color='#3182ce', edgecolor='black', alpha=0.7)
            plt.title(f"Distribution of {col}", fontsize=12, pad=10)
            plt.xlabel(col, fontsize=10)
            plt.ylabel("Frequency", fontsize=10)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            hist_path = img_dir / f"hist_{col.replace(' ', '_')}.png"
            plt.savefig(hist_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            story.append(RLImage(str(hist_path), width=5.5*inch, height=3.6*inch))
            story.append(Spacer(1, 15))
            
            if (idx + 1) % 2 == 0 and (idx + 1) < min(len(valid_cols), 6):
                story.append(PageBreak())

    # 5c. Time-Series Analysis for Top Stocks
    time_col = 'date' if 'date' in df.columns else ('SEANCE' if 'SEANCE' in df.columns else None)
    stock_col = 'code' if 'code' in df.columns else ('CODE' if 'CODE' in df.columns else None)
    price_col = 'close' if 'close' in df.columns else ('CLOTURE' if 'CLOTURE' in df.columns else None)

    if time_col and stock_col and price_col and df[price_col].notna().sum() > 0:
        story.append(PageBreak())
        story.append(Paragraph("Time-Series Analysis (Top Traded Stocks)", subsection_style))
        story.append(Paragraph(f"Closing price history for the 5 most frequently recorded stocks.", normal_style))
        story.append(Spacer(1, 15))
        
        # Parse dates
        df_ts = df.copy()
        df_ts[time_col] = pd.to_datetime(df_ts[time_col], errors='coerce')
        df_ts = df_ts.dropna(subset=[time_col, stock_col, price_col])
        
        # Get top 5 stocks
        top_stocks = df_ts[stock_col].value_counts().head(5).index.tolist()
        
        plt.figure(figsize=(11, 4.5)) # Much wider figure
        for stock in top_stocks:
            stock_data = df_ts[df_ts[stock_col] == stock].sort_values(by=time_col)
            # Ensure price is numeric
            try:
                prices = pd.to_numeric(stock_data[price_col], errors='coerce')
                plt.plot(stock_data[time_col], prices, label=str(stock), alpha=0.8, linewidth=1.5)
            except Exception:
                pass
                
        plt.title("Closing Price Over Time (Top 5 Stocks)", fontsize=14, pad=15)
        plt.xlabel("Date", fontsize=10)
        plt.ylabel("Closing Price", fontsize=10)
        plt.legend(title="Stock Code/Name", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        ts_path = img_dir / "time_series_top_stocks.png"
        plt.savefig(ts_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        # Expand ReportLab image width to fill the A4 width minus margins (A4 is ~8.27 inches wide, margins are ~1 inch each)
        story.append(RLImage(str(ts_path), width=7.5*inch, height=3.2*inch))
        story.append(Spacer(1, 15))

    doc.build(story)
    print("✅ Professional Styled Data Understanding Report Generated")
    print("Saved at:", PDF_OUT.resolve())

if __name__ == "__main__":
    CSV_PATH = Path("data/merged_csv/cleaned_file.csv")
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, low_memory=False)
        df.columns = df.columns.str.strip()
        generate_pdf_report(df)
    else:
        print(f"File not found: {CSV_PATH}")
