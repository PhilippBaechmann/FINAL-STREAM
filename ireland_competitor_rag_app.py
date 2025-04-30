import streamlit as st
import pandas as pd
import numpy as np
import os
from docx import Document as DocxDocument
from io import BytesIO
from datetime import datetime

# Load Excel data 
@st.cache_data
def load_data(file_path='ireland_cleaned_CHGF.xlsx'):
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Filter high-growth firms
        df['ConsistentHighGrowthFirm 2023'] = pd.to_numeric(df['ConsistentHighGrowthFirm 2023'], errors='coerce')
        high_growth_firms = df[df['ConsistentHighGrowthFirm 2023'] == 1].copy()
        
        if high_growth_firms.empty:
            st.error("No high-growth firms found in the dataset.")
            return None
            
        # Fill NAs for better processing
        high_growth_firms['Description'] = high_growth_firms['Description'].fillna('No description available')
        
        for col in ['NACE_Industry', 'Business_Model', 'City', 'Growth 2023']:
            if col in high_growth_firms.columns:
                high_growth_firms[col] = high_growth_firms[col].fillna('Not available')
                
        return high_growth_firms
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Simple text similarity function (using TF-IDF approach)
def calculate_similarity(text1, text2):
    """Calculate similarity between two text strings using TF-IDF like approach"""
    import re
    from collections import Counter
    
    # Lowercase and tokenize
    def tokenize(text):
        return re.findall(r'\w+', text.lower())
    
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    # Count tokens
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)
    
    # Get unique terms
    all_terms = set(counter1.keys()).union(set(counter2.keys()))
    
    # Calculate dot product
    dot_product = sum(counter1.get(term, 0) * counter2.get(term, 0) for term in all_terms)
    
    # Calculate magnitudes
    magnitude1 = sum(count * count for count in counter1.values()) ** 0.5
    magnitude2 = sum(count * count for count in counter2.values()) ** 0.5
    
    # Calculate cosine similarity
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    else:
        return dot_product / (magnitude1 * magnitude2)

def find_competitors(user_company_desc, df, num_results=5, industry_filter=None, business_model_filter=None):
    """Find competitors based on description similarity"""
    if df is None or len(df) == 0:
        return None, None
    
    # Apply filters if specified
    filtered_df = df.copy()
    if industry_filter:
        filtered_df = filtered_df[filtered_df['NACE_Industry'].isin(industry_filter)]
    if business_model_filter:
        filtered_df = filtered_df[filtered_df['Business_Model'].isin(business_model_filter)]
    
    if len(filtered_df) == 0:
        st.warning("No companies match the selected filters. Showing results without filters.")
        filtered_df = df.copy()
    
    # Calculate similarity scores
    similarities = []
    for idx, row in filtered_df.iterrows():
        company_desc = row['Description']
        similarity = calculate_similarity(user_company_desc, company_desc)
        similarities.append((idx, similarity))
    
    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top matches
    top_indices = [idx for idx, _ in similarities[:num_results]]
    top_scores = [score for _, score in similarities[:num_results]]
    
    return filtered_df.loc[top_indices], top_scores

def generate_analysis(user_company_name, user_company_desc, competitors, similarity_scores):
    """Generate analysis text based on competitors"""
    if competitors is None or len(competitors) == 0:
        return "No relevant competitors found in the database."
    
    analysis = f"# Competitor Analysis for {user_company_name}\n\n"
    analysis += "Based on your company description, I've identified the following potential competitors from Ireland's high-growth firms:\n\n"
    
    for i, (_, row) in enumerate(competitors.iterrows()):
        similarity_percentage = similarity_scores[i] * 100
        analysis += f"## {i+1}. {row['Company Name']} (Similarity: {similarity_percentage:.1f}%)\n\n"
        analysis += f"**Industry:** {row['NACE_Industry']}\n\n"
        analysis += f"**Business Model:** {row['Business_Model']}\n\n"
        
        if 'Estimated_Revenue_mn' in row and pd.notna(row['Estimated_Revenue_mn']):
            analysis += f"**Estimated Revenue:** €{row['Estimated_Revenue_mn']} million\n\n"
            
        if 'Growth 2023' in row and pd.notna(row['Growth 2023']):
            analysis += f"**Growth (2023):** {row['Growth 2023']}\n\n"
            
        analysis += f"**Description:** {row['Description']}\n\n"
        analysis += f"**Relevance:** This company is in the {row['NACE_Industry']} industry"
        
        if row['Business_Model'] != 'Not available':
            analysis += f" with a {row['Business_Model']} business model"
        
        analysis += f". The similarity between your company description and theirs suggests potential competitive overlap.\n\n"
        
        if i < len(competitors) - 1:
            analysis += "---\n\n"
    
    analysis += "## Summary\n\n"
    analysis += f"These {len(competitors)} companies represent potential competitors or comparable businesses in Ireland's high-growth landscape. "
    analysis += "Consider analyzing their strategies, market positioning, and customer base to identify competitive advantages and potential threats."
    
    return analysis

def generate_docx_report(analysis_text, user_company_name, user_company_desc, competitors):
    """Generate a Word document report"""
    try:
        doc = DocxDocument()
        
        # Add header
        doc.add_heading('Competitor Analysis Report', level=0)
        doc.add_paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}")
        
        # Company Information
        doc.add_heading('Your Company', level=1)
        doc.add_paragraph(f"Company Name: {user_company_name}")
        doc.add_paragraph(f"Description: {user_company_desc}")
        
        # Add competitors
        doc.add_heading('Potential Competitors', level=1)
        
        for i, (_, row) in enumerate(competitors.iterrows(), 1):
            doc.add_heading(f"{i}. {row['Company Name']}", level=2)
            
            # Company details table
            table = doc.add_table(rows=1, cols=2)
            table.style = 'Table Grid'
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Attribute'
            hdr_cells[1].text = 'Value'
            
            # Add key data to table
            for key in ['NACE_Industry', 'Business_Model', 'Estimated_Revenue_mn', 
                       'Number of employees 2023', 'Growth 2023', 'City']:
                if key in row and pd.notna(row[key]):
                    row_cells = table.add_row().cells
                    row_cells[0].text = key.replace('_', ' ')
                    row_cells[1].text = str(row[key])
            
            # Add description
            doc.add_paragraph(f"Description: {row['Description']}")
            doc.add_paragraph()  # Spacing
        
        # Recommendations
        doc.add_heading('Recommendations', level=1)
        doc.add_paragraph('Based on this competitor analysis, consider the following actions:')
        recommendations = [
            "Analyze these competitors' strengths and weaknesses",
            "Identify gaps in the market that your company could fill",
            "Study their business models and growth strategies",
            "Consider how to differentiate your offering from these competitors",
            "Monitor their product/service developments regularly"
        ]
        for rec in recommendations:
            p = doc.add_paragraph()
            p.add_run("• " + rec)
        
        # Save to BytesIO
        doc_bytes = BytesIO()
        doc.save(doc_bytes)
        doc_bytes.seek(0)
        
        return doc_bytes
    except Exception as e:
        st.error(f"Error generating report: {e}")
        return None

# Main Streamlit App
st.set_page_config(page_title="Competitor Finder", page_icon="🔍", layout="wide")

st.title("🔍 High-Growth Competitor Finder")
st.subheader("Find potential competitors from Ireland's high-growth firms")

# Load data
high_growth_firms = load_data()

if high_growth_firms is not None:
    # Sidebar filters
    with st.sidebar:
        st.header("Settings")
        
        num_competitors = st.slider(
            "Number of competitors to find:", 
            min_value=3, 
            max_value=10, 
            value=5
        )
        
        st.subheader("Optional Filters")
        
        # Industry filter
        industry_options = sorted(high_growth_firms['NACE_Industry'].unique())
        selected_industries = st.multiselect(
            "Filter by Industry:",
            options=industry_options,
            default=None
        )
        
        # Business model filter
        if 'Business_Model' in high_growth_firms.columns:
            bm_options = sorted(high_growth_firms['Business_Model'].unique())
            selected_bm = st.multiselect(
                "Filter by Business Model:",
                options=bm_options,
                default=None
            )
        else:
            selected_bm = None
    
    # Main area - Company input
    st.header("Your Company Details")
    user_company_name = st.text_input("Company Name:")
    user_company_desc = st.text_area("Company Description:", height=150)
    
    # Find competitors button
    if st.button("Find Competitors", type="primary"):
        if not user_company_name or not user_company_desc:
            st.warning("Please enter both company name and description.")
        else:
            with st.spinner("Finding competitors..."):
                # Find competitors
                competitors, similarity_scores = find_competitors(
                    user_company_desc, 
                    high_growth_firms,
                    num_results=num_competitors,
                    industry_filter=selected_industries if selected_industries else None,
                    business_model_filter=selected_bm if selected_bm else None
                )
                
                if competitors is not None and len(competitors) > 0:
                    # Generate analysis
                    analysis = generate_analysis(user_company_name, user_company_desc, competitors, similarity_scores)
                    
                    # Display analysis
                    st.markdown(analysis)
                    
                    # Generate and offer report download
                    report_bytes = generate_docx_report(analysis, user_company_name, user_company_desc, competitors)
                    if report_bytes:
                        st.download_button(
                            label="📄 Download Detailed Report (.docx)",
                            data=report_bytes,
                            file_name=f"{user_company_name}_competitor_analysis.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                else:
                    st.error("No relevant competitors found. Try adjusting your description or filters.")
else:
    st.error("Failed to load data. Please check if the Excel file exists and has the expected format.")

# Footer
st.markdown("---")
st.caption("High-Growth Competitor Finder | Ireland Database")