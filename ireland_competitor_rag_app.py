import streamlit as st
import pandas as pd
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from docx import Document as DocxDocument
from PIL import Image
from datetime import datetime
# Required libraries: pip install streamlit pandas requests numpy python-docx langchain langchain-community langchain-groq faiss-cpu sentence-transformers openpyxl matplotlib seaborn pillow

from langchain.schema import Document as LangchainDocument
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Configuration ---
DATA_FILE = 'ireland_cleaned_CHGF.xlsx'

# Main columns
CHGF_COL = 'ConsistentHighGrowthFirm 2023'
DESC_COL = 'Description'
NAME_COL = 'Company Name'

# Extended metadata for more comprehensive analysis based on actual Excel columns
METADATA_COLS = [
    'Company Name', 'City', 'NACE_Industry', 'Topic', 'Business_Model', 
    'Estimated_Revenue_mn', 'Founded Year', 'Number of employees 2023',
    'Growth 2023', 'aagr 2023', 'Public_or_Private', 'CEO', 'Description'
]

# Model settings - changed to use simpler embedding approach for better compatibility
USE_SIMPLE_EMBEDDINGS = True  # Set to True to use a simpler embedding method without external models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Used only if USE_SIMPLE_EMBEDDINGS is False
LLM_MODEL = "llama3-8b-8192"  # Default model
FAISS_INDEX_PATH = "faiss_ireland_chgf_index"

# Available models for dropdown
AVAILABLE_MODELS = {
    "Llama 3 (8B)": "llama3-8b-8192",
    "Mixtral (8x7B)": "mixtral-8x7b-32768",
    "Gemma-7b-it": "gemma-7b-it",
    "Claude 3 Haiku": "claude-3-haiku-20240307"
}

# Visualization colors
COLOR_PALETTE = sns.color_palette("viridis", 10)

# --- Helper Functions ---

def serper_search(query: str, api_key: str, k: int = 5):
    """Performs a web search using the Serper API."""
    if not api_key: 
        return "Serper API key not provided."
    
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": k}
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        results = response.json().get('organic', [])
        
        # Enhanced formatting with URL and source
        formatted_results = []
        for r in results:
            title = r.get('title', 'N/A')
            snippet = r.get('snippet', 'N/A')
            url = r.get('link', 'N/A')
            source = r.get('source', 'Unknown Source')
            formatted_results.append(f"Title: {title}\nSource: {source}\nURL: {url}\nSnippet: {snippet}")
            
        return "\n\n".join(formatted_results) if formatted_results else "No web search results found."
    except requests.exceptions.RequestException as e: 
        return f"Web search failed: {e}"
    except Exception as e: 
        return f"An error occurred during web search: {e}"

@st.cache_data(show_spinner="Loading and processing data...")
def load_and_prepare_data(file_path):
    """Loads data from Excel, filters CHGF, creates Langchain Documents."""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Fill NaN values for better processing
        for col in METADATA_COLS:
            if col in df.columns:
                df[col] = df[col].fillna('Not Available')

        # --- Filtering ---
        df[CHGF_COL] = pd.to_numeric(df[CHGF_COL], errors='coerce')
        df_chgf = df.dropna(subset=[CHGF_COL]).copy()
        df_chgf[CHGF_COL] = df_chgf[CHGF_COL].astype(int)
        df_chgf = df_chgf[df_chgf[CHGF_COL] == 1].copy()

        if df_chgf.empty:
            st.error(f"No companies found with '{CHGF_COL}' = 1 in the data.")
            return None, None

        df_chgf[DESC_COL] = df_chgf[DESC_COL].fillna('No description available')

        # --- Create Langchain Documents ---
        documents = []
        for _, row in df_chgf.iterrows():
            metadata = {}
            for col in METADATA_COLS:
                if col in df_chgf.columns:
                    val = row[col]
                    # Ensure data types are JSON serializable for metadata
                    if pd.isna(val): 
                        metadata[col] = "Not Available"
                    elif isinstance(val, (np.int64, np.float64)): 
                        metadata[col] = val.item()  # Convert numpy types
                    else: 
                        metadata[col] = str(val)  # Convert others to string
                        
            page_content = f"Company: {row[NAME_COL]}\nDescription: {row[DESC_COL]}"
            doc = LangchainDocument(page_content=page_content, metadata=metadata)
            documents.append(doc)
            
        if not documents:
            st.error("Failed to create documents from the data.")
            return None, None
            
        st.success(f"Loaded and prepared {len(documents)} consistent high-growth firms from Excel file.")
        return documents, df_chgf
        
    except FileNotFoundError:
        st.error(f"Error: Data file '{file_path}' not found. Please ensure it's in the correct directory.")
        return None, None
    except ImportError:
        st.error("Error: The 'openpyxl' library is required to read Excel files. Please install it: pip install openpyxl")
        return None, None
    except KeyError as e:
        st.error(f"Error: Column '{e}' not found in the data file. Please check column names in the Excel sheet.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None, None

@st.cache_resource(show_spinner="Initializing vector store...")
def create_or_load_vector_store(_documents, index_path):
    """Creates FAISS vector store from documents or loads if exists."""
    try:
        # Choose embedding method based on configuration
        if USE_SIMPLE_EMBEDDINGS:
            st.info("Using simple embeddings (no external model required)")
            embeddings = SimpleEmbeddings()
        else:
            try:
                st.info(f"Using HuggingFace embeddings: {EMBEDDING_MODEL}")
                embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            except Exception as e:
                st.warning(f"Failed to load HuggingFace embeddings: {e}. Falling back to simple embeddings.")
                embeddings = SimpleEmbeddings()
    
        if os.path.exists(index_path) and os.path.isdir(index_path):
            try:
                vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
                st.info(f"Loaded existing FAISS index from {index_path}")
                return vectorstore
            except Exception as e:
                st.warning(f"Failed to load existing index ({e}). Rebuilding index.")
                
        if not _documents:
            st.error("No documents available to build vector store.")
            return None
            
        try:
            vectorstore = FAISS.from_documents(_documents, embeddings)
            vectorstore.save_local(index_path)
            st.success(f"Created and saved FAISS index to {index_path}")
            return vectorstore
        except Exception as e:
            st.error(f"Failed to create vector store: {e}")
            return None
    except Exception as e:
        st.error(f"Unexpected error in vector store creation: {e}")
        return None

def setup_rag_chain(vectorstore, api_key, model_name=LLM_MODEL):
    """Sets up the ConversationalRetrievalChain with custom prompt."""
    if not vectorstore or not api_key: 
        return None
        
    try:
        # Set environment variable for Groq SDK
        os.environ['GROQ_API_KEY'] = api_key
        
        # Initialize LLM with selected model
        llm = ChatGroq(temperature=0.2, model_name=model_name)
        
        # Memory to store chat history
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
        
        # Custom prompt template for competitor analysis
        prompt_template = """
        You are an expert business analyst specializing in competitive intelligence and market analysis for Irish companies.
        
        Context from database:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Please provide an insightful, structured analysis that includes:
        1. Clear identification of relevant competitors based on the provided data
        2. Explanation of why these companies are competitors (industry overlap, similar offerings, etc.)
        3. Key differentiators and potential competitive advantages
        4. Market positioning insights when available
        
        Base your analysis only on the provided information. If information is not available in the context, clearly state that instead of making assumptions.
        
        Analysis:
        """
        
        # Create prompt template
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "chat_history"]
        )
        
        # Create the chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),  # Retrieve top 6 most similar documents
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            verbose=False
        )
        
        return chain
        
    except Exception as e:
        st.error(f"Failed to initialize RAG chain: {e}. Ensure your Groq API key is valid.")
        # Clean up env var if setup fails
        if 'GROQ_API_KEY' in os.environ: 
            del os.environ['GROQ_API_KEY']
        return None

def create_industry_chart(df):
    """Creates a visual chart showing industry distribution of CHGF firms."""
    try:
        industry_counts = df['NACE_Industry'].value_counts().head(10)
        
        # Instead of using seaborn, use matplotlib directly to avoid warnings
        plt.figure(figsize=(10, 6))
        bars = plt.barh(y=industry_counts.index, width=industry_counts.values, color='teal')
        plt.title('Top 10 Industries with Consistent High-Growth Firms', fontsize=14)
        plt.xlabel('Number of Companies', fontsize=12)
        plt.tight_layout()
        
        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        return img
    except Exception as e:
        st.error(f"Failed to create industry chart: {e}")
        return None

def create_revenue_chart(df):
    """Creates a revenue distribution chart for CHGF firms."""
    try:
        # Ensure revenue is numeric
        df['Estimated_Revenue_mn'] = pd.to_numeric(df['Estimated_Revenue_mn'], errors='coerce')
        
        # Filter out very large outliers for better visualization
        revenue_data = df[df['Estimated_Revenue_mn'] < df['Estimated_Revenue_mn'].quantile(0.95)]
        
        plt.figure(figsize=(10, 6))
        # Use matplotlib instead of seaborn
        plt.hist(revenue_data['Estimated_Revenue_mn'], bins=15, color='teal', alpha=0.7)
        plt.title('Revenue Distribution of High-Growth Firms (EUR millions)', fontsize=14)
        plt.xlabel('Revenue (million EUR)', fontsize=12)
        plt.ylabel('Number of Companies', fontsize=12)
        plt.tight_layout()
        
        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        return img
    except Exception as e:
        st.error(f"Failed to create revenue chart: {e}")
        return None

def create_growth_chart(df):
    """Creates a chart showing growth rates of CHGF firms."""
    try:
        # Ensure growth is numeric
        df['Growth 2023'] = pd.to_numeric(df['Growth 2023'], errors='coerce')
        
        # Filter out extreme outliers for better visualization
        growth_data = df[df['Growth 2023'] < df['Growth 2023'].quantile(0.95)]
        
        plt.figure(figsize=(10, 6))
        # Use matplotlib instead of seaborn
        plt.hist(growth_data['Growth 2023'], bins=15, color='purple', alpha=0.7)
        plt.title('Growth Rate Distribution of High-Growth Firms (2023)', fontsize=14)
        plt.xlabel('Growth Rate (%)', fontsize=12)
        plt.ylabel('Number of Companies', fontsize=12)
        plt.tight_layout()
        
        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        return img
    except Exception as e:
        st.error(f"Failed to create growth chart: {e}")
        return None

def create_location_chart(df):
    """Creates a chart showing geographical distribution of CHGF firms."""
    try:
        location_counts = df['City'].value_counts().head(10)
        
        # Use matplotlib directly instead of seaborn to avoid warnings
        plt.figure(figsize=(10, 6))
        bars = plt.barh(y=location_counts.index, width=location_counts.values, color='purple')
        plt.title('Top 10 Locations of High-Growth Firms in Ireland', fontsize=14)
        plt.xlabel('Number of Companies', fontsize=12)
        plt.tight_layout()
        
        # Convert plot to image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        return img
    except Exception as e:
        st.error(f"Failed to create location chart: {e}")
        return None

def generate_docx_report(analysis_text, company_name, company_desc, competitors):
    """Generates a detailed Word document report with the analysis."""
    try:
        doc = DocxDocument()
        
        # Add header with date
        doc.add_heading('Competitor Analysis Report', level=0)
        doc.add_paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}")
        
        # Company Information Section
        doc.add_heading('Company Information', level=1)
        doc.add_paragraph(f"Company Name: {company_name}")
        doc.add_paragraph(f"Description: {company_desc}")
        
        # Competitor Analysis Section
        doc.add_heading('Competitor Analysis', level=1)
        doc.add_paragraph(analysis_text)
        
        # Source Documents Section
        if competitors:
            doc.add_heading('Detailed Competitor Information', level=1)
            for i, comp in enumerate(competitors, 1):
                if hasattr(comp, 'metadata'):
                    metadata = comp.metadata
                    doc.add_heading(f"{i}. {metadata.get('Company Name', 'Unknown Company')}", level=2)
                    
                    # Create a table for company details
                    table = doc.add_table(rows=1, cols=2)
                    table.style = 'Table Grid'
                    hdr_cells = table.rows[0].cells
                    hdr_cells[0].text = 'Attribute'
                    hdr_cells[1].text = 'Value'
                    
                    # Add key metadata to table - adjust keys to match actual Excel columns
                    key_fields = [
                        'NACE_Industry', 
                        'Business_Model', 
                        'Estimated_Revenue_mn', 
                        'Number of employees 2023', 
                        'Founded Year',
                        'Growth 2023',
                        'CEO',
                        'Public_or_Private',
                        'City'
                    ]
                    
                    for key in key_fields:
                        if key in metadata:
                            row_cells = table.add_row().cells
                            row_cells[0].text = key.replace('_', ' ')
                            row_cells[1].text = str(metadata[key])
                    
                    # Add description - adjust for the format in page_content
                    if "Description: " in comp.page_content:
                        doc.add_paragraph(comp.page_content.split("Description: ")[-1])
                    else:
                        doc.add_paragraph(comp.page_content)
                    doc.add_paragraph("")  # Add spacing between competitors
        
        # Recommendations Section
        doc.add_heading('Recommendations', level=1)
        doc.add_paragraph('Based on the competitor analysis, consider the following actions:')
        recommendations = [
            "Evaluate your unique value proposition against identified competitors",
            "Identify market gaps that your company could fill",
            "Consider potential collaboration opportunities with complementary businesses",
            "Monitor these competitors' marketing strategies and digital presence",
            "Analyze the growth strategies of these high-growth firms for insights",
            "Consider the business models that are succeeding in your industry segment"
        ]
        for rec in recommendations:
            p = doc.add_paragraph()
            p.add_run("• " + rec)
        
        # Executive summary
        doc.add_heading('Executive Summary', level=1)
        doc.add_paragraph("This report identifies potential competitors to your business from Ireland's database of consistent high-growth firms. These companies have demonstrated sustained growth and may represent both competitive threats and opportunities for benchmarking, partnership, or strategic insights.")
        
        # Save the document to a BytesIO object
        doc_bytes = BytesIO()
        doc.save(doc_bytes)
        doc_bytes.seek(0)
        
        return doc_bytes
    except Exception as e:
        st.error(f"Failed to generate DOCX report: {e}")
        return None

# --- Tab Functions ---

def render_dashboard_tab(df):
    """Renders the dashboard tab with visualizations."""
    st.header("📊 Market Landscape Dashboard")
    
    if df is None or len(df) == 0:
        st.error("Dashboard data is not available. Please check your data file.")
        return
    
    # Summary statistics
    st.subheader("Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total High-Growth Firms", len(df))
    with col2:
        avg_revenue = pd.to_numeric(df['Estimated_Revenue_mn'], errors='coerce').mean()
        st.metric("Average Revenue (EUR mn)", f"{avg_revenue:.2f}")
    with col3:
        industries = df['NACE_Industry'].nunique()
        st.metric("Industries Represented", industries)
    with col4:
        avg_growth = pd.to_numeric(df['Growth 2023'], errors='coerce').mean()
        st.metric("Average Growth 2023", f"{avg_growth:.1f}%")
    
    # Visualizations
    st.subheader("Market Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Industry Distribution", "Revenue Distribution", "Growth Distribution", "Geographic Distribution"])
    
    with tab1:
        industry_chart = create_industry_chart(df)
        if industry_chart:
            # Updated to use use_container_width instead of use_column_width
            st.image(industry_chart, caption="Industry Distribution", use_container_width=True)
        
    with tab2:
        revenue_chart = create_revenue_chart(df)
        if revenue_chart:
            # Updated to use use_container_width instead of use_column_width
            st.image(revenue_chart, caption="Revenue Distribution", use_container_width=True)
    
    with tab3:
        growth_chart = create_growth_chart(df)
        if growth_chart:
            # Updated to use use_container_width instead of use_column_width
            st.image(growth_chart, caption="Growth Rate Distribution", use_container_width=True)
            
    with tab4:
        location_chart = create_location_chart(df)
        if location_chart:
            # Updated to use use_container_width instead of use_column_width
            st.image(location_chart, caption="Geographic Distribution", use_container_width=True)
            
    # Business Model Distribution
    st.subheader("Business Model Distribution")
    try:
        if 'Business_Model' in df.columns:
            bm_counts = df['Business_Model'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            # Use direct matplotlib instead of pandas/seaborn plotting
            wedges, texts, autotexts = ax.pie(
                bm_counts, 
                labels=bm_counts.index,
                autopct='%1.1f%%',
                colors=plt.cm.tab10.colors
            )
            plt.title('Top Business Models', fontsize=14)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.tight_layout()
            
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            # Updated to use use_container_width instead of use_column_width
            st.image(img, caption="Business Model Distribution", use_container_width=True)
    except Exception as e:
        st.error(f"Could not create business model chart: {e}")

def render_analysis_tab(vector_store, groq_api_key, serper_api_key, model_name):
    """Renders the competitor analysis tab."""
    st.header("🔍 Competitor Analysis")
    
    if vector_store is None:
        st.error("Vector store initialization failed. Please check the Settings tab and try clearing the cache.")
        st.info("This may be due to a connection issue with model servers or a problem with the data file.")
        return
    
    # Company input section
    col1, col2 = st.columns([2, 1])
    with col1:
        user_company_name = st.text_input("Your Company Name:", key="analysis_company_name")
        user_description = st.text_area("Your Company Description:", height=150, key="analysis_company_desc")
    with col2:
        st.markdown("### Analysis Options")
        num_competitors = st.slider("Number of competitors to analyze:", min_value=3, max_value=10, value=5)
        analysis_depth = st.select_slider(
            "Analysis Depth:",
            options=["Brief", "Standard", "Detailed"],
            value="Standard"
        )
        enable_web_search = st.checkbox("Enable Web Search", value=False, disabled=(not serper_api_key))
        
    # Analysis parameters
    col1, col2 = st.columns(2)
    with col1:
        industry_filter = st.multiselect(
            "Filter by Industries (Optional):",
            options=sorted(pd.unique(df['NACE_Industry'].dropna())),
            default=None
        )
    with col2:
        business_model_filter = st.multiselect(
            "Filter by Business Model (Optional):",
            options=sorted(pd.unique(df['Business_Model'].dropna())),
            default=None
        )
    
    # Execute analysis button
    search_button = st.button("🚀 Find Competitors", type="primary", disabled=(not groq_api_key))
    
    # Handle search button click
    if search_button and user_company_name and user_description and groq_api_key:
        if not vector_store: 
            st.error("Vector store not available. Cannot perform search.")
        else:
            # Setup or reuse the RAG chain
            st.session_state.rag_chain = setup_rag_chain(vector_store, groq_api_key, model_name)
            if st.session_state.rag_chain:
                # Display user input block in chat
                user_query_display = f"**Finding competitors for:** {user_company_name}\n\n**Description:** {user_description}"
                st.session_state.messages.append({"role": "user", "content": user_query_display})
                with st.chat_message("user"): 
                    st.markdown(user_query_display)

                with st.spinner("Analyzing... Fetching data and generating response..."):
                    try:
                        web_context = ""
                        if enable_web_search and serper_api_key:
                            st.info("Performing web search...")
                            web_search_query = f"{user_company_name} {user_description} competitors Ireland"
                            web_context = serper_search(web_search_query, serper_api_key)
                            if "failed" in web_context or "not provided" in web_context or "No web search results" in web_context:
                                st.warning(f"Web search issue: {web_context}")
                                web_context = "Web search was enabled but failed or returned no results."
                            else: 
                                st.success("Web search completed.")
                        elif enable_web_search and not serper_api_key:
                            web_context = "Web search was enabled but the Serper API key is missing."

                        # Add filters to the query if selected
                        filter_str = ""
                        if industry_filter:
                            filter_str += f"Focus on companies in these industries: {', '.join(industry_filter)}. "
                        if business_model_filter:
                            filter_str += f"Focus on companies with these business models: {', '.join(business_model_filter)}. "

                        # Construct the structured question with depth parameter
                        depth_instructions = {
                            "Brief": "Provide a concise overview of the key competitors.",
                            "Standard": "Provide a balanced analysis with moderate detail.",
                            "Detailed": "Provide an in-depth analysis with extensive details on each competitor."
                        }

                        structured_question = (
                            f"Identify {num_competitors} potential competitors for the company '{user_company_name}' "
                            f"with description '{user_description}'. {filter_str}"
                            f"Use the database context primarily. "
                            f"Consider the following web search results if relevant: {web_context} "
                            f"List competitors from the database and explain the relevance based ONLY on the provided database context. "
                            f"{depth_instructions[analysis_depth]} "
                            f"Do not make up external information."
                        )

                        # Invoke the RAG chain
                        result = st.session_state.rag_chain.invoke({
                            "question": structured_question,
                            "chat_history": []
                        })
                        
                        response = result.get('answer', "Sorry, I couldn't generate an analysis.")
                        source_docs = result.get('source_documents', [])
                        st.session_state.last_response = response
                        st.session_state.last_source_docs = source_docs

                        # Display LLM response
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"): 
                            st.markdown(response)

                        # Offer Download Button only after a successful response
                        if st.session_state.last_response:
                            report_bytes = generate_docx_report(
                                st.session_state.last_response, 
                                user_company_name, 
                                user_description,
                                st.session_state.last_source_docs
                            )
                            
                            if report_bytes:
                                st.download_button(
                                    label="📝 Download Detailed Report (.docx)",
                                    data=report_bytes,
                                    file_name=f"{user_company_name}_competitor_analysis.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                )

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
                        error_msg = f"An error occurred: {e}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        with st.chat_message("assistant"): 
                            st.markdown(error_msg)
            else:
                st.error("RAG Chain could not be initialized. Please check configuration and Groq API key.")
    elif search_button and not (user_company_name and user_description):
        st.warning("Please enter both your company name and description.")
    elif search_button and not groq_api_key:
        st.error("Groq API Key is required. Please provide it via secrets, environment variable, or the sidebar.")

def render_chat_tab(vector_store, groq_api_key, model_name):
    """Renders the chat interface tab."""
    st.header("💬 Ask Questions About the Market")
    
    if vector_store is None:
        st.error("Vector store initialization failed. Please check the Settings tab and try clearing the cache.")
        st.info("This may be due to a connection issue with model servers or a problem with the data file.")
        return
    
    # Initialize the chain if not already done
    if "chat_rag_chain" not in st.session_state or st.session_state.get("current_model") != model_name:
        st.session_state.chat_rag_chain = setup_rag_chain(vector_store, groq_api_key, model_name)
        st.session_state.current_model = model_name
    
    # Chat interface
    if st.session_state.chat_rag_chain:
        # Display chat history
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        user_question = st.chat_input("Ask about Irish high-growth companies...", key="chat_input")
        
        if user_question:
            # Add user message to history
            st.session_state.chat_messages.append({"role": "user", "content": user_question})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Generate response
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chat_rag_chain.invoke({
                        "question": user_question,
                        "chat_history": [(msg["content"], response["content"]) 
                                        for msg, response in zip(
                                            [m for m in st.session_state.chat_messages if m["role"] == "user"][:-1],
                                            [m for m in st.session_state.chat_messages if m["role"] == "assistant"]
                                        )]
                    })
                    
                    response = result.get('answer', "I'm sorry, I couldn't generate a response.")
                    
                    # Add assistant response to history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(response)
                        
                except Exception as e:
                    st.error(f"An error occurred while processing your question: {e}")
    else:
        st.warning("Chat interface not available. Please ensure your Groq API key is valid.")

def render_settings_tab():
    """Renders the settings tab."""
    st.header("⚙️ Settings")
    
    # API Key settings
    st.subheader("API Keys")
    groq_api_key = st.text_input(
        "Groq API Key:", 
        value=st.session_state.get("groq_api_key", ""), 
        type="password",
        key="settings_groq_key"
    )
    serper_api_key = st.text_input(
        "Serper API Key (for web search):", 
        value=st.session_state.get("serper_api_key", ""), 
        type="password",
        key="settings_serper_key"
    )
    
    # Model selection
    st.subheader("Model Settings")
    selected_model_name = st.selectbox(
        "Select LLM Model:",
        options=list(AVAILABLE_MODELS.keys()),
        index=0,
        key="settings_model"
    )
    selected_model = AVAILABLE_MODELS[selected_model_name]
    
    # Save settings button
    if st.button("Save Settings", type="primary"):
        st.session_state.groq_api_key = groq_api_key
        st.session_state.serper_api_key = serper_api_key
        st.session_state.model_name = selected_model
        st.success("Settings saved successfully!")
        
        # Clear chain instances to force re-initialization with new settings
        if "rag_chain" in st.session_state:
            st.session_state.rag_chain = None
        if "chat_rag_chain" in st.session_state:
            st.session_state.chat_rag_chain = None
        
        # Clean cache button
    if st.button("Clear Cache", type="secondary"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared successfully!")
        st.warning("Please refresh the page to reload data and models.")

# --- Main App ---

st.set_page_config(
    page_title="Ireland Competitor Analysis Pro", 
    page_icon="🇮🇪", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved appearance
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='main-header'>🇮🇪 Ireland High-Growth Competitor Analysis Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Identify potential competitors from a curated database of Irish high-growth firms</p>", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "rag_chain" not in st.session_state: 
    st.session_state.rag_chain = None
if "last_response" not in st.session_state: 
    st.session_state.last_response = None
if "last_source_docs" not in st.session_state:
    st.session_state.last_source_docs = []
if "groq_api_key" not in st.session_state:
    # Check Streamlit secrets first
    try:
        st.session_state.groq_api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, 'secrets') else None
    except Exception:
        st.session_state.groq_api_key = None
    # Fallback to environment variables
    if not st.session_state.groq_api_key:
        st.session_state.groq_api_key = os.environ.get('GROQ_API_KEY', "")
        
if "serper_api_key" not in st.session_state:
    # Check Streamlit secrets first
    try:
        st.session_state.serper_api_key = st.secrets.get("SERPER_API_KEY") if hasattr(st, 'secrets') else None
    except Exception:
        st.session_state.serper_api_key = None
    # Fallback to environment variables
    if not st.session_state.serper_api_key:
        st.session_state.serper_api_key = os.environ.get('SERPER_API_KEY', "")
        
if "model_name" not in st.session_state:
    st.session_state.model_name = LLM_MODEL

# Load data and create/load vector store
documents, df = load_and_prepare_data(DATA_FILE)
if documents: 
    vector_store = create_or_load_vector_store(documents, FAISS_INDEX_PATH)
else: 
    vector_store = None
    df = None

# --- Main Navigation ---
tabs = st.tabs(["🏢 Analysis", "📊 Dashboard", "💬 Chat Assistant", "⚙️ Settings"])

with tabs[0]:  # Analysis Tab
    render_analysis_tab(
        vector_store, 
        st.session_state.groq_api_key, 
        st.session_state.serper_api_key, 
        st.session_state.model_name
    )

with tabs[1]:  # Dashboard Tab
    if df is not None:
        render_dashboard_tab(df)
    else:
        st.error("Data not available. Please check your data file.")

with tabs[2]:  # Chat Tab
    render_chat_tab(
        vector_store, 
        st.session_state.groq_api_key, 
        st.session_state.model_name
    )

with tabs[3]:  # Settings Tab
    render_settings_tab()

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Ireland High-Growth Competitor Analysis Pro | ©️ 2025</div>", unsafe_allow_html=True)