import streamlit as st
import pandas as pd
import os
import requests
import numpy as np
from docx import Document as DocxDocument
from io import BytesIO
from datetime import datetime

# Required libraries: pip install streamlit pandas requests numpy python-docx langchain langchain-community langchain-groq faiss-cpu openpyxl

from langchain.schema import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# Updated import path for ChatMessageHistory to avoid deprecation warning
from langchain_community.chat_message_histories import ChatMessageHistory

# --- Configuration ---
DATA_FILE = 'ireland_cleaned_CHGF.xlsx'
CHGF_COL = 'ConsistentHighGrowthFirm 2023'
DESC_COL = 'Description'
NAME_COL = 'Company Name'

# Metadata columns from your Excel file
METADATA_COLS = [
    'Company Name', 'City', 'NACE_Industry', 'Topic', 'Business_Model', 
    'Estimated_Revenue_mn', 'Founded Year', 'Number of employees 2023',
    'Growth 2023', 'aagr 2023', 'Public_or_Private', 'CEO', 'Description'
]

# Use simple embeddings to avoid external dependencies
USE_SIMPLE_EMBEDDINGS = True
FAISS_INDEX_PATH = "faiss_ireland_chgf_index"
LLM_MODEL = "llama3-8b-8192"  # Default model

# --- Simple Embeddings Class (works offline) ---
class SimpleEmbeddings:
    """A simple embedding class that doesn't require external models."""
    
    def __init__(self):
        """Initialize the simple embeddings class."""
        pass
        
    def embed_documents(self, texts):
        """Create simple embeddings for a list of texts using TF-IDF like approach."""
        import re
        import numpy as np
        from collections import Counter
        
        # Process all texts to build vocabulary
        all_words = []
        processed_texts = []
        
        for text in texts:
            # Convert to lowercase and split into words
            words = re.findall(r'\w+', text.lower())
            processed_texts.append(words)
            all_words.extend(words)
            
        # Build vocabulary (unique words)
        vocabulary = list(set(all_words))
        vocab_size = len(vocabulary)
        word_to_idx = {word: i for i, word in enumerate(vocabulary)}
        
        # Create embeddings
        embeddings = []
        for words in processed_texts:
            # Count word frequencies
            word_counts = Counter(words)
            
            # Create vector
            vector = np.zeros(vocab_size, dtype=np.float32)
            for word, count in word_counts.items():
                if word in word_to_idx:
                    vector[word_to_idx[word]] = count
                    
            # Normalize
            if np.sum(vector) > 0:
                vector = vector / np.sqrt(np.sum(vector**2))
                
            embeddings.append(vector)
            
        return embeddings
        
    def embed_query(self, text):
        """Embed a single query text."""
        return self.embed_documents([text])[0]

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
        
        formatted_results = []
        for r in results:
            title = r.get('title', 'N/A')
            snippet = r.get('snippet', 'N/A')
            url = r.get('link', 'N/A')
            formatted_results.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")
            
        return "\n\n".join(formatted_results) if formatted_results else "No web search results found."
    except Exception as e: 
        return f"Web search failed: {e}"

@st.cache_data(show_spinner="Loading and processing data...")
def load_and_prepare_data(file_path):
    """Loads data from Excel, filters CHGF, creates Langchain Documents."""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Fill NaN values for better processing
        for col in METADATA_COLS:
            if col in df.columns:
                df[col] = df[col].fillna('Not Available')

        # --- Filtering for high-growth firms ---
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
                        metadata[col] = val.item()
                    else: 
                        metadata[col] = str(val)
                        
            page_content = f"Company: {row[NAME_COL]}\nDescription: {row[DESC_COL]}"
            doc = LangchainDocument(page_content=page_content, metadata=metadata)
            documents.append(doc)
            
        if not documents:
            st.error("Failed to create documents from the data.")
            return None, None
            
        st.success(f"Loaded and prepared {len(documents)} consistent high-growth firms from Excel file.")
        return documents, df_chgf
        
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None, None

@st.cache_resource(show_spinner="Initializing vector store...")
def create_or_load_vector_store(_documents, index_path):
    """Creates FAISS vector store from documents or loads if exists."""
    try:
        # Choose embedding method - simple embeddings for offline use
        embeddings = SimpleEmbeddings()
        st.info("Using simple embeddings for semantic search (no external models required)")
    
        if os.path.exists(index_path) and os.path.isdir(index_path):
            try:
                # Using allow_dangerous_deserialization for compatibility
                vectorstore = FAISS.load_local(
                    index_path, 
                    embeddings=embeddings,  # Explicitly name the parameter
                    allow_dangerous_deserialization=True
                )
                st.info(f"Loaded existing FAISS index from {index_path}")
                return vectorstore
            except Exception as e:
                st.warning(f"Failed to load existing index ({e}). Rebuilding index.")
                
        if not _documents:
            st.error("No documents available to build vector store.")
            return None
            
        try:
            # Create new vector store with explicit embeddings parameter
            vectorstore = FAISS.from_documents(
                documents=_documents,
                embedding=embeddings  # Use embedding parameter name
            )
            vectorstore.save_local(index_path)
            st.success(f"Created and saved FAISS index to {index_path}")
            return vectorstore
        except Exception as e:
            st.error(f"Failed to create vector store: {e}")
            return None
    except Exception as e:
        st.error(f"Unexpected error in vector store creation: {e}")
        return None

def setup_rag_chain(vectorstore, api_key):
    """Sets up the ConversationalRetrievalChain."""
    if not vectorstore or not api_key: 
        return None
        
    try:
        # Set environment variable for Groq SDK
        os.environ['GROQ_API_KEY'] = api_key
        
        # Initialize LLM
        llm = ChatGroq(temperature=0.2, model_name=LLM_MODEL)
        
        # Updated memory initialization to address deprecation warning
        # Using the newer approach with correct import
        
        # Create a memory instance using the new pattern
        memory = ConversationBufferMemory(
            chat_memory=ChatMessageHistory(),
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        # Create the chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        
        return chain
        
    except Exception as e:
        st.error(f"Failed to initialize RAG chain: {e}. Ensure your Groq API key is valid.")
        # Clean up env var if setup fails
        if 'GROQ_API_KEY' in os.environ: 
            del os.environ['GROQ_API_KEY']
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
                        'City'
                    ]
                    
                    for key in key_fields:
                        if key in metadata:
                            row_cells = table.add_row().cells
                            row_cells[0].text = key.replace('_', ' ')
                            row_cells[1].text = str(metadata[key])
                    
                    # Add description
                    if "Description: " in comp.page_content:
                        doc.add_paragraph(comp.page_content.split("Description: ")[-1])
                    else:
                        doc.add_paragraph(comp.page_content)
                    doc.add_paragraph("")  # Add spacing between competitors
        
        # Save the document to a BytesIO object
        doc_bytes = BytesIO()
        doc.save(doc_bytes)
        doc_bytes.seek(0)
        
        return doc_bytes
    except Exception as e:
        st.error(f"Failed to generate DOCX report: {e}")
        return None

# --- Main App ---

st.set_page_config(
    page_title="High-Growth Competitors Analyzer", 
    page_icon="🔍", 
    layout="wide"
)

# Custom CSS for improved appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1a5276;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #566573;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.markdown("<h1 class='main-header'>High-Growth Competitor Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Find potential competitors from Ireland's consistent high-growth firms</p>", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state: 
    st.session_state.messages = []
if "rag_chain" not in st.session_state: 
    st.session_state.rag_chain = None
if "last_response" not in st.session_state: 
    st.session_state.last_response = None
if "last_source_docs" not in st.session_state:
    st.session_state.last_source_docs = []

# Sidebar for API Keys
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1. Enter your Groq API key below
    2. Fill in your company name and description
    3. Adjust any optional settings
    4. Click 'Find Competitors' to get results
    5. Download a detailed report if needed
    """)
    
    st.header("🔑 API Keys")
    
    # API key handling
    groq_api_key = st.text_input(
        "Groq API Key:", 
        type="password",
        help="Required. Get one at console.groq.com"
    )
    serper_api_key = st.text_input(
        "Serper API Key (optional):", 
        type="password",
        help="Optional. For web search enhancement."
    )
    enable_web_search = st.checkbox("Enable Web Search", value=False, disabled=(not serper_api_key))
    
    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.rag_chain = None
        st.session_state.last_response = None
        st.session_state.last_source_docs = []
        st.rerun()

# Load data and create vector store
documents, df = load_and_prepare_data(DATA_FILE)
if documents: 
    vector_store = create_or_load_vector_store(documents, FAISS_INDEX_PATH)
else: 
    vector_store = None
    df = None

# Main content - Company input section
col1, col2 = st.columns([2, 1])
with col1:
    user_company_name = st.text_input("Your Company Name:", key="company_name")
    user_description = st.text_area("Your Company Description:", height=150, key="company_desc")
with col2:
    st.subheader("Analysis Options")
    num_competitors = st.slider("Number of competitors:", 3, 10, 5)
    
    analysis_depth = st.select_slider(
        "Analysis Depth:",
        options=["Brief", "Standard", "Detailed"],
        value="Standard"
    )

# Optional filters
with st.expander("Advanced Filters (Optional)"):
    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            industry_filter = st.multiselect(
                "Filter by Industry:",
                options=sorted(pd.unique(df['NACE_Industry'].dropna())),
                default=None
            )
        with col2:
            business_model_filter = st.multiselect(
                "Filter by Business Model:",
                options=sorted(pd.unique(df['Business_Model'].dropna())),
                default=None
            )
    else:
        st.warning("Data not loaded. Filters unavailable.")

# Execute analysis button
search_button = st.button("🔍 Find Competitors", type="primary", disabled=(not groq_api_key or not user_company_name))

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle search button click
if search_button and user_company_name and user_description and groq_api_key:
    if not vector_store: 
        st.error("Vector store not available. Please check your data file.")
    else:
        # Setup or reuse the RAG chain
        st.session_state.rag_chain = setup_rag_chain(vector_store, groq_api_key)
        if st.session_state.rag_chain:
            # Display user input block in chat
            user_query_display = f"**Finding competitors for:** {user_company_name}\n\n**Description:** {user_description}"
            st.session_state.messages.append({"role": "user", "content": user_query_display})
            with st.chat_message("user"): 
                st.markdown(user_query_display)

            with st.spinner("Analyzing... This may take a moment..."):
                try:
                    web_context = ""
                    if enable_web_search and serper_api_key:
                        with st.status("Searching web for additional context..."):
                            web_search_query = f"{user_company_name} {user_description} competitors Ireland"
                            web_context = serper_search(web_search_query, serper_api_key)
                            if "failed" in web_context or "not provided" in web_context or "No web search results" in web_context:
                                st.warning(f"Web search issue: {web_context}")
                                web_context = "Web search was enabled but failed or returned no results."

                    # Add filters to the query if selected
                    filter_str = ""
                    if 'industry_filter' in locals() and industry_filter:
                        filter_str += f"Focus on companies in these industries: {', '.join(industry_filter)}. "
                    if 'business_model_filter' in locals() and business_model_filter:
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
                        f"List competitors from the database and explain the relevance based on the provided context. "
                        f"{depth_instructions[analysis_depth]} "
                        f"Focus on how these companies compete with or complement the user's business."
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

                    # Offer download button for report
                    if st.session_state.last_response:
                        report_bytes = generate_docx_report(
                            st.session_state.last_response, 
                            user_company_name, 
                            user_description,
                            st.session_state.last_source_docs
                        )
                        
                        if report_bytes:
                            st.download_button(
                                label="📄 Download Detailed Report (.docx)",
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
            st.error("RAG Chain could not be initialized. Please check your Groq API key.")
elif search_button and not (user_company_name and user_description):
    st.warning("Please enter both your company name and description.")
elif search_button and not groq_api_key:
    st.error("Groq API Key is required.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>High-Growth Competitor Analyzer | Powered by RAG technology</div>", unsafe_allow_html=True)