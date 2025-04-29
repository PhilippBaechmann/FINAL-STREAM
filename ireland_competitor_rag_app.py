# ireland_competitor_rag_app.py

import streamlit as st
import pandas as pd
import os
import requests
import numpy as np
from docx import Document as DocxDocument
# Required libraries: pip install streamlit pandas requests numpy python-docx langchain langchain-community langchain-groq faiss-cpu sentence-transformers openpyxl
# (Replace faiss-cpu with faiss-gpu if needed)
from langchain.schema import Document as LangchainDocument
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate # Although not directly used in ConversationalRetrievalChain kwargs here, good practice to import if needed
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Configuration ---
# *** Updated DATA_FILE constant based on user feedback ***
DATA_FILE = 'ireland_cleaned_CHGF.xlsx'

CHGF_COL = 'ConsistentHighGrowthFirm 2023'
DESC_COL = 'Description'
NAME_COL = 'Company Name'
# Add other relevant cols to include in Document metadata if needed
METADATA_COLS = ['Company Name', 'NACE_Industry', 'Topic', 'City', 'Business_Model', 'Estimated_Revenue_mn']
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3-8b-8192" # Or choose another model like "mixtral-8x7b-32768", "gemma-7b-it"
FAISS_INDEX_PATH = "faiss_ireland_chgf_index" # Path to save/load the index

# --- Helper Functions ---

def serper_search(query: str, api_key: str, k: int = 5):
    """Performs a web search using the Serper API."""
    if not api_key: return "Serper API key not provided."
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": k}
    headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        results = response.json().get('organic', [])
        # Format results: Take title and snippet
        formatted_results = [f"Title: {r.get('title', 'N/A')}\nSnippet: {r.get('snippet', 'N/A')}" for r in results]
        return "\n\n".join(formatted_results) if formatted_results else "No web search results found."
    except requests.exceptions.RequestException as e: return f"Web search failed: {e}"
    except Exception as e: return f"An error occurred during web search: {e}"

@st.cache_data(show_spinner="Loading and processing data...")
def load_and_prepare_data(file_path):
    """Loads data from Excel, filters CHGF, creates Langchain Documents."""
    try:
        # *** Use pd.read_excel as established ***
        # Requires 'openpyxl': pip install openpyxl
        df = pd.read_excel(file_path, engine='openpyxl')

        # --- Filtering ---
        df[CHGF_COL] = pd.to_numeric(df[CHGF_COL], errors='coerce')
        df_chgf = df.dropna(subset=[CHGF_COL]).copy()
        df_chgf[CHGF_COL] = df_chgf[CHGF_COL].astype(int)
        df_chgf = df_chgf[df_chgf[CHGF_COL] == 1].copy()

        if df_chgf.empty:
            st.error(f"No companies found with '{CHGF_COL}' = 1 in the data.")
            return None

        df_chgf[DESC_COL] = df_chgf[DESC_COL].fillna('No description available')

        # --- Create Langchain Documents ---
        documents = []
        for _, row in df_chgf.iterrows():
            metadata = {}
            for col in METADATA_COLS:
                 if col in df_chgf.columns:
                      val = row[col]
                      # Ensure data types are JSON serializable for metadata
                      if pd.isna(val): metadata[col] = "N/A"
                      elif isinstance(val, (np.int64, np.float64)): metadata[col] = val.item() # Convert numpy types
                      else: metadata[col] = str(val) # Convert others to string
            page_content = str(row[DESC_COL]) if pd.notna(row[DESC_COL]) else 'No description available'
            doc = LangchainDocument(page_content=page_content, metadata=metadata)
            documents.append(doc)
        if not documents:
             st.error("Failed to create documents from the data.")
             return None
        st.success(f"Loaded and prepared {len(documents)} consistent high-growth firms from Excel file.")
        return documents
    except FileNotFoundError:
        st.error(f"Error: Data file '{file_path}' not found. Please ensure it's in the correct directory.")
        return None
    except ImportError:
         st.error("Error: The 'openpyxl' library is required to read Excel files. Please install it: pip install openpyxl")
         return None
    except KeyError as e:
        st.error(f"Error: Column '{e}' not found in the data file. Please check column names in the Excel sheet.")
        return None
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return None

@st.cache_resource(show_spinner="Initializing vector store...")
def create_or_load_vector_store(_documents, index_path):
    """Creates FAISS vector store from documents or loads if exists."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if os.path.exists(index_path) and os.path.isdir(index_path):
        try:
            # Allow dangerous deserialization if loading FAISS index created with older pickle protocol versions
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

def setup_rag_chain(vectorstore, api_key):
    """Sets up the ConversationalRetrievalChain."""
    if not vectorstore or not api_key: return None
    try:
        # Set environment variable for Groq SDK (it typically reads from here)
        os.environ['GROQ_API_KEY'] = api_key
        llm = ChatGroq(temperature=0, model_name=LLM_MODEL)
        # Memory to store chat history - output_key ensures 'answer' is used for memory population
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 most similar documents
            memory=memory,
            return_source_documents=True, # Return the documents used to generate the answer
            verbose=False # Set to True for detailed Langchain logs
        )
        return chain
    except Exception as e:
        st.error(f"Failed to initialize RAG chain: {e}. Ensure your Groq API key is valid.")
        # Clean up env var if setup fails
        if 'GROQ_API_KEY' in os.environ: del os.environ['GROQ_API_KEY']
        return None

def generate_docx_report(analysis_text: str, file_path: str):
    """Generates a Word document with the analysis."""
    try:
        doc = DocxDocument()
        doc.add_heading('Competitor Analysis Report', level=1)
        doc.add_paragraph(analysis_text)
        doc.save(file_path)
        return True
    except Exception as e:
        st.error(f"Failed to generate DOCX report: {e}")
        return False

# --- Streamlit App ---

st.set_page_config(page_title="Ireland Competitor Analysis", page_icon="🇮🇪", layout="wide")
st.title("🇮🇪 High-Growth Firm Competitor Analysis (Ireland)")
st.markdown("Enter your company details and get potential competitors from a dataset of Irish high-growth firms.")

# --- Sidebar for Inputs & Controls ---
with st.sidebar:
    st.header("🔑 API Keys")
    st.markdown("""
    Provide API keys below.
    **Priority Order:**
    1. Streamlit Secrets (for deployed apps)
    2. Environment Variables
    3. Input fields below (least secure)
    """)
    # --- API Key Handling with Priority ---
    try: # Check Streamlit secrets first
        groq_api_key_st = st.secrets.get("GROQ_API_KEY") if hasattr(st, 'secrets') else None
        serper_api_key_st = st.secrets.get("SERPER_API_KEY") if hasattr(st, 'secrets') else None
    except Exception: groq_api_key_st = None; serper_api_key_st = None
    # Fallback to environment variables
    groq_api_key_env = os.environ.get('GROQ_API_KEY')
    serper_api_key_env = os.environ.get('SERPER_API_KEY')
    # Determine default values for input fields based on secrets/env vars
    default_groq = groq_api_key_st or groq_api_key_env or ""
    default_serper = serper_api_key_st or serper_api_key_env or ""
    # Use sidebar input as the final fallback
    groq_api_key_input = st.text_input("Groq API Key:", type="password", value=default_groq, placeholder="Required (starts with gsk_...)")
    serper_api_key_input = st.text_input("Serper API Key (Optional):", type="password", value=default_serper, placeholder="Optional for web search")
    # Final keys to use
    groq_api_key = default_groq or groq_api_key_input
    serper_api_key = default_serper or serper_api_key_input
    # --- End API Key Handling ---

    enable_web_search = st.checkbox("Enable Web Search (requires Serper Key)", value=False, disabled=(not serper_api_key))
    if not groq_api_key: st.warning("Groq API Key is required.")

    st.header("🏢 Your Company Details")
    user_company_name = st.text_input("Your Company Name:")
    user_description = st.text_area("Your Company Description:", height=150)
    search_button = st.button("🚀 Find Competitors", type="primary", disabled=(not groq_api_key))

    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.rag_chain = None
        st.session_state.last_response = None
        st.rerun() # Use st.rerun for modern Streamlit versions

# --- Main Area ---

# Load data and create/load vector store
documents = load_and_prepare_data(DATA_FILE)
if documents: vector_store = create_or_load_vector_store(documents, FAISS_INDEX_PATH)
else: vector_store = None # Ensure vector_store is None if data loading failed

# Initialize session state variables if they don't exist
if "messages" not in st.session_state: st.session_state.messages = []
if "rag_chain" not in st.session_state: st.session_state.rag_chain = None
if "last_response" not in st.session_state: st.session_state.last_response = None

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle search button click: requires company name, desc, Groq key, and vector store
if search_button and user_company_name and user_description and groq_api_key:
    if not vector_store: st.error("Vector store not available. Cannot perform search.")
    else:
        # Setup or reuse the RAG chain (re-initializes if API key changed, safe otherwise)
        st.session_state.rag_chain = setup_rag_chain(vector_store, groq_api_key)
        if st.session_state.rag_chain:
            # Display user input block in chat
            user_query_display = f"**Finding competitors for:** {user_company_name}\n\n**Description:** {user_description}"
            st.session_state.messages.append({"role": "user", "content": user_query_display})
            with st.chat_message("user"): st.markdown(user_query_display)

            with st.spinner("Analyzing... Fetching data and generating response..."):
                try:
                    web_context = "" # Initialize web context
                    if enable_web_search and serper_api_key:
                        st.info("Performing web search...")
                        web_search_query = f"{user_company_name} {user_description} competitors Ireland"
                        web_context = serper_search(web_search_query, serper_api_key)
                        if "failed" in web_context or "not provided" in web_context or "No web search results" in web_context:
                             st.warning(f"Web search issue: {web_context}")
                             web_context = "Web search was enabled but failed or returned no results." # Standardize message
                        else: st.success("Web search completed.")
                    elif enable_web_search and not serper_api_key:
                         web_context = "Web search was enabled but the Serper API key is missing."

                    # Construct the structured question for the LLM
                    structured_question = (
                        f"Identify potential competitors for the company '{user_company_name}' "
                        f"with description '{user_description}'. Use the database context primarily. "
                        f"Consider the following web search results if relevant: {web_context} "
                        f"List 3-5 competitors from the database and explain the relevance based ONLY on the provided database context (description, industry, topic, revenue, etc.). Do not make up external information."
                    )

                    # Invoke the RAG chain. Pass empty chat history for a focused analysis each time.
                    result = st.session_state.rag_chain.invoke({
                        "question": structured_question,
                        "chat_history": [] # Ensures analysis is based on current input, not past turns
                        })
                    response = result.get('answer', "Sorry, I couldn't generate an analysis.")
                    st.session_state.last_response = response # Save response for potential download

                    # Display LLM response
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"): st.markdown(response)

                    # Offer Download Button only after a successful response
                    if st.session_state.last_response:
                         report_file_path = "competitor_analysis_report.docx"
                         if generate_docx_report(st.session_state.last_response, report_file_path):
                              try:
                                   with open(report_file_path, "rb") as f:
                                        st.download_button(
                                             label="📝 Download Report (.docx)",
                                             data=f,
                                             file_name=report_file_path,
                                             mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                             key="download_report_button" # Add a key for potential state management
                                        )
                              except FileNotFoundError: st.error("Report file could not be found for download.")
                              # Consider removing the local file after offering download if needed
                              # finally:
                              #      if os.path.exists(report_file_path): os.remove(report_file_path)

                except Exception as e: # Catch potential errors during RAG chain execution
                    st.error(f"An error occurred during analysis: {e}")
                    error_msg = f"An error occurred: {e}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    with st.chat_message("assistant"): st.markdown(error_msg)
        else: # Handle case where RAG chain setup failed
            st.error("RAG Chain could not be initialized. Please check configuration and Groq API key.")

# Handle cases where search button is clicked but inputs are missing
elif search_button and not (user_company_name and user_description):
    st.warning("Please enter both your company name and description.")
elif search_button and not groq_api_key:
     st.error("Groq API Key is required. Please provide it via secrets, environment variable, or the sidebar.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Competitor Analysis Assistant - Ireland</div>", unsafe_allow_html=True)