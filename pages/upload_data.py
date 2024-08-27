import streamlit as st
import logging
import os
from langchain_community.retrievers import BM25Retriever
from pathlib import Path
from local_loader import load_file
from full_chain import create_full_chain

def process_uploaded_file(uploaded_file, chain, ensemble_retriever, openai_api_key=None):
    """
    Processes the uploaded file and adds it to the vector database.

    Args:
        uploaded_file: The uploaded file object from Streamlit.
        openai_api_key: The OpenAI API key for embedding generation.
    """
    # try:
    if uploaded_file is not None:
        logging.info(f'run upload {uploaded_file}')
        # Get the file path from the uploaded file object
        file_path = os.path.join("new_data", uploaded_file.name) 

        # Save the uploaded file to the 'data' directory
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the document using the saved file path
        docs = load_file(Path(file_path))

        all_docs = ensemble_retriever.retrievers[0].docs
        all_docs.extend(docs)

        ensemble_retriever.retrievers[1].add_documents(docs)

        new_bm25 = BM25Retriever.from_texts([t.page_content for t in all_docs])

        ensemble_retriever.retrievers[0] = new_bm25

        chain = create_full_chain(
            ensemble_retriever,
            openai_api_key=openai_api_key,
        )

        logging.info("File uploaded and added to the knowledge base!")
        st.success("File uploaded and added to the knowledge base!")
        logging.info("File uploaded and added to the knowledge base!")

    return ensemble_retriever, chain, None
        
    # except Exception as e:
    #     logging.error(f"Error processing uploaded file: {e}")
    #     st.error("Error processing the file. Please check the logs.")


openai_api_key = os.getenv('OPENAI_API_KEY')

if not st.session_state.get('init', False):
    st.switch_page('streamlit_app.py')

# File Uploader
with st.form("my-form", clear_on_submit=True):
    st.subheader("Upload a Document to the Knowledge Base:")
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["txt", "pdf", "csv", "xls", "xlsx", "json"]
    )
    chain = st.session_state['chain']
    ensemble_retriever = st.session_state['ensemble_retriever']
    st.session_state['ensemble_retriever'], st.session_state['chain'], uploaded_file = process_uploaded_file(uploaded_file, chain, ensemble_retriever, openai_api_key)
    submitted = st.form_submit_button("submit")