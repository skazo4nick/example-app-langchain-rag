import os
import sys
import streamlit as st
import logging
from dotenv import load_dotenv
from pathlib import Path
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.chains import RetrievalQA
from basic_chain import get_model

# Override sqlite3 before importing langchain_chroma
if 'pysqlite3' in sys.modules:
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from langchain_chroma import Chroma # Import Chroma from langchain_chroma

from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_data_files, load_file
from vector_store import EmbeddingProxy
from memory import clean_session_history, create_memory_chain  # Ensure this import is added
from filter import create_retriever  # Import create_retriever from filter.py

# Load environment variables from .env file
load_dotenv()

# Get the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

openai_api_key = OPENAI_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Equity Bank's Assistant")

def show_ui(qa, prompt_to_user="How may I help you?"):
    """
    Displays the Streamlit chat UI and handles user interactions.

    Args:
        qa: The LangChain chain for question answering.
        prompt_to_user: The initial prompt to display to the user.
    """
    logging.info(f"show_ui ru: {prompt_to_user}")
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Say something")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    response = None  # Initialize response to None
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ask_question(qa, st.session_state.messages[-1]["content"])
        message_content = response["content"] if response and isinstance(response, dict) and "content" in response else response if response else "Error"
        message = {"role": "assistant", "content": message_content}
        st.session_state.messages.append(message)

@st.cache_resource
def get_retriever(openai_api_key=openai_api_key, data_dir="data"):
    """
    Creates and caches the document retriever.

    Args:
        openai_api_key: The OpenAI API key.
        data_dir: The directory where data files are located.

    Returns:
        An ensemble document retriever.
    """
    try:
        docs = load_data_files(data_dir=data_dir)
        retriever = create_retriever(docs, openai_api_key=openai_api_key)
        return retriever
    except Exception as e:
        logging.error(f"Error creating retriever: {e}")
        st.error("Error initializing the retriever. Please check the logs.")
        st.stop()

def get_chain(openai_api_key=openai_api_key, data_dir="data"):
    """
    Creates the question answering chain.

    Args:
        openai_api_key: The OpenAI API key.
        data_dir: The directory where data files are located.

    Returns:
        A LangChain question answering chain.
    """
    try:
        logging.info('Start creating chain')
        retriever = get_retriever(openai_api_key=openai_api_key, data_dir=data_dir)
        llm = get_model(openai_api_key=openai_api_key)
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
        logging.info('Chain creating complete')
        return chain
    except Exception as e:
        logging.error(f"Error creating chain: {e}")
        logging.exception("message")
        st.error("Error initializing the application. Please check the logs.")
        st.stop()

def get_secret_or_input(secret_key, secret_name, info_link=None):
    """
    Retrieves a secret from Streamlit secrets or prompts the user for input.

    Args:
        secret_key: The key of the secret in Streamlit secrets.
        secret_name: The user-friendly name of the secret.
        info_link: An optional link to provide information about the secret.

    Returns:
        The secret value.
    """
    if secret_key in st.secrets:
        st.write("Found %s secret" % secret_key)
        secret_value = st.secrets[secret_key]
    else:
        st.write(f"Please provide your {secret_name}")
        secret_value = st.text_input(secret_name, key=f"input_{secret_key}", type="password")
        if secret_value:
            st.session_state[secret_key] = secret_value
        if info_link:
            st.markdown(f"[More info]({info_link})")
    return secret_value

def reset(prompt_to_user="How may I help you?"):
    session_id = get_script_run_ctx().session_id
    clean_session_history(session_id)
    st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

def run():
    """
    Main function to run the Streamlit application.
    """
    ready = True
    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")
    data_dir = "data"  # Set the correct path to your data directory

    with st.sidebar:
        if not openai_api_key:
            openai_api_key = get_secret_or_input("OPENAI_API_KEY", "OpenAI API Key", "https://platform.openai.com/account/api-keys")
        if not huggingfacehub_api_token:
            huggingfacehub_api_token = get_secret_or_input("HUGGINGFACEHUB_API_TOKEN", "Hugging Face Hub API Token", "https://huggingface.co/settings/tokens")

        if not openai_api_key:
            st.error("OpenAI API Key is required.")
            ready = False
        if not huggingfacehub_api_token:
            st.error("Hugging Face Hub API Token is required.")
            ready = False

    if ready:
        logging.info('run loop')

        if not st.session_state.get('init', False):
            st.session_state['init'] = True
            reset()

        chain = get_chain(openai_api_key=openai_api_key, data_dir=data_dir)
        show_ui(chain)
    else:
        st.error("Please provide the required API keys to proceed.")

if __name__ == "__main__":
    run()