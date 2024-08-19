import streamlit as st
import logging
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OpenAIEmbeddings

from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_data_files  # Corrected import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="LangChain & Streamlit RAG")
st.title("LangChain & Streamlit RAG")

def show_ui(qa, prompt_to_user="How may I help you?"):
    """
    Displays the Streamlit chat UI and handles user interactions.

    Args:
        qa: The LangChain chain for question answering.
        prompt_to_user: The initial prompt to display to the user.
    """
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": prompt_to_user}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = ask_question(qa, prompt)
                    st.markdown(response.content)
                except Exception as e:
                    logging.error(f"Error during question answering: {e}")
                    st.write("Sorry, there was an error processing your request.")
        message = {"role": "assistant", "content": response.content if response else "Error"}
        st.session_state.messages.append(message)

@st.cache_resource
def get_retriever(openai_api_key=None):
    """
    Creates and caches the document retriever.

    Args:
        openai_api_key: The OpenAI API key.

    Returns:
        An ensemble document retriever.
    """
    try:
        docs = load_data_files(data_dir="data")  # Add data_dir here 
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
        return ensemble_retriever_from_docs(docs, embeddings=embeddings)
    except Exception as e:
        logging.error(f"Error creating retriever: {e}")
        st.error("Error initializing the application. Please check the logs.")
        st.stop()  # Stop execution if retriever creation fails

def get_chain(openai_api_key=None, huggingfacehub_api_token=None):
    """
    Creates the question answering chain.

    Args:
        openai_api_key: The OpenAI API key.
        huggingfacehub_api_token: The Hugging Face Hub API token.

    Returns:
        A LangChain question answering chain.
    """
    try:
        ensemble_retriever = get_retriever(openai_api_key=openai_api_key)
        chain = create_full_chain(
            ensemble_retriever,
            openai_api_key=openai_api_key,
            chat_memory=StreamlitChatMessageHistory(key="langchain_messages")
        )
        return chain
    except Exception as e:
        logging.error(f"Error creating chain: {e}")
        st.error("Error initializing the application. Please check the logs.")
        st.stop()  # Stop execution if chain creation fails

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
            st.markdown(f"[Get an {secret_name}]({info_link})")
    return secret_value

def run():
    """
    Main function to run the Streamlit application.
    """
    ready = True
    openai_api_key = st.session_state.get("OPENAI_API_KEY")
    huggingfacehub_api_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")

    with st.sidebar:
        if not openai_api_key:
            openai_api_key = get_secret_or_input(
                'OPENAI_API_KEY',
                "OpenAI API key",
                info_link="https://platform.openai.com/account/api-keys"
            )
        if not huggingfacehub_api_token:
            huggingfacehub_api_token = get_secret_or_input(
                'HUGGINGFACEHUB_API_TOKEN',
                "HuggingFace Hub API Token",
                info_link="https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication"
            )

    if not openai_api_key:
        st.warning("Missing OPENAI_API_KEY")
        ready = False
    if not huggingfacehub_api_token:
        st.warning("Missing HUGGINGFACEHUB_API_TOKEN")
        ready = False

    if ready:
        try:
            chain = get_chain(
                openai_api_key=openai_api_key,
                huggingfacehub_api_token=huggingfacehub_api_token
            )
            st.subheader("Ask questions about Equity Bank's products and services:")
            show_ui(chain, "How can I assist you today?")
        except Exception as e:
            logging.error(f"Error initializing application: {e}")
            st.error("Error initializing the application. Please check the logs.")
    else:
        st.stop()

run()