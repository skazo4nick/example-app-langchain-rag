import os
import streamlit as st
import logging
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

from ensemble import ensemble_retriever_from_docs
from full_chain import create_full_chain, ask_question
from local_loader import load_data_files, load_file
from vector_store import EmbeddingProxy 
from memory import clean_session_history
from pathlib import Path

import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def show_ui(message, history, request: gr.Request):
    """
    Displays the Streamlit chat UI and handles user interactions.

    Args:
        qa: The LangChain chain for question answering.
        prompt_to_user: The initial prompt to display to the user.
    """
    global chain
    session_id = request.session_hash
    response = ask_question(chain, message, session_id)
    # logging.info(f"Response: {response}")
    return response.content


def get_retriever(openai_api_key=None):
    """
    Creates and caches the document retriever.

    Args:
        openai_api_key: The OpenAI API key.

    Returns:
        An ensemble document retriever.
    """
    try:
        docs = load_data_files(data_dir="data")  
        # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
        embeddings = HuggingFaceEmbeddings()
        return ensemble_retriever_from_docs(docs, embeddings=embeddings)
    except Exception as e:
        logging.error(f"Error creating retriever: {e}")
        logging.exception(f"message")
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
        )
        return ensemble_retriever, chain
    except Exception as e:
        logging.error(f"Error creating chain: {e}")
        logging.exception(f"message")
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

def process_uploaded_file(uploaded_file):
    """
    Processes the uploaded file and adds it to the vector database.

    Args:
        uploaded_file: The uploaded file object from Streamlit.
        openai_api_key: The OpenAI API key for embedding generation.
    """
    # try:
    if uploaded_file is not None:
        logging.info(f'run upload {uploaded_file}')

        if isinstance(uploaded_file, str):
            filename = uploaded_file
        else:
            filename = str(uploaded_file.name)

        # Load the document using the saved file path
        docs = load_file(Path(filename))

        global ensemble_retriever
        global chain

        all_docs = ensemble_retriever.retrievers[0].docs
        all_docs.extend(docs)

        ensemble_retriever.retrievers[1].add_documents(docs)

        new_bm25 = BM25Retriever.from_texts([t.page_content for t in all_docs])

        ensemble_retriever.retrievers[0] = new_bm25

        chain = create_full_chain(
            ensemble_retriever,
            openai_api_key=open_api_key,
        )

        logging.info("File uploaded and added to the knowledge base!")
        gr.Info('File uploaded and added to the knowledge base!', duration=3)
    
    return None
        
    # except Exception as e:
    #     logging.error(f"Error processing uploaded file: {e}")
    #     st.error("Error processing the file. Please check the logs.")

SUPPORTED_FORMATS = ['.txt', '.json', '.pdf']

def activate():
    return gr.update(interactive=True)

def deactivate():
    return gr.update(interactive=False)

def reset(z, request: gr.Request):
    session_id = request.session_hash
    clean_session_history(session_id)
    return [], []

def main():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# Equity Bank AI assistant \n"
            "Ask questions about Equity Bank's products and services:"
        )
        with gr.Tab('Chat'):
            clean_btn = gr.Button(value="Clean history", variant="secondary", size='sm', render=False)
            bot = gr.Chatbot(elem_id="chatbot", render=False)

            chat = gr.ChatInterface(
                show_ui,
                chatbot=bot,
                undo_btn=None,
                retry_btn=None,
                clear_btn=clean_btn,
            )
        with gr.Tab('Documents'):
            file_input = gr.File(
                label=f'{", ".join([str(f) for f in SUPPORTED_FORMATS])}',
                file_types=SUPPORTED_FORMATS,
            )
            submit_btn = gr.Button(value="Index file", variant="primary", interactive=False)

        clean_btn.click(fn=reset, inputs=clean_btn, outputs=[bot, chat.chatbot_state])

        submit_btn.click(
                fn=process_uploaded_file,
                inputs=file_input,
                outputs=file_input,
                api_name="Index file"
            )
        
        file_input.upload(fn=activate, outputs=[submit_btn])
        file_input.clear(fn=deactivate, outputs=[submit_btn])

    demo.launch(share=True)


open_api_key = os.getenv('OPEN_API_KEY')

ensemble_retriever, chain = get_chain(
    openai_api_key=open_api_key,
    huggingfacehub_api_token=None
)



if __name__ == "__main__":
    main()