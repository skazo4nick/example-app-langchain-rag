import logging
import os
from typing import List
import shutil

# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from local_loader import load_data_files
from splitter import split_documents
from dotenv import load_dotenv
from time import sleep

EMBED_DELAY = 0.02  # 20 milliseconds


# This is to get the Streamlit app to use less CPU while embedding documents into Chromadb.
class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)


# This happens all at once, not ideal for large datasets.
def create_vector_db(texts, embeddings=None, collection_name="chroma"):
    if not texts:
        logging.warning("Empty texts passed in to create vector database")
    # Select embeddings
    if not embeddings:
        openai_api_key = os.environ["OPENAI_API_KEY"]
        # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
        embeddings = HuggingFaceEmbeddings()

    proxy_embeddings = EmbeddingProxy(embeddings)

    persist_directory = os.path.join("store/", collection_name)
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    db = Chroma(collection_name=collection_name,
                embedding_function=proxy_embeddings,
                persist_directory=persist_directory)
    try:
        db.add_documents(texts)
    except Exception as e:
        logging.error(f"Error adding documents to Chroma: {e}")
        # You might want to handle the error more specifically here, 
        # such as retrying or returning an error indicator.
    return db


def find_similar(vs, query):
    docs = vs.similarity_search(query)
    return docs


def main():
    load_dotenv()
    docs = load_data_files(data_dir="data")  # Load data from your 'data' folder
    texts = split_documents(docs)
    vs = create_vector_db(texts)

    # Use a relevant query from your financial domain
    results = find_similar(vs, query="What are the fees for an Equity Ordinary Account?")
    MAX_CHARS = 300
    print("=== Results ===")
    for i, text in enumerate(results):
        content = text.page_content
        n = max(content.find(' ', MAX_CHARS), MAX_CHARS)
        content = text.page_content[:n]
        print(f"Result {i + 1}:\n {content}\n")

if __name__ == "__main__":
    main()