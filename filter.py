import os
import logging

from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
from langchain_community.embeddings import OpenAIEmbeddings 
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever, MergerRetriever
from langchain.chains import RetrievalQA

from basic_chain import get_model
from ensemble import ensemble_retriever_from_docs
from local_loader import load_data_files
from vector_store import create_vector_db

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_retriever(texts, openai_api_key):
    """Creates a retriever with dense and sparse embeddings and filtering."""
    try:
        dense_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")
        sparse_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small") 

        dense_vs = create_vector_db(texts, collection_name="dense", embeddings=dense_embeddings)
        sparse_vs = create_vector_db(texts, collection_name="sparse", embeddings=sparse_embeddings)
        vector_stores = [dense_vs, sparse_vs]

        emb_filter = EmbeddingsRedundantFilter(embeddings=sparse_embeddings)
        reordering = LongContextReorder()
        pipeline = DocumentCompressorPipeline(transformers=[emb_filter, reordering])

        base_retrievers = [vs.as_retriever() for vs in vector_stores]
        lotr = MergerRetriever(retrievers=base_retrievers)

        compression_retriever_reordered = ContextualCompressionRetriever(
            base_compressor=pipeline, base_retriever=lotr, search_kwargs={"k": 5, "include_metadata": True}
        )
        return compression_retriever_reordered
    except Exception as e:
        logging.error(f"Error creating retriever: {e}")
        # Handle the error appropriately (e.g., return a simpler retriever or raise an exception)


def main():
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    try:
        docs = load_data_files(data_dir="data") 
        retriever = create_retriever(docs, openai_api_key=openai_api_key)  
        llm = get_model(openai_api_key=openai_api_key)  
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

        # Financial consultant relevant query
        results = qa.invoke("What are the benefits of an Equity Ordinary Account?")  
        print(results)

    except Exception as e:
        logging.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main()