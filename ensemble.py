import os
import logging

from langchain_community.retrievers import BM25Retriever, TavilySearchAPIRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings

from basic_chain import get_model
from rag_chain import make_rag_chain
from remote_loader import load_web_page
from remote_loader import load_online_pdf  # Import load_online_pdf
from splitter import split_documents
from vector_store import create_vector_db
from dotenv import load_dotenv


def ensemble_retriever_from_docs(docs, embeddings=None):
    texts = split_documents(docs)
    vs = create_vector_db(texts, embeddings)
    vs_retriever = vs.as_retriever()

    bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])

    tavily_retriever = MyTavilySearchAPIRetriever(k=3, include_domains=['https://equitygroupholdings.com/ke'])

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vs_retriever, tavily_retriever],
        weights=[0.33, 0.33, 0.33])

    return ensemble_retriever


class MyTavilySearchAPIRetriever(TavilySearchAPIRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager
    ):
        try:
            return super()._get_relevant_documents(query, run_manager=run_manager)
        except Exception as e:
            logging.error(f"TavilySearch error: {e}")
            return []


def main():
    load_dotenv()

    equity_bank_annual_report_2022 = "https://equitygroupholdings.com/ke/uploads/Equity-Investment-Bank-2022-Annual-Report.pdf"
    
    # Use load_online_pdf to load the PDF directly
    docs = load_online_pdf(equity_bank_annual_report_2022)  
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-3-small")
    ensemble_retriever = ensemble_retriever_from_docs(docs, embeddings)
    model = get_model("ChatGPT")
    chain = make_rag_chain(model, ensemble_retriever) | StrOutputParser()

    result = chain.invoke("What are the key findings of Equity Bank Annual Report?")
    print(result)


if __name__ == "__main__":
    # this is to quite parallel tokenizers warning.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()