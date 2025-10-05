from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI

from langchain_community.document_loaders import PyPDFLoader


from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    download_loader,
    RAKEKeywordTableIndex,
)

import os
from dotenv import load_dotenv

load_dotenv()
from langchain_community.llms import OpenAI


def process_pdf(pdf_file):
    documents = SimpleDirectoryReader(pdf_file).load_data()
    index1 = VectorStoreIndex.from_documents(documents, show_progress=True)
    vectorstore = index1.as_query_engine()
    return vectorstore


def queryEngine(self, query):
    query_engine = self.vectorstore
    return query_engine
