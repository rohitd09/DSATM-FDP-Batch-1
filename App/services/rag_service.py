from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingService:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            show_progress=True
        )
        print("-------------EMBEDDING INI--------------")

        self.vector_store = Chroma(
            persist_directory="./rag_service_db",
            collection_name="teaching_assistant_collection",
            embedding_function=embeddings
        )

        print("-------------CHROMA INI--------------")

    def process_pdf_document(self, file_path: str = "./Assets/Let us c - Summary.pdf"):
        loader = PyPDFLoader(file_path)
        document = loader.load()

        print("-------------LOADER INI--------------")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150
        )

        print("-------------SPLITTER INI--------------")

        split_documents = splitter.split_documents(document)

        self.vector_store.add_documents(split_documents)

    def retrieve_from_pdf(self, query: str):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        response = retriever.invoke(query)
        return [doc.page_content for doc in response]

if __name__ == "__main__":
    service = EmbeddingService()
    service.process_pdf_document()
    # print(f'\n\nCHUNKS RETRIVED \n{service.retrieve_from_pdf("What are pointers?")}')

    chunks = service.retrieve_from_pdf("What are pointers?")

    print("HERE ARE THE CHUNKS: ")
    for chunk in chunks:
        print("\n\n")
        print(chunk)