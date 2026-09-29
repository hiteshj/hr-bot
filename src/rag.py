import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from src.retriever import NativeFAISSRetriever
import streamlit as st

from src.hr_docs import HR_DOCUMENTS


CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100

def _make_documents():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs = []
    for item in HR_DOCUMENTS:
        chunks = splitter.create_documents(
            texts=[item["content"]],
            metadatas=[{"source": item["title"]}],
        )
        docs.extend(chunks)
    return docs


@st.cache_resource(show_spinner="Building HR policy index… (first run downloads ~40 MB embedding model)")
def build_vectorstore():
    #embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",       # Or "gemini-embedding-2-preview"
        output_dimensionality=768           # Optional: 768, 1536, or 3072
    )    
    docs = _make_documents()
    vectorstore = NativeFAISSRetriever.from_documents(documents=docs, embeddings=embeddings, k=4)
    return vectorstore


def retrieve(query: str, vectorstore, k: int = 3):
    """Return top-k chunks with relevance scores."""
    results = vectorstore.invoke(query, k=k)
    return [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "content": doc.page_content,
        }
        for doc in results
    ]

