from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.ingest import load_and_chunk_pdf
import os

def create_vector_store(chunks):
    print("Creating embeddings and storing in FAISS...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    vector_store.save_local("outputs/faiss_index")
    print("Vector store saved to outputs/faiss_index")
    
    return vector_store

def load_vector_store():
    print("Loading existing vector store...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    vector_store = FAISS.load_local(
        "outputs/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vector_store

if __name__ == "__main__":
    pdf_path = os.path.join("data", "Resume_Pavithran M.pdf")
    chunks = load_and_chunk_pdf(pdf_path)
    vector_store = create_vector_store(chunks)
    
    print("\nTesting search...")
    results = vector_store.similarity_search("What are my skills?", k=2)
    
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content)