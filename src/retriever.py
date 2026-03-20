from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

def load_and_chunk_pdf(pdf_path: str):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Total pages loaded: {len(documents)}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

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

def ask_question(question, vector_store):
    print(f"\nQuestion: {question}")
    
    # Step 1 - Retrieve relevant chunks from FAISS
    chunks = vector_store.similarity_search(question, k=5)
    
    # Step 2 - Combine chunks into one context string
    context = "\n\n".join([chunk.page_content for chunk in chunks])
    
    # Step 3 - Build prompt with context + question
    prompt = f"""You are a helpful assistant. Answer the question 
based ONLY on the context provided below.
If the answer is not found in the context, say 
"I don't know based on the provided document."

Context:
{context}

Question: {question}

Answer:"""
    
    # Step 4 - Send to GPT-4o-mini and get answer
    try:
        token = st.secrets["GITHUB_TOKEN"]
    except Exception:
        token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN not found in secrets or environment!")

    client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=token
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content
    print(f"Answer: {answer}")
    return answer


if __name__ == "__main__":
    pdf_path = os.path.join("data", "Resume_Pavithran M.pdf")
    
    # Phase 2 - Load and chunk PDF
    chunks = load_and_chunk_pdf(pdf_path)
    
    # Phase 3 - Create vector store
    vector_store = create_vector_store(chunks)
    
    # Phase 4 - Ask questions
    ask_question("What are my technical skills?", vector_store)
    ask_question("What projects have I done?", vector_store)