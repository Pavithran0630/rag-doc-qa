from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_and_chunk_pdf(pdf_path: str):
    print(f"Loading PDF: {pdf_path}")
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Total pages loaded: {len(documents)}")
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    
    # Preview first chunk
    print("\n--- First Chunk Preview ---")
    print(chunks[0].page_content)
    print("---------------------------\n")
    
    return chunks

if __name__ == "__main__":
    pdf_path = os.path.join("data", "Resume_Pavithran M.pdf")
    chunks = load_and_chunk_pdf(pdf_path)