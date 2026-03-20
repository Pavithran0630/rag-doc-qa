import streamlit as st
import os
import tempfile
from retriever import load_and_chunk_pdf, create_vector_store, ask_question

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")

st.title("📄 RAG Document Q&A App")
st.write("Upload a PDF and ask questions about it!")

# Step 1 - File uploader
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    
    # Step 2 - Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    # Step 3 - Process PDF only once using session state
    if "vector_store" not in st.session_state:
        with st.spinner("Reading and processing your PDF..."):
            chunks = load_and_chunk_pdf(tmp_path)
            st.session_state.vector_store = create_vector_store(chunks)
        st.success(f"PDF processed! {len(chunks)} chunks created.")
    
    # Step 4 - Question input
    question = st.text_input("Ask a question about your PDF:")
    
    if question:
        with st.spinner("Searching for answer..."):
            answer = ask_question(question, st.session_state.vector_store)
        
        st.subheader("Answer:")
        st.write(answer)
        
    # Clear button
    if st.button("Upload a new PDF"):
        del st.session_state.vector_store
        st.rerun()