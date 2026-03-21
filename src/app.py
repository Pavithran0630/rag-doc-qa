import streamlit as st
import os
import tempfile
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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

    # Step 3 - Process PDF only once
    if "vector_store" not in st.session_state or \
       st.session_state.get("pdf_name") != uploaded_file.name:

        with st.spinner("Reading and processing your PDF..."):
            try:
                chunks = load_and_chunk_pdf(tmp_path)

                if len(chunks) == 0:
                    st.error("No text found in PDF. It might be a scanned image.")
                    st.stop()

                st.session_state.vector_store = create_vector_store(chunks)
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chunk_count = len(chunks)
                st.success(f"PDF processed! {len(chunks)} chunks created.")

            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.stop()

    # Show PDF info
    st.info(f"Currently loaded: **{st.session_state.pdf_name}** "
            f"({st.session_state.chunk_count} chunks)")

    # Step 4 - Question input
    question = st.text_input("Ask a question about your PDF:")

    if question:

        if len(question.strip()) < 5:
            st.warning("Please ask a more specific question!")
        else:
            with st.spinner("Searching for answer..."):
                try:
                    # Get answer
                    answer = ask_question(
                        question,
                        st.session_state.vector_store
                    )

                    # Show answer
                    st.subheader("Answer:")
                    st.write(answer)

                    # Show retrieved chunks (transparency)
                    with st.expander("See retrieved chunks"):
                        chunks = st.session_state.vector_store\
                            .similarity_search(question, k=5)
                        for i, chunk in enumerate(chunks):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.write(chunk.page_content)
                            st.divider()

                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")

    # Clear button
    # Clear button
    if st.button("Upload a new PDF"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()