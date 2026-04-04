import streamlit as st
from app import initialize, ask
import os

st.set_page_config(page_title="PaperLens", page_icon="📄")
st.header("PaperLens - AI Research Assistant")

input_type = st.radio("Select Input Type", ["Upload PDF", "ArXiv URL"], horizontal=True)

file_input = None

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

# File Upload
if input_type == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'])
    if uploaded_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_input = "temp.pdf"

    else:
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")
        if st.session_state.vector_store is not None:
            st.session_state.vector_store = None
        st.info("Please upload a PDF to begin.")

elif input_type == "ArXiv URL":
    arxiv_url = st.text_input("Enter ArXiv URL (e.g. https://arxiv.org/abs/2301.00001)")
    if arxiv_url:
        file_input = arxiv_url

if file_input:
    if st.session_state.vector_store is None:
        with st.spinner("Processing document..."):
            st.session_state.vector_store = initialize(file_input)
        st.success("Document processed! Ask your question below.")
    
    question = st.text_input("Ask a question: ")

    if question:
        with st.spinner("Fetching answer..."):
            answer = ask(question, st.session_state.vector_store)
        st.write(f"{answer}")

# Footer
st.markdown(
    """
    <div style="margin-top: 50px; padding: 15px; text-align: center; border-top: 1px solid #ddd;">
        © <a href="https://github.com/tarunsehgal27" target="_blank">Tarun Sehgal</a> | <strong>Made with ❤️ </strong>
    </div>
    """,
    unsafe_allow_html=True
)
