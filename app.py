from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import re
import os

load_dotenv()

# preprocessing text
def preprocess_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    return text.strip()

# doc ingestion
def load_documents(file):
    loader = PyMuPDFLoader(file)
    documents = loader.load()

    return documents
    
# text splitting
def get_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1200,
        chunk_overlap = 120,
        separators = ['\n\n', '\n', '.'],
        add_start_index = True
    )

    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.page_content = preprocess_text(chunk.page_content)

    return chunks

# Embedding generation and storing in vector store
def create_vectorstore(chunks, save_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        encode_kwargs={'batch_size':64}    
    )
    if os.path.exists(save_path):
        print("Already exists")
        vector_store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)

    return vector_store

# indexing
def initialize(file):
    documents = load_documents(file)
    chunks = get_chunks(documents)
    vector_store = create_vectorstore(chunks)

    return vector_store

# Retrieval
def retrieve(query, vector_store):
    retriever = vector_store.as_retriever(
        search_type='mmr',
        search_kwargs={'k':2, 'lambda_mult': 0.6}
    )
    retrieved_docs = retriever.invoke(query)

    return retrieved_docs

# Augmentation (query + retrieved_docs)
def create_prompt(retrieved_docs, query):

    prompt = PromptTemplate(
        template = """
        You are PaperLens, an AI research assistant that answers questions strictly 
        from retrieved academic paper chunks provided in the context below.

        RULES:
        - Ground every answer in the context. Never hallucinate facts or citations.
        - If the answer isn't in the context, say: "Not found in uploaded papers."
        - Cite inline as: [Paper Title, p.X] or [Author et al., Year]
        - Flag contradictions across papers if found.
        - Do not use general knowledge unless explicitly asked.

        RESPONSE FORMAT:
        Answer: <direct answer>
        Source: <bullet points with inline citations>
        Confidence: <direct answer> High | Medium | Low

        AUTO-DETECT MODE:
        - Question asked       → Q&A mode
        - "summarize"          → Problem / Contributions / Method / Results / Limitations
        - "compare"            → Side-by-side comparison table
        - "extract"            → Structured list (datasets, metrics, models, etc.)
        - "critique"           → Gaps, reproducibility, unsupported claims
        - "literature review"  → Synthesize themes across all papers

        TONE: Academic but clear. Hedge when uncertain ("The paper suggests...").

        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        CONTEXT: {context}
        QUESTION: {question}
        """,
        input_variables=['context', 'question']
    )

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    final_prompt = prompt.invoke({'context': context_text, 'question': query})

    return final_prompt

# Generation
def get_result(final_prompt):
    llm = HuggingFaceEndpoint(
        repo_id = "Qwen/Qwen3-Coder-Next",
        task = "text-generation",
        streaming=True
    )

    model = ChatHuggingFace(llm=llm)
    answer = ''
    for word in model.stream(final_prompt):
        print(word.content, end='', flush=True)
        answer += word.content
    print()

    return answer

def ask(query, vector_store):
    retrieved_docs = retrieve(query, vector_store)
    final_prompt = create_prompt(retrieved_docs, query)
    answer = get_result(final_prompt)
    
    return answer

if __name__ == "__main__":
    vector_store = initialize("The Deepfake Dilemma.pdf")
    print(f"Vector Store Created")

    while True:
        question = input("Q: ")
        if question == 'exit':
            break
        result = ask(question, vector_store)


    

