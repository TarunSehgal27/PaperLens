from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from datasets import Dataset
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
    if "arxiv.org" in file:
        if "/abs/" in file:
            file = file.replace("/abs/", "/pdf/")
        if not file.endswith(".pdf"):
            file+= ".pdf"
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

# Evaluation
def evaluate_rag(questions, ground_truths):
    answers = []
    contexts = []

    for ques in questions:
        retrieved_docs = retrieve(ques, vector_store)
        answer = ask(question, vector_store)

        answers.append(answer)
        contexts.append([doc.page_content for doc in retrieved_docs])
    
    data = {
        "user_input" : questions,
        "response" : answers,
        "retrieved_contexts" : contexts,
        "reference" : ground_truths
    }

    dataset = Dataset.from_dict(data)
  
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]
    
    llm = HuggingFaceEndpoint(
        repo_id = "Qwen/Qwen3-Coder-Next",
        task = "text-generation",
        streaming=True
    )

    model = ChatHuggingFace(llm=llm)

    result = evaluate(
        dataset = dataset,
        metrics = metrics,
        llm = model,
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    )

    return result

if __name__ == "__main__":
    vector_store = initialize("https://arxiv.org/abs/2210.03629")
    print(f"Vector Store Created")

    while True:
        question = input("Q: ")
        if question == 'exit':
            break
        result = ask(question, vector_store)

    questions = [
    "What is a deepfake?",
    "Who developed Generative Adversarial Networks (GANs) and when?",
    "What are the two neural network components in a GAN and what are their roles?",
    "What are the four categories of deepfakes described in the paper?",
    "What distinguishes Conditional GANs (cGANs) from standard GANs?",
    "What are the five architectural guidelines for DCGANs to remain stable during training?",
    "What is the key idea behind CycleGANs?",
    "What is the Inception Score (IS) and who proposed it?",
    "Why is the Inception Score (IS) considered insufficient as the sole evaluation metric for GANs?",
    "Why do GANs produce more convincing deepfakes compared to traditional data manipulation methods?",
    "What problem does mode collapse cause in GAN training and how do DCGANs address it?",
    "What are the key challenges and future directions in deepfake detection identified in the paper's conclusion?"
    ]

    ground_truths = [
        "A deepfake is synthetic media generated by an AI model where one person's identity is changed into another person in visual content like images or videos. The term was coined by a Reddit user in 2017 who demonstrated face-swapping in videos.",
        "GANs were developed by Ian J. Goodfellow and his colleagues in 2014.",
        "A GAN consists of a Generator and a Discriminator. The Generator creates synthetic data, while the Discriminator verifies whether the generated data looks like real data. They compete against each other — the Discriminator keeps rejecting generated samples until it can no longer distinguish them from real ones.",
        "The four categories are: (1) Synthesis — completely generated faces of people that do not exist in the real world; (2) Attribute Manipulation — modifying features like hair, skin tone, or eye color without changing the person's identity; (3) Reenactment — using one person's face to manipulate another person's facial expressions, body posture, and gaze; (4) Replacement — transferring or swapping one person's identity onto another.",
        "In Conditional GANs, both the generator and discriminator receive additional information — such as class labels or data from other modalities — as input during training. This conditioning information is represented as 'y' in the objective function, allowing the model to generate outputs conditioned on specific attributes.",
        "The five guidelines are: (a) Use strided convolutions in the discriminator and fractional-strided convolutions in the generator instead of pooling layers; (b) Both networks use batch normalization; (c) No fully connected hidden layers; (d) Use ReLU activation for all generator layers and Tanh for the output layer; (e) Use LeakyReLU activation for all discriminator layers.",
        "CycleGANs, introduced by Zhu et al., enable image-to-image translation between two domains by learning two mappings (G: X→Y and F: Y→X) and enforcing cycle consistency — meaning an image translated from one domain should be translatable back to closely resemble the original.",
        "The Inception Score (IS) was introduced by Salimans et al. It evaluates GAN performance by calculating the Kullback-Leibler divergence between the predicted class probabilities of generated images and the overall distribution of generated images. It measures both the realism and diversity of generated content.",
        "IS is insufficient as a sole metric because it struggles to capture diversity within individual classes and over-relies on the ImageNet dataset, making it unreliable for GANs trained on other datasets. Additionally, high IS scores don't always guarantee high image quality, and vice versa.",
        "GANs use adversarial processes and pixel-wise loss functions that allow models to learn meaningful representations and minute patterns. Unlike traditional methods, GANs are trained on large corpora of image or video data, enabling them to create highly convincing synthetic content that is often indistinguishable to the human eye.",
        "Mode collapse causes the generator to produce identical or repetitive outputs instead of diverse samples. DCGANs address this through specific architectural constraints — including strided convolutions, batch normalization, removal of fully connected layers, and appropriate activation functions — which significantly improve training stability and output quality.",
        "The key challenges and future directions include: (1) developing more effective and resource-constrained detection methods for edge devices and public use; (2) addressing the lack of high-quality, diverse datasets to build generalizable models; (3) adopting zero-shot or few-shot learning to detect unseen forgery methods without extensive retraining; (4) achieving real-time detection for live streaming and surveillance via model pruning and edge computing; (5) fostering collaboration among researchers, policymakers, and industry experts; and (6) building explainable AI models to provide transparency in detection decisions."
    ]
    # result = evaluate_rag(questions, ground_truths)
    # print(result) # {'faithfulness': 1.0000, 'answer_relevancy': 1.0000, 'context_precision': 1.0000, 'context_recall': 1.0000}


    

