import streamlit as st
import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --------------------------------
# STREAMLIT TITLE
# --------------------------------

st.title("🤖 Zyro Dynamics HR Assistant")

# --------------------------------
# API KEY
# --------------------------------

groq_key = os.getenv("GROQ_API_KEY")

st.write("Groq key exists:", groq_key is not None)

if groq_key:
    st.write("Key starts with:", groq_key[:10])

if not groq_key:
    st.error("GROQ_API_KEY not found in Streamlit Secrets")
    st.stop()

# --------------------------------
# GROQ TEST
# --------------------------------

try:
    test_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0
    )

    test_response = test_llm.invoke("Say hello")

    st.success("Groq connection successful")

except Exception as e:
    st.error(f"Groq Error: {e}")
    st.stop()

# --------------------------------
# BUILD RAG PIPELINE
# --------------------------------

@st.cache_resource
def build_rag():

# Load PDFs
    docs = []

    for pdf in glob.glob("*.pdf"):
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())

# Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

chunks = splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
# Vector Store
vectorstore = FAISS.from_documents(chunks, embeddings)

    # Retriever
retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 25}
    )
 # LLM
llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0
    )
# Prompt
template = """
You are the Zyro Dynamics HR Assistant.

Use ONLY the information present in the context.

Rules:
1. If the answer is clearly present in the context, answer directly.
2. If the context does not contain enough information, respond EXACTLY:

I can only answer HR-related questions from Zyro Dynamics policy documents.

3. Do not use outside knowledge.
4. Do not guess.
5. Mention policy details exactly as written.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
# Format docs
def format_docs_with_sources(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[SOURCE: {source}]\n{doc.page_content}")
        return "\n\n".join(formatted)

# RAG Chain
rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, vectorstore, len(docs)

# --------------------------------
# INITIALIZE
# --------------------------------
rag_chain, retriever, vectorstore, num_docs = build_rag()

st.write(f"📄 Number of document pages loaded: {num_docs}")

# --------------------------------
# USER INPUT
# --------------------------------

question = st.text_input(
    "Ask an HR question:"
)
REFUSAL = (
    "I can only answer HR-related questions "
    "from Zyro Dynamics policy documents."
)

def answer_question(question):

    docs_scores = vectorstore.similarity_search_with_score(
        question,
        k=5
    )

    best_score = docs_scores[0][1]

    if best_score > 1.2:
        return REFUSAL

    answer = rag_chain.invoke(question)

    if (
        "not found" in answer.lower()
        or len(answer.strip()) < 20
    ):
        return REFUSAL

    return answer

if question:

    # Retrieved Chunks
    retrieved_docs = retriever.invoke(question)

    with st.expander("Retrieved Chunks"):
        for i, doc in enumerate(retrieved_docs):
            st.write(f"### Chunk {i+1}")
            st.write(doc.page_content[:500])

    # Generate Answer
    answer = answer_question(question)

    st.subheader("Answer")
    st.write(answer)
