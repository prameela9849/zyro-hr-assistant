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
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 25
        }
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

Answer ONLY using the provided HR policy documents.

If the answer exists in the context, answer clearly and concisely.

If the answer is NOT found in the context OR the question is unrelated to HR policies, reply EXACTLY:

I can only answer HR-related questions from Zyro Dynamics policy documents.

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

    # Format Documents
    def format_docs(docs):
        return "\n\n".join(
            doc.page_content for doc in docs
        )

    # RAG Chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, len(docs)

# --------------------------------
# INITIALIZE
# --------------------------------

rag_chain, retriever, num_docs = build_rag()

st.write(f"📄 Number of document pages loaded: {num_docs}")

# --------------------------------
# USER INPUT
# --------------------------------

question = st.text_input(
    "Ask an HR question:"
)

if question:

    # Retrieved Chunks
    retrieved_docs = retriever.invoke(question)

    with st.expander("Retrieved Chunks"):
        for i, doc in enumerate(retrieved_docs):
            st.write(f"### Chunk {i+1}")
            st.write(doc.page_content[:500])

    # Generate Answer
    answer = rag_chain.invoke(question)

    st.subheader("Answer")
    st.write(answer)
