import os
from io import BytesIO
from dotenv import load_dotenv

import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

st.set_page_config(page_title="FAISS RAG Chatbot", page_icon="🤖", layout="centered")
st.title("RAG Assignment — Streamlit (single app)")

def lazy_init_models():
    """Initialize embeddings and LLM once and store in session_state."""
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    if "chat_model" not in st.session_state:
        llm_endpoint = HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-20b",
            huggingfacehub_api_token=os.environ("HF_TOKEN"),
            temperature=0.1,
            timeout=120,
        )
        st.session_state.chat_model = ChatHuggingFace(llm=llm_endpoint)

def initialize_index_from_path(path: str):
    """Load PDF from path, split into chunks, and build FAISS index in memory."""
    lazy_init_models()
    loader = PyPDFLoader(path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    st.session_state.vector_store = FAISS.from_documents(chunks, st.session_state.embeddings)
    st.session_state.index_preview = [c.page_content for c in chunks[:3]]
    st.session_state.indexed_chunks = len(chunks)
    return len(chunks)

def initialize_index_from_bytes(filename: str, file_bytes: bytes):
    """Save uploaded bytes to ./data and index them."""
    os.makedirs("./data", exist_ok=True)
    save_path = os.path.join("./data", filename)
    with open(save_path, "wb") as f:
        f.write(file_bytes)
    return initialize_index_from_path(save_path)

def get_session_history(session_id: str):
    """Retrieve or create a new message history for a specific session."""
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def build_rag_chain():
    """Builds the retrieval + QA chain with history awareness using current vector_store."""
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        raise ValueError("No vector store initialized. Upload a PDF first.")

    chat_model = st.session_state.chat_model
    vector_store = st.session_state.vector_store

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        chat_model, vector_store.as_retriever(), contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "Context: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

st.session_state.setdefault("indexed", False)
st.session_state.setdefault("index_message", "No index available. Upload a PDF to index.")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("session_id", "user1")
st.session_state.setdefault("vector_store", None)
st.session_state.setdefault("index_preview", [])
st.session_state.setdefault("indexed_chunks", 0)

st.sidebar.header("Session")
st.session_state.session_id = st.sidebar.text_input(
    "Session ID (enter manually)", value=st.session_state.session_id
)
st.sidebar.markdown("**Index status**")
st.sidebar.write(st.session_state.index_message)

if not st.session_state.indexed:
    st.header("Upload PDF to index")
    uploaded_file = st.file_uploader("Choose a PDF to upload and index", type=["pdf"])
    if uploaded_file is not None:
        placeholder = st.empty()
        placeholder.info("Uploading and indexing... this may take a while.")
        try:
            file_bytes = uploaded_file.read()
            num_chunks = initialize_index_from_bytes(uploaded_file.name, file_bytes)
            msg = f"Indexed {num_chunks} chunks from {uploaded_file.name}"
            st.success(msg)
            st.session_state.indexed = True
            st.session_state.index_message = msg
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Upload/indexing failed: {e}")
        finally:
            placeholder.empty()
else:
    st.info(st.session_state.index_message)
    if st.button("Reset index / Upload new PDF"):
        st.session_state.indexed = False
        st.session_state.index_message = "No index available. Upload a PDF to index."
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.index_preview = []
        st.session_state.indexed_chunks = 0
        st.session_state.store = {}
        st.experimental_rerun()

if st.session_state.indexed and st.session_state.index_preview:
    with st.expander("Preview first 3 indexed chunks"):
        for i, chunk in enumerate(st.session_state.index_preview, start=1):
            st.markdown(f"**Chunk {i}:**")
            st.code(chunk, language="text")

st.header("Chat with the indexed document")
if not st.session_state.indexed:
    st.info("Index not ready. Upload and index a PDF to enable querying.")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if st.session_state.indexed:
    prompt = st.chat_input("Type your question...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        lazy_init_models()

        try:
            conversational_rag_chain = build_rag_chain()
            with st.spinner("Getting answer from the RAG model..."):
                response = conversational_rag_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )
            answer = response.get("answer", "No answer returned")
        except Exception as e:
            answer = f"⚠️ Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").markdown(answer)
else:
    st.chat_input("Index not ready. Upload a PDF to enable chat.", disabled=True)
