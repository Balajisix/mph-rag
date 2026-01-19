import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
 
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

app = FastAPI(title="FAISS RAG")
load_dotenv()
 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
 
llm_endpoint = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    temperature=0.1,
    timeout=120,
)
chat_model = ChatHuggingFace(llm=llm_endpoint)
 
vector_store = None
store = {}
 
def get_session_history(session_id: str):
    """Retrieve or create a new message history for a specific session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and rebuild FAISS index."""
    try:
        global vector_store
        file_path = f"./data/{file.filename}"
        os.makedirs("./data", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(await file.read())
 
        loader = PyPDFLoader(file_path)
        docs = loader.load()
 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
 
        vector_store = FAISS.from_documents(chunks, embeddings)
        return {"status": "success", "message": f"Indexed {len(chunks)} chunks from {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
def build_rag_chain():
    """Builds the retrieval + QA chain with history awareness."""
    if vector_store is None:
        raise ValueError("No vector store initialized. Upload a PDF first.")
 
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

class ChatRequest(BaseModel):
    query: str
    session_id: str
 
@app.post("/ask")
async def ask_rag(request: ChatRequest):
    try:
        if vector_store is None:
            raise HTTPException(status_code=400, detail="No knowledge base loaded. Upload a PDF first.")
 
        conversational_rag_chain = build_rag_chain()
        response = conversational_rag_chain.invoke(
            {"input": request.query},
            config={"configurable": {"session_id": request.session_id}}
        )
        return {
            "answer": response["answer"],
            "session_id": request.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
