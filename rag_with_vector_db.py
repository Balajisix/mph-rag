import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="RAG Model")

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.1,
        max_new_tokens=512,
    )

model = ChatHuggingFace(llm=llm)

vector_db_path = "./chroma_langchain_db"
vector_store = Chroma(
    collection_name="knowledge_base",
    embedding_function=embeddings,
    persist_directory=vector_db_path
)

def initialize_knowledge_base():
    """Loads documents, chunks them, and adds to Chroma if empty."""
    if not os.path.exists("./data"):
        os.makedirs("./data")
        return
    
    if vector_store._collection.count() == 0:
        print("Indexing documents...")
        loader = DirectoryLoader("./data", glob="./*.pdf", loader_cls=PyPDFLoader)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=150
        )
        final_chunks = text_splitter.split_documents(docs)

        vector_store.add_documents(final_chunks)
        print(f"Successfully indexed {len(final_chunks)} chunks.")

initialize_knowledge_base()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say you don't know.\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(model, prompt_template)
rag_chain = create_retrieval_chain(vector_store.as_retriever(), question_answer_chain)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/ask")
async def ask_rag(request: QueryRequest):
    try:
        response = rag_chain.invoke({"input": request.query})
        
        return {
            "answer": response["answer"],
            "context_used": [doc.page_content for doc in response["context"]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))