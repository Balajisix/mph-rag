import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

app = FastAPI(title="Without Vector DB")
load_dotenv()

access_token = os.getenv("HF_MODEL")
model_id = os.getenv("HF_MODEL")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    huggingfacehub_api_token="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",
    temperature=0.1,
    max_new_tokens=512
)
model = ChatHuggingFace(llm=llm)

all_chunks = []

def initialize_knowledge_base():
    global all_vectors, all_chunks
    loader = PyPDFLoader("./data/troubleshooting.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100
    )
    chunks = text_splitter.split_documents(docs)
    all_chunks = [c.page_content for c in chunks]
    all_vectors = np.array(embeddings.embed_documents(all_chunks))
    print(f"Indexing completed: {len(all_chunks)} chunks in RAM")

initialize_knowledge_base()

def get_relevant_context(query, k=3):
    global all_vectors, all_chunks 
    
    if all_vectors is None or len(all_chunks) == 0:
        print("Warning: Knowledge base is empty. Please check your data folder.")
        return []

    query_vector = np.array(embeddings.embed_query(query))
    dot_product = np.dot(all_vectors, query_vector)
    norms = np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(query_vector)
    similarities = np.divide(dot_product, norms, out=np.zeros_like(dot_product), where=norms!=0)
    current_k = min(k, len(all_chunks))
    top_indices = np.argsort(similarities)[-current_k:]
    top_indices = top_indices[::-1]
    return [all_chunks[int(i)] for i in top_indices]

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_rag(request: QueryRequest):
    try:
        context_chunk = get_relevant_context(request.query)
        context_text = "\n\n".join(context_chunk)
        prompt_template = f"""Use the context below to answer the question.:
        Context: {context_text}
        Question: {request.query}
        Answer:
        Return only the JSON object, no extra text.
        """
        response = model.invoke(prompt_template)
        return {
            "answer": response.content,
            "context_used": context_chunk
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))