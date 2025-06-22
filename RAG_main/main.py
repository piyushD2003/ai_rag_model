# main.py (Advanced Version)

import uuid
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from fastapi.middleware.cors import CORSMiddleware
# from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import google.generativeai as genai

# --- Settings and API Key Configuration ---
# class Settings(BaseSettings):
#     google_api_key: str
#     class Config:
#         env_file = ".env"
# settings = Settings()
# genai.configure(api_key=settings.google_api_key)

# --- 1. Initialize FastAPI App ---
'''
app = FastAPI(title="Advanced RAG Interview Backend")

# --- ADD CORS MIDDLEWARE (This is the crucial part) ---
# This must be added before you define your routes.
app.add_middleware(
    CORSMiddleware,
    # IMPORTANT: Add your frontend's origin URL here.
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    # This explicitly allows POST, GET, and the preflight OPTIONS method.
    allow_methods=["*"], 
    # This allows headers like Content-Type.
    allow_headers=["*"], 
)

# --- 2. Load Models and Clients ---
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")
chroma_client = chromadb.Client()

# --- 3. Define Pydantic Models ---
class ResumeUploadRequest(BaseModel):
    resume_text: str
    api_key: str


class SessionResponse(BaseModel):
    session_id: str

### UPDATED ###
# The chat history is no longer needed from the client
class ChatRequest(BaseModel):
    session_id: str
    chat_history: list[dict]
    api_key: str

class ChatResponse(BaseModel):
    answer: str

# --- 4. Create API Endpoints ---

@app.post("/start-session", response_model=SessionResponse)
def start_session_and_process_resume(request: ResumeUploadRequest):
    print(request.resume_text)
    session_id = str(uuid.uuid4())
    collection = chroma_client.create_collection(name=session_id)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(request.resume_text)
    embeddings = embedding_model.encode(chunks)
    # genai.configure(api_key=request.api_key)
    # result = genai.embed_content(
    #     model="gemini-embedding-exp-03-07",
    #     content=chunks,
    #     task_type="RETRIEVAL_DOCUMENT" # Important: tells the model we are embedding docs for search
    # )
    # embeddings = result['embedding']
    
    ### UPDATED ###
    # Add metadata to distinguish resume chunks from chat chunks
    metadatas = [{"source": "resume"}] * len(chunks)
    chunk_ids = [f"resume_{i}" for i in range(len(chunks))]

    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=metadatas,
        ids=chunk_ids
    )
    print(f"✅ Session '{session_id}' created. Resume processed into {len(chunks)} chunks.")
    return SessionResponse(session_id=session_id)

### UPDATED AND ADVANCED ###
@app.post("/chat", response_model=ChatResponse)
def chat_with_rag(request: ChatRequest):
    try:
        genai.configure(api_key=request.api_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid API Key provided. Error: {e}")

    try:
        collection = chroma_client.get_collection(name=request.session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found.")
    print(request.chat_history[-2]['role'])
    ### MODIFIED: Get the latest user message from the history ###
    if not request.chat_history or request.chat_history[-2]['role'] != 'user':
        raise HTTPException(status_code=400, detail="Invalid chat history: last message must be from the user.")
    
    latest_user_message = request.chat_history[-2]['parts'][0]['text']
    print("latest_user_message",latest_user_message)
    # --- RAG Retrieval Step (This part is the same) ---
    query_embedding = embedding_model.encode([latest_user_message]).tolist()
    # result = genai.embed_content(
    #     model="models/text-embedding-004",
    #     content=latest_user_message,
    #     task_type="RETRIEVAL_QUERY" # Important: tells the model this is a search query
    # )
    # query_embedding = [result['embedding']]
    results = collection.query(query_embeddings=query_embedding, n_results=4)
    retrieved_chunks = results['documents'][0]
    retrieved_context = "\n\n---\n\n".join(retrieved_chunks)

    ### MODIFIED: Updated Prompt Template ###
    # The prompt no longer has a separate "Question" section.
    prompt_template = f"""
You are a strict but professional AI Interviewer. Your task is to conduct an interview based on the candidate's resume.

**Candidate's Resume Information (Use this as your primary source of truth):**
---
{retrieved_context}
---

**Full Conversation History:**
(You are the 'model', the candidate is the 'user')
{request.chat_history}

Based on the provided resume information and the full conversation history (especially the last user message), provide your response or next question.
"""

    # --- Call Gemini API (This part is the same) ---
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        gemini_response = model.generate_content(prompt_template)
        ai_answer = gemini_response.text
        # print(f"Retrieved context for session '{request.session_id}': {retrieved_chunks}")
        # return ChatResponse(answer=gemini_response.text)
    
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from Gemini.")
    
    # --- 3. NEW: UPDATE MEMORY ---
    # Now, we embed the latest exchange and add it to our vector memory for future turns.
    try:
        # Create a text chunk representing the last turn
        conversation_chunk = f"User asked: '{latest_user_message}'\nAI responded: '{ai_answer}'"
        
        # Embed this new chunk
        new_embedding = embedding_model.encode([conversation_chunk]).tolist()
        
        # Generate a unique ID for this chat chunk
        # We can use the number of items already in the collection to ensure uniqueness
        new_chunk_id = f"chat_{collection.count()}"

        # Add it to the collection
        collection.add(
            embeddings=new_embedding,
            documents=[conversation_chunk],
            ids=[new_chunk_id]
        )
        print(f"📝 Memory updated for session '{request.session_id}': {collection.count()}, chunks with chunk ID '{new_chunk_id}'.")
        # log_collection_data(collection, request.session_id, stage="After Memory Update")
    
    except Exception as e:
        # If updating memory fails, we should still return the answer but log the error.
        # This is a non-critical part of the user experience.
        print(f"⚠️ Could not update memory for session '{request.session_id}': {e}")

    # --- 4. RETURN RESPONSE (Same as before) ---
    return ChatResponse(answer=ai_answer)
'''


# main.py

import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from fastapi.middleware.cors import CORSMiddleware

# --- 1. Initialize FastAPI App ---
app = FastAPI(title="RAG Interview Backend")

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-app.vercel.app"], # Add your deployed frontend URL here later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. Initialize In-Memory Vector Store ---
# We no longer load a heavy embedding model on startup.
chroma_client = chromadb.Client()
print("ChromaDB client initialized for in-memory storage.")

# --- 3. Define Pydantic Models for API Data Validation ---

class ResumeUploadRequest(BaseModel):
    resume_text: str

class SessionResponse(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    session_id: str
    chat_history: list[dict]
    api_key: str

class ChatResponse(BaseModel):
    answer: str

# --- 4. Create API Endpoints ---

@app.post("/start-session", response_model=SessionResponse)
def start_session_and_process_resume(request: ResumeUploadRequest):
    """
    Starts a new session, chunks the resume, gets embeddings via API,
    and stores them in a temporary in-memory vector store.
    """
    session_id = str(uuid.uuid4())
    collection = chroma_client.create_collection(name=session_id)
    
    # Split the resume text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(request.resume_text)

    # Use Google's API to get embeddings for the resume chunks
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=chunks,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings = result['embedding']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get resume embeddings from Google API. Error: {e}")

    # Store chunks and embeddings in the session's collection
    chunk_ids = [str(i) for i in range(len(chunks))]
    collection.add(embeddings=embeddings, documents=chunks, ids=chunk_ids)
    
    print(f"✅ Session '{session_id}' created. Resume processed into {len(chunks)} chunks.")
    return SessionResponse(session_id=session_id)


@app.post("/chat", response_model=ChatResponse)
def chat_with_rag(request: ChatRequest):
    """
    Handles a chat turn by performing RAG using the Google API for embeddings
    and Gemini for generation.
    """
    # 1. Configure APIs with the user-provided key for this request
    try:
        genai.configure(api_key=request.api_key)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid API Key provided. Error: {e}")

    # 2. Retrieve the session's vector collection
    try:
        collection = chroma_client.get_collection(name=request.session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new session.")

    # 3. Get the user's latest message and perform RAG retrieval
    if not request.chat_history or request.chat_history[-2]['role'] != 'user':
        raise HTTPException(status_code=400, detail="Invalid chat history: last message must be from the user.")
    
    latest_user_message = request.chat_history[-2]['parts'][0]['text']

    try:
        # Get embedding for the user's question
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=latest_user_message,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = [result['embedding']]

        # Find relevant documents in our vector store
        results = collection.query(query_embeddings=query_embedding, n_results=4)
        retrieved_chunks = results['documents'][0]
        retrieved_context = "\n\n---\n\n".join(retrieved_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed during RAG retrieval. Error: {e}")

    # 4. Construct the prompt and call the generative model
    prompt_template = f"""
You are a professional AI Interviewer. Your persona is strict but fair.
Base your responses on the provided resume facts and the conversation history.

**Relevant Facts (from Resume or past conversation):**
---
{retrieved_context}
---

**Full Conversation History:**
(You are the 'model', the candidate is the 'user')
{request.chat_history}

Based on the relevant facts and the full conversation history, provide your response.
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        gemini_response = model.generate_content(prompt_template)
        ai_answer = gemini_response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get response from Gemini. Error: {e}")

    # 5. Update the vector memory with the latest conversation turn
    try:
        conversation_chunk = f"User asked: '{latest_user_message}'\nAI responded: '{ai_answer}'"
        memory_result = genai.embed_content(
            model="models/text-embedding-004",
            content=conversation_chunk,
            task_type="RETRIEVAL_DOCUMENT"
        )
        new_embedding = [memory_result['embedding']]
        new_chunk_id = f"chat_{collection.count()}"
        collection.add(embeddings=new_embedding, documents=[conversation_chunk], ids=[new_chunk_id])
    except Exception as e:
        # This is a non-critical step, so we just log the error and continue.
        print(f"⚠️ Could not update memory for session '{request.session_id}': {e}")

    return ChatResponse(answer=ai_answer)
