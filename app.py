from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import uvicorn
from langchain_google_genai import ChatGoogleGenerativeAI

# Import your helper function (make sure this import works)
try:
    from helper import process_question  # Direct import since it's in same directory
except ImportError:
    try:
        from src.helper import process_question  # If it's in src folder
    except ImportError:
        print("Error: Could not import process_question. Make sure helper.py is in the correct location.")
        raise

# Initialize FastAPI app
app = FastAPI(title="Medical Bot API with LangGraph", description="A medical chatbot API using LangGraph workflow with RAG")

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests with JSON body using LangGraph workflow"""
    try:
        print(f"Input: {request.message}")
        
        # Use the LangGraph workflow to process the question
        response = process_question(request.message, llm)
        
        print(f"Response: {response}")
        return ChatResponse(answer=response)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Medical Bot API with LangGraph is running"}

# Additional endpoints for better API functionality
@app.get("/api/info")
async def api_info():
    """Get API information"""
    return {
        "name": "Medical Bot API with LangGraph",
        "version": "2.0.0",
        "description": "A medical chatbot API using LangGraph workflow with RAG, question grading, and web search",
        "model": "Google Gemini 2.0 Flash",
        "workflow_features": [
            "Medical question validation",
            "Document retrieval from Pinecone",
            "Question rewriting for better search",
            "Web search integration with Tavily",
            "RAG-based answer generation"
        ],
        "endpoints": {
            "/": "Web interface",
            "/chat": "JSON chat endpoint (recommended)",
            "/health": "Health check",
            "/api/info": "API information"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )