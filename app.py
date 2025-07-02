from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import uvicorn
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional


from src.helper_text import process_question  # Text-only queries
from src.helper_image import process_medical_image  # Image queries

# Initialize FastAPI app
app = FastAPI(
    title="Medical Bot API with Image & Text Support", 
    description="A medical chatbot API using LangGraph workflow with RAG, supporting both text and image queries"
)

# Load environment variables
load_dotenv()

# Environment variables setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = "MedQuery_LangGraph"

# Set environment variables
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

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
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse)
async def chat_text(request: ChatRequest):
    """Handle text-only chat requests using text helper"""
    try:
        print(f"Text Input: {request.message}")
        
        # Use text helper for text-only queries
        response = process_question(request.message, llm)
        
        print(f"Text Response: {response}")
        return ChatResponse(answer=response)
    except Exception as e:
        print(f"Text Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing text request: {str(e)}")

@app.post("/chat/image")
async def chat_image(
    message: str = Form(...),
    image: UploadFile = File(...)
):
    """Handle image + text queries using image helper"""
    try:
        print(f"Image Input - Message: {message}, Image: {image.filename}")
        
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Check file size (max 10MB)
        if image.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
        
        # Use image helper for image queries
        response = process_medical_image(message, image.file, llm)
        
        print(f"Image Response: {response}")
        return JSONResponse(content={"answer": response})
    except Exception as e:
        print(f"Image Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image request: {str(e)}")

@app.post("/chat/mixed")
async def chat_mixed(
    message: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    """Handle mixed requests - route to appropriate helper based on image presence"""
    try:
        if image is not None:
            # If image is provided, use image helper
            print(f"Mixed Input (with image) - Message: {message}, Image: {image.filename}")
            
            # Validate image file
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            if image.size > 10 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
            
            response = process_medical_image(message, image.file, llm)
        else:
            # If no image, use text helper
            print(f"Mixed Input (text only) - Message: {message}")
            response = process_question(message, llm)
        
        print(f"Mixed Response: {response}")
        return JSONResponse(content={"answer": response})
    except Exception as e:
        print(f"Mixed Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing mixed request: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Medical Bot API with Image & Text Support is running"}

# API information endpoint
@app.get("/api/info")
async def api_info():
    """Get API information"""
    return {
        "name": "Medical Bot API with Image & Text Support",
        "version": "3.0.0",
        "description": "A medical chatbot API using LangGraph workflow with RAG, supporting both text and image queries",
        "model": "Google Gemini 2.5 Flash",
        "supported_inputs": [
            "Text-only medical questions",
            "Medical images with questions",
            "Mixed text and image queries"
        ],
        "workflow_features": {
            "text_workflow": [
                "Medical question validation",
                "Document retrieval from Pinecone",
                "Question rewriting for better search",
                "Web search integration with Tavily",
                "RAG-based answer generation"
            ],
            "image_workflow": [
                "Medical image validation",
                "Medical image analysis",
                "Question enhancement with image context",
                "Web search with image-enhanced queries",
                "RAG-based answer generation"
            ]
        },
        "endpoints": {
            "/": "Web interface",
            "/chat": "JSON text-only chat endpoint",
            "/chat/image": "Form-based image + text endpoint",
            "/chat/mixed": "Mixed endpoint (auto-routes based on image presence)",
            "/health": "Health check",
            "/api/info": "API information"
        },
        "supported_image_formats": ["JPEG", "PNG", "GIF", "WebP"],
        "max_image_size": "10MB"
    }

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )