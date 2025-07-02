from typing import List
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
import base64
from PIL import Image as PILImage
import io
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph, START

# Configuration
INDEX_NAME = "medicalbot"

class GradeImage(BaseModel):
    """Binary score for relevance check on Image whether it is valid medical related Image or Not."""
    
    binary_score: str = Field(
        description="Image is relevant to medical aspects, 'yes' or 'no'"
    )

class State(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        image_info: str
        Image_Valid: str
        documents: list of documents
        image_data: base64 encoded image data
    """
    question: str
    generation: str
    image_info: str
    Image_Valid: str
    documents: List[str]
    image_data: str

def preprocess_image(image_file, max_size: tuple = (512, 512)) -> PILImage:
    """Preprocess image for model input"""
    
    image = PILImage.open(image_file)
    
    image.thumbnail(max_size, PILImage.LANCZOS)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

def image_to_base64(image: PILImage) -> str:
    """Convert PIL Image to base64 encoded string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()

def encode_image_to_base64(image_file):
    """Encode uploaded image file to base64 string for Gemini model input"""
    
    # Preprocess the image
    processed_image = preprocess_image(image_file)  
    
    # Convert to base64
    base64_image = image_to_base64(processed_image)
    
    return base64_image

def get_embeddings():
    """Initialize HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    return embeddings

def get_retriever():
    """Initialize retriever from existing Pinecone index."""
    embeddings = get_embeddings()
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )
    return docsearch.as_retriever()

def format_docs(docs):
    """Format documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_image_grader(llm):
    """Create image grader for medical relevance"""
    structured_llm_grader = llm.with_structured_output(GradeImage)
    
    # System prompt
    system = """You are a grader assessing relevance of an input image to medical aspects. 
    Analyze the image content and determine if it contains medical-related elements such as:
    - Medical equipment, scans, anatomy, symptoms, conditions
    - Hospital/clinic environments, medical procedures
    - Medications or medical supplies
    
    Give a binary score 'yes' or 'no' to indicate whether the image is relevant to medical aspects."""
    
    # Create prompt with image format
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", [
            {
                "type": "text",
                "text": "Analyze this image for medical relevance:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,{image_data}"
                }
            }
        ])
    ])
    
    return grade_prompt | structured_llm_grader

def grade_medical_image(state, llm):
    """Grade whether an image is medical-related or not"""
    
    # Create the grader chain
    retrieval_grader = create_image_grader(llm)
    
    Image_Valid = "Yes"

    # Invoke with image data
    score = retrieval_grader.invoke({"image_data": state["image_data"]})
    grade = score.binary_score
    
    if grade != "yes":
        Image_Valid = "No"
        
    return {"Image_Valid": Image_Valid}

def decide_to_generate(state):
    if state.get("Image_Valid") == "Yes":
        return "Get_Image_Info"
    else:
        return "reject"

def reject_question(state):
    return {
        "question": state["question"],
        "documents": [],
        "generation": "Given Image Doesn't belong to Medical Domain"
    }

def analyze_medical_image(state, llm):
    """
    Analyze medical image and store detailed information in image_info string.
    
    Args:
        state: Current state of the workflow
        llm: Language model with vision capabilities
    
    Returns:
        Updated state with image_info populated
    """
    
    # Medical analysis prompt
    medical_analysis_prompt = """You are an expert medical image analyst. Analyze this medical image and provide detailed information about:

1. Disease/Condition: What disease, condition, or medical finding is depicted?
2. Medical Terminology: List relevant medical terms and terminology
3. Anatomical Location: What body part or anatomical structure is affected?
4. Severity Assessment: Assess the severity or stage if determinable
5. Additional Findings: Any other medical observations

Provide a short analysis suitable for medical evaluation. Format your response in an optimal way."""

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", medical_analysis_prompt),
        ("human", [
            {
                "type": "text", 
                "text": "Please analyze this medical image and provide detailed medical information:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,{image_data}"
                }
            }
        ])
    ])
    
    # Create chain with StrOutputParser
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"image_data": state["image_data"]})

    # Store the analysis in image_info as string
    state["image_info"] = response
    
    return state

def create_question_rewriter(llm):
    """Create question rewriter."""
    system = """You a question re-writer that converts an input question and given image-info and getting key understanding of the image, converts an input question to a better version that is optimized \n 
         for web search. Give only 1 optimized question. Look at the input and try to reason about the underlying semantic intent / meaning."""
         
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    
    return re_write_prompt | llm | StrOutputParser()

def transform_question(state, question_rewriter):
    """
    Transform the query to produce a better question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrase question
    """
    question = state["question"] + state["image_info"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    # If the result is a dict, extract the actual question string
    if isinstance(better_question, dict):
        better_question = better_question.get("question", "")
    
    return {"question": better_question}

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """
    web_search_tool = TavilySearchResults(k=5)
    
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    
    web_results = "\n".join([d["content"] for d in docs])
    
    web_results = Document(page_content=web_results)
    
    documents.append(web_results)

    return {"documents": documents, "question": question}

def generate(state, llm):
    """
    Generate answer using RAG chain and web search

    Args:
        state (dict): The current graph state with 'question' and 'documents'
        llm: Language model instance
    
    Returns:
        state (dict): Updated state with 'generation' added
    """
    question = state["question"]
    documents = state["documents"]  # This contains web search results
    
    if documents is None or len(documents) == 0:
        if "generation" in state:
            return {
                "question": question,
                "documents": documents,
                "generation": state["generation"]  
            }
        else:
            return {
                "question": question,
                "documents": documents,
                "generation": "Error: No documents were retrieved before generation."
            }

    # Combine web search results with vector store retrieval
    retriever = get_retriever()
    vector_docs = retriever.invoke(question)  # Get from Pinecone
    
    # Combine both sources
    all_docs = documents + vector_docs  # Web search + Vector store
    
    # Format all documents
    context = format_docs(all_docs)
    
    # Create prompt manually since we're combining sources
    final_prompt = f"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    Question: {question}
    Context: {context}
    Answer:"""
    
    # Generate response
    generation = llm.invoke(final_prompt).content
    
    return {
        "documents": all_docs,
        "generation": generation,
        "question": question
    }

def create_workflow(llm):
    """Create and compile the workflow graph."""
    # Initialize components
    question_rewriter = create_question_rewriter(llm)
    
    # Create workflow
    workflow = StateGraph(State)
    
    # Define nodes with closures to pass dependencies
    workflow.add_node("Get_Image_Info", lambda state: analyze_medical_image(state, llm))
    workflow.add_node("grade_image", lambda state: grade_medical_image(state, llm))
    workflow.add_node("generate", lambda state: generate(state, llm))
    workflow.add_node("transform_question", lambda state: transform_question(state, question_rewriter))
    workflow.add_node("web_search_node", web_search)
    workflow.add_node("reject", reject_question)
    
    # Build graph
    workflow.add_edge(START, "grade_image")

    workflow.add_conditional_edges(
        "grade_image",
        decide_to_generate,
        {
            "Get_Image_Info": "Get_Image_Info",
            "reject": "reject"
        }
    )
    
    workflow.add_edge("reject", "generate")
    workflow.add_edge("Get_Image_Info", "transform_question")
    workflow.add_edge("transform_question", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def process_medical_image(question: str, image_file, llm):
    """
    Process a medical image question through the workflow.
    
    Args:
        question: User's question about the image
        image_file: Uploaded image file (FastAPI UploadFile)
        llm: Language model instance
    
    Returns:
        Generated answer string
    """
    # Encode image to base64
    image_data = encode_image_to_base64(image_file)
    
    # Create workflow
    app = create_workflow(llm)
    
    # Prepare inputs
    inputs = {
        "question": question,
        "image_data": image_data,
        "documents": [],
        "generation": "",
        "image_info": "",
        "Image_Valid": ""
    }
    
    # Run the workflow
    result = None
    for output in app.stream(inputs):
        for key, value in output.items():
            result = value
    
    return result.get("generation", "No answer generated")