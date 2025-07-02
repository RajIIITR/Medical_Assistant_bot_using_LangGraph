import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document


# Configuration
INDEX_NAME = "medicalbot"

class GradeQuestion(BaseModel):
    """Binary score for relevance check on Question given by user whether is it valid medical related Question or Not."""
    binary_score: str = Field(
        description="Question are relevant to medical, 'yes' or 'no'"
    )

class State(TypedDict):
    """Represents the state of our graph."""
    question: str
    generation: str
    Question_Valid: str
    documents: List[str]

def get_embeddings():
    """Initialize HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    return embeddings

def get_retriever(llm=None):
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

def create_rag_chain(llm):
    """Create RAG chain."""
    retriever = get_retriever()
    prompt = get_prompt()  
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def create_question_grader(llm):
    """Create question grader for medical relevance."""
    structured_llm_grader = llm.with_structured_output(GradeQuestion)
    
    system = """You are a grader assessing relevance of a user question. \n 
        If the user question contains keyword(s) or semantic meaning related to the medical aspects, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the question is relevant to medical aspects."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "User question: {question}"),
    ])
    
    return grade_prompt | structured_llm_grader

def create_question_rewriter(llm):
    """Create question rewriter."""
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
         for web search. Give only 1 optimized question. Look at the input and try to reason about the underlying semantic intent / meaning."""
    
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    
    return re_write_prompt | llm | StrOutputParser()

def retrieve(state):
    """Retrieve documents."""
    question = state["question"]
    retriever = get_retriever()
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}

def grade_question(state, retrieval_grader):
    """Determine whether the user question is relevant to medical domain."""
    question = state["question"]
    Question_Valid = "Yes"
    
    score = retrieval_grader.invoke({"question": question})
    grade = score.binary_score
    
    if grade != "yes":
        Question_Valid = "No"
    
    return {"question": question, "Question_Valid": Question_Valid}

def decide_to_generate(state):
    """Decide whether to proceed with generation or reject."""
    if state.get("Question_Valid") == "Yes":
        return "retrieve"
    else:
        return "reject"

def reject_question(state):
    """Handle rejection of non-medical questions."""
    return {
        "question": state["question"],
        "documents": [],
        "generation": "Given Question Doesn't belong to Medical Domain"
    }

def web_search(state):
    web_search_tool = TavilySearchResults(k=5)
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    
    web_results = "\n".join([d["content"] for d in docs])
    
    web_results = Document(page_content=web_results)
    
    documents.append(web_results)

    return {"documents": documents, "question": question}

def generate(state, llm, prompt):
    """
    Generate answer using both RAG documents and web search results

    Args:
        state (dict): The current graph state with 'question' and 'documents'
        llm: The language model instance
        prompt: The RAG prompt template
    
    Returns:
        state (dict): Updated state with 'generation' added
    """
    print("---GENERATE---")

    question = state["question"]
    documents = state["documents"]

    # Check if documents is None or empty
    if documents is None or len(documents) == 0:
        # If there's already a generation (from reject), return it
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

    # Format documents from state (includes both RAG and web search results)
    formatted_context = format_docs(documents)
    
    # Create a chain that uses the documents from state instead of retriever
    generation_chain = (
        prompt
        | llm
        | StrOutputParser()
    )
    
    # Generate answer using both RAG and web search context
    generation = generation_chain.invoke({
        "context": formatted_context,
        "question": question
    })

    print(f"---USING CONTEXT FROM {len(documents)} DOCUMENTS---")
    
    return {
        "documents": documents,
        "generation": generation,
        "question": question
    }
def get_prompt():
    """Get the RAG prompt for generation."""
    return hub.pull("rlm/rag-prompt")

def transform_question(state, question_rewriter):
    """Transform the query to produce a better question."""
    question = state["question"]
    documents = state["documents"]
    
    better_question = question_rewriter.invoke({"question": question})
    if isinstance(better_question, dict):
        better_question = better_question.get("question", "")
    
    return {"documents": documents, "question": better_question}

def create_workflow(llm):
    """Create and compile the workflow graph."""
    # Initialize components
    rag_chain = create_rag_chain(llm)
    retrieval_grader = create_question_grader(llm)
    question_rewriter = create_question_rewriter(llm)
    prompt = get_prompt()  # ADD THIS LINE
    
    # Create workflow
    workflow = StateGraph(State)
    
    # Define nodes with closures to pass dependencies
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_question", lambda state: grade_question(state, retrieval_grader))
    workflow.add_node("generate", lambda state: generate(state, llm, prompt))  # MODIFY THIS LINE
    workflow.add_node("transform_question", lambda state: transform_question(state, question_rewriter))
    workflow.add_node("web_search_node", lambda state: web_search(state, )) # web_search
    workflow.add_node("reject", reject_question)
    
    # Build graph
    workflow.add_edge(START, "grade_question")
    
    workflow.add_conditional_edges(
        "grade_question",
        decide_to_generate,
        {
            "retrieve": "retrieve",
            "reject": "reject"
        }
    )
    
    workflow.add_edge("reject", "generate")
    workflow.add_edge("retrieve", "transform_question")
    workflow.add_edge("transform_question", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def process_question(question: str, llm):
    """Process a single question through the workflow."""
    app = create_workflow(llm)
    inputs = {"question": question}
    
    # Run the workflow
    result = None
    for output in app.stream(inputs):
        for key, value in output.items():
            result = value
    
    return result.get("generation", "No answer generated")