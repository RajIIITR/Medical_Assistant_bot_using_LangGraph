import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Set your API key as environment variable
DATA_PATH = "Data"  # Update this path as needed
INDEX_NAME = "medicalbot"

def load_pdf_file(data_path):
    """Extract data from PDF files in the specified directory."""
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def text_split(extracted_data):
    """Split documents into text chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def get_embeddings():
    """Initialize HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    return embeddings

def create_pinecone_index():
    """Create Pinecone index if it doesn't exist."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index already exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384, 
            metric="cosine", 
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-east-1"
            ) 
        )
        print(f"Created new index: {INDEX_NAME}")
    else:
        print(f"Index {INDEX_NAME} already exists")

def store_documents_to_pinecone():
    """Main function to process documents and store them in Pinecone."""
    try:
        # Load PDF files
        print("Loading PDF files...")
        extracted_data = load_pdf_file(DATA_PATH)
        print(f"Loaded {len(extracted_data)} documents")
        
        # Split into chunks
        print("Splitting documents into chunks...")
        text_chunks = text_split(extracted_data)
        print(f"Created {len(text_chunks)} text chunks")
        
        # Get embeddings
        print("Initializing embeddings...")
        embeddings = get_embeddings()
        
        # Create Pinecone index
        print("Creating/checking Pinecone index...")
        create_pinecone_index()
        
        # Store documents in Pinecone
        print("Storing documents in Pinecone...")
        docsearch = PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=INDEX_NAME,
            embedding=embeddings
        )
        
        print("Successfully stored all documents in Pinecone!")
        return True
        
    except Exception as e:
        print(f"Error storing documents: {str(e)}")
        return False

if __name__ == "__main__":
    if not PINECONE_API_KEY:
        print("Please set PINECONE_API_KEY environment variable")
    else:
        store_documents_to_pinecone()