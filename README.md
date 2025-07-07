# 🧠 MedQueryAI: Advanced Medical Knowledge Assistant

MedQueryAI is a multi-modal medical AI system designed to deliver accurate and insightful responses to both text-based and image-based medical queries. By combining Retrieval-Augmented Generation (RAG), Hugging Face embeddings, Pinecone vector search, Tavily web search, and Google Gemini 2.5 Flash, MedQueryAI offers robust, context-aware responses for healthcare applications.

## 🔧 Features

- Multi-modal support for text and image queries
- RAG-based architecture with internal and external context retrieval
- Integration with Google Gemini 2.5 Flash for medical reasoning and generation
- Real-time web augmentation using Tavily Search API
- Image understanding via a custom `Get_Image_Info` pipeline
- Scalable deployment using FastAPI, Docker, and GitHub Actions

## 🧠 How It Works

1. **Grading Phase**:
   - Text queries pass through `grade_question`
   - Image inputs go through `grade_image`
   - Invalid or low-quality inputs are filtered via the `reject` node

2. **Retrieval Phase**:
   - Valid inputs are routed to either:
     - `retrieve` node for semantic vector search using Hugging Face and Pinecone (text)
     - `Get_Image_Info` node for extracting clinical features from image input

3. **Transformation Phase**:
   - Questions are reformatted using `transform_question` to match the prompt expectations of the LLM

4. **Web Search Phase**:
   - A `web_search_node` integrates external medical knowledge using the Tavily toolkit

5. **Generation Phase**:
   - A context-rich response is generated using Google Gemini 2.5 Flash
   - The response is returned via the `generate` node

## 📊 Workflow Diagrams

### 🔤 Text-Only Query Pipeline

```mermaid
flowchart TD
    A[__start__] --> B[grade_question]
    B -->|valid| C[retrieve] --> D[transform_question] --> E[web_search_node] --> F[generate]
    B -->|invalid| G[reject] --> F
    F --> H[__end__]


