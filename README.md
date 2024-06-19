# Medical-RAG-using-Bio-Mistral-7B

This project develops a sophisticated API for document ingestion and retrieval-augmented question answering, leveraging advanced natural language processing techniques. It consists of two main components:

## Document Ingestion System (`ingest.py`)
The ingestion module is responsible for loading, processing, and storing documents using the following tools:
- **SentenceTransformerEmbeddings**: Utilizes embeddings from the `langchain` library, specifically the "NeuML/pubmedbert-base-embeddings" model for biomedical texts.
- **DirectoryLoader**: Loads documents from directories, supporting various file types like PDFs, part of `langchain_community.document_loaders`.
- **RecursiveCharacterTextSplitter**: Splits large documents into manageable chunks, enhancing text processing efficiency.
- **Chroma**: A vector store for persisting and querying document embeddings for fast retrieval.

## API Service (`app.py`)
Built with FastAPI, the API service incorporates:
- **FastAPI**: Framework for building APIs with Python 3.7+.
- **Jinja2Templates**: Manages HTML templates for rendering user interfaces.
- **StaticFiles**: Serves static files like CSS and JavaScript.
- **LlamaCpp**: Executes large language models, used here in a retrieval-augmented setup.
- **RetrievalQA**: Combines language models and retrieval systems to provide contextually accurate answers.
- **PromptTemplate**: Structures prompts for querying language models.
- **Hugging Face Hub (hf_hub_download)**: Downloads necessary models from the Hugging Face Hub, specifically "MaziyarPanahi/BioMistral-7B-GGUF".
