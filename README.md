# Medical-RAG-using-Bio-Mistral-7B

This project develops a sophisticated API for document ingestion and retrieval-augmented question answering, leveraging advanced natural language processing techniques. It consists of two main components:

## Document Ingestion System
The ingestion module is responsible for loading, processing, and storing documents. It utilizes:
- **SentenceTransformerEmbeddings** from the langchain library for embedding generation, specifically using the "NeuML/pubmedbert-base-embeddings" model suited for biomedical texts.
- **DirectoryLoader** to load documents from directories with support for various file types, including PDFs.
- **Text Splitting Functionality** to divide large documents into manageable chunks.
- **Chroma Vector Store** for persisting document embeddings, facilitating efficient similarity searches.

## API Service
The API service is built using FastAPI and provides:
- **HTML Interface** for user interactions.
- **Retrieval-Augmented Question Answering (QA) System** powered by LlamaCpp and RetrievalQA from langchain, with the model sourced from "MaziyarPanahi/BioMistral-7B-GGUF".
- **Retrieval System** that uses the Chroma vector store to fetch relevant documents based on the embeddings.
- **Endpoint** that accepts user queries and returns detailed, contextually accurate responses, including the source document and specific excerpts.
