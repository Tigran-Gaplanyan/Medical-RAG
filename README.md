# Medical-RAG-using-Bio-Mistral-7B
This project develops a sophisticated API for document ingestion and retrieval-augmented question answering, leveraging advanced natural language processing techniques. It consists of two main components:

The ingestion module is responsible for loading, processing, and storing documents. It utilizes:

SentenceTransformerEmbeddings from the langchain library for embedding generation, specifically using the "NeuML/pubmedbert-base-embeddings" model suited for biomedical texts.
DirectoryLoader to load documents from directories with support for various file types, including PDFs.
Text splitting functionality to divide large documents into manageable chunks.
Chroma vector store for persisting document embeddings, facilitating efficient similarity searches.

The API service is built using FastAPI and provides:

An HTML interface for user interactions.
A retrieval-augmented question answering (QA) system powered by LlamaCpp and RetrievalQA from langchain, with the model sourced from "MaziyarPanahi/BioMistral-7B-GGUF".
A retrieval system that uses the Chroma vector store to fetch relevant documents based on the embeddings.
An endpoint that accepts user queries and returns detailed, contextually accurate responses, including the source document and specific excerpts.
