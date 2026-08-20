# 📚 PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions based on their content.

## 🚀 Features

- Upload PDF documents
- Automatically extract PDF text
- Split documents into chunks
- Generate embeddings using Hugging Face
- Store embeddings in ChromaDB
- Retrieve relevant document chunks
- Generate grounded answers using Google Gemini
- Display source file names and page numbers
- Prevent duplicate PDF processing during a session
- Streamlit web interface
- Graceful Gemini API quota handling

## 🛠️ Tech Stack

- Python
- LangChain
- Google Gemini
- Hugging Face Sentence Transformers
- ChromaDB
- Streamlit
- PyPDF
- Git / GitHub

## 🧠 RAG Architecture

User Uploads PDF

↓

PDF Loader

↓

Text Chunking

↓

Hugging Face Embeddings

↓

ChromaDB Vector Store

↓

User Question

↓

Similarity Search

↓

Relevant Document Chunks

↓

Gemini LLM

↓

Grounded Answer + Sources

## 📂 Project Structure

```text
basic-rag-streamlit/
│
├── app.py
├── data/
├── chroma_db/
├── notebook/
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── .gitignore
└── README.md
