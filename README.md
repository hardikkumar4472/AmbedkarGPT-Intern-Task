# AmbedkarGPT-Intern-Task

A simple RAG-based command-line Q&A system built as part of the **Kalpit Pvt Ltd UK – AI Intern Assignment**.

## Features
- Loads Ambedkar’s speech from `speech.txt`
- Splits text into chunks
- Creates embeddings using **sentence-transformers/all-MiniLM-L6-v2**
- Stores vectors locally using **ChromaDB**
- Uses **Ollama + Mistral 7B** as the local LLM
- Answers questions ONLY from the provided text

---

## Project Structure
