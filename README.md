![AI chat diagram](/PDF-LangChain.jpg)
# 📚 Chat with your PDFs — Streamlit App
Interact with your PDF documents using AI!
Upload your PDFs and ask questions — get answers based on their content, powered by Hugging Face's bigscience/bloom-560m model and LangChain.

## 🚀 Features

📄 Upload one or multiple PDFs

🧠 Extracts and splits the content into smart text chunks

🔎 Embeds chunks using all-MiniLM-L6-v2

🤖 Uses HuggingFaceHub to answer your questions about your PDFs

🧵 Remembers previous conversation context (memory)

🖥️ Built with Streamlit for easy web deployment

## 🛠️ Tech Stack
Streamlit — Web UI

LangChain — LLMs, memory, retrieval

FAISS — Local vector database

Sentence Transformers — Text embeddings

Hugging Face Hub — Access to bigscience/bloom-560m

Python-dotenv — Load environment secrets

## 📦 Setup Instructions

### Clone this repository

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

### Install dependencies

pip install -r requirements.txt

### Set your Hugging Face API key

Create a .env file (or set it in Streamlit Cloud Secrets) and add:

HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

### Run the Streamlit app

streamlit run app.py

## 📚 How it works
Extracts all text from uploaded PDFs

Splits the text into manageable chunks

Embeds each chunk and stores in FAISS vectorstore

On user question, retrieves relevant chunks

Uses HuggingFaceHub model to generate a response

Displays chat history with user and bot templates

## ⚠️ Important Notes
Free Hugging Face models like bloom-560m may be slower than commercial APIs.

Streamlit free hosting limits resource usage; prefer smaller PDFs for better performance.

HuggingFaceHub token must be properly set for models requiring authentication.

## 🤝 Acknowledgements
LangChain

Hugging Face

Streamlit

Sentence Transformers
