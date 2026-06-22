# RAG
This project is a **Retrieval-Augmented Generation (RAG) system** built using deep learning techniques. It allows users to query PDF documents using natural language by combining vector search (ChromaDB) with a large language model.

The system:
- Converts documents into embeddings
- Stores them in a vector database
- Retrieves relevant context for user queries
- Generates responses using an LLM

## 📁 Project Structure
    project/
    │
    ├── populate_database.py   
    ├── get_embedding_function.py 
    ├── query_data.py 
    ├── app.py 
    │
    ├── chroma/ 
    ├── raw_data/
    └── requirements.txt


## 📄 Files Description

| File | Description |
|------|-------------|
| `get_embedding_function.py` | Configures embedding model for converting text into vector representations |
| `populate_database.py` | Loads PDFs, chunks text, generates embeddings, and stores them in ChromaDB |
| `query_data.py` | Handles retrieval + LLM response generation |
| `app.py` | Streamlit UI for chatbot interaction |
| `requirements.txt` | Project dependencies |

## 📂 Folders Description

| Folder | Description |
|--------|-------------|
| `chroma/` | Stores vector database files (ChromaDB) |
| `raw_data/` | Contains PDF files used as knowledge base |

## 🚀 How to Run the Project
Follow these steps to set up and run the Logistics RAG Assistant on your local machine.


### 1. Setup Environment
First, ensure you have Python installed. Then, create a virtual environment to keep your dependencies isolated.
```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
venv\Scripts\activate # Windows

source venv/bin/activate #Mac/Linux:

# Install required dependencies
pip install -r requirements.txt
```

## 2. Run the Application
Launch the Streamlit interface to start interacting
```bash
streamlit run app.py
```
<p align="center">
  <img src="logos\icon.png" width="200"/>
</p>