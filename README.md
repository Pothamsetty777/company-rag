# Company Policy RAG Chatbot

A free Retrieval-Augmented Generation (RAG) chatbot that helps users ask questions from company policy documents. This project uses local embeddings and vector search, so it does not require an OpenAI API key or paid API service.

## Project Overview

The Company Policy RAG Chatbot allows users to search company policy information such as leave policy, sick leave, remote work rules, reimbursement process, and employee benefits.

The app reads policy content, splits it into smaller text chunks, converts those chunks into embeddings, stores them in a FAISS vector index, and retrieves the most relevant policy section based on the user's question.

## Features

- Ask questions from company policy documents
- Use a sample company policy document
- Upload PDF policy documents
- Free local embedding model using Sentence Transformers
- FAISS vector search for relevant source matching
- Source match display with similarity score
- Attractive Streamlit frontend
- No OpenAI API key required
- No paid API required

## Tech Stack

- Python
- Streamlit
- Sentence Transformers
- FAISS
- PyPDF
- NumPy

## How RAG Works In This Project

1. The policy document is loaded from a sample text file or uploaded PDF.
2. The document is split into smaller chunks.
3. Each chunk is converted into an embedding using Sentence Transformers.
4. FAISS stores the embeddings for fast similarity search.
5. When the user asks a question, the question is also converted into an embedding.
6. FAISS retrieves the most relevant policy chunks.
7. The app displays the best answer and source matches.

## Example Questions

- What is the vacation policy?
- How many sick leave days are allowed?
- What is the remote work policy?
- What are the core business hours for remote employees?
- What is the expense reimbursement policy?
- Are receipts required for reimbursement?
- What benefits are available for full-time employees?

## Run Locally

Clone the repository:

```bash
git clone https://github.com/your-username/company-policy-rag-chatbot.git
cd company-policy-rag-chatbot
```

Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Open the local URL in your browser:

```text
http://localhost:8501
```

## Project Structure

```text
company-policy-rag-chatbot/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── docs/
    └── sample_policy.txt
```

## Resume Bullet

Built a free Company Policy RAG Chatbot using Python, Streamlit, Sentence Transformers, and FAISS to retrieve accurate answers from HR and policy documents with source-based matches.

## Author

Rishi Pothamsetty