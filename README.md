# ☕ ZUS Coffee Chatbot API (LLM + FastAPI)

A chatbot API that answers questions about ZUS Coffee's products and outlets using RAG and Text2SQL, powered by LangChain and FastAPI.

---

## 🚀 Features

- 🔍 Product question answering with vector search (Chroma)
- 🧠 Outlet info via SQL + LLM Text2SQL
- ⚡ Streaming chat endpoint using Server-Sent Events (SSE)
- 🤖 Tool-calling with LangChain or LangGraph agents

---

## 🛠️ Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/zus-chatbot

# Go into the folder
cd zus-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the FastAPI server
uvicorn main:app --reload
