from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional



app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI is working!"}

@app.get("/products")
def get_product_summary(query: str = Query(..., description="User question about ZUS products")):
    # 🔍 Pretend this is RAG / vector store logic
    mock_answer = f"This is a summary related to: '{query}'. The ZUS OG Cup is RM55 and comes in Thunder Blue."
    return {"query": query, "summary": mock_answer}
# uvicorn api_calls:app --reload

"""
Objective: Build and consume FastAPI endpoints for domain data and test them as your
external APIs.
1. Product-KB Retrieval Endpoint
○ Ingest ZUS product docs into a vector store (e.g., FAISS, Pinecone).
i. Source: https://shop.zuscoffee.com/ > Drinkware only
○ Expose /products?query=<user_question> that retrieves top-k and
returns an AI-generated summary.
2. Outlets Text2SQL Endpoint
○ Maintain a SQL DB of ZUS outlets (location, hours, services).
i. Source: https://zuscoffee.com/category/store/kuala-lumpur-selangor/
○ Expose /outlets?query=<nl_query> that translates to SQL, executes it,
and returns results.


so to my understanding, its still a function
how would the nodes be built?

so api calls as a tool


"""