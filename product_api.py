import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from utils.llm import llm_groq
from pprint import pprint
from utils.prompts import *
from langchain_core.documents import Document
from langchain_chroma import Chroma
import numpy as np
from utils.embedding_model import embedding_minilm
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import convert_to_messages
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool


llm = llm_groq()
embedding_model = embedding_minilm()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class MathTools:
    def __init__(self):
        pass

    @tool
    def add(self, a: float, b: float) -> float:
        """Adds two numbers"""
        return a + b

    @tool
    def subtract(self, a: float, b: float) -> float:
        """Subtracts two numbers"""
        return a - b

    @tool
    def multiply(self, a: float, b: float) -> float:
        """Multiplies two numbers"""
        return a * b


vectorstore = Chroma(
    collection_name="zus-products",
    embedding_function=embedding_model,
    persist_directory="./zus_products_vectorstore"
)

retriever = vectorstore.as_retriever()


def retriever_tool(message):
    retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_product_information",
    "Search and return information about ZUS coffee shop products.",
)
    results = retriever_tool.invoke({"query": message})

    return results
