from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


def llm_groq():
    llm = ChatGroq(model="llama-3.3-70b-versatile")
    return llm


def llm_qwen():
    llm = ChatGroq(model="qwen/qwen3-32b")
    return llm

def llm_qwen2():
    llm = ChatGroq(model="qwen-qwq-32b")
    return llm


def llm_meta():
    llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")
    return llm
