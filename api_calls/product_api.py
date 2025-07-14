from utils.llm import llm_groq
from utils.prompts import *
from langchain_chroma import Chroma
from utils.embedding_model import embedding_minilm
from langchain.tools.retriever import create_retriever_tool


llm = llm_groq()
embedding_model = embedding_minilm()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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
