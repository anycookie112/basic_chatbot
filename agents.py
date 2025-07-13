from prompts import *
from utils.printing import *
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

# Create your LLM instance
llm = ChatOllama(model="qwen2.5:14b")



class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:  # type: ignore
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]: # type: ignore
        return self.model.encode([query])[0].tolist()

embedding_model = CustomEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



vectorstore = Chroma(
    collection_name="zus-products",
    embedding_function=embedding_model,
    persist_directory="./zus_products_vectorstore"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)




def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

def retrieve_documents(question: str) -> str:
    """Return revelevant documents for a given question"""
    docs = vectorstore.as_retriever().invoke(question)
    return format_docs(docs)

retriever = vectorstore.as_retriever()
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_product_information",
    "Search and return information about ZUS coffee shop products.",
)


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


def prompt(state, config):
    system_msg = (
        "You are a data retriver, you have a tool that retrives data from a database\n "
        "Never include data that is not presented in the information queried"
    )
    return [SystemMessage(content=system_msg)] + state["messages"]


# Create React agents (these are already properly structured)
math_agent = create_react_agent(
    model=llm,
    tools=[add, multiply, divide],
    prompt="You are a math agent. Solve mathematical problems and calculations.",
    name="math_agent"
)

customer_service_agent = create_react_agent(
    model=llm,
    tools=[retriever_tool],  # Your customer service tools
    prompt=prompt,
    name="customer_service_agent"
)

def create_super():
    return create_supervisor(
        model=llm,
        agents=[math_agent, customer_service_agent],
        prompt=(
            "You are a supervisor managing two agents for ZUS Coffee. Zus coffee is a company that sells coffee, but they also sell drinkwares:\n"
            "- a math agent. Assign math-related tasks to this agent\n"
            "- a customer service agent. Assign task when user inquires some information about products or the company\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "Do not do any work yourself."
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()



# supervisor = create_super()

# messages = [HumanMessage(content="Hi what mugs do you offer can you show me some and the price of them?")]
# messages = supervisor.invoke({"messages": messages})
# # print(messages)

# last_message = messages["messages"][-1]  # get the last message
# print(last_message.content)
