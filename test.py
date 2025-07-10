from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import TypedDict
from utils.show_graph import show_mermaid, show_supervisor
import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")



class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:  # type: ignore
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]: # type: ignore
        return self.model.encode([query])[0].tolist()

embedding_model = CustomEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")


vectorstore = Chroma(
    collection_name="zus-products",
    embedding_function=embedding_model,
    persist_directory="./zus_products_vectorstore"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")





from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


math_agent = create_react_agent(
    model= llm,
    tools=[add, multiply, divide],
    prompt=(
        "You are a math agent.\n\n"
        "INSTRUCTIONS:\n"
        "- Assist ONLY with math-related tasks\n"
        "- After you're done with your tasks, respond to the supervisor directly\n"
        "- Respond ONLY with the results of your work, do NOT include ANY other text."
    ),
    name="math_agent",
)


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

tools = [retriever_tool]

def prompt(state, config):
    system_msg = (
        "You are a data retriver, you have a tool that retrives data from a database\n "
        "Never include data that is not presented in the information queried"
    )
    return [SystemMessage(content=system_msg)] + state["messages"]

# Create the agent with correct parameter name and prompt function
# customer_service_agent = create_react_agent(
#     model =llm,
#     tools=tools,
#     prompt=(
#         "You are a data retriver, you query a vector database with a question from the user. Do not make up any information \n"
#         "Only return the documents that are queried without any explainations"
#         # "You are a customer support agent for ZUS Coffee. Your job is to answer customer questions only"
#         # "based on the product information available in the vector store. You can provide additional "
#         # "information that is listed in the context given to you even when the question does not explicitly ask for it. "
#         # "But do not overload the customer with too much information. Emphasize key details like special offers or features. "
#         # "If the information is not available, you can say 'I don't know' or 'Not available'. "
#         # "Use the following product context to help answer the customer's question. Be concise, friendly, and helpful."
#         ),
#     name="customer_service_agent"
# )
customer_service_agent = create_react_agent(
    model =llm,
    tools=tools,
    prompt=prompt,
    name="customer_service_agent"
)

# for chunk in customer_service_agent.stream(
#     # {"messages": [{"role": "user", "content": "I am looking for a affordable mug with big capacity which one would you recommend, and how much would it cost if i were to buy 2 of them."}]}
#     {"messages": [{"role": "user", "content": "2+2*8"}]}

# ):
#     pretty_print_messages(chunk)




# Invoke the agent with user message
# response = customer_service_agent.invoke(
#     {"messages": [{"role": "user", "content": "What are the most expensive products?"}]}
# )

# print(response["messages"][-1].content)

# retrival_agent = create_react_agent(
#     model = llm,
#     tools = [retrieve_documents],
#     prompt=("You are a data retrieval agent. Your job is to retrieve relevant documents from database based on the user's question. Just return the retrived documents without any explaination. No not make any information up, just return the documents that are relevant to the question asked."),
#     name = "retrieval_agent"
# )


# customer_service_agent = create_react_agent(
#     model = llm,
#     tools = [generate_answer],
#     prompt=("You are a customer support agent for ZUS Coffee. Your job is to answer customer questions based on the product information available in the vector store. "
#             "You can provide additional information that is listed in the context given to you even when the question does not explicitly ask for it. But do not overload the customer with too much information. "
#             "Emphasize key details like special offers or features. If the information is not available, you can say 'I don't know' or 'Not available'. "
#             "Use the following product context to help answer the customer's question. Be concise, friendly, and helpful."),
#     name = "customer_service_agent"
# )




supervisor = create_supervisor(
    model=llm,
    agents=[math_agent, customer_service_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "- a customer service agent. Assign task when user inquires some information about products or the company\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself ."
    ),
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

# show_supervisor(supervisor)



for chunk in supervisor.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": " What is the total price of a RM 5 mug and two RM 7 mug",
            }
        ]
    },
):
    pretty_print_messages(chunk, last_message=True)

final_message_history = chunk["supervisor"]["messages"]


for message in final_message_history:
    message.pretty_print()


"""

so do not need seperate agent for generation


"""