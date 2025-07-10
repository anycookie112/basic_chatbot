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
from rich import print
load_dotenv()
from langgraph.prebuilt import ToolNode


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

retriever = vectorstore.as_retriever()
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_product_information",
    "Search and return information about ZUS coffee shop products.",
)


def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b

tools = [retriever_tool]

# results = retriever_tool.invoke({"query": "Mugs"})
# print(results)

from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        llm
        # highlight-next-line
        .bind_tools(tools).invoke(state["messages"])
    )
    return {"messages": [response]}
from langchain_core.tools import tool



GENERATE_PROMPT = (
    "You are a customer service agent. "
    "Use the following pieces of retrieved context to answer the question. "
    "Do not make up any data, If you don't know the answer, just say that you don't know. Only use data given to you from the database"
    "Be concise, friendly and helpful. Emphasize key details like special offers or features."
    "Try to recommend products that are similar to the question asked.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}



input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What are you most expensive mugs?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_product_information",
                        "args": {"query": "most expensive products"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": """   
                    Product: ZUS OG CUP 2.0 With Screw-On Lid
                                Sale Price: RM55.00
                                Regular Price: RM79.00
                                Description: A high capacity drinkware with screw-on lid, made of durable material, affordable
                                Colors: Thunder Blue, Space Black, Lucky Pink
                                Category: Drinkware
                                Brand: ZUS

                    Product: ZUS All Day Cup - Mountain Collection
                                Sale Price: RM79.00
                                Regular Price: None
                                Description: A high capacity cup, expensive
                                Colors: Soft Fern, Pine Green, Terrain Green, Forest Green
                                Category: Drinkware
                                Brand: ZUS
                                
                                """,
                "tool_call_id": "1",
            },
        ]
    )
}

# response = generate_answer(input)
# response["messages"][-1].pretty_print()

from pydantic import BaseModel, Field
from typing import Literal

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


# highlight-next-line
class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


grader_model = llm


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        # highlight-next-line
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
    
REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": response.content}]}


from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

workflow = StateGraph(MessagesState)
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
# workflow.add_node("calculator", calculator_node)
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START,"generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END
    },
)

workflow.add_conditional_edges("retrieve", grade_documents,)
workflow.add_conditional_edges("calculator", grade_documents,)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()
#################################################3
# workflow.add_node(generate_query_or_respond)
# workflow.add_node("retrieve", ToolNode([retriever_tool]))
# workflow.add_node(rewrite_question)
# workflow.add_node(generate_answer)

# workflow.add_edge(START, "generate_query_or_respond")
# workflow.add_conditional_edges(
#     "generate_query_or_respond",
#     tools_condition,
#     {
#         "tools": "retrieve",
#         END: END,
#     },
# )

# workflow.add_conditional_edges("retrieve",grade_documents,)
# workflow.add_edge("generate_answer", END)
# workflow.add_edge("rewrite_question", "generate_query_or_respond")

# # Compile
# graph = workflow.compile()
from utils.show_graph import show_mermaid

show_mermaid(graph)

# for chunk in graph.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "What affordable colourful cups can you recommend me? I also have a Rm 100 budget.",
#             }
#         ]
#     }
# ):
#     for node, update in chunk.items():
#         print("Update from node", node)
#         update["messages"][-1].pretty_print()
#         print("\n\n")