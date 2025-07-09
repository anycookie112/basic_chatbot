import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from llm import llm_groq
from pprint import pprint
from prompts import prompt_scrapping, prompt_cs
from langchain_core.documents import Document
from langchain_chroma import Chroma
import numpy as np
from embedding_model import embedding_minilm

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

math_tools = MathTools()
agent_tools = [math_tools.add, math_tools.subtract, math_tools.multiply]


llm_with_tools = llm.bind_tools(agent_tools)

# response = llm_with_tools.invoke("What is 5 + 3? What is 10 - 2? What is 4 * 6?")


class RetrievalGeneration:
    def __init__(self, llm):
        self.retriever = None
        self.retrieved_docs = None
        self.embedding_model = None
        self.model = llm
        self.prompt = None
        self.url = None

    def load_documents(self, directory):
        persist_directory="./zus_products_vectorstore"
        url = self.url

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        # Get all visible text
        text = soup.get_text(separator="\n")

        # Clean and print non-empty lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        # print(clean_text)
        doc = [Document(
            page_content=clean_text,
            metadata={
                "source": "https://shop.zuscoffee.com/collections/drinkware",
                "category": "Drinkware"
            }
        )]
        prompt = self.prompt
        chain = prompt | llm

        response = chain.invoke({"doc": doc})


        # print(response)
        import json
        # json_str = 
        data = json.loads(response.content) 
        # print(data)

        docs = []
        for product in data:
            # Format document content
            content = f"""
            Product: {product['product_name']}
            Sale Price: {product['sale_price']}
            Regular Price: {product['regular_price']}
            Description: {product['description']}
            Colors: {", ".join(product['colors']) if product.get('colors') else "N/A"}
            Category: {product['category']}
            Brand: {product['brand']}
            """

            docs.append(Document(
                page_content=content.strip(),
                metadata={"source": product.get("source_url", "N/A")}
            ))
        

        # Insert data
        vectorstore = Chroma.from_documents(
        documents=docs,
        collection_name="zus-products",
        embedding= self.embedding_model,
        persist_directory= persist_directory
        )

        self.retriever = vectorstore.as_retriever()
        return self.retriever

    
    def get_documents(self, question, top_k):
        if not self.retriever:
            raise ValueError("Documents must be loaded before they can be searched")
        # get relevant docs
        self.retrieved_docs = self.retriever.invoke(question) or []  # Get initial retrieved docs
        # Re-rank and select top K
        ranked_docs = self.rerank_documents(question, top_k=top_k) if self.retrieved_docs else []
        formatted_docs = format_docs(ranked_docs) if ranked_docs else []
        return formatted_docs
    
    def load_existing_vectorstore(self, persist_directory="./zus_products_vectorstore"):
        vectorstore = Chroma(
            collection_name="zus-products",
            embedding_function = self.embedding_model,
            persist_directory=persist_directory
        )
        self.retriever = vectorstore.as_retriever()
        print("Vectorstore loaded and retriever initialized.")
    
    def rerank_documents(self, query, top_k=2):
        """Ranks documents based on cosine similarity to the query embedding."""
        if not self.embedding_model or hasattr(self.embedding_model, 'embed_query') is False:
            raise ValueError("Tokenizer must be loaded before configuring model kwargs")

        query_embedding = self.embedding_model.embed_query(query)  # Get query vector
        doc_embeddings = np.array(
            [self.embedding_model.embed_query(doc.page_content) for doc in self.retrieved_docs])  # Get doc vectors
        # Compute cosine similarity
        similarities = np.dot(doc_embeddings, query_embedding) / (
                np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        # Rank by similarity and select top_k
        ranked_indices = np.argsort(similarities)[::-1]  # Sort in descending order
        top_docs = [self.retrieved_docs[i] for i in ranked_indices[:top_k]]
        return top_docs
    



vectorstore = Chroma(
    collection_name="zus-products",
    embedding_function=embedding_model,
    persist_directory="./zus_products_vectorstore"
)

retriever = vectorstore.as_retriever()


# pprint(doc_splits)
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# prompt_cs = ChatPromptTemplate.from_template("""
# You are a customer support agent for ZUS Coffee. Your job is to answer customer questions based on the product information available in the vector store.
# You can provide additional information that is listed in the context given to you even when the question does not explicitly ask for it. But do not overload the customer with too much information.
# Emphasize key details like special offers or features.
# If the information is not available, you can say "I don't know" or "Not available
# Use the following product context to help answer the customer's question. Be concise, friendly, and helpful.

# Context:
# {context}

# Customer Question:
# {question}

# Answer:
# """)
docs_txt = format_docs(docs)
rag_chain = prompt_cs | llm | StrOutputParser()
generation = rag_chain.invoke({"context": docs_txt, "question": question})
print(generation)


class AnswerGeneration:
    def __init__ (self):
        self.prompt = None

    def generate_answer(self, docs):

        rag_chain = prompt_cs | llm | StrOutputParser()
        generation = rag_chain.invoke({"context": docs_txt, "question": question})



class LLMmanager:
    def __init__(self, llm):
        self.llm = llm
        self.embedding_model = embedding_model
        self.prompt = None

    def _create_prompt(self)
        return prompt_cs

persist_directory="./zus_products_vectorstore"
rag = RetrievalGeneration(llm=llm)
rag.prompt = prompt_scrapping
rag.url = "https://shop.zuscoffee.com/collections/drinkware"
rag.embedding_model = embedding_model
# rag.load_documents(directory=persist_directory)  


rag.load_existing_vectorstore()
docs = rag.get_documents("Most expensive products you have", top_k=3)
print(docs)

#     def retrieve(state, vectorstore):
#         """
#         Retrieve documents

#         Args:
#             state (dict): The current graph state

#         Returns:
#             state (dict): New key added to state, documents, that contains retrieved documents
#         """
#         print("---RETRIEVE---")
#         question = state["question"]

#         # Retrieval
#         retriever = vectorstore.as_retriever()
#         documents = retriever.invoke(question)
#         return {"documents": documents, "question": question}

#     def generate(state):
#         """
#         Generate answer

#         Args:
#             state (dict): The current graph state

#         Returns:
#             state (dict): New key added to state, generation, that contains LLM generation
#         """
#         print("---GENERATE---")
#         question = state["question"]
#         documents = state["documents"]

#         # RAG generation
#         docs_txt = format_docs(documents)
#         generation = rag_chain.invoke({"context": docs_txt, "question": question})
#         return {"documents": documents, "question": question, "generation": generation}


# def web_search(state):
#     """
#     Web search based on the re-phrased question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Updates documents key with appended web results
#     """

#     print("---WEB SEARCH---")
#     question = state["question"]

#     # Web search
#     docs = web_search_tool.invoke({"query": question})
#     web_results = "\n".join([d["content"] for d in docs])
#     web_results = Document(page_content=web_results)

#     return {"documents": web_results, "question": question}

















### Edges ###


# def route_question(state):
#     """
#     Route question to web search or RAG.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Next node to call
#     """

#     print("---ROUTE QUESTION---")
#     question = state["question"]
#     source = question_router.invoke({"question": question})
#     if source.datasource == "web_search":
#         print("---ROUTE QUESTION TO WEB SEARCH---")
#         return "web_search"
#     elif source.datasource == "vectorstore":
#         print("---ROUTE QUESTION TO RAG---")
#         return "vectorstore"
    


# from pydantic import BaseModel, Field


# class Plan(BaseModel):
#     """Plan to follow in future"""

#     steps: List[str] = Field(
#         description="different steps to follow, should be in sorted order"
#     )

# from langchain_core.prompts import ChatPromptTemplate

# planner_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             """For the given objective, come up with a simple step by step plan. \
# This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
# The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
#         ),
#         ("placeholder", "{messages}"),
#     ]
# )
# planner = planner_prompt | llm(
#     model="gpt-4o", temperature=0
# ).with_structured_output(Plan)


# from langgraph.graph import MessagesState
# from langchain_core.messages import HumanMessage, SystemMessage

# # System message
# sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

# # Node
# def assistant(state: MessagesState):
#    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}



# from langgraph.graph import START, StateGraph
# from langgraph.prebuilt import tools_condition
# from langgraph.prebuilt import ToolNode
# from IPython.display import Image, display
# from utils.show_graph import show_mermaid


# """

# so we bind the llm to the tools, this only tells the llm what tools it can use, it does not execute them
# so we need to create a node that will execute the tools when called
# binding them its not enough, we also need to link it to a tool node and input the tools into the node as well

# """
# builder = StateGraph(MessagesState)

# builder.add_node( "assistant", assistant)
# builder.add_node("tools", ToolNode(agent_tools))


# builder.add_edge(START, "assistant")
# builder.add_conditional_edges(
#     "assistant",
#     # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
#     # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
#     tools_condition,
# )
# builder.add_edge("tools", "assistant")
# react_graph = builder.compile()

# # Show
# show_mermaid(react_graph)





