import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from pprint import pprint
from llm_list import llm_groq
from langchain.prompts import ChatPromptTemplate
from embedding_model import embedding_minilm
from langchain_community.vectorstores import Chroma

url = "https://shop.zuscoffee.com/collections/drinkware"

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


# Use an LLM to split the doc into meaningful sections
llm = llm_groq()
embedding_model = embedding_minilm()

# prompt = ChatPromptTemplate.from_template("""
# You are a smart text processor. the text you are given is a collection of product information from an e-commerce website. Your task is to summarize the content and extract key information.
# the most important information is the product name, price, and description, colour/variation, text like "Free Shipping", "Limited Edition", "New Arrival" are extarcted and related to the respective product 
# this data will be insereted into a vector store for future retrieval. so include metadata , and any other relevant information that can help in identifying the product.
# do not make up any information, only extract what is present in the text.
# Text:
# {doc}
# """)

###########################################################################################################################################################################################
# prompt = ChatPromptTemplate.from_template("""
# You are a smart text processor. Given e-commerce product listings, extract the following:
# - product_name
# - sale_price
# - regular_price (if any)
# - discount (just the discount percentage if any)
# - description (if available)
# - colors or variations
# - category
# - tags (e.g., Free Shipping, New Arrival, etc.)
# - brand
# - source_url (if available)

# Do not fabricate information. Only extract what's present.
# Respond with a pure JSON array of objects. Do not include any explanations, code formatting, or comments. Only output the JSON.
# if any of the fields are not available, set them to None.

# Text:
# {doc}
# """)


# chain = prompt | llm

# response = chain.invoke({"doc": doc})


# # print(response)
# import json
# # json_str = 
# data = json.loads(response.content) 
# # print(data)

# docs = []
# for product in data:
#     # Format document content
#     content = f"""
#     Product: {product['product_name']}
#     Sale Price: {product['sale_price']}
#     Regular Price: {product['regular_price']}
#     Description: {product['description']}
#     Colors: {", ".join(product['colors']) if product.get('colors') else "N/A"}
#     Category: {product['category']}
#     Brand: {product['brand']}
#     """

#     docs.append(Document(
#         page_content=content.strip(),
#         metadata={"source": product.get("source_url", "N/A")}
#     ))



# # Now insert directly
# vectorstore = Chroma.from_documents(
#     documents=docs,
#     collection_name="zus-products",
#     embedding=embedding_model,
#     persist_directory="./zus_products_vectorstore"
# )
###########################################################################################################################################################################################


vectorstore = Chroma(
    collection_name="zus-products",
    embedding_function=embedding_model,
    persist_directory="./zus_products_vectorstore"
)

retriever = vectorstore.as_retriever()


question = "what are the available drinkwares under rm 70?"
docs = retriever.invoke(question)
print(docs)
# # print(docs)

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=500, chunk_overlap=0
# )
# doc_splits = text_splitter.split_documents(docs)

# pprint(doc_splits)
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

prompt_cs = ChatPromptTemplate.from_template("""
You are a customer support agent for ZUS Coffee. Your job is to answer customer questions based on the product information available in the vector store.
You can provide additional information that is listed in the context given to you even when the question does not explicitly ask for it. But do not overload the customer with too much information.
Emphasize key details like special offers or features.
If the information is not available, you can say "I don't know" or "Not available
Use the following product context to help answer the customer's question. Be concise, friendly, and helpful.

Context:
{context}

Customer Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

docs_txt = format_docs(docs)
rag_chain = prompt_cs | llm | StrOutputParser()
generation = rag_chain.invoke({"context": docs_txt, "question": question})
print(generation)


from typing import List, TypedDict
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]






"""

so have a llm agent of 

to store or not to store
so always query the vector store first, if not found then scrape the website
insert the scraped data into the vector store
and answer the question based on vector store data

so i need to give a list of urls? or just use tavily

check db 
if similarity scores are low, find new information
so i need a grader for this answer?

i also need a planning agent to plan the steps to take



"""