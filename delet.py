
from langchain_chroma import Chroma
from embedding_model import embedding_minilm

embedding_model = embedding_minilm()

vectorstore = Chroma(
    persist_directory="./zus_products_vectorstore",
    embedding_function=embedding_model,
    collection_name="zus-products"
)



vectorstore.delete_collection()


# import chromadb

# # Load the persistent ChromaDB database
# client = chromadb.PersistentClient(path="./zus_products_vectorstore")

# # List all collections
# collections = client.list_collections()

# # Print collection names
# for c in collections:
#     print(f"🗂️ Collection name: {c.name}")
