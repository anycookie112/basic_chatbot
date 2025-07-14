from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from typing import List

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:  # type: ignore
        return [self.model.encode(d).tolist() for d in documents]

    def embed_query(self, query: str) -> List[float]: # type: ignore
        return self.model.encode([query])[0].tolist()


def embedding_minilm():
    embedding_model = CustomEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model