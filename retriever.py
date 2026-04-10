from config import TOP_K
from langchain_community.retrievers import BM25Retriever
from typing import List
from langchain_core.documents import Document

class CustomHybridRetriever:
    """
    A custom retriever that combines Qdrant (MMR), BM25, and Cross-Encoder Reranking
    without needing missing 'langchain.retrievers' imports.
    """
    def __init__(self, vector_store, chunks):
        fetch_k = TOP_K * 4
        
        # 1. MMR Retriever
        self.mmr_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": fetch_k, "fetch_k": fetch_k * 2}
        )
        
        # 2. BM25 Retriever
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = fetch_k
        
        self.top_k = TOP_K
        
        # Load cross-encoder for reranking
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def invoke(self, query: str, **kwargs) -> List[Document]:
        # Fetch candidates from both retrievers
        mmr_docs = self.mmr_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # Combine uniquely based on page_content
        unique_docs_map = {}
        for doc in mmr_docs + bm25_docs:
            if doc.page_content not in unique_docs_map:
                unique_docs_map[doc.page_content] = doc
                
        candidate_docs = list(unique_docs_map.values())
        
        if not candidate_docs:
            return []
            
        # Rerank candidates
        pairs = [[query, doc.page_content] for doc in candidate_docs]
        scores = self.reranker.predict(pairs)
        
        # Sort by score descending
        scored_docs = list(zip(candidate_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N documents
        return [doc for doc, score in scored_docs[:self.top_k]]

def get_retriever(vector_store, chunks):
    return CustomHybridRetriever(vector_store, chunks)