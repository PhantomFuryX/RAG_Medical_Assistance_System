from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import torch

class RelevanceScorer:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device}
        )
    
    def compute_similarity(self, query: str, document: str) -> float:
        """Compute cosine similarity between query and document."""
        query_embedding = self.embeddings.embed_query(query)
        doc_embedding = self.embeddings.embed_query(document)
        
        # Compute cosine similarity
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        return float(similarity)
    
    def filter_relevant_documents(self, query: str, documents: list, threshold: float = 0.6):
        """Filter documents based on relevance to the query."""
        relevant_docs = []
        scores = []
        
        for doc in documents:
            score = self.compute_similarity(query, doc.page_content)
            if score >= threshold:
                relevant_docs.append(doc)
                scores.append(score)
        
        # Sort by relevance score
        sorted_docs = [doc for _, doc in sorted(zip(scores, relevant_docs), reverse=True)]
        return sorted_docs
