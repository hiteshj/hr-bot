import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from pydantic import Field
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class NativeFAISSRetriever(BaseRetriever):
    """Production FAISS retriever built directly on langchain-core and faiss-cpu."""

    embeddings: Embeddings
    index: Any = Field(exclude=True)
    docstore: List[Document] = Field(default_factory=list)
    k: int = 4
    filter_metadata: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embeddings: Embeddings,
        k: int = 4,
        use_hnsw: bool = False,
    ) -> "NativeFAISSRetriever":
        texts = [doc.page_content for doc in documents]
        vectors = np.array(embeddings.embed_documents(texts), dtype=np.float32)

        # L2-normalize vectors so Inner Product equals Cosine Similarity
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]

        if use_hnsw:
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 64
            index.hnsw.efSearch = 32
        else:
            index = faiss.IndexFlatIP(dim)

        index.add(vectors)
        return cls(embeddings=embeddings, index=index, docstore=documents, k=k)

    def save_local(self, folder_path: str) -> None:
        path = Path(folder_path)
        path.mkdir(parents=True, exist_ok=True)
        # 1. Save native C++ FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        # 2. Save documents safely as JSON (no pickle vulnerability)
        serialized_docs = [
            {"page_content": d.page_content, "metadata": d.metadata, "id": d.id}
            for d in self.docstore
        ]
        (path / "docstore.json").write_text(json.dumps(serialized_docs), encoding="utf-8")

    @classmethod
    def load_local(
        cls, folder_path: str, embeddings: Embeddings, k: int = 4
    ) -> "NativeFAISSRetriever":
        path = Path(folder_path)
        index = faiss.read_index(str(path / "index.faiss"))
        raw_docs = json.loads((path / "docstore.json").read_text(encoding="utf-8"))
        docstore = [
            Document(page_content=d["page_content"], metadata=d["metadata"], id=d.get("id"))
            for d in raw_docs
        ]
        return cls(embeddings=embeddings, index=index, docstore=docstore, k=k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        query_vec = np.array([self.embeddings.embed_query(query)], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # Over-fetch if metadata filtering is active
        fetch_k = self.k * 5 if self.filter_metadata else self.k
        scores, indices = self.index.search(query_vec, fetch_k)

        results: List[Document] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 when fewer than fetch_k items exist
                continue
            doc = self.docstore[idx]
            if self.filter_metadata:
                if not all(doc.metadata.get(k) == v for k, v in self.filter_metadata.items()):
                    continue
            # Attach similarity score to metadata for downstream inspection/reranking
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "similarity_score": float(score)},
                id=doc.id,
            )
            results.append(doc_copy)
            if len(results) >= self.k:
                break
        return results




class HybridFAISSBM25Retriever:
    def __init__(
        self,
        faiss_retriever: NativeFAISSRetriever,
        documents: List[Document],
        faiss_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        self.faiss_retriever = faiss_retriever
        self.documents = documents
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def invoke(self, query: str, top_k: int = 4) -> List[Document]:
        # 1. Dense search via FAISS
        dense_docs = self.faiss_retriever.invoke(query)

        # 2. Sparse search via BM25
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k]
        sparse_docs = [self.documents[i] for i in top_bm25_indices if bm25_scores[i] > 0]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs):
            key = doc.id or doc.page_content
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (
                self.faiss_weight / (self.rrf_k + rank + 1)
            )

        for rank, doc in enumerate(sparse_docs):
            key = doc.id or doc.page_content
            doc_map[key] = doc
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (
                self.bm25_weight / (self.rrf_k + rank + 1)
            )

        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return [doc_map[k] for k in sorted_keys[:top_k]]