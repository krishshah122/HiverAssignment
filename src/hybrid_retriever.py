"""
Hybrid Retriever: BM25 + Vector Search → RRF → Cross-Encoder Reranker

This module implements production-grade retrieval that combines:
1. BM25 (keyword/lexical match) for exact term matching
2. Vector Search (semantic match) for meaning-level matching
3. Reciprocal Rank Fusion (RRF) to merge both ranked lists
4. Cross-Encoder Reranker to pick the true top-K cases
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import chromadb
from chromadb.utils import embedding_functions


def clean_text(text: str) -> str:
    """Basic tweet cleaning for BM25 tokenization."""
    text = re.sub(r'@\w+', '', text)        # remove mentions
    text = re.sub(r'https?://\S+', '', text) # remove URLs
    text = re.sub(r'#', '', text)            # remove hashtag symbol
    text = text.lower().strip()
    return text


class HybridRetriever:
    def __init__(
        self,
        knowledge_csv: str = "data/AppleSupport_massive_pairs.csv",
        chroma_db_path: str = "data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        kb_sample_size: int = 5000,
        random_state: int = 100,
    ):
        print("Loading Knowledge Base...")
        self.kb_df = pd.read_csv(knowledge_csv)

        # CRITICAL: Remove DM-redirect replies from BM25 KB too
        dm_keywords = ['send us a dm', 'dm us', 'reply in dm', 'join us in dm',
                       'let us know in dm', 'dm your', 'https://t.co/']
        self.kb_df['is_dm'] = self.kb_df['text_brand'].str.lower().apply(
            lambda x: any(k in str(x) for k in dm_keywords)
        )
        self.kb_df = self.kb_df[~self.kb_df['is_dm']].copy()
        print(f"  Removed DM-redirect replies. Remaining: {len(self.kb_df)} substantive pairs.")

        self.kb_df = self.kb_df.sample(n=min(kb_sample_size, len(self.kb_df)), random_state=random_state).reset_index(drop=True)

        self.customer_texts: List[str] = self.kb_df['text_customer'].astype(str).tolist()
        self.brand_responses: List[str] = self.kb_df['text_brand'].astype(str).tolist()

        # ── BM25 index ──────────────────────────────────────────────
        print("Building BM25 index...")
        tokenized_corpus = [clean_text(t).split() for t in self.customer_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # ── Vector index (ChromaDB) ─────────────────────────────────
        print("Connecting to ChromaDB vector index...")
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = client.get_collection(
            name="apple_support_resolutions",
            embedding_function=self.ef,
        )

        # ── Cross-Encoder reranker ──────────────────────────────────
        print("Loading Cross-Encoder reranker...")
        self.reranker = CrossEncoder(reranker_model)

        print("Hybrid Retriever ready.\n")

    # ----------------------------------------------------------------
    # BM25 search
    # ----------------------------------------------------------------
    def _bm25_search(self, query: str, top_n: int = 20) -> List[Tuple[int, float]]:
        """Returns list of (doc_index, bm25_score) sorted descending."""
        tokens = clean_text(query).split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    # ----------------------------------------------------------------
    # Vector search
    # ----------------------------------------------------------------
    def _vector_search(self, query: str, top_n: int = 20) -> List[Tuple[int, float]]:
        """Returns list of (doc_index, distance) sorted ascending (lower=closer)."""
        results = self.collection.query(query_texts=[query], n_results=top_n)
        ids = results['ids'][0]
        distances = results['distances'][0]

        # Map ChromaDB IDs back to KB indices
        id_to_idx = {}
        for i, tid in enumerate(self.kb_df['tweet_id_customer'].astype(str)):
            id_to_idx[tid] = i

        ranked: List[Tuple[int, float]] = []
        for doc_id, dist in zip(ids, distances):
            if doc_id in id_to_idx:
                ranked.append((id_to_idx[doc_id], float(dist)))
        return ranked

    # ----------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ----------------------------------------------------------------
    @staticmethod
    def _rrf(ranked_lists: List[List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
        """
        Reciprocal Rank Fusion (RRF) merges multiple ranked lists.
        score(doc) = sum over lists of  1 / (k + rank_in_list)
        """
        fused_scores: Dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (doc_idx, _score) in enumerate(ranked):
                fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs

    # ----------------------------------------------------------------
    # Reranker
    # ----------------------------------------------------------------
    def _rerank(self, query: str, candidate_indices: List[int], top_k: int = 3) -> List[Dict]:
        """
        Uses a Cross-Encoder to rerank the candidate documents.
        Returns top_k results with score.
        """
        pairs = [(query, self.customer_texts[idx]) for idx in candidate_indices]
        scores = self.reranker.predict(pairs)

        scored = list(zip(candidate_indices, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]

        results = []
        for idx, score in scored:
            results.append({
                "customer_query": self.customer_texts[idx],
                "brand_response": self.brand_responses[idx],
                "reranker_score": float(score),
            })
        return results

    # ----------------------------------------------------------------
    # Public API: full hybrid retrieval pipeline
    # ----------------------------------------------------------------
    def retrieve(self, query: str, bm25_top_n: int = 20, vector_top_n: int = 20, final_top_k: int = 3) -> List[Dict]:
        """
        Full pipeline: BM25 + Vector → RRF → Reranker → Top-K cases.
        """
        # 1. BM25
        bm25_results = self._bm25_search(query, top_n=bm25_top_n)

        # 2. Vector
        vector_results = self._vector_search(query, top_n=vector_top_n)

        # 3. RRF
        rrf_ranked = self._rrf([bm25_results, vector_results])

        # Take top candidates for reranking (more than final_top_k for the reranker to work with)
        candidate_indices = [idx for idx, _score in rrf_ranked[:min(20, len(rrf_ranked))]]

        if not candidate_indices:
            return []

        # 4. Rerank
        top_cases = self._rerank(query, candidate_indices, top_k=final_top_k)
        return top_cases


if __name__ == "__main__":
    retriever = HybridRetriever()
    query = "My battery drains so fast after the iOS update, phone dies in 2 hours"
    print(f"Query: {query}\n")
    results = retriever.retrieve(query)
    for i, r in enumerate(results):
        print(f"--- Result {i+1} (Reranker Score: {r['reranker_score']:.4f}) ---")
        print(f"  Past Customer: {r['customer_query'][:100]}...")
        print(f"  Brand Reply:   {r['brand_response'][:100]}...")
        print()
