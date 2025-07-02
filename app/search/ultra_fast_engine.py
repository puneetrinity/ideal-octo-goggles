import numpy as np
import time
import os
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import asyncio
from sentence_transformers import SentenceTransformer
import faiss

from app.math.lsh_index import LSHIndex
from app.math.hnsw_index import HNSWIndex
from app.math.product_quantization import ProductQuantizer
from app.logger import get_logger
from app.config import settings

logger = get_logger(__name__)

@dataclass
class SearchResult:
    doc_id: str
    similarity_score: float
    bm25_score: float
    combined_score: float
    metadata: Dict

class UltraFastSearchEngine:

    def __init__(self, embedding_dim: int, use_gpu: bool):
        self.embedding_model = SentenceTransformer(settings.embedding_model_name, device='cuda' if use_gpu else 'cpu')
        self.embedding_dim = embedding_dim
        self.index_path = settings.index_path
        self._initialize_indexes()
        self.load_indexes()

    def _initialize_indexes(self):
        self.lsh_index = LSHIndex(num_hashes=128, num_bands=16)
        self.hnsw_index = HNSWIndex(dimension=self.embedding_dim)
        self.pq_quantizer = ProductQuantizer(dimension=self.embedding_dim)
        self.document_vectors = {}
        self.document_codes = {}
        self.document_metadata = {}
        self.document_text_features = {}
        self.bm25_index = {}
        self.doc_frequencies = {}
        self.corpus_size = 0
        self.avg_doc_length = 0
        self.search_stats = {'total_searches': 0, 'avg_response_time': 0, 'cache_hits': 0}
        self.query_cache = {}
        self.cache_max_size = 1000

    def save_indexes(self):
        logger.info(f"Saving indexes to {self.index_path}")
        os.makedirs(self.index_path, exist_ok=True)
        faiss.write_index(self.hnsw_index.index, os.path.join(self.index_path, "hnsw.index"))
        with open(os.path.join(self.index_path, "other_data.pkl"), "wb") as f:
            pickle.dump({
                "lsh_index": self.lsh_index,
                "pq_quantizer": self.pq_quantizer,
                "document_vectors": self.document_vectors,
                "document_codes": self.document_codes,
                "document_metadata": self.document_metadata,
                "document_text_features": self.document_text_features,
                "bm25_index": self.bm25_index,
                "doc_frequencies": self.doc_frequencies,
                "corpus_size": self.corpus_size,
                "avg_doc_length": self.avg_doc_length,
                "doc_ids": self.hnsw_index.doc_ids
            }, f)

    def load_indexes(self):
        if not os.path.exists(os.path.join(self.index_path, "hnsw.index")):
            logger.info("No existing indexes found. Ready for building.")
            return
        logger.info(f"Loading indexes from {self.index_path}")
        self.hnsw_index.index = faiss.read_index(os.path.join(self.index_path, "hnsw.index"))
        with open(os.path.join(self.index_path, "other_data.pkl"), "rb") as f:
            data = pickle.load(f)
            self.lsh_index = data["lsh_index"]
            self.pq_quantizer = data["pq_quantizer"]
            self.document_vectors = data["document_vectors"]
            self.document_codes = data["document_codes"]
            self.document_metadata = data["document_metadata"]
            self.document_text_features = data["document_text_features"]
            self.bm25_index = data["bm25_index"]
            self.doc_frequencies = data["doc_frequencies"]
            self.corpus_size = data["corpus_size"]
            self.avg_doc_length = data["avg_doc_length"]
            self.hnsw_index.doc_ids = data["doc_ids"]

    async def build_indexes(self, documents: List[Dict]):
        logger.info(f"Building ultra-fast indexes for {len(documents)} documents...")
        start_time = time.time()
        self._initialize_indexes()

        texts_to_embed = [self._get_document_text(doc) for doc in documents]
        vectors = self.embedding_model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
        doc_ids = [doc['id'] for doc in documents]

        for i, doc in enumerate(documents):
            doc_id = doc['id']
            text_features = self._extract_text_features(doc)
            self.document_text_features[doc_id] = text_features
            self.document_vectors[doc_id] = vectors[i]
            self.document_metadata[doc_id] = {
                'name': doc.get('name', ''),
                'experience_years': doc.get('experience_years', 0),
                'skills': doc.get('skills', []),
                'seniority_level': doc.get('seniority_level', 'unknown')
            }

        await asyncio.gather(
            self._build_lsh_index(documents, [self.document_text_features[did] for did in doc_ids]),
            self._build_hnsw_index(doc_ids, vectors),
            self._build_pq_index(vectors),
            self._build_bm25_index(documents)
        )
        self.save_indexes()
        build_time = time.time() - start_time
        logger.info(f"Index building completed in {build_time:.2f} seconds")

    async def search(self, query: str, num_results: int = 10, filters: Optional[Dict] = None) -> List[SearchResult]:
        # ... (rest of the search logic is the same)
        search_start = time.time()

        cache_key = f"{query}:{num_results}:{str(filters)}"
        if cache_key in self.query_cache:
            self.search_stats['cache_hits'] += 1
            return self.query_cache[cache_key]

        query_vector = self.embedding_model.encode([query], convert_to_numpy=True)
        query_features = self._extract_query_features(query)

        lsh_candidates = self.lsh_index.query_candidates(query_features, num_candidates=200)
        hnsw_results = self.hnsw_index.search(query_vector, k=100)
        hnsw_candidates = [doc_id for doc_id, _ in hnsw_results]

        all_candidates = list(set(lsh_candidates + hnsw_candidates))

        if filters:
            all_candidates = self._apply_filters(all_candidates, filters)

        scored_results = await self._score_candidates(all_candidates, query, query_vector[0], query_features)

        scored_results.sort(key=lambda x: x.combined_score, reverse=True)
        final_results = scored_results[:num_results]

        if len(self.query_cache) >= self.cache_max_size:
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[cache_key] = final_results

        response_time = (time.time() - search_start) * 1000
        self.search_stats['total_searches'] += 1
        self.search_stats['avg_response_time'] = (self.search_stats['avg_response_time'] * (self.search_stats['total_searches'] - 1) + response_time) / self.search_stats['total_searches']

        logger.info(f"Search completed in {response_time:.1f}ms, found {len(final_results)} results from {len(all_candidates)} candidates")
        return final_results

    async def _score_candidates(self, candidates: List[str], query: str, query_vector: np.ndarray, query_features: List[str]) -> List[SearchResult]:
        tasks = [self._score_single_candidate(candidate, query, query_vector, query_features) for candidate in candidates]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    async def _score_single_candidate(self, doc_id: str, query: str, query_vector: np.ndarray, query_features: List[str]) -> Optional[SearchResult]:
        if doc_id not in self.document_vectors:
            return None

        doc_vector = self.document_vectors[doc_id]
        vector_similarity = 1 - self._cosine_distance(query_vector, doc_vector)
        jaccard_similarity = self.lsh_index.jaccard_similarity(doc_id, query_features)
        bm25_score = self._compute_bm25_score(doc_id, query)

        combined_score = (0.4 * vector_similarity + 0.3 * jaccard_similarity + 0.3 * bm25_score)

        return SearchResult(
            doc_id=doc_id,
            similarity_score=vector_similarity,
            bm25_score=bm25_score,
            combined_score=combined_score,
            metadata=self.document_metadata.get(doc_id, {})
        )

    def _cosine_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return 1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    async def _build_lsh_index(self, documents: List[Dict], text_features_list: List[List[str]]):
        logger.info("Building LSH index...")
        for doc, features in zip(documents, text_features_list):
            self.lsh_index.add_document(doc['id'], features)

    async def _build_hnsw_index(self, doc_ids: List[str], vectors: np.ndarray):
        logger.info("Building HNSW index...")
        self.hnsw_index.add_documents(vectors, doc_ids)

    async def _build_pq_index(self, vectors: np.ndarray):
        logger.info("Building PQ index...")
        self.pq_quantizer.train(vectors)
        for doc_id, vector in self.document_vectors.items():
            self.document_codes[doc_id] = self.pq_quantizer.encode(vector.reshape(1, -1))[0]

    async def _build_bm25_index(self, documents: List[Dict]):
        logger.info("Building BM25 index...")
        total_length = 0
        for doc in documents:
            doc_id = doc['id']
            text = self._get_document_text(doc)
            tokens = text.lower().split()
            total_length += len(tokens)
            tf = {token: tokens.count(token) for token in set(tokens)}
            for token in set(tokens):
                self.doc_frequencies[token] = self.doc_frequencies.get(token, 0) + 1
            self.bm25_index[doc_id] = {'tf': tf, 'length': len(tokens)}
        self.corpus_size = len(documents)
        self.avg_doc_length = total_length / self.corpus_size

    def _compute_bm25_score(self, doc_id: str, query: str) -> float:
        if doc_id not in self.bm25_index:
            return 0.0
        k1 = 1.5
        b = 0.75
        doc_data = self.bm25_index[doc_id]
        doc_tf = doc_data['tf']
        doc_length = doc_data['length']
        query_terms = query.lower().split()
        score = 0.0
        for term in query_terms:
            if term in doc_tf:
                tf = doc_tf[term]
                df = self.doc_frequencies.get(term, 0)
                idf = np.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
                score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / self.avg_doc_length))
        return score

    def _extract_text_features(self, doc: Dict) -> List[str]:
        features = []
        if 'skills' in doc: features.extend([s.lower() for s in doc['skills']])
        if 'technologies' in doc: features.extend([t.lower() for t in doc['technologies']])
        text = self._get_document_text(doc).lower()
        features.extend(text.split())
        return list(set(features))

    def _extract_query_features(self, query: str) -> List[str]:
        return list(set(query.lower().split()))

    def _get_document_text(self, doc: Dict) -> str:
        text_parts = []
        for field in ['name', 'title', 'description', 'experience', 'projects']:
            if field in doc:
                text_parts.append(str(doc[field]))
        if 'skills' in doc: text_parts.extend(doc['skills'])
        if 'technologies' in doc: text_parts.extend(doc['technologies'])
        return ' '.join(text_parts)

    def _apply_filters(self, candidates: List[str], filters: Dict) -> List[str]:
        filtered = []
        for doc_id in candidates:
            if doc_id not in self.document_metadata: continue
            doc_meta = self.document_metadata[doc_id]
            if 'min_experience' in filters and doc_meta['experience_years'] < filters['min_experience']: continue
            if 'seniority_levels' in filters and doc_meta['seniority_level'] not in filters['seniority_levels']: continue
            if 'required_skills' in filters and not set(s.lower() for s in filters['required_skills']).issubset(set(s.lower() for s in doc_meta['skills'])): continue
            filtered.append(doc_id)
        return filtered

    def get_performance_stats(self) -> Dict:
        cache_hit_rate = self.search_stats['cache_hits'] / self.search_stats['total_searches'] if self.search_stats['total_searches'] > 0 else 0
        return {
            'total_searches': self.search_stats['total_searches'],
            'avg_response_time_ms': self.search_stats['avg_response_time'],
            'cache_hit_rate': cache_hit_rate
        }