# retriever.py - Hybrid Retriever (BM25 + Vector)
# ==============================================

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import re
import jieba  # For better tokenization, can be replaced with Vietnamese tokenizer

class BGEEmbeddings(Embeddings):
    """
    BGE-M3 embeddings wrapper
    """
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name)
        print(f" Loaded BGE embeddings: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

class VietnameseTokenizer:
    """
    Vietnamese tokenizer cho BM25
    """
    
    def __init__(self):
        # Vietnamese stopwords
        self.stopwords = {
            'là', 'của', 'và', 'có', 'được', 'theo', 'cho', 'về', 'từ', 'với',
            'trong', 'như', 'để', 'khi', 'này', 'đó', 'các', 'những', 'một',
            'không', 'cũng', 'sẽ', 'đã', 'hay', 'hoặc', 'nhưng', 'nếu', 'thì',
            'vì', 'do', 'bằng', 'trên', 'dưới', 'giữa', 'ngoài', 'trong', 'trước'
        }
        
        # Legal term patterns
        self.legal_patterns = [
            r'điều\s+\d+',           # Điều 1, 2, 3...
            r'khoản\s+\d+',          # Khoản 1, 2, 3...
            r'điểm\s+[a-z]',         # Điểm a, b, c...
            r'luật\s+[\w\s]+',       # Luật ...
            r'nghị\s+định\s+[\w\s]+', # Nghị định ...
            r'thông\s+tư\s+[\w\s]+', # Thông tư ...
            r'\d+/\d+/[A-Z-]+',      # 01/2023/NĐ-CP
        ]
        
        self.legal_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.legal_patterns]
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text với focus on legal terms
        """
        text = text.lower().strip()
        
        # Extract legal terms first
        legal_terms = []
        for regex in self.legal_regex:
            matches = regex.findall(text)
            legal_terms.extend([match.replace(' ', '_') for match in matches])
        
        # Basic word tokenization (can be improved with Vietnamese NLP tools)
        words = re.findall(r'\b\w+\b', text)
        
        # Remove stopwords
        words = [w for w in words if w not in self.stopwords and len(w) > 1]
        
        # Add legal terms
        words.extend(legal_terms)
        
        return words

class HybridRetriever:
    """
    Hybrid retriever kết hợp BM25 và vector search
    - BM25: exact keyword matching (legal terms, article numbers)
    - Vector: semantic similarity
    - Hybrid scoring: weighted combination
    """
    
    def __init__(self, weaviate_client, embedding_model: str = "BAAI/bge-m3", 
                 bm25_weight: float = 0.3, vector_weight: float = 0.7):
        self.weaviate_client = weaviate_client
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
        # Initialize components
        self.embeddings = BGEEmbeddings(embedding_model)
        self.tokenizer = VietnameseTokenizer()
        self.bm25 = None
        self.documents = []
        
        print(f" Initialized HybridRetriever (BM25: {bm25_weight}, Vector: {vector_weight})")
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to both BM25 and vector store
        """
        print(f" Adding {len(documents)} documents to retriever...")
        
        self.documents = documents
        
        # Prepare for BM25
        tokenized_docs = []
        vector_docs = []
        vectors = []
        
        for doc in documents:
            # Tokenize for BM25
            tokens = self.tokenizer.tokenize(doc.page_content)
            tokenized_docs.append(tokens)
            
            # Prepare for vector store
            doc_dict = {
                "content": doc.page_content,
                "source": doc.metadata.get('source', 'unknown'),
                "qid": doc.metadata.get('qid', ''),
                "cid": doc.metadata.get('original_cid', ''),
                "chunk_idx": doc.metadata.get('chunk_idx', 0),
                "document_type": doc.metadata.get('document_type', 'unknown'),
                "dieu_so": doc.metadata.get('dieu_so', ''),
                "ten_van_ban": doc.metadata.get('ten_van_ban', ''),
                "so_van_ban": doc.metadata.get('so_van_ban', ''),
                "word_count": doc.metadata.get('word_count', 0),
                "char_count": doc.metadata.get('char_count', 0),
                "related_question": doc.metadata.get('related_question', ''),
                "extra_metadata": {k: v for k, v in doc.metadata.items() 
                                 if k not in ['source', 'qid', 'original_cid', 'chunk_idx']}
            }
            vector_docs.append(doc_dict)
            
            # Generate embeddings
            vector = self.embeddings.embed_query(doc.page_content)
            vectors.append(vector)
        
        # Build BM25 index
        if tokenized_docs:
            self.bm25 = BM25Okapi(tokenized_docs)
            print(" Built BM25 index")
        
        # Add to vector store
        if vectors:
            self.weaviate_client.add_documents(vector_docs, vectors)
            print(" Added to vector store")
        
        print(f" Retriever ready with {len(documents)} documents")
    
    def retrieve(self, query: str, k: int = 5, return_scores: bool = False) -> List[Document]:
        """
        Hybrid retrieval: BM25 + Vector search với score fusion
        """
        if not self.documents:
            return []
        
        # BM25 retrieval
        bm25_results = self._bm25_search(query, k * 2)  # Get more for fusion
        
        # Vector retrieval  
        vector_results = self._vector_search(query, k * 2)
        
        # Score fusion
        fused_results = self._fuse_scores(bm25_results, vector_results, k)
        
        # Convert to Document objects
        documents = []
        for result in fused_results:
            doc_idx = result['doc_idx']
            if 0 <= doc_idx < len(self.documents):
                doc = self.documents[doc_idx]
                
                # Add score info to metadata
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata.copy()
                )
                doc_copy.metadata.update({
                    'hybrid_score': result['score'],
                    'bm25_score': result.get('bm25_score', 0),
                    'vector_score': result.get('vector_score', 0),
                    'method': 'hybrid'
                })
                
                documents.append(doc_copy)
        
        return documents if not return_scores else [(doc, doc.metadata['hybrid_score']) for doc in documents]
    
    def _bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """BM25 search"""
        if not self.bm25:
            return []
        
        # Tokenize query
        query_tokens = self.tokenizer.tokenize(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only positive scores
                results.append({
                    'doc_idx': int(idx),
                    'bm25_score': float(scores[idx]),
                    'method': 'bm25'
                })
        
        return results
    
    def _vector_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Vector similarity search"""
        # Get query embedding
        query_vector = self.embeddings.embed_query(query)
        
        # Search in vector store
        vector_docs = self.weaviate_client.search(query_vector, limit=k)
        
        results = []
        for doc in vector_docs:
            # Find corresponding document index
            doc_idx = self._find_doc_index(doc)
            if doc_idx >= 0:
                results.append({
                    'doc_idx': doc_idx,
                    'vector_score': doc.get('similarity_score', 0.5),
                    'method': 'vector'
                })
        
        return results
    
    def _find_doc_index(self, vector_doc: Dict[str, Any]) -> int:
        """Find document index in self.documents"""
        source = vector_doc.get('source', '')
        content_preview = vector_doc.get('content', '')[:100]
        
        for idx, doc in enumerate(self.documents):
            if (doc.metadata.get('source') == source and 
                doc.page_content[:100] == content_preview):
                return idx
        
        return -1
    
    def _fuse_scores(self, bm25_results: List[Dict], vector_results: List[Dict], k: int) -> List[Dict]:
        """
        Fuse BM25 and vector scores using weighted combination
        """
        # Normalize scores
        bm25_scores = self._normalize_scores([r['bm25_score'] for r in bm25_results])
        vector_scores = self._normalize_scores([r['vector_score'] for r in vector_results])
        
        # Update normalized scores
        for i, result in enumerate(bm25_results):
            if i < len(bm25_scores):
                result['bm25_score_norm'] = bm25_scores[i]
        
        for i, result in enumerate(vector_results):
            if i < len(vector_scores):
                result['vector_score_norm'] = vector_scores[i]
        
        # Combine results by document index
        combined = {}
        
        # Add BM25 results
        for result in bm25_results:
            doc_idx = result['doc_idx']
            combined[doc_idx] = {
                'doc_idx': doc_idx,
                'bm25_score': result['bm25_score'],
                'bm25_score_norm': result.get('bm25_score_norm', 0),
                'vector_score': 0,
                'vector_score_norm': 0
            }
        
        # Add vector results
        for result in vector_results:
            doc_idx = result['doc_idx']
            if doc_idx in combined:
                combined[doc_idx]['vector_score'] = result['vector_score']
                combined[doc_idx]['vector_score_norm'] = result.get('vector_score_norm', 0)
            else:
                combined[doc_idx] = {
                    'doc_idx': doc_idx,
                    'bm25_score': 0,
                    'bm25_score_norm': 0,
                    'vector_score': result['vector_score'],
                    'vector_score_norm': result.get('vector_score_norm', 0)
                }
        
        # Calculate hybrid scores
        for doc_idx, result in combined.items():
            hybrid_score = (self.bm25_weight * result['bm25_score_norm'] + 
                          self.vector_weight * result['vector_score_norm'])
            result['score'] = hybrid_score
        
        # Sort by hybrid score and return top k
        sorted_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        return sorted_results[:k]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)  # All scores are the same
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def explain_retrieval(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Explain why documents were retrieved (for MCP tool)
        """
        if not self.documents:
            return []
        
        # Get detailed results
        bm25_results = self._bm25_search(query, k * 2)
        vector_results = self._vector_search(query, k * 2)
        fused_results = self._fuse_scores(bm25_results, vector_results, k)
        
        explanations = []
        for result in fused_results:
            doc_idx = result['doc_idx']
            if 0 <= doc_idx < len(self.documents):
                doc = self.documents[doc_idx]
                
                # Find matching query terms
                query_tokens = self.tokenizer.tokenize(query)
                doc_tokens = self.tokenizer.tokenize(doc.page_content)
                matching_terms = list(set(query_tokens) & set(doc_tokens))
                
                explanation = {
                    'doc_idx': doc_idx,
                    'doc_preview': doc.page_content[:120],
                    'source': doc.metadata.get('source', 'unknown'),
                    'document_type': doc.metadata.get('document_type', 'unknown'),
                    'hybrid_score': result['score'],
                    'bm25_score': result['bm25_score'],
                    'vector_score': result['vector_score'],
                    'bm25_contribution': self.bm25_weight * result.get('bm25_score_norm', 0),
                    'vector_contribution': self.vector_weight * result.get('vector_score_norm', 0),
                    'matching_terms': matching_terms[:5],  # Top 5 matching terms
                    'method': 'hybrid',
                    'dieu_so': doc.metadata.get('dieu_so', ''),
                    'ten_van_ban': doc.metadata.get('ten_van_ban', '')
                }
                
                explanations.append(explanation)
        
        return explanations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            'total_documents': len(self.documents),
            'bm25_ready': self.bm25 is not None,
            'vector_store_ready': self.weaviate_client is not None,
            'bm25_weight': self.bm25_weight,
            'vector_weight': self.vector_weight,
            'embedding_model': getattr(self.embeddings.model, 'model_name', 'unknown')
        }

if __name__ == "__main__":
    # Test HybridRetriever
    from weaviate_setup import WeaviateSetup
    from data_loader import YuITCDataLoader
    
    print(" Testing HybridRetriever...")
    
    # Setup components
    weaviate_setup = WeaviateSetup()
    data_loader = YuITCDataLoader(max_documents=5)
    
    # Load test data
    documents = data_loader.load_dataset()
    
    # Initialize retriever
    retriever = HybridRetriever(weaviate_setup.client)
    retriever.add_documents(documents)
    
    # Test queries
    test_queries = [
        "Điều 1 luật hình sự",
        "Quyền và nghĩa vụ của công dân",
        "Tù chung thân",
        "Hôn nhân và gia đình"
    ]
    
    for query in test_queries:
        print(f"\n Query: {query}")
        results = retriever.retrieve(query, k=3)
        
        print(f" Found {len(results)} results:")
        for i, doc in enumerate(results, 1):
            score = doc.metadata.get('hybrid_score', 0)
            method = doc.metadata.get('method', 'unknown')
            preview = doc.page_content[:80].replace('\n', ' ')
            print(f"  {i}. [{method}|{score:.3f}] {preview}...")
        
        # Test explanation
        explanations = retriever.explain_retrieval(query, k=2)
        print(f" Explanations:")
        for exp in explanations[:2]:
            print(f"  - BM25: {exp['bm25_score']:.3f}, Vector: {exp['vector_score']:.3f}")
            print(f"    Matching terms: {exp['matching_terms']}")
    
    print(f"\n Stats: {retriever.get_statistics()}")