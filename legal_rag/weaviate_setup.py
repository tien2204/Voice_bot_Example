# weaviate_setup.py - Vector Store Setup
# ====================================

import weaviate
import numpy as np
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

class WeaviateSetup:
    """
    Setup Weaviate vector database cho legal documents
    Fallback to simple in-memory store nếu Weaviate không available
    """
    
    def __init__(self, weaviate_url: str = "http://localhost:8080"):
        self.weaviate_url = weaviate_url
        self.client = None
        self.is_weaviate = False
        
        # Thử kết nối Weaviate
        self._connect()
        
        # Setup schema
        if self.is_weaviate:
            self._setup_weaviate_schema()
    
    def _connect(self):
        """Thử kết nối Weaviate, fallback to simple store"""
        try:
            # Thử kết nối Weaviate local
            self.client = weaviate.Client(self.weaviate_url)
            
            # Test connection
            self.client.schema.get()
            self.is_weaviate = True
            print(f" Kết nối Weaviate thành công: {self.weaviate_url}")
            
        except Exception as e:
            print(f" Không thể kết nối Weaviate: {e}")
            print(" Sử dụng SimpleVectorStore fallback...")
            
            # Fallback to simple vector store
            self.client = SimpleVectorStore()
            self.is_weaviate = False
    
    def _setup_weaviate_schema(self):
        """Setup Weaviate schema cho Vietnamese legal documents"""
        try:
            # Xóa class cũ nếu tồn tại
            if self.client.schema.exists("LegalDocument"):
                self.client.schema.delete_class("LegalDocument")
                print(" Đã xóa schema cũ")
            
            # Tạo schema mới
            schema = {
                "class": "LegalDocument",
                "description": "Vietnamese legal document chunks",
                "vectorizer": "none",  # Sử dụng custom vectors
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "Document content",
                        "tokenization": "word"
                    },
                    {
                        "name": "source",
                        "dataType": ["string"],
                        "description": "Document source identifier"
                    },
                    {
                        "name": "qid",
                        "dataType": ["string"],
                        "description": "Question ID from dataset"
                    },
                    {
                        "name": "cid",
                        "dataType": ["string"],
                        "description": "Context ID from dataset"
                    },
                    {
                        "name": "chunk_idx",
                        "dataType": ["int"],
                        "description": "Chunk index"
                    },
                    {
                        "name": "document_type",
                        "dataType": ["string"],
                        "description": "Type of legal document"
                    },
                    {
                        "name": "dieu_so",
                        "dataType": ["string"],
                        "description": "Article number if available"
                    },
                    {
                        "name": "ten_van_ban",
                        "dataType": ["string"],
                        "description": "Document name"
                    },
                    {
                        "name": "so_van_ban",
                        "dataType": ["string"],
                        "description": "Document number"
                    },
                    {
                        "name": "word_count",
                        "dataType": ["int"],
                        "description": "Word count of chunk"
                    },
                    {
                        "name": "char_count",
                        "dataType": ["int"],
                        "description": "Character count of chunk"
                    },
                    {
                        "name": "related_question",
                        "dataType": ["text"],
                        "description": "Related question from dataset"
                    },
                    {
                        "name": "metadata_json",
                        "dataType": ["string"],
                        "description": "JSON string of additional metadata"
                    }
                ]
            }
            
            self.client.schema.create_class(schema)
            print(" Đã tạo Weaviate schema cho LegalDocument")
            
        except Exception as e:
            print(f" Lỗi tạo Weaviate schema: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]], vectors: List[List[float]]):
        """Thêm documents với vectors vào store"""
        if self.is_weaviate:
            return self._add_to_weaviate(documents, vectors)
        else:
            return self._add_to_simple_store(documents, vectors)
    
    def _add_to_weaviate(self, documents: List[Dict[str, Any]], vectors: List[List[float]]):
        """Thêm documents vào Weaviate"""
        success_count = 0
        
        with self.client.batch as batch:
            batch.batch_size = 100
            
            for doc, vector in zip(documents, vectors):
                try:
                    # Chuẩn bị data object
                    data_object = {
                        "content": doc["content"],
                        "source": doc.get("source", "unknown"),
                        "qid": doc.get("qid", ""),
                        "cid": doc.get("cid", ""),
                        "chunk_idx": doc.get("chunk_idx", 0),
                        "document_type": doc.get("document_type", "unknown"),
                        "dieu_so": doc.get("dieu_so", ""),
                        "ten_van_ban": doc.get("ten_van_ban", ""),
                        "so_van_ban": doc.get("so_van_ban", ""),
                        "word_count": doc.get("word_count", 0),
                        "char_count": doc.get("char_count", 0),
                        "related_question": doc.get("related_question", ""),
                        "metadata_json": json.dumps(doc.get("extra_metadata", {}), ensure_ascii=False)
                    }
                    
                    batch.add_data_object(
                        data_object=data_object,
                        class_name="LegalDocument",
                        vector=vector
                    )
                    
                    success_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Lỗi thêm document: {e}")
        
        print(f" Đã thêm {success_count}/{len(documents)} documents vào Weaviate")
        return success_count
    
    def _add_to_simple_store(self, documents: List[Dict[str, Any]], vectors: List[List[float]]):
        """Thêm documents vào SimpleVectorStore"""
        return self.client.add_documents(documents, vectors)
    
    def search(self, query_vector: List[float], limit: int = 5, 
              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Tìm kiếm documents"""
        if self.is_weaviate:
            return self._search_weaviate(query_vector, limit, filters)
        else:
            return self._search_simple_store(query_vector, limit, filters)
    
    def _search_weaviate(self, query_vector: List[float], limit: int = 5, 
                        filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Tìm kiếm trong Weaviate"""
        try:
            # Build query
            query = (
                self.client.query
                .get("LegalDocument", [
                    "content", "source", "qid", "cid", "chunk_idx",
                    "document_type", "dieu_so", "ten_van_ban", "so_van_ban",
                    "word_count", "related_question", "metadata_json"
                ])
                .with_near_vector({
                    "vector": query_vector,
                    "certainty": 0.7
                })
                .with_limit(limit)
            )
            
            # Add filters if provided
            if filters:
                where_filter = self._build_weaviate_filter(filters)
                if where_filter:
                    query = query.with_where(where_filter)
            
            # Execute query
            result = query.do()
            
            # Process results
            documents = []
            if "data" in result and "Get" in result["data"]:
                for item in result["data"]["Get"]["LegalDocument"]:
                    doc = {
                        "content": item["content"],
                        "source": item["source"],
                        "qid": item["qid"],
                        "cid": item["cid"],
                        "chunk_idx": item["chunk_idx"],
                        "document_type": item["document_type"],
                        "dieu_so": item["dieu_so"],
                        "ten_van_ban": item["ten_van_ban"],
                        "so_van_ban": item["so_van_ban"],
                        "word_count": item["word_count"],
                        "related_question": item["related_question"]
                    }
                    
                    # Parse metadata JSON
                    if item.get("metadata_json"):
                        try:
                            extra_metadata = json.loads(item["metadata_json"])
                            doc.update(extra_metadata)
                        except:
                            pass
                    
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f" Lỗi search Weaviate: {e}")
            return []
    
    def _search_simple_store(self, query_vector: List[float], limit: int = 5, 
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Tìm kiếm trong SimpleVectorStore"""
        return self.client.search(query_vector, limit, filters)
    
    def _build_weaviate_filter(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build Weaviate where filter"""
        conditions = []
        
        for key, value in filters.items():
            if key == "document_type":
                conditions.append({
                    "path": ["document_type"],
                    "operator": "Equal",
                    "valueString": value
                })
            elif key == "dieu_so":
                conditions.append({
                    "path": ["dieu_so"],
                    "operator": "Equal",
                    "valueString": str(value)
                })
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions[0]
        else:
            return {
                "operator": "And",
                "operands": conditions
            }

class SimpleVectorStore:
    """
    Simple in-memory vector store as fallback
    """
    
    def __init__(self):
        self.documents = []
        self.vectors = []
        print(" Khởi tạo SimpleVectorStore")
    
    def add_documents(self, documents: List[Dict[str, Any]], vectors: List[List[float]]):
        """Thêm documents và vectors"""
        for doc, vector in zip(documents, vectors):
            self.documents.append(doc)
            self.vectors.append(np.array(vector))
        
        print(f" Đã thêm {len(documents)} documents vào SimpleVectorStore")
        return len(documents)
    
    def search(self, query_vector: List[float], limit: int = 5, 
              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Tìm kiếm similarity"""
        if not self.vectors:
            return []
        
        query_vec = np.array(query_vector)
        similarities = []
        
        for i, doc_vector in enumerate(self.vectors):
            # Skip if filters don't match
            if filters and not self._match_filters(self.documents[i], filters):
                continue
            
            # Compute cosine similarity
            cos_sim = self._cosine_similarity(query_vec, doc_vector)
            similarities.append((cos_sim, i))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k results
        results = []
        for sim, idx in similarities[:limit]:
            doc = self.documents[idx].copy()
            doc["similarity_score"] = float(sim)
            results.append(doc)
        
        return results
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Tính cosine similarity"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _match_filters(self, doc: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document matches filters"""
        for key, value in filters.items():
            if doc.get(key) != value:
                return False
        return True
    
    def get_count(self) -> int:
        """Lấy số lượng documents"""
        return len(self.documents)
    
    def save_to_file(self, filepath: str):
        """Save store to file"""
        data = {
            "documents": self.documents,
            "vectors": [vec.tolist() for vec in self.vectors]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f" Đã lưu SimpleVectorStore vào {filepath}")
    
    def load_from_file(self, filepath: str):
        """Load store from file"""
        if not Path(filepath).exists():
            print(f" File không tồn tại: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = data["documents"]
        self.vectors = [np.array(vec) for vec in data["vectors"]]
        
        print(f" Đã load {len(self.documents)} documents từ {filepath}")

if __name__ == "__main__":
    # Test WeaviateSetup
    setup = WeaviateSetup()
    
    # Test data
    test_docs = [
        {
            "content": "Điều 1. Phạm vi điều chỉnh của luật",
            "source": "test_1",
            "qid": "q1",
            "document_type": "luat",
            "word_count": 7
        }
    ]
    
    test_vectors = [
        [0.1] * 768  # Mock vector
    ]
    
    # Test add
    setup.add_documents(test_docs, test_vectors)
    
    # Test search
    results = setup.search([0.1] * 768, limit=5)
    print(f"Search results: {len(results)}")
    for result in results:
        print(f"  - {result.get('content', '')[:50]}...")