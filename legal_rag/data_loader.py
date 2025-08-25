# data_loader.py - YuITC Dataset Loader
# =====================================

import re
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json
import hashlib

class YuITCDataLoader:
    """
    Loader cho YuITC Vietnamese Legal Document Retrieval Data
    Dataset structure: question, context_list, qid, cid
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100, max_documents: int = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_documents = max_documents
        
        # Text splitter cho chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\nĐiều ",  # Ưu tiên chia theo điều luật
                "\n\nKhoản ",  # Chia theo khoản
                "\n\nĐiểm ",   # Chia theo điểm
                "\n\n",        # Chia theo paragraph
                "\n",          # Chia theo dòng
                ". ",          # Chia theo câu
                "! ",
                "? ",
                " ",           # Chia theo từ
                ""
            ]
        )
        
        self.legal_pattern = self._compile_legal_patterns()
    
    def _compile_legal_patterns(self):
        """Compile regex patterns để nhận diện cấu trúc pháp luật"""
        patterns = {
            'dieu': re.compile(r'Điều\s+(\d+)\.?\s*(.+?)(?=Điều\s+\d+|$)', re.DOTALL),
            'khoan': re.compile(r'(\d+)\.?\s*(.+?)(?=\d+\.|$)', re.DOTALL),
            'diem': re.compile(r'([a-z])\)\s*(.+?)(?=[a-z]\)|$)', re.DOTALL),
            'luat': re.compile(r'(Luật|Nghị định|Thông tư|Quyết định)\s+([^,\n]+)', re.IGNORECASE),
            'so_van_ban': re.compile(r'(\d+/\d+/[A-Z-]+)', re.IGNORECASE)
        }
        return patterns
    
    def load_dataset(self) -> List[Document]:
        """
        Load YuITC dataset và convert thành documents
        """
        print("Đang tải YuITC Vietnamese Legal Document Retrieval Data...")
        
        try:
            # Load dataset từ HuggingFace
            dataset = load_dataset(
                "YuITC/Vietnamese-Legal-Doc-Retrieval-Data",
                split="train"
            )
            
            print(f"Đã tải dataset với {len(dataset)} records")
            
            # Process dataset
            documents = []
            context_cache = set()  # Để tránh duplicate context
            
            for idx, item in enumerate(dataset):
                if self.max_documents and idx >= self.max_documents:
                    break
                
                # Extract data từ item
                question = item.get('question', '')
                context_list = item.get('context_list', [])
                qid = item.get('qid', f'q_{idx}')
                cid = item.get('cid', [])
                
                # Process context_list
                processed_docs = self._process_context_list(
                    context_list, qid, cid, question, idx
                )
                
                # Add unique contexts
                for doc in processed_docs:
                    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                    if content_hash not in context_cache:
                        context_cache.add(content_hash)
                        documents.append(doc)
                
                if (idx + 1) % 100 == 0:
                    print(f"📖 Đã xử lý {idx + 1}/{min(len(dataset), self.max_documents or len(dataset))} records...")
            
            print(f"Tổng cộng tạo ra {len(documents)} chunks từ dataset")
            return documents
            
        except Exception as e:
            print(f"Lỗi khi tải YuITC dataset: {e}")
            print("Tạo dữ liệu mẫu...")
            return self._create_sample_data()
    
    def _process_context_list(self, context_list: List[str], qid: str, cid: List[str], 
                            question: str = "", item_idx: int = 0) -> List[Document]:
        """
        Process context_list thành chunks
        """
        documents = []
        
        for ctx_idx, context in enumerate(context_list):
            if not context or not isinstance(context, str):
                continue
                
            context = context.strip()
            if len(context) < 50:  # Skip context quá ngắn
                continue
            
            # Detect loại văn bản pháp luật
            doc_type = self._detect_document_type(context)
            
            # Chunking strategy
            if doc_type in ['luat', 'nghi_dinh'] and len(context) > self.chunk_size:
                # Chia theo cấu trúc pháp luật
                chunks = self._chunk_by_legal_structure(context)
            else:
                # Chia theo token size thông thường
                chunks = self.text_splitter.split_text(context)
            
            # Create Document objects
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.strip()) < 30:  # Skip chunk quá ngắn
                    continue
                
                # Extract metadata
                metadata = self._extract_metadata(chunk, context, doc_type)
                metadata.update({
                    'qid': qid,
                    'original_cid': cid[ctx_idx] if ctx_idx < len(cid) else f'c_{item_idx}_{ctx_idx}',
                    'context_idx': ctx_idx,
                    'chunk_idx': chunk_idx,
                    'source': f"yuítc_item_{item_idx}_ctx_{ctx_idx}_chunk_{chunk_idx}",
                    'related_question': question,
                    'document_type': doc_type,
                    'chunk_size': len(chunk),
                    'original_context_size': len(context)
                })
                
                doc = Document(
                    page_content=chunk.strip(),
                    metadata=metadata
                )
                
                documents.append(doc)
        
        return documents
    
    def _detect_document_type(self, text: str) -> str:
        """
        Detect loại văn bản pháp luật
        """
        text_upper = text.upper()
        
        if any(keyword in text_upper for keyword in ['LUẬT', 'LAW']):
            return 'luat'
        elif any(keyword in text_upper for keyword in ['NGHỊ ĐỊNH', 'DECREE']):
            return 'nghi_dinh'
        elif any(keyword in text_upper for keyword in ['THÔNG TƯ', 'CIRCULAR']):
            return 'thong_tu'
        elif any(keyword in text_upper for keyword in ['QUYẾT ĐỊNH', 'DECISION']):
            return 'quyet_dinh'
        elif any(keyword in text_upper for keyword in ['BỘ LUẬT', 'CODE']):
            return 'bo_luat'
        elif 'HIẾN PHÁP' in text_upper:
            return 'hien_phap'
        else:
            return 'van_ban_khac'
    
    def _chunk_by_legal_structure(self, text: str) -> List[str]:
        """
        Chia chunk theo cấu trúc pháp luật (Điều, Khoản, Điểm)
        """
        chunks = []
        
        # Thử chia theo Điều trước
        dieu_matches = list(self.legal_pattern['dieu'].finditer(text))
        
        if len(dieu_matches) > 1:
            # Có nhiều điều -> chia theo điều
            for i, match in enumerate(dieu_matches):
                dieu_num = match.group(1)
                dieu_content = match.group(2).strip()
                
                chunk_title = f"Điều {dieu_num}"
                chunk_content = f"{chunk_title}. {dieu_content}"
                
                # Nếu điều quá dài, chia nhỏ hơn nữa
                if len(chunk_content) > self.chunk_size * 1.5:
                    sub_chunks = self.text_splitter.split_text(chunk_content)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(chunk_content)
        else:
            # Không có cấu trúc điều rõ ràng -> dùng text_splitter thường
            chunks = self.text_splitter.split_text(text)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _extract_metadata(self, chunk: str, original_context: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract metadata từ chunk
        """
        metadata = {}
        
        # Extract số điều nếu có
        dieu_match = self.legal_pattern['dieu'].search(chunk)
        if dieu_match:
            metadata['dieu_so'] = dieu_match.group(1)
        
        # Extract tên luật/văn bản
        luat_match = self.legal_pattern['luat'].search(chunk)
        if luat_match:
            metadata['loai_van_ban'] = luat_match.group(1)
            metadata['ten_van_ban'] = luat_match.group(2)
        
        # Extract số văn bản
        so_match = self.legal_pattern['so_van_ban'].search(chunk)
        if so_match:
            metadata['so_van_ban'] = so_match.group(1)
        
        # Tính toán thống kê
        metadata['word_count'] = len(chunk.split())
        metadata['char_count'] = len(chunk)
        
        # Check nếu là chunk đầu của document
        metadata['is_first_chunk'] = chunk in original_context[:len(chunk)]
        
        return metadata
    
    def _create_sample_data(self) -> List[Document]:
        """
        Tạo dữ liệu mẫu phong phú về pháp luật Việt Nam
        """
        sample_contexts = [
            {
                "context": "Điều 1. Phạm vi điều chỉnh\n1. Bộ luật này quy định về tội phạm và hình phạt.\n2. Bộ luật này áp dụng đối với mọi người phạm tội trên lãnh thổ nước Cộng hòa xã hội chủ nghĩa Việt Nam.\nĐiều 2. Nhiệm vụ của Bộ luật hình sự\nBộ luật hình sự có nhiệm vụ bảo vệ độc lập, chủ quyền, thống nhất, toàn vẹn lãnh thổ của Tổ quốc, bảo vệ chế độ chính trị.",
                "question": "Bộ luật hình sự có phạm vi điều chỉnh như thế nào?",
                "doc_type": "bo_luat"
            },
            {
                "context": "Điều 15. Tuổi chịu trách nhiệm hình sự\n1. Người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự về mọi tội phạm.\n2. Người từ đủ 14 tuổi đến dưới 16 tuổi phải chịu trách nhiệm hình sự về tội rất nghiêm trọng do cố ý, tội đặc biệt nghiêm trọng quy định tại các điều 123, 134, 141, 142, 143, 144, 150, 151, 168, 169, 170, 171, 173, 174, 178, 248, 249, 250, 251, 252, 266, 286, 287, 289, 290, 299, 303, 304 của Bộ luật này.",
                "question": "Tuổi chịu trách nhiệm hình sự được quy định như thế nào?",
                "doc_type": "bo_luat"
            },
            {
                "context": "Điều 33. Tù chung thân\n1. Tù chung thân là hình phạt tù không thời hạn.\n2. Tù chung thân chỉ áp dụng đối với người phạm tội đặc biệt nghiêm trọng trong trường hợp không áp dụng hình phạt tử hình và được quy định tại các điều của Phần đặc biệt của Bộ luật này.\n3. Tù chung thân không áp dụng đối với người dưới 18 tuổi khi phạm tội; phụ nữ có thai; người từ đủ 70 tuổi trở lên.",
                "question": "Hình phạt tù chung thân được áp dụng như thế nào?",
                "doc_type": "bo_luat"
            },
            {
                "context": "Điều 143. Tội giết người\n1. Người nào cố ý làm chết người khác, thì bị phạt tù từ 12 năm đến 20 năm, tù chung thân hoặc tử hình.\n2. Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 20 năm, tù chung thân hoặc tử hình:\na) Có tổ chức;\nb) Giết nhiều người;\nc) Giết người dưới 16 tuổi;\nd) Giết phụ nữ mà biết là có thai;\nđ) Giết người trong tình trạng thần kinh bất thường do tác động của chất ma túy.",
                "question": "Tội giết người có những mức phạt như thế nào?",
                "doc_type": "bo_luat"
            },
            {
                "context": "Điều 1. Phạm vi điều chỉnh\nLuật này quy định về hôn nhân và gia đình; quyền và nghĩa vụ của thành viên gia đình; hôn nhân và gia đình có yếu tố nước ngoài.\nĐiều 2. Nguyên tắc của hôn nhân và gia đình\n1. Hôn nhân tự do, tiến bộ, một vợ một chồng, vợ chồng bình đẳng.\n2. Gia đình bình đẳng, hòa thuận, hạnh phúc và bền vững.",
                "question": "Luật hôn nhân và gia đình có những nguyên tắc gì?",
                "doc_type": "luat"
            },
            {
                "context": "Điều 8. Điều kiện kết hôn\n1. Nam từ đủ 20 tuổi, nữ từ đủ 18 tuổi.\n2. Việc kết hôn do nam, nữ tự nguyện quyết định.\n3. Không bị cấm kết hôn theo quy định tại Điều 10 của Luật này.\nĐiều 9. Đăng ký kết hôn\n1. Việc kết hôn phải được đăng ký tại cơ quan có thẩm quyền.\n2. Chỉ công nhận hôn nhân được đăng ký theo quy định của pháp luật.",
                "question": "Điều kiện và thủ tục kết hôn được quy định như thế nào?",
                "doc_type": "luat"
            }
        ]
        
        documents = []
        
        for idx, item in enumerate(sample_contexts):
            context = item["context"]
            question = item["question"]
            doc_type = item["doc_type"]
            
            # Chunk context if needed
            if len(context) > self.chunk_size:
                chunks = self._chunk_by_legal_structure(context)
            else:
                chunks = [context]
            
            for chunk_idx, chunk in enumerate(chunks):
                metadata = self._extract_metadata(chunk, context, doc_type)
                metadata.update({
                    'qid': f'sample_q_{idx}',
                    'original_cid': f'sample_c_{idx}',
                    'context_idx': 0,
                    'chunk_idx': chunk_idx,
                    'source': f"sample_legal_{idx}_chunk_{chunk_idx}",
                    'related_question': question,
                    'document_type': doc_type,
                    'chunk_size': len(chunk),
                    'original_context_size': len(context),
                    'is_sample_data': True
                })
                
                doc = Document(
                    page_content=chunk.strip(),
                    metadata=metadata
                )
                
                documents.append(doc)
        
        print(f"✅ Đã tạo {len(documents)} sample documents về pháp luật Việt Nam")
        return documents
    
    def get_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Tính toán thống kê về dataset
        """
        if not documents:
            return {}
        
        doc_types = {}
        chunk_sizes = []
        word_counts = []
        has_dieu = 0
        
        for doc in documents:
            # Document type stats
            doc_type = doc.metadata.get('document_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            # Size stats
            chunk_sizes.append(doc.metadata.get('chunk_size', len(doc.page_content)))
            word_counts.append(doc.metadata.get('word_count', len(doc.page_content.split())))
            
            # Structure stats
            if doc.metadata.get('dieu_so'):
                has_dieu += 1
        
        return {
            'total_documents': len(documents),
            'document_types': doc_types,
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'documents_with_dieu': has_dieu,
            'coverage_percentage': (has_dieu / len(documents)) * 100 if documents else 0
        }

if __name__ == "__main__":
    # Test data loader
    loader = YuITCDataLoader(chunk_size=512, chunk_overlap=100, max_documents=10)
    documents = loader.load_dataset()
    
    print(f"\n Dataset Statistics:")
    stats = loader.get_statistics(documents)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n Sample Documents:")
    for i, doc in enumerate(documents[:3]):
        print(f"\n--- Document {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Type: {doc.metadata.get('document_type', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {dict(list(doc.metadata.items())[:5])}")  # First 5 metadata items