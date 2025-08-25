# main.py - Entry Point cho Vietnamese Legal RAG System với MCP
# ================================================================

import torch
import warnings
from pathlib import Path
import json
import sys
import os

warnings.filterwarnings('ignore')

# Import các modules của project
from data_loader import YuITCDataLoader
from weaviate_setup import WeaviateSetup
from retriever import HybridRetriever
from llm import VinaLlamaLLM
from tts import VietnameseF5TTS

# Import MCP tools
from mcp.meta_reasoning import MetaReasoningTool
from mcp.logging_tools import LoggingTool
from mcp.memory import SessionMemory
from mcp.hybrid_sources import HybridKnowledgeTool
from mcp.tts_tool import TTSTool
from mcp.feedback import FeedbackTool

class EnhancedVietnameseLegalRAG:
    """
    Enhanced Vietnamese Legal RAG System với MCP integration
    Sử dụng dataset YuITC Vietnamese Legal Document Retrieval Data
    """
    
    def __init__(self, config_path: str = "config.json"):
        print(" Khởi tạo Enhanced Vietnamese Legal RAG System với MCP...")
        
        # Load config
        self.config = self.load_config(config_path)
        
        # Setup core components
        self.setup_core_components()
        
        # Setup MCP tools
        self.setup_mcp_tools()
        
        # Load and process YuITC dataset
        self.load_yuítc_dataset()
        
        print(" Hệ thống sẵn sàng với MCP tools")
        self.print_available_tools()
    
    def load_config(self, config_path: str) -> dict:
        """Load cấu hình hệ thống"""
        default_config = {
            "chunk_size": 512,
            "chunk_overlap": 100,
            "embedding_model": "BAAI/bge-m3",
            "llm_model": "vilm/vinallama-2.7b-chat",
            "tts_model": "hynt/F5-TTS-Vietnamese-ViVoice",
            "vector_store": "weaviate",
            "retrieval_k": 5,
            "max_documents": 1000,  # Giới hạn để demo
            "log_file": "logs/session.jsonl",
            "feedback_file": "logs/feedback.jsonl"
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def setup_core_components(self):
        """Setup các component cốt lõi"""
        print(" Setup core components...")
        
        # Data loader
        self.data_loader = YuITCDataLoader(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            max_documents=self.config["max_documents"]
        )
        
        # Vector store
        self.weaviate_setup = WeaviateSetup()
        
        # Retriever
        self.retriever = HybridRetriever(
            weaviate_client=self.weaviate_setup.client,
            embedding_model=self.config["embedding_model"]
        )
        
        # LLM
        self.llm = VinaLlamaLLM(
            model_name=self.config["llm_model"]
        )
        
        # TTS with F5-TTS
        self.tts = VietnameseF5TTS(
            model_name=self.config["tts_model"]
        )
    
    def setup_mcp_tools(self):
        """Setup MCP tools"""
        print(" Setup MCP tools...")
        
        # Tạo thư mục logs nếu chưa có
        Path("logs").mkdir(exist_ok=True)
        
        # Meta-reasoning tools
        self.meta_reasoning = MetaReasoningTool(
            retriever=self.retriever,
            llm=self.llm
        )
        
        # Logging và analytics
        self.logger = LoggingTool(
            log_file=self.config["log_file"]
        )
        
        # Session memory
        self.memory = SessionMemory()
        
        # Hybrid knowledge sources
        self.hybrid_sources = HybridKnowledgeTool()
        
        # TTS tool
        self.tts_tool = TTSTool(self.tts)
        
        # Feedback system
        self.feedback = FeedbackTool(
            feedback_file=self.config["feedback_file"]
        )
    
    def load_yuítc_dataset(self):
        """Load và process YuITC dataset"""
        print(" Loading YuITC Vietnamese Legal Document Retrieval Data...")
        
        # Load dataset
        documents = self.data_loader.load_dataset()
        
        # Add to retriever
        self.retriever.add_documents(documents)
        
        print(f" Đã load {len(documents)} chunks từ YuITC dataset")
    
    def print_available_tools(self):
        """In danh sách MCP tools có sẵn"""
        print("\n MCP TOOLS CÓ SẴN:")
        print("=" * 50)
        print("1.explain_retrieval(query) - Giải thích tại sao tài liệu được chọn")
        print("2.debug_prompt(query) - Hiển thị full prompt context")
        print("3.view_session_log() - Xem lịch sử session")
        print("4.get_context_memory() - Xem memory ngữ cảnh")
        print("5.fetch_law_news() - Lấy tin tức pháp luật mới")
        print("6.speak_text(text) - Convert text to speech")
        print("7.submit_feedback(rating, comment) - Đánh giá câu trả lời")
        print("8.export_analytics() - Export dữ liệu phân tích")
        print("=" * 50)
    
    def query(self, question: str, use_mcp: bool = True, return_audio: bool = True):
        """
        Main query function với MCP integration
        """
        print(f" Câu hỏi: {question}")
        
        # Add to session memory
        if use_mcp:
            self.memory.add_message("user", question)
        
        try:
            # Retrieve relevant documents
            print(" Đang tìm kiếm tài liệu liên quan...")
            relevant_docs = self.retriever.retrieve(question, k=self.config["retrieval_k"])
            
            if not relevant_docs:
                print(" Không tìm thấy tài liệu liên quan")
                return self._create_response(question, "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu pháp luật.")
            
            # Debug info nếu sử dụng MCP
            if use_mcp:
                print(f" Tìm thấy {len(relevant_docs)} tài liệu liên quan:")
                for i, doc in enumerate(relevant_docs[:3], 1):
                    method = doc.metadata.get('method', 'unknown')
                    score = doc.metadata.get('score', 0)
                    preview = doc.page_content[:100].replace('\n', ' ')
                    print(f"  {i}. [{method}|{score:.3f}] {preview}...")
            
            # Create context
            context = self._create_context(relevant_docs)
            
            # Debug prompt nếu sử dụng MCP
            if use_mcp:
                debug_prompt = self.meta_reasoning.debug_prompt(question, context)
                print(f" Debug prompt length: {len(debug_prompt)} chars")
            
            # Generate response
            print(" Đang tạo câu trả lời...")
            prompt = self._create_prompt(question, context)
            response = self.llm.generate(prompt)
            
            # Add to session memory
            if use_mcp:
                self.memory.add_message("assistant", response)
                
                # Log session
                self.logger.log_session(
                    question=question,
                    answer=response,
                    context=context[:500],  # Truncate for storage
                    retrieved_docs=[doc.page_content[:200] for doc in relevant_docs]
                )
            
            print(f" Câu trả lời: {response}")
            
            # Generate audio nếu được yêu cầu
            audio_file = None
            if return_audio:
                print(" Đang tạo file âm thanh...")
                audio_file = self.tts_tool.speak(response)
                if audio_file:
                    print(f" Đã tạo file âm thanh: {audio_file}")
            
            return self._create_response(question, response, context, relevant_docs, audio_file)
            
        except Exception as e:
            error_msg = f"Đã xảy ra lỗi: {str(e)}"
            print(f" {error_msg}")
            return self._create_response(question, error_msg)
    
    def _create_context(self, documents):
        """Tạo context từ retrieved documents"""
        contexts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Không xác định')
            content = doc.page_content.strip()
            contexts.append(f"[Tài liệu {i} - {source}]\n{content}")
        
        return "\n\n".join(contexts)
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Tạo prompt cho LLM"""
        return f"""Câu hỏi: {question}

Văn bản pháp luật liên quan:
{context}

Hãy trả lời câu hỏi dựa trên văn bản pháp luật được cung cấp. Nếu thông tin không đủ để trả lời chính xác, hãy nói rõ điều đó. Trả lời bằng tiếng Việt và viện dẫn đúng điều khoản pháp luật nếu có.

Trả lời:"""
    
    def _create_response(self, question, answer, context=None, docs=None, audio_file=None):
        """Tạo response object"""
        return {
            "question": question,
            "answer": answer,
            "context": context,
            "retrieved_docs": len(docs) if docs else 0,
            "audio_file": audio_file,
            "timestamp": self.logger.get_timestamp(),
            "retrieved_content": [doc.page_content for doc in docs] if docs else []
        }
    
    # MCP Tool Methods
    def explain_retrieval(self, query: str):
        """MCP Tool: Giải thích tại sao tài liệu được chọn"""
        return self.meta_reasoning.explain_retrieval(query, k=self.config["retrieval_k"])
    
    def debug_prompt(self, query: str):
        """MCP Tool: Debug prompt context"""
        relevant_docs = self.retriever.retrieve(query, k=self.config["retrieval_k"])
        context = self._create_context(relevant_docs)
        return self.meta_reasoning.debug_prompt(query, context)
    
    def view_session_log(self):
        """MCP Tool: Xem session log"""
        return self.logger.get_recent_sessions(n=10)
    
    def get_context_memory(self):
        """MCP Tool: Xem context memory"""
        return self.memory.get_context()
    
    def fetch_law_news(self):
        """MCP Tool: Lấy tin tức pháp luật"""
        return self.hybrid_sources.fetch_law_news()
    
    def speak_text(self, text: str):
        """MCP Tool: Convert text to speech"""
        return self.tts_tool.speak(text)
    
    def submit_feedback(self, rating: int, comment: str = None):
        """MCP Tool: Submit feedback"""
        last_qa = self.memory.get_last_qa()
        if last_qa:
            return self.feedback.submit_feedback(
                question=last_qa["question"],
                answer=last_qa["answer"],
                rating=rating,
                comment=comment
            )
        return {"error": "Không có QA gần đây để đánh giá"}
    
    def export_analytics(self):
        """MCP Tool: Export analytics data"""
        return self.logger.export_analytics()

def run_demo():
    """Chạy demo với các câu hỏi mẫu"""
    print("\n" + "="*60)
    print(" DEMO ENHANCED VIETNAMESE LEGAL RAG SYSTEM với MCP")
    print("="*60)
    
    # Khởi tạo hệ thống
    rag_system = EnhancedVietnameseLegalRAG()
    
    # Câu hỏi demo
    demo_questions = [
        "Quyền và nghĩa vụ cơ bản của công dân Việt Nam là gì?",
        "Luật hình sự quy định những tội phạm nào?",
        "Thủ tục đăng ký kết hôn như thế nào?",
        "Điều kiện để được hưởng bảo hiểm xã hội là gì?",
        "Quyền sở hữu trí tuệ được bảo vệ như thế nào?"
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n--- Demo {i} ---")
        try:
            result = rag_system.query(question, return_audio=False)
            
            # Demo MCP tools
            print("\n MCP Tools Demo:")
            print("1. Explain Retrieval:")
            explain = rag_system.explain_retrieval(question)
            for j, item in enumerate(explain[:2], 1):
                print(f"   {j}. [{item['method']}|{item['score']:.3f}] {item['doc_preview']}...")
            
            print("-" * 50)
            
        except Exception as e:
            print(f" Lỗi trong demo {i}: {e}")

def run_interactive():
    """Chế độ interactive với MCP tools"""
    print("\n" + "="*60)
    print(" CHẾ ĐỘ TÂM SỰ PHÁP LUẬT với MCP TOOLS")
    print("="*60)
    print("Lệnh đặc biệt:")
    print("  'mcp <tool_name>' - Gọi MCP tool")
    print("  'help' - Hiển thị trợ giúp")
    print("  'quit' - Thoát")
    print("="*60)
    
    # Khởi tạo hệ thống
    rag_system = EnhancedVietnameseLegalRAG()
    
    while True:
        try:
            user_input = input("\n Câu hỏi/Lệnh: ").strip()
            
            if not user_input:
                continue
                
            # Check exit
            if user_input.lower() in ['quit', 'exit', 'thoat', 'q']:
                break
                
            # Check help
            if user_input.lower() == 'help':
                rag_system.print_available_tools()
                continue
            
            # Check MCP commands
            if user_input.startswith('mcp '):
                command = user_input[4:].strip()
                result = handle_mcp_command(rag_system, command)
                print(f" MCP Result: {result}")
                continue
            
            # Regular query
            result = rag_system.query(user_input)
            
            # Offer feedback
            feedback = input("\n Đánh giá (1-5, enter để skip): ").strip()
            if feedback.isdigit() and 1 <= int(feedback) <= 5:
                comment = input(" Nhận xét (optional): ").strip()
                rag_system.submit_feedback(int(feedback), comment or None)
                print(" Cảm ơn feedback!")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f" Lỗi: {e}")
    
    print("\n Cảm ơn bạn đã sử dụng hệ thống!")

def handle_mcp_command(rag_system, command):
    """Xử lý MCP commands"""
    parts = command.split()
    if not parts:
        return "Lệnh không hợp lệ"
    
    tool_name = parts[0]
    args = parts[1:] if len(parts) > 1 else []
    
    try:
        if tool_name == 'explain_retrieval' and args:
            return rag_system.explain_retrieval(' '.join(args))
        elif tool_name == 'debug_prompt' and args:
            return rag_system.debug_prompt(' '.join(args))
        elif tool_name == 'view_session_log':
            return rag_system.view_session_log()
        elif tool_name == 'get_context_memory':
            return rag_system.get_context_memory()
        elif tool_name == 'fetch_law_news':
            return rag_system.fetch_law_news()
        elif tool_name == 'speak_text' and args:
            return rag_system.speak_text(' '.join(args))
        elif tool_name == 'export_analytics':
            return rag_system.export_analytics()
        else:
            return f"Tool không tồn tại hoặc thiếu tham số: {tool_name}"
    except Exception as e:
        return f"Lỗi MCP tool: {e}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            run_demo()
        elif sys.argv[1] == "interactive":
            run_interactive()
        else:
            print("Usage: python main.py [demo|interactive]")
    else:
        run_interactive()