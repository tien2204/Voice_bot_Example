# mcp/memory.py - Session Memory Management
# ========================================

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import hashlib
from collections import deque
from dataclasses import dataclass, asdict

@dataclass
class ConversationMessage:
    """Structured conversation message"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SessionMemory:
    """
    MCP Tool cho Session Memory & Context Management:
    1. Conversation history management
    2. Context window optimization  
    3. Long-term memory storage
    4. Smart context retrieval
    """
    
    def __init__(self, max_messages: int = 50, context_window: int = 10, 
                 memory_file: str = "memory/session_memory.json"):
        self.max_messages = max_messages
        self.context_window = context_window
        self.memory_file = memory_file
        
        # In-memory storage
        self.conversation_history = deque(maxlen=max_messages)
        self.long_term_memory = {}  # topic -> messages
        self.session_metadata = {}
        
        # Session info
        self.session_id = self._generate_session_id()
        self.session_start = datetime.now()
        
        # Load existing memory if available
        self._load_memory()
        
        print(f" SessionMemory initialized - ID: {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        MCP Tool: Thêm message vào conversation history
        """
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.conversation_history.append(message)
        
        # Update long-term memory nếu có topics
        if metadata and 'topics' in metadata:
            self._update_long_term_memory(message, metadata['topics'])
        
        # Auto-save periodically
        if len(self.conversation_history) % 10 == 0:
            self._save_memory()
        
        return {
            'added': True,
            'message_count': len(self.conversation_history),
            'session_id': self.session_id
        }
    
    def get_context(self, last_n: Optional[int] = None, include_system: bool = False) -> str:
        """
        MCP Tool: Lấy context cho LLM (formatted string)
        """
        n = last_n or self.context_window
        recent_messages = list(self.conversation_history)[-n:]
        
        if not include_system:
            recent_messages = [msg for msg in recent_messages if msg.role != 'system']
        
        # Format context
        context_parts = []
        for msg in recent_messages:
            timestamp = msg.timestamp[:19]  # Remove microseconds
            context_parts.append(f"[{timestamp}] {msg.role.upper()}: {msg.content}")
        
        return "\n".join(context_parts)
    
    def get_context_messages(self, last_n: Optional[int] = None, 
                           include_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        MCP Tool: Lấy context dưới dạng structured messages
        """
        n = last_n or self.context_window
        recent_messages = list(self.conversation_history)[-n:]
        
        if include_metadata:
            return [msg.to_dict() for msg in recent_messages]
        else:
            return [
                {
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp
                }
                for msg in recent_messages
            ]
    
    def get_last_qa(self) -> Optional[Dict[str, str]]:
        """
        MCP Tool: Lấy Q&A cuối cùng
        """
        messages = list(self.conversation_history)
        
        # Tìm user message cuối và assistant response
        last_user = None
        last_assistant = None
        
        for msg in reversed(messages):
            if msg.role == 'assistant' and last_assistant is None:
                last_assistant = msg
            elif msg.role == 'user' and last_user is None:
                last_user = msg
            
            if last_user and last_assistant:
                break
        
        if last_user and last_assistant:
            return {
                'question': last_user.content,
                'answer': last_assistant.content,
                'question_time': last_user.timestamp,
                'answer_time': last_assistant.timestamp
            }
        
        return None
    
    def _update_long_term_memory(self, message: ConversationMessage, topics: List[str]):
        """Update long-term memory với topics"""
        for topic in topics:
            if topic not in self.long_term_memory:
                self.long_term_memory[topic] = deque(maxlen=20)  # Max 20 per topic
            
            self.long_term_memory[topic].append({
                'role': message.role,
                'content': message.content[:200],  # Truncate for storage
                'timestamp': message.timestamp,
                'session_id': self.session_id
            })
    
    def search_memory(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        MCP Tool: Tìm kiếm trong memory
        """
        results = []
        query_lower = query.lower()
        
        # Search conversation history
        for msg in self.conversation_history:
            if query_lower in msg.content.lower():
                results.append({
                    'source': 'conversation_history',
                    'role': msg.role,
                    'content': msg.content,
                    'timestamp': msg.timestamp,
                    'relevance': 'high' if query_lower in msg.content.lower()[:100] else 'medium'
                })
        
        # Search long-term memory
        for topic, messages in self.long_term_memory.items():
            if query_lower in topic.lower():
                for msg in messages:
                    if query_lower in msg['content'].lower():
                        results.append({
                            'source': 'long_term_memory',
                            'topic': topic,
                            'role': msg['role'],
                            'content': msg['content'],
                            'timestamp': msg['timestamp'],
                            'relevance': 'medium'
                        })
        
        # Sort by relevance and timestamp
        results.sort(key=lambda x: (x['relevance'] == 'high', x['timestamp']), reverse=True)
        
        return results[:max_results]
    
    def get_topic_context(self, topic: str, max_messages: int = 5) -> List[Dict[str, Any]]:
        """
        MCP Tool: Lấy context theo topic từ long-term memory
        """
        if topic not in self.long_term_memory:
            return []
        
        topic_messages = list(self.long_term_memory[topic])[-max_messages:]
        return topic_messages
    
    def analyze_conversation_patterns(self) -> Dict[str, Any]:
        """
        MCP Tool: Phân tích patterns trong conversation
        """
        if not self.conversation_history:
            return {'error': 'No conversation history'}
        
        messages = list(self.conversation_history)
        
        # Basic stats
        user_messages = [msg for msg in messages if msg.role == 'user']
        assistant_messages = [msg for msg in messages if msg.role == 'assistant']
        
        # Question types analysis
        question_types = {}
        legal_topics = {}
        
        for msg in user_messages:
            content_lower = msg.content.lower()
            
            # Classify question type
            qtype = self._classify_question_type(content_lower)
            question_types[qtype] = question_types.get(qtype, 0) + 1
            
            # Extract legal topics
            topics = self._extract_legal_topics(content_lower)
            for topic in topics:
                legal_topics[topic] = legal_topics.get(topic, 0) + 1
        
        # Conversation flow analysis
        conversation_length = len(messages)
        avg_response_length = sum(len(msg.content) for msg in assistant_messages) / len(assistant_messages) if assistant_messages else 0
        avg_question_length = sum(len(msg.content) for msg in user_messages) / len(user_messages) if user_messages else 0
        
        # Time analysis
        if len(messages) > 1:
            time_span = datetime.fromisoformat(messages[-1].timestamp) - datetime.fromisoformat(messages[0].timestamp)
            session_duration = time_span.total_seconds() / 60  # minutes
        else:
            session_duration = 0
        
        analysis = {
            'conversation_stats': {
                'total_messages': conversation_length,
                'user_messages': len(user_messages),
                'assistant_messages': len(assistant_messages),
                'session_duration_minutes': round(session_duration, 2),
                'avg_question_length': round(avg_question_length, 1),
                'avg_response_length': round(avg_response_length, 1)
            },
            'question_types': dict(sorted(question_types.items(), key=lambda x: x[1], reverse=True)),
            'legal_topics': dict(sorted(legal_topics.items(), key=lambda x: x[1], reverse=True)[:10]),
            'long_term_topics': list(self.long_term_memory.keys()),
            'session_info': {
                'session_id': self.session_id,
                'session_start': self.session_start.isoformat(),
                'messages_in_memory': len(self.conversation_history)
            }
        }
        
        return analysis
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type"""
        if any(word in question for word in ['điều', 'khoản']):
            return 'specific_article'
        elif any(word in question for word in ['là gì', 'định nghĩa']):
            return 'definition'
        elif any(word in question for word in ['thủ tục', 'cách']):
            return 'procedure'
        elif any(word in question for word in ['quyền', 'nghĩa vụ']):
            return 'rights'
        elif any(word in question for word in ['so sánh', 'khác']):
            return 'comparison'
        else:
            return 'general'
    
    def _extract_legal_topics(self, text: str) -> List[str]:
        """Extract legal topics from text"""
        topics = []
        legal_keywords = {
            'hình sự': 'criminal_law',
            'dân sự': 'civil_law', 
            'hôn nhân': 'family_law',
            'lao động': 'labor_law',
            'hiến pháp': 'constitutional_law',
            'hành chính': 'administrative_law',
            'kinh tế': 'economic_law',
            'môi trường': 'environmental_law',
            'đầu tư': 'investment_law',
            'thuế': 'tax_law'
        }
        
        for keyword, topic in legal_keywords.items():
            if keyword in text:
                topics.append(topic)
        
        return topics
    
    def optimize_context_window(self, target_tokens: int = 2000) -> Dict[str, Any]:
        """
        MCP Tool: Tối ưu context window cho LLM
        """
        messages = list(self.conversation_history)
        
        # Estimate tokens (rough calculation: 1 token ≈ 4 characters Vietnamese)
        estimated_tokens = 0
        selected_messages = []
        
        # Start from most recent messages
        for msg in reversed(messages):
            msg_tokens = len(msg.content) // 4
            if estimated_tokens + msg_tokens <= target_tokens:
                selected_messages.insert(0, msg)
                estimated_tokens += msg_tokens
            else:
                break
        
        # Ensure we have complete conversations (user -> assistant pairs)
        optimized_messages = self._ensure_complete_conversations(selected_messages)
        
        result = {
            'optimized_message_count': len(optimized_messages),
            'estimated_tokens': sum(len(msg.content) // 4 for msg in optimized_messages),
            'target_tokens': target_tokens,
            'dropped_messages': len(messages) - len(optimized_messages),
            'context_text': self._messages_to_text(optimized_messages)
        }
        
        return result
    
    def _ensure_complete_conversations(self, messages: List[ConversationMessage]) -> List[ConversationMessage]:
        """Ensure context has complete user->assistant pairs"""
        complete_messages = []
        i = 0
        
        while i < len(messages):
            msg = messages[i]
            complete_messages.append(msg)
            
            # If user message, try to include corresponding assistant response
            if msg.role == 'user' and i + 1 < len(messages):
                next_msg = messages[i + 1]
                if next_msg.role == 'assistant':
                    complete_messages.append(next_msg)
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        return complete_messages
    
    def _messages_to_text(self, messages: List[ConversationMessage]) -> str:
        """Convert messages to formatted text"""
        text_parts = []
        for msg in messages:
            text_parts.append(f"{msg.role.upper()}: {msg.content}")
        return "\n\n".join(text_parts)
    
    def clear_memory(self, keep_last_n: int = 0) -> Dict[str, Any]:
        """
        MCP Tool: Xóa memory (giữ lại n messages cuối)
        """
        old_count = len(self.conversation_history)
        
        if keep_last_n > 0:
            # Keep last n messages
            messages_to_keep = list(self.conversation_history)[-keep_last_n:]
            self.conversation_history.clear()
            for msg in messages_to_keep:
                self.conversation_history.append(msg)
        else:
            # Clear all
            self.conversation_history.clear()
        
        # Save state
        self._save_memory()
        
        return {
            'cleared': True,
            'old_message_count': old_count,
            'current_message_count': len(self.conversation_history),
            'kept_messages': keep_last_n
        }
    
    def _save_memory(self):
        """Save memory to file"""
        try:
            memory_data = {
                'session_id': self.session_id,
                'session_start': self.session_start.isoformat(),
                'last_updated': datetime.now().isoformat(),
                'conversation_history': [msg.to_dict() for msg in self.conversation_history],
                'long_term_memory': {
                    topic: list(messages) for topic, messages in self.long_term_memory.items()
                },
                'session_metadata': self.session_metadata
            }
            
            import os
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            print(f" Failed to save memory: {e}")
    
    def _load_memory(self):
        """Load memory from file"""
        try:
            import os
            if not os.path.exists(self.memory_file):
                return
            
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # Load conversation history
            if 'conversation_history' in memory_data:
                for msg_data in memory_data['conversation_history']:
                    msg = ConversationMessage(**msg_data)
                    self.conversation_history.append(msg)
            
            # Load long-term memory
            if 'long_term_memory' in memory_data:
                for topic, messages in memory_data['long_term_memory'].items():
                    self.long_term_memory[topic] = deque(messages, maxlen=20)
            
            # Load metadata
            if 'session_metadata' in memory_data:
                self.session_metadata = memory_data['session_metadata']
            
            print(f" Loaded {len(self.conversation_history)} messages from memory")
        
        except Exception as e:
            print(f" Failed to load memory: {e}")

if __name__ == "__main__":
    # Test SessionMemory
    print(" Testing SessionMemory...")
    
    # Initialize
    memory = SessionMemory(
        max_messages=20,
        context_window=5,
        memory_file="test_memory/session_memory.json"
    )
    
    # Test conversation
    test_conversation = [
        ('user', 'Tuổi chịu trách nhiệm hình sự là bao nhiêu?', {'topics': ['criminal_law']}),
        ('assistant', 'Theo Điều 15 Bộ luật Hình sự, người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự.', None),
        ('user', 'Điều kiện kết hôn như thế nào?', {'topics': ['family_law']}),
        ('assistant', 'Theo Luật Hôn nhân và Gia đình, nam từ đủ 20 tuổi, nữ từ đủ 18 tuổi.', None),
        ('user', 'So sánh hình phạt tù có thời hạn và tù chung thân?', {'topics': ['criminal_law']}),
        ('assistant', 'Tù có thời hạn từ 3 tháng đến 20 năm, tù chung thân là hình phạt không thời hạn.', None)
    ]
    
    print(f"\n Adding test conversation...")
    for role, content, metadata in test_conversation:
        result = memory.add_message(role, content, metadata)
        print(f" Added {role} message: {result['message_count']} total")
    
    # Test context retrieval
    print(f"\n Testing context retrieval...")
    context_str = memory.get_context(last_n=4)
    print(f"  Context (4 messages):\n{context_str[:200]}...")
    
    # Test structured context
    context_msgs = memory.get_context_messages(last_n=3, include_metadata=True)
    print(f"  Structured context: {len(context_msgs)} messages")
    
    # Test last Q&A
    last_qa = memory.get_last_qa()
    if last_qa:
        print(f"  Last Q&A: {last_qa['question'][:30]}... -> {last_qa['answer'][:30]}...")
    
    # Test memory search
    print(f"\n Testing memory search...")
    search_results = memory.search_memory("hình phạt", max_results=3)
    print(f"  Found {len(search_results)} results for 'hình phạt'")
    
    # Test conversation analysis
    print(f"\n Testing conversation analysis...")
    analysis = memory.analyze_conversation_patterns()
    print(f"  - Total messages: {analysis['conversation_stats']['total_messages']}")
    print(f"  - Question types: {analysis['question_types']}")
    print(f"  - Legal topics: {analysis['legal_topics']}")
    
    # Test context optimization
    print(f"\n Testing context optimization...")
    optimized = memory.optimize_context_window(target_tokens=500)
    print(f"  - Optimized to {optimized['optimized_message_count']} messages")
    print(f"  - Estimated tokens: {optimized['estimated_tokens']}")
    
    print(f"\n SessionMemory test completed!")