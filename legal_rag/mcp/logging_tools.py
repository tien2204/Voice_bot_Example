# mcp/logging_tools.py - Logging và Analytics Tools
# =================================================

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib
import statistics

class LoggingTool:
    """
    MCP Tools cho logging và analytics:
    1. Session logging - lưu Q&A, context, performance
    2. Analytics - phân tích patterns, performance metrics
    3. Export capabilities - xuất dữ liệu cho monitoring systems
    """
    
    def __init__(self, log_file: str = "logs/session.jsonl", 
                 metrics_file: str = "logs/metrics.jsonl",
                 analytics_file: str = "logs/analytics.json"):
        self.log_file = Path(log_file)
        self.metrics_file = Path(metrics_file)
        self.analytics_file = Path(analytics_file)
        
        # Create directories
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.analytics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize session
        self.session_id = self._generate_session_id()
        self.session_start_time = datetime.now()
        self.current_session_logs = []
        
        print(f" LoggingTool initialized - Session: {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]
    
    def get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()
    
    def log_session(self, question: str, answer: str, context: str, 
                   retrieved_docs: List[str], metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        MCP Tool: Log một session Q&A với full context
        """
        log_entry = {
            'session_id': self.session_id,
            'timestamp': self.get_timestamp(),
            'question': question,
            'answer': answer,
            'context': context[:1000],  # Truncate long context
            'context_length': len(context),
            'retrieved_docs_count': len(retrieved_docs),
            'retrieved_docs_preview': [doc[:200] for doc in retrieved_docs[:3]],  # First 3 docs preview
            'question_length': len(question),
            'answer_length': len(answer),
            'question_word_count': len(question.split()),
            'answer_word_count': len(answer.split()),
            'metadata': metadata or {}
        }
        
        # Add to current session
        self.current_session_logs.append(log_entry)
        
        # Write to file
        self._write_log_entry(log_entry)
        
        # Log metrics
        self._log_metrics(log_entry)
        
        return {
            'logged': True,
            'session_id': self.session_id,
            'log_count': len(self.current_session_logs)
        }
    
    def _write_log_entry(self, log_entry: Dict[str, Any]):
        """Write log entry to JSONL file"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f" Failed to write log: {e}")
    
    def _log_metrics(self, log_entry: Dict[str, Any]):
        """Log performance metrics"""
        metrics = {
            'session_id': self.session_id,
            'timestamp': log_entry['timestamp'],
            'question_length': log_entry['question_length'],
            'answer_length': log_entry['answer_length'],
            'context_length': log_entry['context_length'],
            'retrieved_docs_count': log_entry['retrieved_docs_count'],
            'processing_time': None,  # Would be filled by caller
            'question_type': self._classify_question_type(log_entry['question']),
            'answer_quality_estimate': self._estimate_answer_quality(log_entry)
        }
        
        try:
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f" Failed to write metrics: {e}")
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type for metrics"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['điều', 'khoản']):
            return 'specific_article'
        elif any(word in question_lower for word in ['là gì', 'định nghĩa']):
            return 'definition'
        elif any(word in question_lower for word in ['thủ tục', 'cách']):
            return 'procedure'
        elif any(word in question_lower for word in ['quyền', 'nghĩa vụ']):
            return 'rights'
        else:
            return 'general'
    
    def _estimate_answer_quality(self, log_entry: Dict[str, Any]) -> str:
        """Estimate answer quality based on heuristics"""
        answer = log_entry['answer']
        context_length = log_entry['context_length']
        retrieved_count = log_entry['retrieved_docs_count']
        
        score = 0
        
        # Length checks
        if 50 <= len(answer) <= 500:
            score += 1
        
        # Context availability
        if context_length > 100:
            score += 1
        
        # Retrieved documents
        if retrieved_count >= 3:
            score += 1
        
        # Content checks
        if any(word in answer.lower() for word in ['điều', 'luật', 'quy định']):
            score += 1
        
        # Avoid generic responses
        if not any(phrase in answer.lower() for phrase in ['xin lỗi', 'không thể', 'không biết']):
            score += 1
        
        if score >= 4:
            return 'high'
        elif score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def get_recent_sessions(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        MCP Tool: Lấy n sessions gần đây
        """
        if not self.log_file.exists():
            return []
        
        try:
            sessions = []
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        session = json.loads(line.strip())
                        sessions.append(session)
                    except json.JSONDecodeError:
                        continue
            
            # Return recent sessions
            return sessions[-n:] if sessions else []
            
        except Exception as e:
            print(f" Failed to read sessions: {e}")
            return []
    
    def get_session_analytics(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        MCP Tool: Phân tích sessions trong X giờ gần đây
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        sessions = self.get_recent_sessions(1000)  # Get more for analysis
        
        # Filter by time
        recent_sessions = []
        for session in sessions:
            try:
                session_time = datetime.fromisoformat(session['timestamp'])
                if session_time >= cutoff_time:
                    recent_sessions.append(session)
            except:
                continue
        
        if not recent_sessions:
            return {'error': 'No sessions found in time range'}
        
        # Analyze
        analytics = {
            'time_period': f"Last {hours_back} hours",
            'total_sessions': len(recent_sessions),
            'unique_session_ids': len(set(s['session_id'] for s in recent_sessions)),
            'question_types': self._analyze_question_types(recent_sessions),
            'performance_metrics': self._analyze_performance(recent_sessions),
            'common_topics': self._analyze_topics(recent_sessions),
            'quality_distribution': self._analyze_quality_distribution(recent_sessions),
            'generated_at': self.get_timestamp()
        }
        
        return analytics
    
    def _analyze_question_types(self, sessions: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of question types"""
        types = {}
        for session in sessions:
            qtype = self._classify_question_type(session['question'])
            types[qtype] = types.get(qtype, 0) + 1
        return types
    
    def _analyze_performance(self, sessions: List[Dict]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        if not sessions:
            return {}
        
        question_lengths = [s['question_length'] for s in sessions]
        answer_lengths = [s['answer_length'] for s in sessions]
        context_lengths = [s['context_length'] for s in sessions]
        doc_counts = [s['retrieved_docs_count'] for s in sessions]
        
        return {
            'avg_question_length': statistics.mean(question_lengths),
            'avg_answer_length': statistics.mean(answer_lengths),
            'avg_context_length': statistics.mean(context_lengths),
            'avg_retrieved_docs': statistics.mean(doc_counts),
            'median_question_length': statistics.median(question_lengths),
            'median_answer_length': statistics.median(answer_lengths)
        }
    
    def _analyze_topics(self, sessions: List[Dict]) -> Dict[str, int]:
        """Analyze common topics/keywords"""
        topics = {}
        legal_keywords = [
            'hình sự', 'dân sự', 'hôn nhân', 'gia đình', 'lao động', 
            'hiến pháp', 'tù', 'phạt', 'quyền', 'nghĩa vụ',
            'luật', 'nghị định', 'thông tư', 'điều', 'khoản'
        ]
        
        for session in sessions:
            question = session['question'].lower()
            for keyword in legal_keywords:
                if keyword in question:
                    topics[keyword] = topics.get(keyword, 0) + 1
        
        # Sort by frequency
        sorted_topics = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True))
        return dict(list(sorted_topics.items())[:10])  # Top 10
    
    def _analyze_quality_distribution(self, sessions: List[Dict]) -> Dict[str, int]:
        """Analyze quality distribution"""
        quality_dist = {'high': 0, 'medium': 0, 'low': 0}
        
        for session in sessions:
            quality = self._estimate_answer_quality(session)
            quality_dist[quality] += 1
        
        return quality_dist
    
    def export_analytics(self, format_type: str = "json") -> str:
        """
        MCP Tool: Export analytics data
        """
        analytics = {
            'export_timestamp': self.get_timestamp(),
            'session_analytics_24h': self.get_session_analytics(24),
            'session_analytics_7d': self.get_session_analytics(24 * 7),
            'recent_sessions': self.get_recent_sessions(50),
            'current_session_id': self.session_id,
            'current_session_logs': len(self.current_session_logs)
        }
        
        if format_type.lower() == "json":
            # Save to analytics file
            with open(self.analytics_file, 'w', encoding='utf-8') as f:
                json.dump(analytics, f, ensure_ascii=False, indent=2)
            
            print(f" Analytics exported to: {self.analytics_file}")
            return str(self.analytics_file)
        
        else:
            print(f" Unsupported format: {format_type}")
            return ""
    
    def export_to_elastic(self, elasticsearch_url: str = None) -> Dict[str, Any]:
        """
        MCP Tool: Export to ElasticSearch (placeholder)
        """
        # This would implement actual ElasticSearch integration
        recent_sessions = self.get_recent_sessions(100)
        
        # Simulate export
        export_result = {
            'success': False,  # Would be True if actual export succeeds
            'message': 'ElasticSearch integration not implemented',
            'sessions_to_export': len(recent_sessions),
            'elasticsearch_url': elasticsearch_url,
            'would_export': [
                {
                    'index': 'legal-rag-sessions',
                    'doc_type': 'session_log',
                    'documents': len(recent_sessions)
                }
            ]
        }
        
        print(f" ElasticSearch export simulation: {len(recent_sessions)} sessions")
        return export_result
    
    def export_to_prometheus(self, pushgateway_url: str = None) -> Dict[str, Any]:
        """
        MCP Tool: Push metrics to Prometheus (placeholder)
        """
        analytics = self.get_session_analytics(1)  # Last hour
        
        # Simulate Prometheus metrics push
        metrics_to_push = {
            'legal_rag_sessions_total': analytics.get('total_sessions', 0),
            'legal_rag_avg_answer_length': analytics.get('performance_metrics', {}).get('avg_answer_length', 0),
            'legal_rag_quality_high_ratio': self._calculate_quality_ratio(analytics, 'high')
        }
        
        export_result = {
            'success': False,  # Would be True if actual push succeeds
            'message': 'Prometheus integration not implemented',
            'pushgateway_url': pushgateway_url,
            'metrics': metrics_to_push
        }
        
        print(f" Prometheus export simulation: {len(metrics_to_push)} metrics")
        return export_result
    
    def _calculate_quality_ratio(self, analytics: Dict, quality: str) -> float:
        """Calculate ratio of specific quality"""
        quality_dist = analytics.get('quality_distribution', {})
        total = sum(quality_dist.values())
        if total == 0:
            return 0.0
        return quality_dist.get(quality, 0) / total
    
    def clear_old_logs(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """
        MCP Tool: Dọn dẹp logs cũ
        """
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        if not self.log_file.exists():
            return {'message': 'No logs to clean', 'removed': 0}
        
        try:
            # Read all logs
            kept_logs = []
            removed_count = 0
            
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        log_time = datetime.fromisoformat(log_entry['timestamp'])
                        
                        if log_time >= cutoff_time:
                            kept_logs.append(line)
                        else:
                            removed_count += 1
                    except:
                        # Keep malformed lines
                        kept_logs.append(line)
            
            # Rewrite file with kept logs
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.writelines(kept_logs)
            
            result = {
                'success': True,
                'removed_logs': removed_count,
                'kept_logs': len(kept_logs),
                'cutoff_date': cutoff_time.isoformat()
            }
            
            print(f"🧹 Cleaned {removed_count} old logs, kept {len(kept_logs)}")
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    # Test LoggingTool
    print(" Testing LoggingTool...")
    
    # Initialize
    logger = LoggingTool(
        log_file="test_logs/session.jsonl",
        metrics_file="test_logs/metrics.jsonl",
        analytics_file="test_logs/analytics.json"
    )
    
    # Test logging sessions
    test_sessions = [
        {
            'question': 'Tuổi chịu trách nhiệm hình sự là bao nhiêu?',
            'answer': 'Theo Điều 15 Bộ luật Hình sự, người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự về mọi tội phạm.',
            'context': 'Điều 15. Tuổi chịu trách nhiệm hình sự\n1. Người từ đủ 16 tuổi trở lên...',
            'retrieved_docs': ['Doc 1 content', 'Doc 2 content']
        },
        {
            'question': 'Điều kiện kết hôn theo pháp luật?',
            'answer': 'Theo Luật Hôn nhân và Gia đình, nam từ đủ 20 tuổi, nữ từ đủ 18 tuổi.',
            'context': 'Điều 8. Điều kiện kết hôn...',
            'retrieved_docs': ['Marriage doc 1', 'Marriage doc 2', 'Marriage doc 3']
        }
    ]
    
    print(f"\n Testing session logging...")
    for i, session in enumerate(test_sessions, 1):
        result = logger.log_session(
            session['question'],
            session['answer'], 
            session['context'],
            session['retrieved_docs']
        )
        print(f" Session {i} logged: {result['logged']}")
    
    # Test analytics
    print(f"\n Testing analytics...")
    analytics = logger.get_session_analytics(24)
    print(f"  - Total sessions: {analytics['total_sessions']}")
    print(f"  - Question types: {analytics['question_types']}")
    print(f"  - Quality distribution: {analytics['quality_distribution']}")
    
    # Test recent sessions
    print(f"\n Testing recent sessions...")
    recent = logger.get_recent_sessions(5)
    print(f"  - Retrieved {len(recent)} recent sessions")
    
    # Test export
    print(f"\n Testing export...")
    export_path = logger.export_analytics()
    print(f"  - Exported to: {export_path}")
    
    # Test external integrations (simulation)
    print(f"\n Testing external integrations...")
    elastic_result = logger.export_to_elastic("http://localhost:9200")
    print(f"  - ElasticSearch: {elastic_result['message']}")
    
    prometheus_result = logger.export_to_prometheus("http://localhost:9091")
    print(f"  - Prometheus: {prometheus_result['message']}")
    
    print(f"\n LoggingTool test completed!")