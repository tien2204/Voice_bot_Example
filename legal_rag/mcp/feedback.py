# mcp/feedback.py - User Feedback System
# ====================================

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import statistics
import hashlib

class FeedbackTool:
    """
    MCP Tools cho User Feedback Loop:
    1. Submit feedback - người dùng đánh giá câu trả lời
    2. Analyze feedback patterns - phân tích feedback để cải thiện
    3. Generate improvement suggestions - đề xuất cải thiện
    4. Export feedback data - xuất dữ liệu cho training
    """
    
    def __init__(self, feedback_file: str = "logs/feedback.jsonl", 
                 analytics_file: str = "logs/feedback_analytics.json"):
        self.feedback_file = Path(feedback_file)
        self.analytics_file = Path(analytics_file)
        
        # Create directories
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        self.analytics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing feedback
        self.feedback_data = self._load_feedback_data()
        
        print(f" FeedbackTool initialized - {len(self.feedback_data)} existing feedbacks")
    
    def _load_feedback_data(self) -> List[Dict[str, Any]]:
        """Load existing feedback data"""
        feedback_list = []
        
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            feedback = json.loads(line.strip())
                            feedback_list.append(feedback)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f" Failed to load feedback data: {e}")
        
        return feedback_list
    
    def submit_feedback(self, question: str, answer: str, rating: int, 
                       comment: str = None, categories: List[str] = None, 
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        MCP Tool: Submit user feedback cho Q&A pair
        
        Args:
            question: Câu hỏi gốc
            answer: Câu trả lời từ hệ thống  
            rating: Đánh giá 1-5 sao
            comment: Nhận xét chi tiết (optional)
            categories: Danh mục feedback ['accuracy', 'completeness', 'clarity']
            metadata: Thông tin bổ sung
        """
        if not (1 <= rating <= 5):
            return {'error': 'Rating must be between 1 and 5'}
        
        # Generate feedback ID
        feedback_id = self._generate_feedback_id(question, answer)
        
        feedback_entry = {
            'feedback_id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'rating': rating,
            'comment': comment or "",
            'categories': categories or [],
            'metadata': metadata or {},
            'question_length': len(question),
            'answer_length': len(answer),
            'question_hash': hashlib.md5(question.encode()).hexdigest()[:8],
            'answer_hash': hashlib.md5(answer.encode()).hexdigest()[:8]
        }
        
        # Add contextual analysis
        feedback_entry.update(self._analyze_feedback_context(question, answer, rating, comment))
        
        # Store feedback
        self.feedback_data.append(feedback_entry)
        self._save_feedback_entry(feedback_entry)
        
        result = {
            'feedback_submitted': True,
            'feedback_id': feedback_id,
            'rating': rating,
            'total_feedbacks': len(self.feedback_data),
            'timestamp': feedback_entry['timestamp']
        }
        
        print(f" Feedback submitted: {rating} for question '{question[:30]}...'")
        
        # Trigger analytics update if significant feedback received
        if len(self.feedback_data) % 10 == 0:
            self._update_analytics()
        
        return result
    
    def _generate_feedback_id(self, question: str, answer: str) -> str:
        """Generate unique feedback ID"""
        content = f"{question}{answer}{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _analyze_feedback_context(self, question: str, answer: str, rating: int, comment: str) -> Dict[str, Any]:
        """Analyze feedback context for insights"""
        context = {}
        
        # Question type classification
        context['question_type'] = self._classify_question_type(question)
        
        # Answer quality estimation
        context['estimated_answer_quality'] = self._estimate_answer_quality(answer)
        
        # Rating vs estimation comparison
        context['rating_vs_estimation'] = rating - context['estimated_answer_quality']
        
        # Comment sentiment (basic)
        if comment:
            context['comment_sentiment'] = self._analyze_comment_sentiment(comment)
            context['comment_topics'] = self._extract_comment_topics(comment)
        
        # Legal domain detection
        context['legal_domains'] = self._detect_legal_domains(question, answer)
        
        return context
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['điều', 'khoản']):
            return 'specific_article'
        elif any(word in question_lower for word in ['là gì', 'định nghĩa']):
            return 'definition'
        elif any(word in question_lower for word in ['thủ tục', 'cách']):
            return 'procedure'
        elif any(word in question_lower for word in ['so sánh', 'khác nhau']):
            return 'comparison'
        else:
            return 'general'
    
    def _estimate_answer_quality(self, answer: str) -> int:
        """Estimate answer quality (1-5 scale)"""
        score = 1
        
        # Length check
        if 50 <= len(answer) <= 500:
            score += 1
        
        # Legal terminology
        legal_terms = ['điều', 'luật', 'quy định', 'nghị định', 'khoản']
        if any(term in answer.lower() for term in legal_terms):
            score += 1
        
        # Structure check
        if any(marker in answer for marker in ['1.', '2.', '-', '•']):
            score += 1
        
        # Avoid generic responses
        generic_phrases = ['xin lỗi', 'không biết', 'không thể']
        if not any(phrase in answer.lower() for phrase in generic_phrases):
            score += 1
        
        return min(score, 5)
    
    def _analyze_comment_sentiment(self, comment: str) -> str:
        """Basic sentiment analysis of comment"""
        comment_lower = comment.lower()
        
        positive_words = ['tốt', 'hay', 'chính xác', 'hữu ích', 'rõ ràng', 'đúng']
        negative_words = ['sai', 'không đúng', 'thiếu', 'khó hiểu', 'không rõ', 'kém']
        
        positive_count = sum(1 for word in positive_words if word in comment_lower)
        negative_count = sum(1 for word in negative_words if word in comment_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _extract_comment_topics(self, comment: str) -> List[str]:
        """Extract topics from comment"""
        topics = []
        comment_lower = comment.lower()
        
        topic_keywords = {
            'accuracy': ['chính xác', 'đúng', 'sai', 'không đúng'],
            'completeness': ['thiếu', 'đầy đủ', 'chi tiết', 'cần thêm'],
            'clarity': ['rõ ràng', 'khó hiểu', 'dễ hiểu', 'mơ hồ'],
            'relevance': ['liên quan', 'phù hợp', 'không liên quan'],
            'format': ['trình bày', 'định dạng', 'cấu trúc']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in comment_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _detect_legal_domains(self, question: str, answer: str) -> List[str]:
        """Detect legal domains in Q&A"""
        domains = []
        combined_text = (question + ' ' + answer).lower()
        
        domain_keywords = {
            'criminal_law': ['hình sự', 'tội phạm', 'tù', 'phạt'],
            'civil_law': ['dân sự', 'hợp đồng', 'sở hữu'],
            'family_law': ['hôn nhân', 'gia đình', 'kết hôn', 'ly hôn'],
            'labor_law': ['lao động', 'việc làm', 'lương', 'bảo hiểm'],
            'administrative_law': ['hành chính', 'thủ tục', 'giấy phép'],
            'economic_law': ['kinh tế', 'doanh nghiệp', 'đầu tư', 'thương mại']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                domains.append(domain)
        
        return domains
    
    def _save_feedback_entry(self, feedback_entry: Dict[str, Any]):
        """Save feedback entry to file"""
        try:
            with open(self.feedback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f" Failed to save feedback: {e}")
    
    def analyze_feedback_patterns(self, days_back: int = 30) -> Dict[str, Any]:
        """
        MCP Tool: Phân tích patterns trong feedback
        """
        print(f" Analyzing feedback patterns (last {days_back} days)...")
        
        # Filter recent feedback
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_feedback = [
            fb for fb in self.feedback_data
            if datetime.fromisoformat(fb['timestamp']) >= cutoff_date
        ]
        
        if not recent_feedback:
            return {'message': f'No feedback found in last {days_back} days'}
        
        # Basic statistics
        ratings = [fb['rating'] for fb in recent_feedback]
        avg_rating = statistics.mean(ratings)
        rating_distribution = {i: ratings.count(i) for i in range(1, 6)}
        
        # Question type analysis
        question_types = {}
        for fb in recent_feedback:
            qtype = fb.get('question_type', 'unknown')
            question_types[qtype] = question_types.get(qtype, 0) + 1
        
        # Legal domain analysis
        legal_domains = {}
        for fb in recent_feedback:
            for domain in fb.get('legal_domains', []):
                legal_domains[domain] = legal_domains.get(domain, 0) + 1
        
        # Rating vs estimation analysis
        rating_accuracy = []
        for fb in recent_feedback:
            if 'rating_vs_estimation' in fb:
                rating_accuracy.append(abs(fb['rating_vs_estimation']))
        
        avg_rating_accuracy = statistics.mean(rating_accuracy) if rating_accuracy else 0
        
        # Comment sentiment analysis
        sentiment_distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        comment_topics = {}
        
        for fb in recent_feedback:
            if fb.get('comment'):
                sentiment = fb.get('comment_sentiment', 'neutral')
                sentiment_distribution[sentiment] += 1
                
                for topic in fb.get('comment_topics', []):
                    comment_topics[topic] = comment_topics.get(topic, 0) + 1
        
        # Identify problem areas
        low_rated_feedback = [fb for fb in recent_feedback if fb['rating'] <= 2]
        problem_patterns = self._identify_problem_patterns(low_rated_feedback)
        
        analysis = {
            'analysis_period': f'{days_back} days',
            'total_feedback': len(recent_feedback),
            'rating_statistics': {
                'average_rating': round(avg_rating, 2),
                'rating_distribution': rating_distribution,
                'satisfaction_rate': round(sum(rating_distribution.get(i, 0) for i in [4, 5]) / len(recent_feedback) * 100, 1)
            },
            'question_type_distribution': dict(sorted(question_types.items(), key=lambda x: x[1], reverse=True)),
            'legal_domain_distribution': dict(sorted(legal_domains.items(), key=lambda x: x[1], reverse=True)),
            'comment_analysis': {
                'total_comments': sum(1 for fb in recent_feedback if fb.get('comment')),
                'sentiment_distribution': sentiment_distribution,
                'common_topics': dict(sorted(comment_topics.items(), key=lambda x: x[1], reverse=True)[:5])
            },
            'system_performance': {
                'average_rating_accuracy': round(avg_rating_accuracy, 2),
                'low_rated_count': len(low_rated_feedback),
                'problem_areas': problem_patterns
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return analysis
    
    def _identify_problem_patterns(self, low_rated_feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify patterns in low-rated feedback"""
        if not low_rated_feedback:
            return {'message': 'No low-rated feedback to analyze'}
        
        # Common question types in low ratings
        problem_question_types = {}
        for fb in low_rated_feedback:
            qtype = fb.get('question_type', 'unknown')
            problem_question_types[qtype] = problem_question_types.get(qtype, 0) + 1
        
        # Common legal domains in problems
        problem_domains = {}
        for fb in low_rated_feedback:
            for domain in fb.get('legal_domains', []):
                problem_domains[domain] = problem_domains.get(domain, 0) + 1
        
        # Common complaint topics
        complaint_topics = {}
        for fb in low_rated_feedback:
            for topic in fb.get('comment_topics', []):
                complaint_topics[topic] = complaint_topics.get(topic, 0) + 1
        
        return {
            'total_low_ratings': len(low_rated_feedback),
            'problematic_question_types': dict(sorted(problem_question_types.items(), key=lambda x: x[1], reverse=True)),
            'problematic_domains': dict(sorted(problem_domains.items(), key=lambda x: x[1], reverse=True)),
            'common_complaints': dict(sorted(complaint_topics.items(), key=lambda x: x[1], reverse=True))
        }
    
    def generate_improvement_suggestions(self, analysis_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        MCP Tool: Tạo đề xuất cải thiện dựa trên feedback analysis
        """
        if not analysis_data:
            analysis_data = self.analyze_feedback_patterns(30)
        
        suggestions = []
        priority_scores = []
        
        # Analyze rating statistics
        avg_rating = analysis_data.get('rating_statistics', {}).get('average_rating', 3)
        satisfaction_rate = analysis_data.get('rating_statistics', {}).get('satisfaction_rate', 50)
        
        if avg_rating < 3.5:
            suggestions.append({
                'category': 'overall_quality',
                'issue': 'Low average rating',
                'suggestion': 'Review and improve answer generation prompts and retrieval quality',
                'priority': 'high',
                'metrics': {'current_avg_rating': avg_rating, 'target': 4.0}
            })
            priority_scores.append(3)
        
        if satisfaction_rate < 70:
            suggestions.append({
                'category': 'user_satisfaction',
                'issue': 'Low satisfaction rate',
                'suggestion': 'Focus on improving answer completeness and accuracy',
                'priority': 'high',
                'metrics': {'current_satisfaction': satisfaction_rate, 'target': 80}
            })
            priority_scores.append(3)
        
        # Analyze problem question types
        problem_types = analysis_data.get('system_performance', {}).get('problem_areas', {}).get('problematic_question_types', {})
        
        if problem_types:
            most_problematic = max(problem_types.items(), key=lambda x: x[1])
            suggestions.append({
                'category': 'question_handling',
                'issue': f'High failure rate for {most_problematic[0]} questions',
                'suggestion': f'Improve handling of {most_problematic[0]} questions through specialized prompts or additional training data',
                'priority': 'medium',
                'metrics': {'problem_count': most_problematic[1], 'question_type': most_problematic[0]}
            })
            priority_scores.append(2)
        
        # Analyze comment topics
        complaint_topics = analysis_data.get('comment_analysis', {}).get('common_topics', {})
        
        if 'accuracy' in complaint_topics:
            suggestions.append({
                'category': 'accuracy',
                'issue': 'Accuracy complaints in user feedback',
                'suggestion': 'Improve fact-checking and source verification in answers',
                'priority': 'high',
                'metrics': {'accuracy_complaints': complaint_topics['accuracy']}
            })
            priority_scores.append(3)
        
        if 'completeness' in complaint_topics:
            suggestions.append({
                'category': 'completeness',
                'issue': 'Completeness complaints in feedback',
                'suggestion': 'Increase retrieved document count and improve context utilization',
                'priority': 'medium',
                'metrics': {'completeness_complaints': complaint_topics['completeness']}
            })
            priority_scores.append(2)
        
        if 'clarity' in complaint_topics:
            suggestions.append({
                'category': 'clarity',
                'issue': 'Clarity issues in answers',
                'suggestion': 'Simplify language and improve answer structure for better readability',
                'priority': 'medium',
                'metrics': {'clarity_complaints': complaint_topics['clarity']}
            })
            priority_scores.append(2)
        
        # Analyze problematic domains
        problem_domains = analysis_data.get('system_performance', {}).get('problem_areas', {}).get('problematic_domains', {})
        
        if problem_domains:
            most_problematic_domain = max(problem_domains.items(), key=lambda x: x[1])
            suggestions.append({
                'category': 'domain_expertise',
                'issue': f'Poor performance in {most_problematic_domain[0]} domain',
                'suggestion': f'Enhance knowledge base and specialized prompts for {most_problematic_domain[0]}',
                'priority': 'medium',
                'metrics': {'domain': most_problematic_domain[0], 'problem_count': most_problematic_domain[1]}
            })
            priority_scores.append(2)
        
        # Sort suggestions by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        suggestions.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
        
        # Generate action plan
        action_plan = self._generate_action_plan(suggestions)
        
        improvement_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_suggestions': len(suggestions),
            'high_priority_count': sum(1 for s in suggestions if s['priority'] == 'high'),
            'suggestions': suggestions,
            'action_plan': action_plan,
            'estimated_impact': self._estimate_improvement_impact(suggestions),
            'recommended_next_steps': self._get_next_steps(suggestions[:3])  # Top 3 suggestions
        }
        
        return improvement_report
    
    def _generate_action_plan(self, suggestions: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate actionable plan from suggestions"""
        action_plan = {
            'immediate_actions': [],
            'short_term_goals': [],
            'long_term_improvements': []
        }
        
        for suggestion in suggestions:
            category = suggestion['category']
            priority = suggestion['priority']
            
            if priority == 'high':
                if category == 'accuracy':
                    action_plan['immediate_actions'].append("Review and update fact-checking procedures")
                    action_plan['immediate_actions'].append("Implement source verification for legal citations")
                elif category == 'overall_quality':
                    action_plan['immediate_actions'].append("Audit current answer generation prompts")
                    action_plan['immediate_actions'].append("Increase retrieval quality thresholds")
            
            elif priority == 'medium':
                if category == 'question_handling':
                    action_plan['short_term_goals'].append("Develop specialized handlers for problematic question types")
                elif category == 'completeness':
                    action_plan['short_term_goals'].append("Optimize context window and document selection")
                elif category == 'clarity':
                    action_plan['short_term_goals'].append("Implement answer formatting and simplification")
            
            # Long-term improvements
            action_plan['long_term_improvements'].append(f"Continuous monitoring and improvement for {category}")
        
        return action_plan
    
    def _estimate_improvement_impact(self, suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate impact of implementing suggestions"""
        if not suggestions:
            return {'estimated_rating_increase': 0, 'estimated_satisfaction_increase': 0}
        
        high_priority_count = sum(1 for s in suggestions if s['priority'] == 'high')
        medium_priority_count = sum(1 for s in suggestions if s['priority'] == 'medium')
        
        # Rough impact estimation
        rating_increase = high_priority_count * 0.3 + medium_priority_count * 0.15
        satisfaction_increase = high_priority_count * 10 + medium_priority_count * 5
        
        return {
            'estimated_rating_increase': round(min(rating_increase, 1.0), 2),
            'estimated_satisfaction_increase': round(min(satisfaction_increase, 25), 1),
            'confidence': 'medium' if len(suggestions) >= 3 else 'low'
        }
    
    def _get_next_steps(self, top_suggestions: List[Dict[str, Any]]) -> List[str]:
        """Get recommended next steps"""
        next_steps = [
            "Implement feedback collection improvements",
            "Monitor feedback trends weekly",
            "Set up automated alerts for rating drops"
        ]
        
        for suggestion in top_suggestions:
            category = suggestion['category']
            if category == 'accuracy':
                next_steps.append("Implement fact-checking pipeline")
            elif category == 'completeness':
                next_steps.append("Optimize retrieval parameters")
            elif category == 'clarity':
                next_steps.append("Develop answer formatting guidelines")
        
        return next_steps[:5]  # Top 5 next steps
    
    def export_training_data(self, min_rating: int = 4, output_file: str = None) -> str:
        """
        MCP Tool: Export high-quality Q&A pairs for training
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"training_data_{timestamp}.jsonl"
        
        print(f" Exporting training data (rating >= {min_rating})...")
        
        # Filter high-quality feedback
        high_quality_feedback = [
            fb for fb in self.feedback_data
            if fb['rating'] >= min_rating and fb.get('comment_sentiment') != 'negative'
        ]
        
        # Create training examples
        training_examples = []
        
        for fb in high_quality_feedback:
            example = {
                'input': fb['question'],
                'output': fb['answer'],
                'rating': fb['rating'],
                'feedback_metadata': {
                    'question_type': fb.get('question_type'),
                    'legal_domains': fb.get('legal_domains', []),
                    'comment_sentiment': fb.get('comment_sentiment'),
                    'timestamp': fb['timestamp']
                }
            }
            training_examples.append(example)
        
        # Save training data
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for example in training_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            
            export_stats = {
                'total_exported': len(training_examples),
                'min_rating_filter': min_rating,
                'output_file': output_file,
                'export_timestamp': datetime.now().isoformat(),
                'quality_distribution': self._get_quality_distribution(training_examples)
            }
            
            print(f" Exported {len(training_examples)} training examples to {output_file}")
            return output_file
            
        except Exception as e:
            print(f" Failed to export training data: {e}")
            return ""
    
    def _get_quality_distribution(self, examples: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get quality distribution of training examples"""
        distribution = {}
        for example in examples:
            rating = example['rating']
            distribution[f"{rating}_star"] = distribution.get(f"{rating}_star", 0) + 1
        return distribution
    
    def _update_analytics(self):
        """Update analytics file with latest analysis"""
        try:
            analytics = {
                'last_updated': datetime.now().isoformat(),
                'total_feedback_count': len(self.feedback_data),
                'recent_analysis': self.analyze_feedback_patterns(7),  # Last 7 days
                'improvement_suggestions': self.generate_improvement_suggestions()
            }
            
            with open(self.analytics_file, 'w', encoding='utf-8') as f:
                json.dump(analytics, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f" Failed to update analytics: {e}")
    
    def get_feedback_summary(self, question_hash: str = None) -> Dict[str, Any]:
        """
        MCP Tool: Lấy tóm tắt feedback cho question cụ thể hoặc tổng quan
        """
        if question_hash:
            # Get feedback for specific question
            related_feedback = [
                fb for fb in self.feedback_data 
                if fb.get('question_hash') == question_hash
            ]
            
            if not related_feedback:
                return {'message': f'No feedback found for question hash {question_hash}'}
            
            ratings = [fb['rating'] for fb in related_feedback]
            
            return {
                'question_hash': question_hash,
                'feedback_count': len(related_feedback),
                'average_rating': round(statistics.mean(ratings), 2),
                'rating_distribution': {i: ratings.count(i) for i in range(1, 6)},
                'latest_feedback': related_feedback[-1] if related_feedback else None,
                'common_comments': [fb['comment'] for fb in related_feedback if fb.get('comment')][:3]
            }
        
        else:
            # Overall summary
            if not self.feedback_data:
                return {'message': 'No feedback data available'}
            
            ratings = [fb['rating'] for fb in self.feedback_data]
            recent_feedback = self.feedback_data[-10:]  # Last 10
            
            return {
                'total_feedback': len(self.feedback_data),
                'overall_average_rating': round(statistics.mean(ratings), 2),
                'overall_rating_distribution': {i: ratings.count(i) for i in range(1, 6)},
                'recent_trend': round(statistics.mean([fb['rating'] for fb in recent_feedback]), 2),
                'feedback_with_comments': sum(1 for fb in self.feedback_data if fb.get('comment')),
                'most_recent_feedback': self.feedback_data[-1] if self.feedback_data else None
            }

if __name__ == "__main__":
    # Test FeedbackTool
    print(" Testing FeedbackTool...")
    
    # Initialize
    feedback_tool = FeedbackTool(
        feedback_file="test_logs/feedback.jsonl",
        analytics_file="test_logs/feedback_analytics.json"
    )
    
    # Test feedback submissions
    test_feedbacks = [
        {
            'question': 'Tuổi chịu trách nhiệm hình sự là bao nhiêu?',
            'answer': 'Theo Điều 15 Bộ luật Hình sự, người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự về mọi tội phạm.',
            'rating': 5,
            'comment': 'Câu trả lời rất chính xác và rõ ràng',
            'categories': ['accuracy', 'clarity']
        },
        {
            'question': 'Điều kiện kết hôn theo pháp luật Việt Nam?',
            'answer': 'Nam từ đủ 20 tuổi, nữ từ đủ 18 tuổi. Việc kết hôn do nam, nữ tự nguyện quyết định.',
            'rating': 4,
            'comment': 'Đúng nhưng thiếu chi tiết về thủ tục',
            'categories': ['accuracy', 'completeness']
        },
        {
            'question': 'Hình phạt tù chung thân như thế nào?',
            'answer': 'Tù chung thân là hình phạt tù không thời hạn.',
            'rating': 2,
            'comment': 'Câu trả lời quá ngắn, thiếu thông tin chi tiết',
            'categories': ['completeness', 'clarity']
        },
        {
            'question': 'Quyền cơ bản của công dân là gì?',
            'answer': 'Hiến pháp quy định quyền cơ bản của công dân bao gồm quyền sống, quyền tự do, quyền bình đẳng.',
            'rating': 4,
            'comment': 'Tốt nhưng có thể chi tiết hơn',
            'categories': ['accuracy']
        }
    ]
    
    print(f"\n👍 Testing feedback submissions...")
    for i, feedback in enumerate(test_feedbacks, 1):
        result = feedback_tool.submit_feedback(
            feedback['question'],
            feedback['answer'],
            feedback['rating'],
            feedback['comment'],
            feedback['categories']
        )
        print(f" Feedback {i} submitted: {result['rating']}⭐")
    
    # Test feedback analysis
    print(f"\n Testing feedback analysis...")
    analysis = feedback_tool.analyze_feedback_patterns(days_back=30)
    print(f"  - Total feedback: {analysis['total_feedback']}")
    print(f"  - Average rating: {analysis['rating_statistics']['average_rating']}")
    print(f"  - Satisfaction rate: {analysis['rating_statistics']['satisfaction_rate']}%")
    
    # Test improvement suggestions
    print(f"\n Testing improvement suggestions...")
    improvements = feedback_tool.generate_improvement_suggestions(analysis)
    print(f"  - Total suggestions: {improvements['total_suggestions']}")
    print(f"  - High priority: {improvements['high_priority_count']}")
    
    # Test training data export
    print(f"\n Testing training data export...")
    export_file = feedback_tool.export_training_data(min_rating=4)
    if export_file:
        print(f"  - Training data exported: {export_file}")
    
    # Test feedback summary
    print(f"\n Testing feedback summary...")
    summary = feedback_tool.get_feedback_summary()
    print(f"  - Total feedback: {summary['total_feedback']}")
    print(f"  - Overall rating: {summary['overall_average_rating']}")
    
    print(f"\n FeedbackTool test completed!")