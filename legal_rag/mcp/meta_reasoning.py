# mcp/meta_reasoning.py - Meta-Reasoning Tools
# ===========================================

from typing import List, Dict, Any, Optional, Tuple
import json
import time
from datetime import datetime

class MetaReasoningTool:
    """
    MCP Tools cho Meta-Reasoning:
    1. Explain Retrieval - giải thích tại sao tài liệu được chọn
    2. Debug Prompt - hiển thị full prompt context
    3. Analyze Query - phân tích câu hỏi
    """
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.last_retrieval_results = []
        self.last_prompt = ""
        self.last_query = ""
        self.retrieval_history = []
        
        print(" MetaReasoningTool initialized")
    
    def explain_retrieval(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        MCP Tool: Giải thích chi tiết quá trình retrieval
        """
        print(f" Explaining retrieval for: '{query}'")
        
        # Get detailed retrieval results
        explanations = self.retriever.explain_retrieval(query, k)
        
        # Store for later reference
        self.last_retrieval_results = explanations
        self.last_query = query
        
        # Add to history
        self.retrieval_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'results_count': len(explanations),
            'top_score': explanations[0]['hybrid_score'] if explanations else 0
        })
        
        # Format explanations for better readability
        formatted_explanations = []
        for i, exp in enumerate(explanations, 1):
            formatted_exp = {
                'rank': i,
                'doc_preview': exp['doc_preview'],
                'source': exp['source'],
                'document_type': exp['document_type'],
                'scores': {
                    'hybrid_score': round(exp['hybrid_score'], 4),
                    'bm25_score': round(exp['bm25_score'], 4),
                    'vector_score': round(exp['vector_score'], 4),
                    'bm25_contribution': round(exp['bm25_contribution'], 4),
                    'vector_contribution': round(exp['vector_contribution'], 4)
                },
                'matching_terms': exp['matching_terms'],
                'legal_info': {
                    'dieu_so': exp.get('dieu_so', ''),
                    'ten_van_ban': exp.get('ten_van_ban', '')
                },
                'explanation': self._generate_explanation(exp)
            }
            formatted_explanations.append(formatted_exp)
        
        return formatted_explanations
    
    def _generate_explanation(self, exp: Dict[str, Any]) -> str:
        """Generate human-readable explanation"""
        explanations = []
        
        # Score analysis
        if exp['bm25_score'] > exp['vector_score']:
            explanations.append(f"Tài liệu này được chọn chủ yếu do matching keywords (BM25: {exp['bm25_score']:.3f})")
        else:
            explanations.append(f"Tài liệu này được chọn chủ yếu do similarity ngữ nghĩa (Vector: {exp['vector_score']:.3f})")
        
        # Matching terms
        if exp['matching_terms']:
            terms_str = ', '.join(exp['matching_terms'][:3])
            explanations.append(f"Keywords match: {terms_str}")
        
        # Legal structure
        if exp.get('dieu_so'):
            explanations.append(f"Chứa Điều {exp['dieu_so']}")
        
        if exp.get('ten_van_ban'):
            explanations.append(f"Từ: {exp['ten_van_ban']}")
        
        return ". ".join(explanations)
    
    def debug_prompt(self, query: str, context: str = None) -> Dict[str, Any]:
        """
        MCP Tool: Debug prompt context trước khi gửi đến LLM
        """
        print(f" Debugging prompt for query: '{query[:50]}...'")
        
        # Get context nếu không có
        if not context:
            relevant_docs = self.retriever.retrieve(query, k=5)
            context = self._create_context_from_docs(relevant_docs)
        
        # Generate full prompt
        if hasattr(self.llm, 'format_legal_prompt'):
            full_prompt = self.llm.format_legal_prompt(query, context)
        else:
            full_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Store for reference
        self.last_prompt = full_prompt
        
        # Analyze prompt
        analysis = self._analyze_prompt(full_prompt, query, context)
        
        debug_info = {
            'query': query,
            'context_length': len(context),
            'full_prompt_length': len(full_prompt),
            'full_prompt': full_prompt,
            'context_preview': context[:500] + "..." if len(context) > 500 else context,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'tokenization_info': self._get_tokenization_info(full_prompt)
        }
        
        return debug_info
    
    def _analyze_prompt(self, prompt: str, query: str, context: str) -> Dict[str, Any]:
        """Analyze prompt structure and content"""
        analysis = {
            'prompt_structure': 'unknown',
            'has_legal_terms': False,
            'context_quality': 'low',
            'potential_issues': [],
            'suggestions': []
        }
        
        # Detect prompt structure
        if '<|im_start|>' in prompt:
            analysis['prompt_structure'] = 'chat_template'
        elif 'Context:' in prompt and 'Question:' in prompt:
            analysis['prompt_structure'] = 'qa_format'
        else:
            analysis['prompt_structure'] = 'custom'
        
        # Check for legal terms
        legal_terms = ['điều', 'khoản', 'luật', 'nghị định', 'thông tư', 'bộ luật']
        found_terms = [term for term in legal_terms if term in query.lower() or term in context.lower()]
        analysis['has_legal_terms'] = len(found_terms) > 0
        analysis['found_legal_terms'] = found_terms
        
        # Assess context quality
        if len(context) < 100:
            analysis['context_quality'] = 'low'
            analysis['potential_issues'].append('Context quá ngắn')
        elif len(context) > 3000:
            analysis['context_quality'] = 'high_but_long'
            analysis['potential_issues'].append('Context có thể quá dài cho LLM')
        else:
            analysis['context_quality'] = 'good'
        
        # Check for repetition
        context_words = context.lower().split()
        unique_words = set(context_words)
        if len(context_words) > 0:
            repetition_ratio = 1 - (len(unique_words) / len(context_words))
            if repetition_ratio > 0.5:
                analysis['potential_issues'].append(f'Context có nhiều từ lặp lại ({repetition_ratio:.2%})')
        
        # Generate suggestions
        if 'Context quá ngắn' in analysis['potential_issues']:
            analysis['suggestions'].append('Tăng số lượng retrieved documents')
        
        if not analysis['has_legal_terms']:
            analysis['suggestions'].append('Query có thể không liên quan đến pháp luật')
        
        if len(analysis['potential_issues']) == 0:
            analysis['suggestions'].append('Prompt structure trông tốt')
        
        return analysis
    
    def _get_tokenization_info(self, prompt: str) -> Dict[str, Any]:
        """Get tokenization info if LLM tokenizer available"""
        token_info = {
            'estimated_tokens': len(prompt.split()),  # Rough estimate
            'character_count': len(prompt),
            'word_count': len(prompt.split())
        }
        
        # Try to get actual token count
        if hasattr(self.llm, 'tokenizer') and self.llm.tokenizer:
            try:
                tokens = self.llm.tokenizer.encode(prompt)
                token_info['actual_tokens'] = len(tokens)
                token_info['tokenizer_vocab_size'] = self.llm.tokenizer.vocab_size
            except:
                pass
        
        return token_info
    
    def _create_context_from_docs(self, documents) -> str:
        """Create context string from documents"""
        contexts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Không xác định')
            content = doc.page_content.strip()
            contexts.append(f"[Tài liệu {i} - {source}]\n{content}")
        
        return "\n\n".join(contexts)
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        MCP Tool: Phân tích chi tiết câu hỏi
        """
        print(f" Analyzing query: '{query}'")
        
        analysis = {
            'query': query,
            'query_length': len(query),
            'word_count': len(query.split()),
            'query_type': self._classify_query_type(query),
            'legal_entities': self._extract_legal_entities(query),
            'complexity': self._assess_query_complexity(query),
            'suggested_retrieval_strategy': self._suggest_retrieval_strategy(query),
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of legal query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['điều', 'khoản', 'điểm']):
            return 'specific_article'
        elif any(word in query_lower for word in ['là gì', 'định nghĩa', 'khái niệm']):
            return 'definition'
        elif any(word in query_lower for word in ['thủ tục', 'cách thức', 'quy trình']):
            return 'procedure'
        elif any(word in query_lower for word in ['hình phạt', 'phạt', 'tù']):
            return 'penalty'
        elif any(word in query_lower for word in ['quyền', 'nghĩa vụ']):
            return 'rights_obligations'
        elif any(word in query_lower for word in ['so sánh', 'khác nhau', 'giống']):
            return 'comparison'
        else:
            return 'general'
    
    def _extract_legal_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract legal entities from query"""
        import re
        
        entities = {
            'articles': [],
            'laws': [],
            'document_numbers': [],
            'penalties': [],
            'procedures': []
        }
        
        # Extract articles (Điều X)
        articles = re.findall(r'điều\s+(\d+)', query.lower())
        entities['articles'] = [f'Điều {art}' for art in articles]
        
        # Extract laws
        laws = re.findall(r'(luật|bộ luật|nghị định|thông tư)\s+([\w\s]+?)(?=\s|$|,)', query.lower())
        entities['laws'] = [f'{law[0].title()} {law[1].strip()}' for law in laws]
        
        # Extract document numbers
        doc_numbers = re.findall(r'(\d+/\d+/[A-Z-]+)', query)
        entities['document_numbers'] = doc_numbers
        
        return entities
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess query complexity"""
        factors = 0
        
        # Length factor
        if len(query) > 100:
            factors += 1
        
        # Multiple legal concepts
        legal_terms = ['điều', 'khoản', 'luật', 'nghị định', 'quyền', 'nghĩa vụ', 'thủ tục']
        term_count = sum(1 for term in legal_terms if term in query.lower())
        if term_count > 2:
            factors += 1
        
        # Question words indicating complex queries
        complex_indicators = ['so sánh', 'phân biệt', 'mối quan hệ', 'ảnh hưởng', 'tại sao']
        if any(indicator in query.lower() for indicator in complex_indicators):
            factors += 1
        
        # Multiple questions
        if query.count('?') > 1:
            factors += 1
        
        if factors >= 3:
            return 'high'
        elif factors >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_retrieval_strategy(self, query: str) -> Dict[str, Any]:
        """Suggest optimal retrieval strategy"""
        strategy = {
            'recommended_k': 5,
            'bm25_weight': 0.3,
            'vector_weight': 0.7,
            'filters': {},
            'reasoning': []
        }
        
        query_lower = query.lower()
        
        # If specific article mentioned, favor BM25
        if any(word in query_lower for word in ['điều', 'khoản']):
            strategy['bm25_weight'] = 0.6
            strategy['vector_weight'] = 0.4
            strategy['reasoning'].append('Favor BM25 for specific legal articles')
        
        # If conceptual question, favor vector search
        if any(word in query_lower for word in ['là gì', 'định nghĩa', 'khái niệm', 'ý nghĩa']):
            strategy['bm25_weight'] = 0.2
            strategy['vector_weight'] = 0.8
            strategy['reasoning'].append('Favor vector search for conceptual questions')
        
        # Adjust k based on complexity
        complexity = self._assess_query_complexity(query)
        if complexity == 'high':
            strategy['recommended_k'] = 8
            strategy['reasoning'].append('Increase k for complex queries')
        elif complexity == 'low':
            strategy['recommended_k'] = 3
            strategy['reasoning'].append('Decrease k for simple queries')
        
        # Document type filters
        if 'hình sự' in query_lower:
            strategy['filters']['document_type'] = 'bo_luat'
            strategy['reasoning'].append('Filter for criminal law documents')
        elif 'hôn nhân' in query_lower:
            strategy['filters']['document_type'] = 'luat'
            strategy['reasoning'].append('Filter for family law documents')
        
        return strategy
    
    def get_retrieval_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get retrieval history"""
        return self.retrieval_history[-limit:] if self.retrieval_history else []
    
    def get_last_results(self) -> Dict[str, Any]:
        """Get last retrieval results with metadata"""
        return {
            'query': self.last_query,
            'results': self.last_retrieval_results,
            'prompt': self.last_prompt[:500] + "..." if len(self.last_prompt) > 500 else self.last_prompt,
            'retrieval_count': len(self.last_retrieval_results)
        }
    
    def compare_queries(self, query1: str, query2: str) -> Dict[str, Any]:
        """
        MCP Tool: So sánh hai queries
        """
        print(f" Comparing queries: '{query1[:30]}...' vs '{query2[:30]}...'")
        
        analysis1 = self.analyze_query(query1)
        analysis2 = self.analyze_query(query2)
        
        # Retrieve for both queries
        results1 = self.explain_retrieval(query1, k=5)
        results2 = self.explain_retrieval(query2, k=5)
        
        comparison = {
            'query1': {
                'text': query1,
                'analysis': analysis1,
                'top_results': results1[:3]
            },
            'query2': {
                'text': query2,
                'analysis': analysis2,
                'top_results': results2[:3]
            },
            'similarities': self._find_similarities(analysis1, analysis2, results1, results2),
            'differences': self._find_differences(analysis1, analysis2, results1, results2),
            'recommendations': self._generate_comparison_recommendations(analysis1, analysis2)
        }
        
        return comparison
    
    def _find_similarities(self, analysis1: Dict, analysis2: Dict, 
                          results1: List, results2: List) -> List[str]:
        """Find similarities between queries"""
        similarities = []
        
        # Query type similarity
        if analysis1['query_type'] == analysis2['query_type']:
            similarities.append(f"Cùng loại câu hỏi: {analysis1['query_type']}")
        
        # Complexity similarity
        if analysis1['complexity'] == analysis2['complexity']:
            similarities.append(f"Cùng độ phức tạp: {analysis1['complexity']}")
        
        # Common legal entities
        entities1 = analysis1['legal_entities']
        entities2 = analysis2['legal_entities']
        
        for entity_type in entities1.keys():
            common = set(entities1[entity_type]) & set(entities2[entity_type])
            if common:
                similarities.append(f"Chung {entity_type}: {', '.join(common)}")
        
        # Common retrieved documents
        sources1 = {r['source'] for r in results1}
        sources2 = {r['source'] for r in results2}
        common_sources = sources1 & sources2
        
        if common_sources:
            similarities.append(f"Chung {len(common_sources)} tài liệu retrieved")
        
        return similarities
    
    def _find_differences(self, analysis1: Dict, analysis2: Dict, 
                         results1: List, results2: List) -> List[str]:
        """Find differences between queries"""
        differences = []
        
        # Query type difference
        if analysis1['query_type'] != analysis2['query_type']:
            differences.append(f"Khác loại: {analysis1['query_type']} vs {analysis2['query_type']}")
        
        # Complexity difference
        if analysis1['complexity'] != analysis2['complexity']:
            differences.append(f"Khác độ phức tạp: {analysis1['complexity']} vs {analysis2['complexity']}")
        
        # Length difference
        len_diff = abs(analysis1['query_length'] - analysis2['query_length'])
        if len_diff > 20:
            differences.append(f"Khác độ dài: {len_diff} ký tự")
        
        # Score differences in retrieval
        if results1 and results2:
            score_diff = abs(results1[0]['scores']['hybrid_score'] - results2[0]['scores']['hybrid_score'])
            if score_diff > 0.1:
                differences.append(f"Khác chất lượng retrieval: {score_diff:.3f}")
        
        return differences
    
    def _generate_comparison_recommendations(self, analysis1: Dict, analysis2: Dict) -> List[str]:
        """Generate recommendations based on comparison"""
        recommendations = []
        
        # If one query is much more complex
        if analysis1['complexity'] == 'high' and analysis2['complexity'] == 'low':
            recommendations.append("Query1 phức tạp hơn, có thể cần tăng số retrieved documents")
        elif analysis2['complexity'] == 'high' and analysis1['complexity'] == 'low':
            recommendations.append("Query2 phức tạp hơn, có thể cần tăng số retrieved documents")
        
        # If different query types
        if analysis1['query_type'] != analysis2['query_type']:
            recommendations.append("Hai câu hỏi khác loại, có thể cần strategy retrieval khác nhau")
        
        # If one has more legal entities
        entities1_count = sum(len(v) for v in analysis1['legal_entities'].values())
        entities2_count = sum(len(v) for v in analysis2['legal_entities'].values())
        
        if abs(entities1_count - entities2_count) > 2:
            recommendations.append("Câu hỏi có nhiều legal entities hơn có thể cần BM25 weight cao hơn")
        
        return recommendations
    
    def export_analysis(self, filepath: str = None) -> str:
        """Export analysis results to file"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"meta_analysis_{timestamp}.json"
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'last_query': self.last_query,
            'last_results_count': len(self.last_retrieval_results),
            'retrieval_history': self.get_retrieval_history(50),
            'last_prompt_length': len(self.last_prompt),
            'last_retrieval_results': self.last_retrieval_results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f" Meta-reasoning analysis exported to: {filepath}")
        return filepath

if __name__ == "__main__":
    # Test MetaReasoningTool
    print(" Testing MetaReasoningTool...")
    
    # Mock retriever and LLM for testing
    class MockRetriever:
        def explain_retrieval(self, query, k=5):
            return [
                {
                    'doc_preview': 'Điều 15. Tuổi chịu trách nhiệm hình sự...',
                    'source': 'test_doc_1',
                    'document_type': 'bo_luat',
                    'hybrid_score': 0.85,
                    'bm25_score': 0.7,
                    'vector_score': 0.9,
                    'bm25_contribution': 0.21,
                    'vector_contribution': 0.63,
                    'matching_terms': ['tuổi', 'trách nhiệm', 'hình sự'],
                    'dieu_so': '15',
                    'ten_van_ban': 'Bộ luật Hình sự'
                }
            ]
        
        def retrieve(self, query, k=5):
            from langchain_core.documents import Document
            return [
                Document(
                    page_content="Test content",
                    metadata={'source': 'test', 'document_type': 'luat'}
                )
            ]
    
    class MockLLM:
        def format_legal_prompt(self, query, context):
            return f"Query: {query}\nContext: {context}\nAnswer:"
    
    # Test the tool
    mock_retriever = MockRetriever()
    mock_llm = MockLLM()
    
    meta_tool = MetaReasoningTool(mock_retriever, mock_llm)
    
    # Test explain_retrieval
    test_query = "Tuổi chịu trách nhiệm hình sự là bao nhiêu?"
    print(f"\n Testing explain_retrieval for: {test_query}")
    
    explanations = meta_tool.explain_retrieval(test_query)
    print(f" Got {len(explanations)} explanations:")
    for exp in explanations:
        print(f"  - {exp['doc_preview'][:50]}... (Score: {exp['scores']['hybrid_score']})")
    
    # Test debug_prompt
    print(f"\n Testing debug_prompt...")
    debug_info = meta_tool.debug_prompt(test_query, "Sample context about legal age")
    print(f" Debug info generated:")
    print(f"  - Prompt length: {debug_info['full_prompt_length']}")
    print(f"  - Analysis: {debug_info['analysis']['context_quality']}")
    
    # Test analyze_query
    print(f"\n Testing analyze_query...")
    analysis = meta_tool.analyze_query(test_query)
    print(f" Query analysis:")
    print(f"  - Type: {analysis['query_type']}")
    print(f"  - Complexity: {analysis['complexity']}")
    print(f"  - Legal entities: {analysis['legal_entities']}")
    
    # Test compare_queries
    query2 = "Điều kiện kết hôn theo pháp luật Việt Nam?"
    print(f"\n Testing compare_queries...")
    comparison = meta_tool.compare_queries(test_query, query2)
    print(f" Comparison completed:")
    print(f"  - Similarities: {len(comparison['similarities'])}")
    print(f"  - Differences: {len(comparison['differences'])}")
    print(f"  - Recommendations: {len(comparison['recommendations'])}")
    
    print("\n MetaReasoningTool test completed!")