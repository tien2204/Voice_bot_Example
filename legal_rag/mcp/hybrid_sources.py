# mcp/hybrid_sources.py - Hybrid Knowledge Sources
# ===============================================

import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import re
import time
from bs4 import BeautifulSoup
import hashlib

class HybridKnowledgeTool:
    """
    MCP Tools cho Hybrid Knowledge Sources:
    1. Fetch law news - tin tức pháp luật mới
    2. Official gazette - văn bản công báo
    3. Live legal updates - cập nhật luật mới
    4. External legal databases - CSDL pháp luật bên ngoài
    """
    
    def __init__(self, cache_duration_hours: int = 6):
        self.cache_duration_hours = cache_duration_hours
        self.cache = {}  # Simple in-memory cache
        
        # URLs for Vietnamese legal sources
        self.sources = {
            'phap_luat_vn': 'https://plo.vn/phap-luat',
            'thanh_nien_phap_luat': 'https://thanhnien.vn/phap-luat',
            'vnexpress_phap_luat': 'https://vnexpress.net/phap-luat',
            'dan_tri_phap_luat': 'https://dantri.com.vn/phap-luat',
            'cong_bao_gov': 'https://congbao.chinhphu.vn',
            'thong_tin_phap_luat': 'https://thongtinphapluat.vn'
        }
        
        print(" HybridKnowledgeTool initialized")
    
    def fetch_law_news(self, max_articles: int = 10, hours_back: int = 24) -> Dict[str, Any]:
        """
        MCP Tool: Lấy tin tức pháp luật mới từ các nguồn
        """
        print(f" Fetching law news (last {hours_back} hours)...")
        
        cache_key = f"law_news_{hours_back}_{max_articles}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            print(" Returning cached law news")
            return self.cache[cache_key]['data']
        
        all_articles = []
        fetch_results = {}
        
        # Fetch from multiple sources
        for source_name, source_url in self.sources.items():
            if 'cong_bao' in source_name:
                continue  # Skip official gazette for news
            
            try:
                articles = self._fetch_from_source(source_name, source_url, max_articles // 2)
                all_articles.extend(articles)
                fetch_results[source_name] = {
                    'success': True,
                    'articles_count': len(articles)
                }
            except Exception as e:
                fetch_results[source_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_articles = [
            article for article in all_articles
            if article.get('published_time') and 
               datetime.fromisoformat(article['published_time']) >= cutoff_time
        ]
        
        # Sort by time and limit
        recent_articles.sort(key=lambda x: x.get('published_time', ''), reverse=True)
        recent_articles = recent_articles[:max_articles]
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'hours_back': hours_back,
            'total_articles': len(recent_articles),
            'sources_fetched': fetch_results,
            'articles': recent_articles,
            'summary': self._generate_news_summary(recent_articles)
        }
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def _fetch_from_source(self, source_name: str, source_url: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch articles from a specific source"""
        # This is a simplified implementation - in reality would need proper parsing for each site
        articles = []
        
        try:
            # Simulate fetching (replace with actual implementation)
            sample_articles = self._get_sample_articles(source_name, limit)
            articles.extend(sample_articles)
            
        except Exception as e:
            print(f" Failed to fetch from {source_name}: {e}")
        
        return articles
    
    def _get_sample_articles(self, source_name: str, limit: int) -> List[Dict[str, Any]]:
        """Generate sample legal news articles"""
        sample_articles = [
            {
                'title': 'Chính phủ ban hành Nghị định mới về quản lý đầu tư công',
                'summary': 'Nghị định 08/2024/NĐ-CP quy định chi tiết về quy trình quản lý đầu tư công...',
                'url': f'https://{source_name}.vn/article-1',
                'published_time': (datetime.now() - timedelta(hours=2)).isoformat(),
                'source': source_name,
                'category': 'investment_law',
                'legal_documents': ['08/2024/NĐ-CP']
            },
            {
                'title': 'Tòa án Nhân dân Tối cao ban hành hướng dẫn mới về xử lý tội phạm kinh tế',
                'summary': 'Thông tư 03/2024/TT-TANDTC hướng dẫn việc áp dụng một số quy định...',
                'url': f'https://{source_name}.vn/article-2', 
                'published_time': (datetime.now() - timedelta(hours=6)).isoformat(),
                'source': source_name,
                'category': 'economic_crime',
                'legal_documents': ['03/2024/TT-TANDTC']
            },
            {
                'title': 'Luật Bảo vệ môi trường sửa đổi có hiệu lực từ 2025',
                'summary': 'Quốc hội đã thông qua Luật sửa đổi, bổ sung một số điều của Luật Bảo vệ môi trường...',
                'url': f'https://{source_name}.vn/article-3',
                'published_time': (datetime.now() - timedelta(hours=12)).isoformat(),
                'source': source_name,
                'category': 'environmental_law',
                'legal_documents': ['Luật Bảo vệ môi trường 2020 sửa đổi']
            }
        ]
        
        return sample_articles[:limit]
    
    def fetch_official_gazette(self, days_back: int = 7) -> Dict[str, Any]:
        """
        MCP Tool: Lấy văn bản từ Công báo chính phủ
        """
        print(f" Fetching official gazette (last {days_back} days)...")
        
        cache_key = f"official_gazette_{days_back}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            print(" Returning cached official gazette")
            return self.cache[cache_key]['data']
        
        # Simulate official gazette documents
        documents = self._get_sample_official_documents(days_back)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'days_back': days_back,
            'total_documents': len(documents),
            'documents': documents,
            'categories': self._categorize_documents(documents),
            'summary': self._generate_gazette_summary(documents)
        }
        
        # Cache result
        self._cache_result(cache_key, result)
        
        return result
    
    def _get_sample_official_documents(self, days_back: int) -> List[Dict[str, Any]]:
        """Generate sample official documents"""
        documents = [
            {
                'document_number': '15/2024/NĐ-CP',
                'title': 'Nghị định về quản lý hoạt động kinh doanh dịch vụ logistics',
                'type': 'Nghị định',
                'issuing_agency': 'Chính phủ',
                'effective_date': '2024-03-01',
                'published_date': (datetime.now() - timedelta(days=2)).isoformat()[:10],
                'summary': 'Quy định về điều kiện, thủ tục kinh doanh dịch vụ logistics...',
                'category': 'business_regulation',
                'status': 'có hiệu lực'
            },
            {
                'document_number': '02/2024/TT-BTP',
                'title': 'Thông tư hướng dẫn thi hành Luật Tổ chức Tòa án nhân dân',
                'type': 'Thông tư',
                'issuing_agency': 'Bộ Tư pháp',
                'effective_date': '2024-02-15',
                'published_date': (datetime.now() - timedelta(days=5)).isoformat()[:10],
                'summary': 'Hướng dẫn chi tiết về tổ chức và hoạt động của tòa án...',
                'category': 'judicial_organization',
                'status': 'có hiệu lực'
            },
            {
                'document_number': '1234/QĐ-TTg',
                'title': 'Quyết định phê duyệt Chiến lược phát triển kinh tế số 2024-2030',
                'type': 'Quyết định',
                'issuing_agency': 'Thủ tướng Chính phủ',
                'effective_date': '2024-01-15',
                'published_date': (datetime.now() - timedelta(days=7)).isoformat()[:10],
                'summary': 'Phê duyệt chiến lược phát triển kinh tế số đến năm 2030...',
                'category': 'economic_strategy',
                'status': 'có hiệu lực'
            }
        ]
        
        # Filter by days_back
        cutoff_date = datetime.now() - timedelta(days=days_back)
        filtered_docs = [
            doc for doc in documents
            if datetime.fromisoformat(doc['published_date']) >= cutoff_date.date()
        ]
        
        return filtered_docs
    
    def _categorize_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize official documents"""
        categories = {}
        for doc in documents:
            category = doc.get('category', 'other')
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _generate_gazette_summary(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of official gazette"""
        doc_types = {}
        agencies = {}
        
        for doc in documents:
            doc_type = doc.get('type', 'Unknown')
            agency = doc.get('issuing_agency', 'Unknown')
            
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            agencies[agency] = agencies.get(agency, 0) + 1
        
        return {
            'document_types': doc_types,
            'issuing_agencies': agencies,
            'latest_document': max(documents, key=lambda x: x['published_date']) if documents else None
        }
    
    def search_legal_updates(self, query: str, date_range: int = 30) -> Dict[str, Any]:
        """
        MCP Tool: Tìm kiếm cập nhật pháp luật theo query
        """
        print(f" Searching legal updates for: '{query}'")
        
        # Combine news and official documents
        news_result = self.fetch_law_news(max_articles=20, hours_back=date_range * 24)
        gazette_result = self.fetch_official_gazette(days_back=date_range)
        
        # Search in news articles
        matching_articles = []
        for article in news_result.get('articles', []):
            if self._match_query(query, article):
                matching_articles.append({
                    'type': 'news_article',
                    'title': article['title'],
                    'summary': article['summary'],
                    'url': article['url'],
                    'published_time': article['published_time'],
                    'source': article['source'],
                    'relevance_score': self._calculate_relevance(query, article)
                })
        
        # Search in official documents
        matching_documents = []
        for doc in gazette_result.get('documents', []):
            if self._match_query(query, doc):
                matching_documents.append({
                    'type': 'official_document',
                    'document_number': doc['document_number'],
                    'title': doc['title'],
                    'summary': doc['summary'],
                    'issuing_agency': doc['issuing_agency'],
                    'published_date': doc['published_date'],
                    'relevance_score': self._calculate_relevance(query, doc)
                })
        
        # Combine and sort by relevance
        all_results = matching_articles + matching_documents
        all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        result = {
            'query': query,
            'date_range_days': date_range,
            'search_timestamp': datetime.now().isoformat(),
            'total_results': len(all_results),
            'news_articles': len(matching_articles),
            'official_documents': len(matching_documents),
            'results': all_results[:15],  # Top 15 results
            'search_summary': self._generate_search_summary(query, all_results)
        }
        
        return result
    
    def _match_query(self, query: str, item: Dict[str, Any]) -> bool:
        """Check if item matches search query"""
        query_lower = query.lower()
        
        # Check title
        if 'title' in item and query_lower in item['title'].lower():
            return True
        
        # Check summary
        if 'summary' in item and query_lower in item['summary'].lower():
            return True
        
        # Check legal documents mentioned
        if 'legal_documents' in item:
            for doc in item['legal_documents']:
                if query_lower in doc.lower():
                    return True
        
        # Check document number
        if 'document_number' in item and query_lower in item['document_number'].lower():
            return True
        
        return False
    
    def _calculate_relevance(self, query: str, item: Dict[str, Any]) -> float:
        """Calculate relevance score (0-1)"""
        score = 0.0
        query_words = query.lower().split()
        
        # Title match (highest weight)
        title = item.get('title', '').lower()
        title_matches = sum(1 for word in query_words if word in title)
        score += (title_matches / len(query_words)) * 0.5
        
        # Summary match
        summary = item.get('summary', '').lower()
        summary_matches = sum(1 for word in query_words if word in summary)
        score += (summary_matches / len(query_words)) * 0.3
        
        # Exact phrase match bonus
        if query.lower() in title or query.lower() in summary:
            score += 0.2
        
        return min(score, 1.0)
    
    def _generate_search_summary(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate search summary"""
        if not results:
            return {'message': 'No results found'}
        
        # Analyze result types
        news_count = sum(1 for r in results if r['type'] == 'news_article')
        doc_count = sum(1 for r in results if r['type'] == 'official_document')
        
        # Find most relevant result
        top_result = results[0] if results else None
        
        return {
            'total_found': len(results),
            'news_articles': news_count,
            'official_documents': doc_count,
            'top_result': {
                'title': top_result['title'],
                'type': top_result['type'],
                'relevance': top_result['relevance_score']
            } if top_result else None,
            'search_effectiveness': 'high' if len(results) > 5 else 'medium' if len(results) > 0 else 'low'
        }
    
    def get_trending_legal_topics(self, days_back: int = 7) -> Dict[str, Any]:
        """
        MCP Tool: Lấy các chủ đề pháp luật đang trending
        """
        print(f" Analyzing trending legal topics (last {days_back} days)...")
        
        # Get recent data
        news_result = self.fetch_law_news(max_articles=50, hours_back=days_back * 24)
        gazette_result = self.fetch_official_gazette(days_back=days_back)
        
        # Extract topics from news
        news_topics = {}
        for article in news_result.get('articles', []):
            topics = self._extract_topics_from_text(article['title'] + ' ' + article['summary'])
            for topic in topics:
                news_topics[topic] = news_topics.get(topic, 0) + 1
        
        # Extract topics from official documents
        official_topics = {}
        for doc in gazette_result.get('documents', []):
            topics = self._extract_topics_from_text(doc['title'] + ' ' + doc['summary'])
            for topic in topics:
                official_topics[topic] = official_topics.get(topic, 0) + 1
        
        # Combine and rank topics
        all_topics = {}
        for topic, count in news_topics.items():
            all_topics[topic] = all_topics.get(topic, 0) + count
        for topic, count in official_topics.items():
            all_topics[topic] = all_topics.get(topic, 0) + count * 1.5  # Weight official docs higher
        
        # Sort by frequency
        trending_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)
        
        result = {
            'analysis_period': f"{days_back} days",
            'analysis_timestamp': datetime.now().isoformat(),
            'data_sources': {
                'news_articles': len(news_result.get('articles', [])),
                'official_documents': len(gazette_result.get('documents', []))
            },
            'trending_topics': [
                {
                    'topic': topic,
                    'frequency': count,
                    'trend_level': self._classify_trend_level(count)
                }
                for topic, count in trending_topics[:10]
            ],
            'topic_categories': self._categorize_trending_topics(trending_topics[:10])
        }
        
        return result
    
    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract legal topics from text"""
        topics = []
        text_lower = text.lower()
        
        # Legal topic keywords
        topic_keywords = {
            'kinh tế': ['kinh tế', 'đầu tư', 'doanh nghiệp', 'thương mại', 'tài chính'],
            'hình sự': ['hình sự', 'tội phạm', 'an ninh', 'trật tự'],
            'dân sự': ['dân sự', 'hợp đồng', 'sở hữu', 'thừa kế'],
            'hành chính': ['hành chính', 'thủ tục', 'giấy phép', 'công vụ'],
            'lao động': ['lao động', 'việc làm', 'bảo hiểm', 'lương'],
            'môi trường': ['môi trường', 'khí hậu', 'ô nhiễm', 'tài nguyên'],
            'giáo dục': ['giáo dục', 'đào tạo', 'học sinh', 'sinh viên'],
            'y tế': ['y tế', 'sức khỏe', 'bệnh viện', 'thuốc'],
            'công nghệ': ['công nghệ', 'số hóa', 'internet', 'AI'],
            'xã hội': ['xã hội', 'phúc lợi', 'trợ cấp', 'an sinh']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _classify_trend_level(self, frequency: int) -> str:
        """Classify trend level based on frequency"""
        if frequency >= 10:
            return 'very_high'
        elif frequency >= 5:
            return 'high'
        elif frequency >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_trending_topics(self, trending_topics: List[Tuple[str, int]]) -> Dict[str, List[str]]:
        """Categorize trending topics"""
        categories = {
            'economic_legal': [],
            'social_legal': [],
            'administrative': [],
            'criminal_law': [],
            'civil_law': []
        }
        
        for topic, _ in trending_topics:
            if topic in ['kinh tế', 'đầu tư', 'doanh nghiệp']:
                categories['economic_legal'].append(topic)
            elif topic in ['xã hội', 'giáo dục', 'y tế']:
                categories['social_legal'].append(topic)
            elif topic in ['hành chính', 'công vụ']:
                categories['administrative'].append(topic)
            elif topic in ['hình sự', 'an ninh']:
                categories['criminal_law'].append(topic)
            elif topic in ['dân sự', 'hợp đồng']:
                categories['civil_law'].append(topic)
        
        return categories
    
    def _generate_news_summary(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of news articles"""
        if not articles:
            return {'message': 'No articles found'}
        
        categories = {}
        sources = {}
        
        for article in articles:
            category = article.get('category', 'other')
            source = article.get('source', 'unknown')
            
            categories[category] = categories.get(category, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        return {
            'total_articles': len(articles),
            'categories': categories,
            'sources': sources,
            'latest_article': max(articles, key=lambda x: x['published_time']) if articles else None,
            'time_span': self._calculate_time_span(articles)
        }
    
    def _calculate_time_span(self, articles: List[Dict[str, Any]]) -> str:
        """Calculate time span of articles"""
        if not articles:
            return "N/A"
        
        times = [datetime.fromisoformat(article['published_time']) for article in articles]
        oldest = min(times)
        newest = max(times)
        span = newest - oldest
        
        if span.days > 0:
            return f"{span.days} days"
        else:
            return f"{span.seconds // 3600} hours"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache is still valid"""
        if cache_key not in self.cache:
            return False
        
        cached_time = datetime.fromisoformat(self.cache[cache_key]['timestamp'])
        expiry_time = cached_time + timedelta(hours=self.cache_duration_hours)
        
        return datetime.now() < expiry_time
    
    def _cache_result(self, cache_key: str, data: Any):
        """Cache result with timestamp"""
        self.cache[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
    
    def clear_cache(self) -> Dict[str, Any]:
        """
        MCP Tool: Clear cached data
        """
        cache_size = len(self.cache)
        self.cache.clear()
        
        return {
            'cleared': True,
            'items_cleared': cache_size,
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Test HybridKnowledgeTool
    print(" Testing HybridKnowledgeTool...")
    
    # Initialize
    hybrid_tool = HybridKnowledgeTool(cache_duration_hours=1)
    
    # Test fetch law news
    print(f"\n Testing fetch_law_news...")
    news_result = hybrid_tool.fetch_law_news(max_articles=5, hours_back=48)
    print(f"  - Found {news_result['total_articles']} articles")
    print(f"  - Sources: {list(news_result['sources_fetched'].keys())}")
    
    # Test official gazette
    print(f"\n Testing fetch_official_gazette...")
    gazette_result = hybrid_tool.fetch_official_gazette(days_back=7)
    print(f"  - Found {gazette_result['total_documents']} documents")
    print(f"  - Categories: {gazette_result['categories']}")
    
    # Test search
    print(f"\n Testing search_legal_updates...")
    search_result = hybrid_tool.search_legal_updates("kinh tế", date_range=30)
    print(f"  - Query: {search_result['query']}")
    print(f"  - Total results: {search_result['total_results']}")
    print(f"  - Search effectiveness: {search_result['search_summary']['search_effectiveness']}")
    
    # Test trending topics
    print(f"\n Testing get_trending_legal_topics...")
    trending_result = hybrid_tool.get_trending_legal_topics(days_back=7)
    print(f"  - Analysis period: {trending_result['analysis_period']}")
    print(f"  - Top topics: {[t['topic'] for t in trending_result['trending_topics'][:3]]}")
    
    print(f"\n HybridKnowledgeTool test completed!")
