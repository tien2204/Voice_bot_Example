# Vietnamese Legal RAG System với MCP Tools

Hệ thống RAG (Retrieval-Augmented Generation) tiếng Việt chuyên biệt cho pháp luật với tích hợp MCP (Model Context Protocol) tools.

## 🎯 Tính năng chính

### Core RAG System
- **Dataset**: YuITC Vietnamese Legal Document Retrieval Data từ HuggingFace
- **Hybrid Retrieval**: Kết hợp BM25 + BGE-M3 embeddings
- **LLM**: VinaLlama-2.7B-Chat với quantization
- **TTS**: F5-TTS-Vietnamese-ViVoice cho text-to-speech
- **Vector Store**: Weaviate với fallback SimpleVectorStore

### MCP Tools Suite
1. **Meta-Reasoning Tools** (`mcp/meta_reasoning.py`)
   - Explain Retrieval: Giải thích tại sao tài liệu được chọn
   - Debug Prompt: Hiển thị full prompt context
   - Query Analysis: Phân tích câu hỏi chi tiết

2. **Logging & Analytics** (`mcp/logging_tools.py`)
   - Session logging với full context
   - Performance metrics tracking
   - Export to ElasticSearch/Prometheus (placeholder)

3. **Memory Management** (`mcp/memory.py`)
   - Conversation history management
   - Context window optimization
   - Long-term memory với topics

4. **Hybrid Knowledge Sources** (`mcp/hybrid_sources.py`)
   - Tin tức pháp luật real-time
   - Công báo chính phủ updates
   - Trending legal topics analysis

5. **TTS as Service** (`mcp/tts_tool.py`)
   - TTS caching system
   - Batch conversion
   - Legal document formatting

6. **Feedback System** (`mcp/feedback.py`)
   - User rating & comments
   - Pattern analysis
   - Improvement suggestions

## 🏗️ Cấu trúc dự án

```
legal_rag/
├── main.py                     # Entry point
├── config.json                 # System configuration
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
│
├── data_loader.py              # YuITC dataset loader với chunking
├── weaviate_setup.py           # Vector store setup
├── retriever.py                # Hybrid retriever (BM25 + Vector)
├── llm.py                      # VinaLlama wrapper
├── tts.py                      # F5-TTS Vietnamese
│
├── mcp/                        # MCP Tools
│   ├── __init__.py
│   ├── meta_reasoning.py       # Meta-reasoning tools
│   ├── logging_tools.py        # Logging & analytics
│   ├── memory.py               # Memory management
│   ├── hybrid_sources.py       # External sources
│   ├── tts_tool.py             # TTS service
│   └── feedback.py             # Feedback system
│
├── logs/                       # Log files
├── memory/                     # Session memory
├── tts_cache/                  # TTS cache
└── test_logs/                  # Test logs
```

## 🚀 Cài đặt và chạy

### 1. Cài đặt dependencies

```bash
pip install transformers torch datasets
pip install langchain langchain-community langchain-core
pip install weaviate-client sentence-transformers rank_bm25
pip install FlagEmbedding soundfile librosa pydub
pip install accelerate bitsandbytes faiss-cpu
pip install edge-tts pyttsx3 requests beautifulsoup4
```

### 2. Setup Weaviate (Optional)

```bash
# Docker Weaviate local
docker run -p 8080:8080 semitechnologies/weaviate:latest
```

### 3. Chạy hệ thống

```bash
# Interactive mode (mặc định)
python main.py

# Demo mode
python main.py demo

# Interactive mode với full MCP tools
python main.py interactive
```

## 🛠️ MCP Tools Usage

### Basic Query
```python
# Simple question
❓ Câu hỏi: Tuổi chịu trách nhiệm hình sự là bao nhiêu?
```

### MCP Commands
```bash
# Explain retrieval
mcp explain_retrieval Tuổi chịu trách nhiệm hình sự

# Debug prompt
mcp debug_prompt Điều kiện kết hôn

# View session log
mcp view_session_log

# Get context memory
mcp get_context_memory

# Fetch law news
mcp fetch_law_news

# Submit feedback
# (Sau khi có câu trả lời)
👍 Đánh giá (1-5): 4
💬 Nhận xét: Câu trả lời chính xác nhưng cần thêm chi tiết
```

## 📊 Dataset Processing Strategy

### YuITC Dataset Structure
```json
{
  "question": "Câu hỏi pháp luật",
  "context_list": ["Văn bản luật 1", "Văn bản luật 2"],
  "qid": "question_id", 
  "cid": ["context_id_1", "context_id_2"]
}
```

### Chunking Strategy
1. **Legal Structure-aware**: Chia theo Điều, Khoản, Điểm
2. **Token-based fallback**: 512 tokens với 100 overlap
3. **Metadata extraction**: Số điều, tên luật, loại văn bản

### Retrieval Strategy
- **BM25 (30%)**: Exact keyword matching (Điều X, Luật Y)
- **Vector (70%)**: Semantic similarity với BGE-M3
- **Hybrid fusion**: Weighted combination + normalization

## 🎵 TTS Integration

### F5-TTS Features
- **Model**: hynt/F5-TTS-Vietnamese-ViVoice
- **Legal text preprocessing**: Điều X → "Điều số X"
- **Smart caching**: MD5-based file caching
- **Batch processing**: Multiple texts to audio
- **Legal document formatting**: Pause after articles

### TTS Usage
```python
# Direct synthesis
audio_file = tts_tool.speak("Điều 15. Tuổi chịu trách nhiệm hình sự...")

# Legal document with metadata
legal_audio = tts_tool.speak_legal_document({
    'document_type': 'Bộ luật',
    'title': 'Bộ luật Hình sự', 
    'content': 'Điều 15...'
})

# Batch conversion
batch_results = tts_tool.batch_speak([
    "Text 1", "Text 2", "Text 3"
], output_dir="audio_output")
```

## 📈 Analytics & Monitoring

### Session Logging
- **Full Q&A context** với retrieved documents
- **Performance metrics**: Response time, accuracy estimation
- **User feedback**: Rating, comments, categories

### Feedback Analysis
```python
# Analyze patterns
analysis = feedback_tool.analyze_feedback_patterns(days_back=30)

# Get improvements
suggestions = feedback_tool.generate_improvement_suggestions()

# Export training data
training_file = feedback_tool.export_training_data(min_rating=4)
```

### External Integration (Planned)
- **ElasticSearch**: Session và metrics indexing
- **Prometheus**: Real-time monitoring metrics
- **Grafana**: Dashboard visualization

## 🔧 Configuration

Chỉnh sửa `config.json` để tùy chỉnh:

```json
{
  "retrieval": {
    "k": 5,                    # Số documents retrieve
    "bm25_weight": 0.3,       # Trọng số BM25
    "vector_weight": 0.7      # Trọng số vector search
  },
  "llm": {
    "max_new_tokens": 512,    # Độ dài response
    "temperature": 0.3        # Creativity level
  },
  "mcp_tools": {
    "feedback": {
      "auto_analysis_interval": 10  # Auto-analyze mỗi N feedback
    }
  }