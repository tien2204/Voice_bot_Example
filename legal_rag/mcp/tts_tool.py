# mcp/tts_tool.py - TTS as MCP Tool
# =================================

from typing import Optional, Dict, Any, List
from pathlib import Path
import time
import hashlib
import json

class TTSTool:
    """
    MCP Tool wrapper cho Vietnamese F5-TTS
    Expose TTS như một service độc lập có thể reuse
    """
    
    def __init__(self, tts_engine, cache_dir: str = "tts_cache", max_cache_files: int = 100):
        self.tts_engine = tts_engine
        self.cache_dir = Path(cache_dir)
        self.max_cache_files = max_cache_files
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load cache index
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
        print(f" TTSTool initialized with cache: {self.cache_dir}")
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """Load TTS cache index"""
        if self.cache_index_file.exists():
            try:
                with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_cache_index(self):
        """Save TTS cache index"""
        try:
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f" Failed to save cache index: {e}")
    
    def _generate_cache_key(self, text: str, voice_settings: Dict[str, Any] = None) -> str:
        """Generate cache key for text and settings"""
        content = text + str(voice_settings or {})
        return hashlib.md5(content.encode()).hexdigest()
    
    def speak(self, text: str, output_file: str = None, voice_settings: Dict[str, Any] = None, 
             use_cache: bool = True) -> Optional[str]:
        """
        MCP Tool: Convert text to speech với caching
        
        Args:
            text: Text to convert
            output_file: Output audio file path (optional)
            voice_settings: Voice configuration (optional)
            use_cache: Use cached audio if available
        """
        if not text.strip():
            return None
        
        print(f"🎵 Converting to speech: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Check cache first
        if use_cache:
            cached_file = self._get_cached_audio(text, voice_settings)
            if cached_file:
                if output_file and cached_file != output_file:
                    # Copy cached file to requested location
                    import shutil
                    shutil.copy2(cached_file, output_file)
                    print(f" Copied cached audio to: {output_file}")
                    return output_file
                else:
                    print(f" Using cached audio: {cached_file}")
                    return cached_file
        
        # Generate audio
        start_time = time.time()
        
        if output_file:
            audio_file = self.tts_engine.synthesize(text, output_file, **voice_settings or {})
        else:
            # Generate with timestamp filename
            timestamp = int(time.time())
            output_file = self.cache_dir / f"tts_{timestamp}.wav"
            audio_file = self.tts_engine.synthesize(text, str(output_file), **voice_settings or {})
        
        generation_time = time.time() - start_time
        
        if audio_file:
            # Cache the result
            if use_cache:
                self._cache_audio(text, voice_settings, audio_file, generation_time)
            
            print(f" Audio generated in {generation_time:.2f}s: {audio_file}")
            return audio_file
        else:
            print(" Failed to generate audio")
            return None
    
    def _get_cached_audio(self, text: str, voice_settings: Dict[str, Any] = None) -> Optional[str]:
        """Get cached audio file if exists"""
        cache_key = self._generate_cache_key(text, voice_settings)
        
        if cache_key in self.cache_index:
            cached_info = self.cache_index[cache_key]
            cached_file = Path(cached_info['file_path'])
            
            # Check if cached file still exists
            if cached_file.exists():
                # Update access time
                cached_info['last_accessed'] = time.time()
                self._save_cache_index()
                return str(cached_file)
            else:
                # Remove from index if file doesn't exist
                del self.cache_index[cache_key]
                self._save_cache_index()
        
        return None
    
    def _cache_audio(self, text: str, voice_settings: Dict[str, Any], audio_file: str, generation_time: float):
        """Cache audio file"""
        cache_key = self._generate_cache_key(text, voice_settings)
        
        # Move file to cache directory if not already there
        audio_path = Path(audio_file)
        if not audio_path.parent.samefile(self.cache_dir):
            cached_path = self.cache_dir / f"cached_{cache_key}.wav"
            import shutil
            shutil.copy2(audio_file, cached_path)
            cached_file = str(cached_path)
        else:
            cached_file = audio_file
        
        # Add to cache index
        self.cache_index[cache_key] = {
            'text': text[:100],  # Store first 100 chars for reference
            'text_length': len(text),
            'file_path': cached_file,
            'voice_settings': voice_settings or {},
            'generation_time': generation_time,
            'created_time': time.time(),
            'last_accessed': time.time(),
            'file_size': Path(cached_file).stat().st_size if Path(cached_file).exists() else 0
        }
        
        # Clean up old cache if needed
        self._cleanup_cache()
        
        # Save index
        self._save_cache_index()
    
    def batch_speak(self, texts: List[str], output_dir: str = None, 
                   voice_settings: Dict[str, Any] = None, use_cache: bool = True) -> List[Optional[str]]:
        """
        MCP Tool: Batch convert multiple texts to speech
        """
        print(f" Batch converting {len(texts)} texts to speech...")
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
        
        results = []
        
        for i, text in enumerate(texts):
            output_file = None
            if output_dir:
                output_file = Path(output_dir) / f"batch_{i+1:03d}.wav"
            
            audio_file = self.speak(
                text, 
                str(output_file) if output_file else None, 
                voice_settings, 
                use_cache
            )
            
            results.append(audio_file)
            
            # Progress update
            if (i + 1) % 5 == 0:
                print(f" Progress: {i+1}/{len(texts)} completed")
        
        successful = sum(1 for r in results if r is not None)
        print(f" Batch conversion completed: {successful}/{len(texts)} successful")
        
        return results
    
    def speak_legal_document(self, document: Dict[str, Any], output_file: str = None, 
                           include_metadata: bool = True) -> Optional[str]:
        """
        MCP Tool: Convert legal document to speech với formatting đặc biệt
        """
        print(f" Converting legal document to speech...")
        
        # Format legal document for TTS
        formatted_text = self._format_legal_document(document, include_metadata)
        
        # Use legal-specific voice settings
        legal_voice_settings = {
            'speed_rate': 0.9,  # Slower for legal text
            'pause_between_articles': True,
            'emphasize_legal_terms': True
        }
        
        return self.speak(formatted_text, output_file, legal_voice_settings)
    
    def _format_legal_document(self, document: Dict[str, Any], include_metadata: bool = True) -> str:
        """Format legal document for better TTS reading"""
        parts = []
        
        # Add metadata if requested
        if include_metadata:
            if 'document_type' in document:
                parts.append(f"Loại văn bản: {document['document_type']}")
            
            if 'document_number' in document:
                parts.append(f"Số văn bản: {document['document_number']}")
            
            if 'title' in document:
                parts.append(f"Tiêu đề: {document['title']}")
            
            parts.append("Nội dung:")
        
        # Add main content
        content = document.get('content', document.get('page_content', ''))
        if content:
            # Enhance legal text formatting
            formatted_content = self._enhance_legal_text_for_tts(content)
            parts.append(formatted_content)
        
        return "\n\n".join(parts)
    
    def _enhance_legal_text_for_tts(self, text: str) -> str:
        """Enhance legal text for better TTS pronunciation"""
        import re
        
        # Add pauses after legal article numbers
        text = re.sub(r'(Điều\s+\d+)', r'\1. ', text)
        text = re.sub(r'(Khoản\s+\d+)', r'\1. ', text)
        text = re.sub(r'(Điểm\s+[a-z])', r'\1. ', text)
        
        # Add emphasis to important terms
        