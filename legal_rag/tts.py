# tts.py - Vietnamese F5-TTS
# =========================

import torch
import torchaudio
import numpy as np
from typing import Optional, Union, Dict, Any
from pathlib import Path
import warnings
import re
import time
warnings.filterwarnings('ignore')

try:
    # Import F5-TTS components (assuming they're available)
    from f5_tts import F5TTS, load_vocoder
    from f5_tts.utils import normalize_text, split_text
    F5_AVAILABLE = True
except ImportError:
    print(" F5-TTS không có sẵn, sử dụng fallback TTS")
    F5_AVAILABLE = False

class VietnameseF5TTS:
    """
    Vietnamese F5-TTS wrapper for legal document text-to-speech
    Fallback to edge-tts hoặc simple TTS nếu F5-TTS không available
    """
    
    def __init__(self, model_name: str = "hynt/F5-TTS-Vietnamese-ViVoice", 
                 device: str = "auto", sample_rate: int = 24000):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = self._get_device(device)
        
        print(f" Initializing Vietnamese F5-TTS: {model_name}")
        
        # Load model
        self.model = None
        self.vocoder = None
        self.tts_available = False
        
        self._load_model()
        
        # Fallback TTS nếu F5 không có
        if not self.tts_available:
            self._setup_fallback_tts()
    
    def _get_device(self, device: str) -> str:
        """Determine device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps" 
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load F5-TTS model"""
        if not F5_AVAILABLE:
            print(" F5-TTS package not available")
            return
        
        try:
            # Load F5-TTS model
            self.model = F5TTS.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load vocoder
            self.vocoder = load_vocoder("vietnamese")
            if self.vocoder:
                self.vocoder = self.vocoder.to(self.device)
                self.vocoder.eval()
            
            self.tts_available = True
            print(f" F5-TTS loaded successfully on {self.device}")
            
        except Exception as e:
            print(f" Failed to load F5-TTS: {e}")
            self.tts_available = False
    
    def _setup_fallback_tts(self):
        """Setup fallback TTS (edge-tts hoặc simple TTS)"""
        try:
            # Thử edge-tts trước
            import edge_tts
            self.fallback_type = "edge-tts"
            print(" Using edge-tts as fallback")
            
        except ImportError:
            try:
                # Thử pyttsx3
                import pyttsx3
                self.fallback_engine = pyttsx3.init()
                self.fallback_type = "pyttsx3"
                print(" Using pyttsx3 as fallback")
                
                # Configure pyttsx3
                voices = self.fallback_engine.getProperty('voices')
                for voice in voices:
                    if 'vietnamese' in voice.name.lower() or 'vi' in voice.id.lower():
                        self.fallback_engine.setProperty('voice', voice.id)
                        break
                
                self.fallback_engine.setProperty('rate', 150)
                
            except ImportError:
                self.fallback_type = "none"
                print(" No TTS fallback available")
    
    def preprocess_legal_text(self, text: str) -> str:
        """
        Preprocess legal text for better TTS pronunciation
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Handle legal terminology
        legal_replacements = {
            r'Điều\s+(\d+)': r'Điều số \1',
            r'Khoản\s+(\d+)': r'Khoản số \1', 
            r'Điểm\s+([a-z])': r'Điểm \1',
            r'(\d+)/(\d+)/([A-Z-]+)': r'\1 trên \2 trên \3',  # số văn bản
            r'NĐ-CP': 'Nghị định Chính phủ',
            r'QĐ': 'Quyết định',
            r'TT': 'Thông tư',
            r'BLHS': 'Bộ luật Hình sự',
            r'BLDS': 'Bộ luật Dân sự',
        }
        
        for pattern, replacement in legal_replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Handle abbreviations
        text = text.replace('CHXHCN', 'Cộng hòa Xã hội Chủ nghĩa')
        text = text.replace('VN', 'Việt Nam')
        text = text.replace('TP.', 'Thành phố ')
        
        # Add pauses for better pronunciation
        text = text.replace('.', '. ')
        text = text.replace(';', ', ')
        text = text.replace(':', ': ')
        
        return text
    
    def synthesize(self, text: str, output_file: str = None, 
                  reference_audio: str = None, reference_text: str = None) -> Optional[str]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            output_file: Output audio file path
            reference_audio: Reference audio for voice cloning
            reference_text: Reference text corresponding to reference_audio
        """
        if not text.strip():
            return None
        
        # Preprocess text
        processed_text = self.preprocess_legal_text(text)
        
        # Generate output filename nếu không có
        if not output_file:
            timestamp = int(time.time())
            output_file = f"tts_output_{timestamp}.wav"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.tts_available:
            return self._synthesize_f5tts(processed_text, str(output_path), reference_audio, reference_text)
        else:
            return self._synthesize_fallback(processed_text, str(output_path))
    
    def _synthesize_f5tts(self, text: str, output_file: str, 
                         reference_audio: str = None, reference_text: str = None) -> Optional[str]:
        """Synthesize using F5-TTS"""
        try:
            # Split text nếu quá dài
            max_length = 500  # characters
            if len(text) > max_length:
                chunks = self._split_text(text, max_length)
            else:
                chunks = [text]
            
            audio_segments = []
            
            for i, chunk in enumerate(chunks):
                print(f"🎵 Synthesizing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
                
                # Generate audio
                with torch.no_grad():
                    # F5-TTS generation
                    if reference_audio and reference_text:
                        # Voice cloning mode
                        audio = self.model.generate(
                            text=chunk,
                            reference_audio=reference_audio,
                            reference_text=reference_text
                        )
                    else:
                        # Default voice
                        audio = self.model.generate(text=chunk)
                    
                    # Convert to numpy nếu cần
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().numpy()
                    
                    audio_segments.append(audio)
            
            # Concatenate audio segments
            if len(audio_segments) > 1:
                final_audio = np.concatenate(audio_segments, axis=0)
            else:
                final_audio = audio_segments[0]
            
            # Normalize audio
            final_audio = final_audio / np.max(np.abs(final_audio))
            
            # Save audio
            torchaudio.save(
                output_file, 
                torch.tensor(final_audio).unsqueeze(0), 
                self.sample_rate
            )
            
            print(f" Audio saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f" F5-TTS synthesis failed: {e}")
            return self._synthesize_fallback(text, output_file)
    
    def _synthesize_fallback(self, text: str, output_file: str) -> Optional[str]:
        """Synthesize using fallback TTS"""
        try:
            if self.fallback_type == "edge-tts":
                return self._synthesize_edge_tts(text, output_file)
            elif self.fallback_type == "pyttsx3":
                return self._synthesize_pyttsx3(text, output_file)
            else:
                print(f" TTS không khả dụng, chỉ hiển thị text:")
                print(f" Text content: {text}")
                return None
                
        except Exception as e:
            print(f" Fallback TTS failed: {e}")
            print(f" Text content: {text}")
            return None
    
    def _synthesize_edge_tts(self, text: str, output_file: str) -> Optional[str]:
        """Synthesize using edge-tts"""
        try:
            import edge_tts
            import asyncio
            
            async def generate_audio():
                communicate = edge_tts.Communicate(text, "vi-VN-HoaiMyNeural")
                await communicate.save(output_file)
            
            # Run async function
            asyncio.run(generate_audio())
            print(f" Edge-TTS audio saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f" Edge-TTS failed: {e}")
            return None
    
    def _synthesize_pyttsx3(self, text: str, output_file: str) -> Optional[str]:
        """Synthesize using pyttsx3"""
        try:
            self.fallback_engine.save_to_file(text, output_file)
            self.fallback_engine.runAndWait()
            print(f" pyttsx3 audio saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f" pyttsx3 failed: {e}")
            return None
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into smaller chunks for TTS"""
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence exceeds max_length, save current chunk
            if len(current_chunk + ". " + sentence) > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split by words
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk + " " + word) > max_length:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            temp_chunk += (" " + word) if temp_chunk else word
                    if temp_chunk:
                        current_chunk = temp_chunk
            else:
                current_chunk += (". " + sentence) if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get TTS model information"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'sample_rate': self.sample_rate,
            'f5_tts_available': self.tts_available,
            'fallback_type': getattr(self, 'fallback_type', 'none'),
            'model_loaded': self.model is not None,
            'vocoder_loaded': self.vocoder is not None
        }
    
    def test_synthesis(self, test_text: str = None) -> Optional[str]:
        """Test TTS synthesis"""
        if not test_text:
            test_text = "Điều 1. Bộ luật này quy định về tội phạm và hình phạt. Xin chào, đây là test tổng hợp tiếng nói tiếng Việt cho hệ thống pháp luật."
        
        print(f" Testing TTS with text: {test_text[:50]}...")
        
        output_file = "test_tts.wav"
        result = self.synthesize(test_text, output_file)
        
        if result:
            print(f" Test synthesis successful: {result}")
        else:
            print(" Test synthesis failed")
        
        return result

if __name__ == "__main__":
    # Test VietnameseF5TTS
    print(" Testing VietnameseF5TTS...")
    
    tts = VietnameseF5TTS()
    
    # Print model info
    info = tts.get_model_info()
    print(f" TTS Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test synthesis
    test_texts = [
        "Điều 15. Tuổi chịu trách nhiệm hình sự. Người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự về mọi tội phạm.",
        "Luật Hôn nhân và Gia đình quy định về điều kiện kết hôn. Nam từ đủ 20 tuổi, nữ từ đủ 18 tuổi.",
        "Bộ luật Hình sự Việt Nam có hiệu lực từ ngày 01 tháng 01 năm 2018."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test Case {i} ---")
        print(f" Text: {text}")
        
        output_file = f"test_legal_{i}.wav"
        result = tts.synthesize(text, output_file)
        
        if result:
            print(f" Audio generated: {result}")
        else:
            print(" Audio generation failed")
        
        print("-" * 50)
    
    # Test long text
    long_text = """
    Điều 1. Phạm vi điều chỉnh
    1. Bộ luật này quy định về tội phạm và hình phạt.
    2. Bộ luật này áp dụng đối với mọi người phạm tội trên lãnh thổ nước Cộng hòa xã hội chủ nghĩa Việt Nam.
    
    Điều 2. Nhiệm vụ của Bộ luật hình sự
    Bộ luật hình sự có nhiệm vụ bảo vệ độc lập, chủ quyền, thống nhất, toàn vẹn lãnh thổ của Tổ quốc.
    """
    
    print(f"\n Testing long text synthesis...")
    long_result = tts.synthesize(long_text.strip(), "test_long_legal.wav")
    if long_result:
        print(f" Long text synthesis successful: {long_result}")
    else:
        print(" Long text synthesis failed")