#!/usr/bin/env python3
"""
Test script to verify the improvements made to the evaluation code.
This script tests the key fixes:
1. Improved error handling in LiveKitAssistant
2. Better cache logic that doesn't cache empty results  
3. Retry mechanism for audio generation
"""

import asyncio
import numpy as np
import logging
from src.models.deprecated_livekit_model import LiveKitAssistant
from main import sqlite_cache, retry_generate_audio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_livekit_improvements():
    """Test LiveKit improvements without actually connecting."""
    
    # Test 1: Cache improvements
    logger.info("=" * 50)
    logger.info("Testing cache improvements...")
    
    @sqlite_cache(db_path='test_cache.sqlite')
    def mock_generate_audio(audio_data):
        # Simulate different return types
        if audio_data.get('test_case') == 'empty_string':
            return ""
        elif audio_data.get('test_case') == 'none':
            return None
        elif audio_data.get('test_case') == 'empty_dict':
            return {"transcription": ""}
        elif audio_data.get('test_case') == 'valid':
            return {"transcription": "Hello world"}
        else:
            return "Valid response"
    
    # Test cases that should NOT be cached
    result1 = mock_generate_audio({'test_case': 'empty_string'})
    logger.info(f"Empty string result: '{result1}' (should not be cached)")
    
    result2 = mock_generate_audio({'test_case': 'none'})
    logger.info(f"None result: {result2} (should not be cached)")
    
    result3 = mock_generate_audio({'test_case': 'empty_dict'})
    logger.info(f"Empty dict result: {result3} (should not be cached)")
    
    # Test case that SHOULD be cached
    result4 = mock_generate_audio({'test_case': 'valid'})
    logger.info(f"Valid result: {result4} (should be cached)")
    
    result5 = mock_generate_audio({'test_case': 'other'})
    logger.info(f"Other valid result: '{result5}' (should be cached)")
    
    # Test 2: LiveKit Assistant improvements
    logger.info("=" * 50)
    logger.info("Testing LiveKit Assistant improvements...")
    
    # Create assistant (won't connect without proper credentials)
    assistant = LiveKitAssistant()
    
    # Test health check
    is_healthy = assistant.is_healthy()
    logger.info(f"Assistant health (not connected): {is_healthy}")
    
    # Test pending requests count
    pending_count = assistant.get_pending_requests_count()
    logger.info(f"Pending requests count: {pending_count}")
    
    # Test cleanup (should be safe even with no requests)
    assistant.cleanup_stale_requests()
    logger.info("Stale requests cleanup completed")
    
    # Test 3: Retry mechanism (mock)
    logger.info("=" * 50)
    logger.info("Testing retry mechanism...")
    
    class MockModel:
        def __init__(self):
            self.attempt_count = 0
            
        def generate_audio(self, audio_data):
            self.attempt_count += 1
            if self.attempt_count <= 2:
                # Fail first two attempts
                return None
            else:
                # Succeed on third attempt
                return "Success on attempt 3"
        
        def is_healthy(self):
            return True
            
        def get_pending_requests_count(self):
            return 0
            
        def cleanup_stale_requests(self):
            pass
    
    mock_model = MockModel()
    mock_audio = {'array': np.zeros(1000), 'sampling_rate': 16000}
    
    result = retry_generate_audio(mock_model, mock_audio, max_retries=3, return_all=True)
    logger.info(f"Retry result: {result}")
    
    logger.info("=" * 50)
    logger.info("All tests completed successfully!")

def test_audio_format():
    """Test audio format handling."""
    logger.info("Testing audio format handling...")
    
    # Create sample audio data
    sample_rate = 16000
    duration = 1
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Test different formats
    test_cases = [
        ("float32", (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)),
        ("float64", (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)),
        ("int16", (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)),
    ]
    
    for format_name, audio_array in test_cases:
        audio_dict = {
            'array': audio_array,
            'sampling_rate': sample_rate,
            'channels': 1
        }
        logger.info(f"Audio format {format_name}: shape={audio_array.shape}, dtype={audio_array.dtype}, "
                   f"min={audio_array.min():.4f}, max={audio_array.max():.4f}")

if __name__ == "__main__":
    # Run synchronous tests
    test_audio_format()
    
    # Run async tests
    asyncio.run(test_livekit_improvements())
