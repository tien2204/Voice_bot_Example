from datasets import load_dataset, Audio
from argparse import ArgumentParser
from src.models import model_cls_mapping
import json
from tqdm import tqdm
from loguru import logger
from src.models import LiveKitAssistant
import sqlite3
import hashlib
import pickle
import functools
import os
import logging
logging.basicConfig(level=logging.DEBUG)
# Silence sqlite3 log messages
logging.getLogger('sqlite3').setLevel(logging.WARNING)
# Silence numba log messages
logging.getLogger('numba').setLevel(logging.WARNING)
def sqlite_cache(db_path='cache.sqlite'):
    def decorator(func):
        
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB
            )
        ''')
        conn.commit()
        conn.close()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            
            key_data = (func.__name__, args, kwargs)
            key = hashlib.sha256(pickle.dumps(key_data)).hexdigest()

            
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute('SELECT value FROM cache WHERE key = ?', (key,))
            row = c.fetchone()
            if row:
                result = pickle.loads(row[0])
                conn.close()
                return result

            
            result = func(*args, **kwargs)
            
            # Only cache valid, non-empty results
            should_cache = True
            if result is None:
                should_cache = False
                logger.warning(f"Not caching None result for {func.__name__}")
            elif isinstance(result, str) and result.strip() == "":
                should_cache = False
                logger.warning(f"Not caching empty string result for {func.__name__}")
            elif isinstance(result, dict):
                # For dict results, check if transcription is empty
                transcription = result.get("transcription", "")
                if isinstance(transcription, str) and transcription.strip() == "":
                    should_cache = False
                    logger.warning(f"Not caching result with empty transcription for {func.__name__}")
            
            if should_cache:
                c.execute('INSERT INTO cache (key, value) VALUES (?, ?)', (key, pickle.dumps(result)))
                conn.commit()
                logger.info(f"Cached result for {func.__name__}")
            
            conn.close()
            return result

        return wrapper
    return decorator
def retry_generate_audio(model, audio_data, max_retries=3, return_all=True):
    """
    Retry wrapper for generate_audio with exponential backoff.
    """
    import time
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Audio generation attempt {attempt + 1}/{max_retries}")
            
            result = model.generate_audio(audio_data)
            
            # Check if result is valid
            if result is None:
                logger.warning(f"Attempt {attempt + 1} returned None")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
            elif isinstance(result, str) and result.strip() == "":
                logger.warning(f"Attempt {attempt + 1} returned empty string")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
            
            # Success case
            logger.info(f"Audio generation succeeded on attempt {attempt + 1}")
            if return_all:
                return {"transcription": result, "attempt": attempt + 1}
            else:
                return result
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed with exception: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed")
                if return_all:
                    return {"transcription": "", "attempt": max_retries, "error": str(e)}
                else:
                    return ""

def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen2', choices=list(model_cls_mapping.keys()))
    parser.add_argument('--data', type=str, default='alpacaeval')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--modality', type=str, default='audio', choices=['audio', 'text', 'ttft'])
    args = parser.parse_args()

    
    data = load_dataset('hlt-lab/voicebench', args.data, split=args.split)
    data = data.cast_column("audio", Audio(sampling_rate=16_000))

    
    # Initialize model
    model = model_cls_mapping[args.model]()
    
    if args.modality == 'ttft':
        
        _ = model.generate_ttft(data[0]['audio'])

    
    # Don't use cache for generate_audio directly - use retry wrapper instead
    # gen_audio=sqlite_cache()(model.generate_audio)
    gen_audio_with_retry = sqlite_cache()(lambda audio_data: retry_generate_audio(model, audio_data, max_retries=3, return_all=True))
    
    results = []
    
    for i, item in enumerate(tqdm(data, total=len(data))):
        logger.info(f"Processing item {i+1}/{len(data)}")
        tmp = {k: v for k, v in item.items() if k != 'audio'}
        
        try:
            if args.modality == 'text':
                response = model.generate_text(item['prompt'])
            elif args.modality == 'audio':
                resp = gen_audio_with_retry(item['audio'])
                logger.info(f"Audio generation result: {resp}")
                
                if isinstance(resp, dict):
                    response = resp.get("transcription", "")
                    if "error" in resp:
                        logger.error(f"Error in audio generation: {resp['error']}")
                else:
                    response = resp if resp else ""
                
            elif args.modality == 'ttft':
                response = model.generate_ttft(item['audio'])
            else:
                raise NotImplementedError
                
            logger.info(f"Prompt: {item['prompt']}")
            logger.info(f"Response: {response}")
            logger.info('====================================')
            
        except Exception as e:
            logger.error(f"Error processing item {i+1}: {e}")
            response = ""
            
        tmp['response'] = response
        results.append(tmp)

    
    # Write results to file
    output_file = f'{args.model}-{args.data}-{args.split}-{args.modality}.jsonl'
    with open(output_file, 'w') as f:
        for record in results:
            json_line = json.dumps(record)  
            f.write(json_line + '\n')
    
    logger.info(f"Results written to {output_file}")
    
    # Cleanup for LiveKit models
    if hasattr(model, 'disconnect'):
        logger.info("Disconnecting from LiveKit...")
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(model.disconnect())
            logger.info("Successfully disconnected from LiveKit")
        except Exception as e:
            logger.error(f"Error disconnecting from LiveKit: {e}")


if __name__ == '__main__':
    main()
