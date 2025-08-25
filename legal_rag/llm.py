# llm.py - VinaLlama LLM Wrapper
# ===============================

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    GenerationConfig
)
from typing import Optional, Dict, Any, List
import re
import warnings
warnings.filterwarnings('ignore')

class VinaLlamaLLM:
    """
    VinaLlama 2.7B Chat model wrapper cho Vietnamese Legal RAG
    Optimized for legal question answering
    """
    
    def __init__(self, model_name: str = "vilm/vinallama-2.7b-chat", 
                 max_length: int = 2048, device_map: str = "auto"):
        self.model_name = model_name
        self.max_length = max_length
        self.device_map = device_map
        
        print(f" Initializing VinaLlama: {model_name}")
        
        # Load model với quantization để tiết kiệm memory
        self._load_model()
        
        # Setup generation config
        self._setup_generation_config()
        
        print(" VinaLlama model ready")
    
    def _load_model(self):
        """Load tokenizer và model với quantization"""
        try:
            # Quantization config cho GPU memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Setup pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=self.device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Set to eval mode
            self.model.eval()
            
        except Exception as e:
            print(f" Failed to load VinaLlama model: {e}")
            print(" Using fallback model...")
            self.model = None
            self.tokenizer = None
    
    def _setup_generation_config(self):
        """Setup generation configuration"""
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            min_new_tokens=10,
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            do_sample=True,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else 0,
            eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else 0,
        )
    
    def format_legal_prompt(self, question: str, context: str) -> str:
        """
        Format prompt specifically for Vietnamese legal questions
        """
        # Check if question is about specific legal articles
        has_dieu = bool(re.search(r'điều\s+\d+', question.lower()))
        has_khoan = bool(re.search(r'khoản\s+\d+', question.lower()))
        has_luat = bool(re.search(r'(luật|bộ luật|nghị định|thông tư)', question.lower()))
        
        # Customize prompt based on question type
        if has_dieu or has_khoan:
            instruction = "Hãy trả lời câu hỏi dựa trên các điều khoản pháp luật được cung cấp. Trích dẫn chính xác số điều, khoản liên quan."
        elif has_luat:
            instruction = "Hãy trả lời câu hỏi về văn bản pháp luật. Nêu rõ tên luật, nghị định liên quan nếu có."
        else:
            instruction = "Hãy trả lời câu hỏi pháp luật một cách chính xác và dễ hiểu."
        
        # Format context
        formatted_context = self._format_context(context)
        
        prompt = f"""<|im_start|>system
Bạn là một chuyên gia pháp luật Việt Nam. Nhiệm vụ của bạn là trả lời các câu hỏi về pháp luật một cách chính xác, chi tiết và dễ hiểu.

Quy tắc trả lời:
1. Dựa vào thông tin pháp luật được cung cấp
2. Trích dẫn chính xác điều, khoản, luật liên quan
3. Giải thích rõ ràng, dễ hiểu
4. Nếu không có đủ thông tin, hãy nói rõ
5. Trả lời bằng tiếng Việt

{instruction}<|im_end|>
<|im_start|>user
Thông tin pháp luật:
{formatted_context}

Câu hỏi: {question}<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def _format_context(self, context: str) -> str:
        """Format context for better readability"""
        if not context:
            return "Không có thông tin pháp luật liên quan."
        
        # Split by documents if multiple
        sections = context.split('[Tài liệu')
        formatted_sections = []
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            if i == 0 and not section.startswith(' '):
                # First section without document marker
                formatted_sections.append(section.strip())
            else:
                # Add document marker back
                formatted_sections.append(f"[Tài liệu{section}")
        
        return "\n\n".join(formatted_sections)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from prompt
        """
        if not self.model or not self.tokenizer:
            return "Xin lỗi, mô hình ngôn ngữ chưa được tải thành công."
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generation_config = self.generation_config
                
                # Override generation config với kwargs
                for key, value in kwargs.items():
                    if hasattr(generation_config, key):
                        setattr(generation_config, key, value)
                
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # Decode response
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Clean up response
            cleaned_response = self._clean_response(response)
            
            return cleaned_response
            
        except Exception as e:
            error_msg = f"Xảy ra lỗi khi tạo câu trả lời: {str(e)}"
            print(f" Generation error: {e}")
            return error_msg
    
    def _clean_response(self, response: str) -> str:
        """Clean up model response"""
        # Remove extra whitespace
        response = response.strip()
        
        # Remove repetitive patterns
        lines = response.split('\n')
        cleaned_lines = []
        prev_line = ""
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and repetitive lines
            if not line or line == prev_line:
                continue
                
            # Skip lines that are too short or too repetitive
            if len(line) < 10:
                continue
                
            cleaned_lines.append(line)
            prev_line = line
            
            # Limit response length
            if len(cleaned_lines) >= 10:
                break
        
        # Join lines
        final_response = '\n'.join(cleaned_lines)
        
        # Remove common artifacts
        artifacts_to_remove = [
            '<|im_end|>', '<|im_start|>', '</s>', '<s>',
            'Assistant:', 'Human:', 'User:'
        ]
        
        for artifact in artifacts_to_remove:
            final_response = final_response.replace(artifact, '')
        
        # Ensure minimum response length
        if len(final_response.strip()) < 20:
            return "Xin lỗi, tôi không thể tạo ra câu trả lời phù hợp cho câu hỏi này."
        
        return final_response.strip()
    
    def generate_legal_answer(self, question: str, context: str, **kwargs) -> str:
        """
        Specialized method for legal question answering
        """
        # Format prompt specifically for legal questions
        prompt = self.format_legal_prompt(question, context)
        
        # Legal-specific generation parameters
        legal_kwargs = {
            'temperature': 0.2,  # Lower temperature for more factual responses
            'top_p': 0.8,
            'repetition_penalty': 1.2,
            'max_new_tokens': 400
        }
        
        # Override with user kwargs
        legal_kwargs.update(kwargs)
        
        return self.generate(prompt, **legal_kwargs)
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts
        """
        if not self.model or not self.tokenizer:
            return ["Mô hình chưa sẵn sàng"] * len(prompts)
        
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'device_map': self.device_map,
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'vocab_size': self.tokenizer.vocab_size if self.tokenizer else 0,
            'generation_config': self.generation_config.__dict__ if self.generation_config else {}
        }
    
    def test_generation(self) -> str:
        """Test model generation với câu hỏi mẫu"""
        test_question = "Quyền cơ bản của công dân là gì?"
        test_context = "Hiến pháp quy định quyền cơ bản của công dân bao gồm quyền sống, quyền tự do, quyền bình đẳng."
        
        return self.generate_legal_answer(test_question, test_context)

if __name__ == "__main__":
    # Test VinaLlamaLLM
    print(" Testing VinaLlamaLLM...")
    
    llm = VinaLlamaLLM()
    
    # Print model info
    info = llm.get_model_info()
    print(f" Model Info:")
    for key, value in info.items():
        if key != 'generation_config':
            print(f"  {key}: {value}")
    
    # Test generation
    if llm.model and llm.tokenizer:
        print("\n Testing generation...")
        
        test_cases = [
            {
                "question": "Tuổi chịu trách nhiệm hình sự là bao nhiêu?",
                "context": "Điều 15. Tuổi chịu trách nhiệm hình sự\n1. Người từ đủ 16 tuổi trở lên phải chịu trách nhiệm hình sự về mọi tội phạm.\n2. Người từ đủ 14 tuổi đến dưới 16 tuổi phải chịu trách nhiệm hình sự về tội rất nghiêm trọng."
            },
            {
                "question": "Điều kiện kết hôn theo pháp luật Việt Nam?",
                "context": "Điều 8. Điều kiện kết hôn\n1. Nam từ đủ 20 tuổi, nữ từ đủ 18 tuổi.\n2. Việc kết hôn do nam, nữ tự nguyện quyết định."
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f" Question: {test_case['question']}")
            
            response = llm.generate_legal_answer(
                test_case['question'], 
                test_case['context']
            )
            
            print(f" Answer: {response}")
            print("-" * 50)
    
    else:
        print(" Model not loaded, testing basic prompt generation...")
        simple_test = llm.test_generation()
        print(f" Simple test result: {simple_test}")