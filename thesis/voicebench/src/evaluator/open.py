import re
from .base import Evaluator
import numpy as np


def extract_rating(llm_output):
    """
    Extracts the rating in the format [[number]] from the LLM output.

    Args:
    - llm_output (str): The response from the LLM containing the evaluation and rating.

    Returns:
    - int: The extracted rating, or None if the rating is not found.
    """
    
    pattern = r"\[\[(\d+)\]\]"

    
    match = re.search(pattern, llm_output)

    if match:
        
        return int(match.group(1))
    else:
        
        
        raise NotImplementedError


class OpenEvaluator(Evaluator):
    def evaluate(self, data):
        scores = []
        for item in data:
            for score in item['score']:
                try:
                    score = float(score)
                except Exception as e:
                    score = extract_rating(score)
                scores.append(score)
        return {'gpt': np.mean(scores)}

