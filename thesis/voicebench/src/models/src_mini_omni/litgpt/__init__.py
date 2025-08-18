

import logging
import re
from litgpt.model import GPT  
from litgpt.config import Config
from litgpt.tokenizer import Tokenizer


pattern = re.compile(".*Profiler function .* will be ignored")
logging.getLogger("torch._dynamo.variables.torch").addFilter(
    lambda record: not pattern.search(record.getMessage())
)


logging.getLogger("torch.distributed.fsdp._optim_utils").disabled = True
logging.getLogger("torch.distributed.fsdp._debug_utils").disabled = True

__all__ = ["GPT", "Config", "Tokenizer"]
