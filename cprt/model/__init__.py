from typing import Dict, Type

from cprt.model.cprt_model import BaseCPrtModel
from cprt.model.softprompt_cprt import SoftPromptCPrt
from cprt.model.xattention_cprt import CrossAttentionCPrt

CPRT_MODELS: Dict[str, Type[BaseCPrtModel]] = {
    "soft-prompt": SoftPromptCPrt,
    "softprompt": SoftPromptCPrt,
    "x-attention": CrossAttentionCPrt,
    "xattention": CrossAttentionCPrt,
    "cross-attention": CrossAttentionCPrt,
}
