from typing import Optional, Union

from transformers import PreTrainedModel, PreTrainedTokenizer

# lm stuff
from mg_behavior_learning.lm.base import BaseMarioLM
from mg_behavior_learning.lm.bert import MarioBert
from mg_behavior_learning.lm.gpt import MarioGPT
from mg_behavior_learning.prompter import Prompter

from mg_behavior_learning.lm.behavior_head import BehaviorHeadCausalLM

def MarioLM(
    lm: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    context_len: int = 700,
    prompter: Optional[Prompter] = None,
    mask_proportion: float = 0.15,
    mask_model: bool = False,
    lm_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    **kwargs
) -> Union[MarioGPT, MarioBert]:
    if not mask_model:
        return MarioGPT(
            lm=lm,
            tokenizer=tokenizer,
            context_len=context_len,
            prompter=prompter,
            lm_path=lm_path,
            tokenizer_path=tokenizer_path,
            **kwargs
        )
    return MarioBert(
        lm=lm,
        tokenizer=tokenizer,
        context_len=context_len,
        mask_proportion=mask_proportion,
        lm_path=lm_path,
        tokenizer_path=tokenizer_path,
        **kwargs
    )


__all__ = ["BaseMarioLM", "MarioGPT", "MarioBert", "MarioLM"]
