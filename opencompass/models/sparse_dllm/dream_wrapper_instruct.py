from typing import Dict, List, Optional, Union

import torch
from mmengine.device import is_npu_available

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

from ..huggingface_above_v4_33 import HuggingFaceBaseModel

from transformers import  AutoConfig
from transformers.generation.utils import GenerationConfig

from transformers import AutoModelForCausalLM
from .llada_generate import generate

import os
import random

import numpy as np


def _get_stopping_criteria(stop_words, tokenizer, batch_size):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class MultiTokenEOSCriteria(StoppingCriteria):
        """Criteria to stop on the specified multi-token sequence."""

        def __init__(self, stop_words: List[str], tokenizer, batch_size: int):
            self.done_tracker = [False] * batch_size
            self.stop_words, self.max_sequence_id_len = [], 0
            for s in stop_words:
                self.stop_words.append(s)
                sequence_ids = tokenizer.encode(s, add_special_tokens=False)
                self.max_sequence_id_len = max(self.max_sequence_id_len, len(sequence_ids))
            self.tokenizer = tokenizer

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            # compare the last len(stop) tokens
            lookback_ids_batch = input_ids[:, -self.max_sequence_id_len:]
            lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
            for i, done in enumerate(self.done_tracker):
                if done:
                    continue
                self.done_tracker[i] = any(s in lookback_tokens_batch[i] for s in self.stop_words)
            return False not in self.done_tracker

    c = MultiTokenEOSCriteria(stop_words, tokenizer, batch_size)
    return StoppingCriteriaList([c])


def _get_possible_max_seq_len(max_seq_len, path):
    if max_seq_len is not None:
        return max_seq_len

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    possible_keys = [
        'max_position_embeddings',
        'seq_length',
        'model_max_length', 
        'max_sequence_length', 
    ]
    for k in possible_keys:
        if hasattr(config, k):
            return getattr(config, k)
    raise ValueError('max_seq_len is not provided and cannot be inferred from the model config.')


def  _convert_base_messages(inputs):
    outputs = []
    for _input in inputs:
        if isinstance(_input, str):
            outputs.append(_input)
        else:
            messages = []
            for item in _input:
                messages.append(item['prompt'])
            outputs.append(''.join(messages))
    return outputs


def _set_model_kwargs_torch_dtype(model_kwargs):
    import torch
    if 'torch_dtype' not in model_kwargs:
        torch_dtype = torch.float16
    else:
        torch_dtype = {
            'torch.float16': torch.float16,
            'torch.bfloat16': torch.bfloat16,
            'torch.float': torch.float,
            'auto': 'auto',
            'None': None,
        }.get(model_kwargs['torch_dtype'])
    if torch_dtype is not None:
        model_kwargs['torch_dtype'] = torch_dtype
    return model_kwargs


@MODELS.register_module()
class Sparse_dLLM_DreamCausalLMInstruct(HuggingFaceBaseModel):

    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 peft_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 generation_kwargs: dict = dict(),
                 max_seq_len: Optional[int] = None,
                 pad_token_id: Optional[int] = None,
                 stop_words: Optional[str] = [],
                 drop_middle: bool = False,

                 scaling_config: dict = None, 
                 diffusion_config: dict = None, 
                 model_type: str = None, 
                 seed: int = None, 

                 ## add parameters
                 kernel_size: Optional[int] = 3,
                 keep_ratio: float = 0.5,
                 block_length: int = 32,

                 **other_kwargs):

        if seed is not None:
            os.environ['PYTHONHASHSEED'] = str(seed) 
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be False
            torch.backends.cudnn.deterministic = True # choose a deterministic algorithm 

        self.logger = get_logger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = LMTemplateParser()
        self.max_seq_len = max_seq_len  # _get_possible_max_seq_len(max_seq_len, path)
        self.drop_middle = drop_middle
        self._load_tokenizer(tokenizer_path or path, tokenizer_kwargs, pad_token_id)

        self.scaling_config = scaling_config
        self.block_len = block_length
        self.keep_ratio = keep_ratio
        self.kernel_size = kernel_size

        if model_type == 'dream':
            self.diffusion_config = {'steps': 256, 'alg': 'entropy', 'return_dict_in_generate': True, 'temperature': 0.2, 'top_p': 0.95}
        else:
            self.diffusion_config = {'steps': 128, 'block_length': 32, 'temperature': 0., 'cfg_scale': 0., 'remasking': 'low_confidence', }
        if diffusion_config is not None:
            self.diffusion_config.update(diffusion_config)

        print(self.diffusion_config, flush=True)

        self.model_type = model_type

        if not tokenizer_only:
            self._load_model(path=path, kwargs=model_kwargs, peft_path=peft_path, peft_kwargs=peft_kwargs)
        self.generation_kwargs = generation_kwargs
        self.stop_words = stop_words

        for k, v in other_kwargs.items():
            if v is not None:
                self.logger.warning(f'Unused argument {k}={v}')
    

    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        from transformers import AutoModel, AutoModelForCausalLM

        DEFAULT_MODEL_KWARGS = dict(device_map='auto', trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs.update(kwargs)
        model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs)
        self.logger.debug(f'using model_kwargs: {model_kwargs}')
        if is_npu_available():
            model_kwargs['device_map'] = 'npu'

        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        # config.flash_attention = True
        config.block_len = self.block_len
        config.kernel_size = self.kernel_size
        config.keep_ratio = self.keep_ratio

        if self.scaling_config is not None:
            scaling_factor = self.scaling_config['scaling_factor'] if 'scaling_factor' in self.scaling_config else 1
            config.rope_theta = config.rope_theta * scaling_factor
            print(f'{config.rope_theta=}', flush=True)

        if self.model_type == 'llama':
            self.model = AutoModelForCausalLM.from_pretrained(path, config=config, device_map='auto', 
                                                        torch_dtype=torch.bfloat16, trust_remote_code=True)
        elif self.model_type == 'dream':
            from .dream.modeling_dream import DreamModel
            self.model = DreamModel.from_pretrained(path, config=config, device_map='auto', 
                                                   torch_dtype=torch.bfloat16, trust_remote_code=True)
        else:
            from .modeling_llada import LLaDAModelLM
            self.model = LLaDAModelLM.from_pretrained(path, config=config, device_map='auto', 
                                                        torch_dtype=torch.bfloat16, trust_remote_code=True)

        if peft_path is not None:
            from peft import PeftModel
            peft_kwargs['is_trainable'] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False


    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:
        messages = _convert_base_messages(inputs)
        batch_size = len(messages)

        tokenize_kwargs = dict(
            return_tensors='pt',
            padding=True,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_seq_len
        )
        if self.drop_middle:
            raise NotImplementedError("drop_middle mode is disabled")
        else:
            m = [[{"role": "user", "content": p}] for p in messages]
            formatted_text = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize = False)
            # print(formatted_text)
            tokens = self.tokenizer(formatted_text, **tokenize_kwargs)

        tokens = {k: v.to(self.model.device) for k, v in tokens.items()}

        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update(kwargs)
        stopping_criteria = list(set(stopping_criteria + self.stop_words))
        if stopping_criteria:
            generation_kwargs['stopping_criteria'] = _get_stopping_criteria(stopping_criteria, self.tokenizer, batch_size)
        if max_out_len is not None:
            generation_kwargs['max_new_tokens'] = max_out_len
        if min_out_len is not None:
            generation_kwargs['min_new_tokens'] = min_out_len
        generation_kwargs['pad_token_id'] = self.tokenizer.pad_token_id

        # step-2: conduct model forward to generate output
        print(tokens['input_ids'].shape, flush=True)

        if self.model_type == 'llama':
            outputs = self.model.generate(**tokens, **generation_kwargs)
        elif self.model_type == 'dream':
            diffusion_config = self.diffusion_config
            if diffusion_config['steps'] > max_out_len:
                diffusion_config['steps'] = max_out_len
                print(diffusion_config, flush=True)
            
            outputs = self.model.diffusion_generate(tokens['input_ids'], attention_mask = tokens['attention_mask'],
                                                    max_new_tokens=max_out_len, **diffusion_config)
        else:
            diffusion_config = self.diffusion_config
            if max_out_len % diffusion_config['block_length'] != 0:
                max_out_len = int((max_out_len // diffusion_config['block_length'] + 1) * diffusion_config['block_length'])
            # print(max_out_len)

            if diffusion_config['steps'] > max_out_len:
                diffusion_config['steps'] = max_out_len
                print(diffusion_config, flush=True)

            outputs = generate(self.model, tokens['input_ids'], 
                               gen_length=max_out_len, **diffusion_config,
                               )

        generations = [
            self.tokenizer.decode(g[len(p) :].tolist())
            for p, g in zip(tokens['input_ids'], outputs.sequences)
        ]

        decodeds = [
            gen.split(self.tokenizer.eos_token)[0]
            for gen in generations
        ]
        # print(decodeds)
        # for stop in stopping_criteria:
        #     decodeds = [token.split(stop)[0] for token in decodeds]

        return decodeds

    def get_token_len(self, prompt: str, add_special_tokens: bool=True) -> int:
        m = _convert_base_messages([prompt])[0]
        t = self.tokenizer(m, add_special_tokens=add_special_tokens)
        return len(t['input_ids'])
