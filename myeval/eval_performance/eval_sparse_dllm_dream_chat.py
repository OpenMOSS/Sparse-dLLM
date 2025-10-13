from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner, SizePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import Sparse_dLLM_DreamCausalLMInstruct

with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1dce88_maxoutlen_256 import gsm8k_datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen_79e572 import mmlu_datasets as mmlu_5_shot_datasets
    from opencompass.configs.datasets.gpqa.gpqa_gen_5shot import gpqa_datasets as gpqa_5_shot_datasets
    from opencompass.configs.datasets.math.math_gen_a58d9d_dream import math_datasets
    from opencompass.configs.datasets.piqa.piqa_gen import piqa_datasets
    from opencompass.configs.datasets.ARC_c.ARC_c_gen import ARC_c_datasets

    ## If evaluating the following benchmarks, please modify the max_seq_len to 4096, max_out_len to 512 and steps to 512
    # from opencompass.configs.datasets.humaneval.humaneval_gen_chat import humaneval_datasets
    # from opencompass.configs.datasets.longbench.longbench import longbench_datasets

datasets = []
datasets += gsm8k_datasets
datasets += mmlu_5_shot_datasets
datasets += gpqa_5_shot_datasets
datasets += math_datasets
datasets += piqa_datasets
datasets += ARC_c_datasets
max_seq_len = 2048
max_out_len = 256

num_gpus = {
    'llada_8b_chat': 1, 'llada_1_5_8b': 1, 
    'dream_v0_7b_base': 1, 'dream_v0_7b_chat': 1, 
}

path_dict = {
    'llada_8b_chat': 'path/to/LLaDA-8B-Instruct', 
    'llada_1_5_8b': 'path/to/LLaDA-1.5', 

    'dream_v0_7b_chat': 'path/to/Dream-v0-7B-Instruct', 
    'dream_v0_7b_base': 'path/to/Dream-v0-7B-Base', 
}

models = [
    ('dream_v0_7b_chat-sparse_dllm', {}, {'steps': 256, }, 3, 0.5), 
]

models = [
    dict(
        type=Sparse_dLLM_DreamCausalLMInstruct, abbr=abbr, path=path_dict[abbr.split('-')[0]],
        kernel_size = kernel_size, keep_ratio = keep_ratio,
        scaling_config=scaling_config, diffusion_config=diffusion_config, seed=2025, model_type=abbr.split('_')[0],
        max_seq_len = max_seq_len, max_out_len=max_out_len, batch_size=1, 
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, scaling_config, diffusion_config, kernel_size, keep_ratio in models
]

work_dir = './outputs/sparse_dllm/'

infer = dict(
    partitioner=dict(type=NaivePartitioner), 
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask), 
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)