<div align="center">
<h1>Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction</h1>
Yuerong Song<sup>1,2</sup>, Xiaoran Liu<sup>1,2</sup>, Ruixiao Li<sup>1,2</sup>, Zhigeng Liu<sup>1,2</sup>, Zengfeng Huang<sup>1,2</sup>, Qipeng Guo<sup>2,3</sup>, Ziwei He<sup>2,†</sup>, Xipeng Qiu<sup>1,2,†</sup>

<sup>1</sup> Fudan Univerisity, <sup>2</sup>Shanghai Innovation Institute, <sup>3</sup>Shanghai AI Laboratory

[<a href="https://arxiv.org/abs/2508.02558">📝 Paper</a>] | [<a href="https://huggingface.co/papers/2508.02558">🤗 HF</a>] | [<a href="https://github.com/OpenMOSS/Sparse-dLLM">🚀 Code</a>]
</div>

## Introduction

In this work, we present ***Sparse-dLLM***, a training-free framework that tackles the core bottleneck of diffusion large language models (dLLMs): quadratic-time computational complexity. While prior caching methods accelerate dLLMs by reusing full-layer KV states, they incur substantial memory overhead that constrains long-context applications. Our analysis reveals a distinctive property of dLLM attention—persistent cross-layer sparsity with stable token saliency over decoding steps—suggesting that many cached entries are low-relevance and can be safely discarded.

Building on these observations, we integrate ***dynamic cache eviction*** with ***sparse attention*** via a ***delayed bidirectional sparse caching*** strategy. Sparse-dLLM retains pivotal tokens and dynamically evicts unimportant prefix and suffix entries using an attention-guided strategy, while delaying cache updates by one step to stabilize selection. This plug-and-play design prunes redundant cache states without retraining, accelerates dLLM decoding, and preserves a near-identical peak memory footprint compared with vanilla dLLMs, enabling practical long-context inference.

On LLaDA and Dream series, Sparse-dLLM delivers up to ***10×*** higher throughput than vanilla dLLMs, maintaining comparable performance and outperforming recent dLLM caching methods in efficiency–effectiveness trade-off. Our study thus establishes ***the first*** method that combines dynamic cache eviction with sparse attention for dLLMs, and provides empirical evidence and analysis that chart a path toward scalable, fast, and memory-efficient dLLM decoding.

<p align="center">
<img src="./img/intro.png" width="500"/>
<p>

## Installation

### Prepare Your OpenCompass

We run our downstream evaluation based on [OpenCompass](https://github.com/open-compass/opencompass).

```bash
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

The necessary Python packages we use and their corresponding versions.

```
opencompass==0.4.2
torch==2.6.0
transformers==4.46.3
```

### Prepare Your Model and Benchmarks

Copy the directory `Sparse-dLLM/opencompass/`to your OpenCompass directory and add the following lines to the end of `opencompass/models/__init__.py`.

```python
from .sparse_dllm.llada_wrapper import Sparse_dLLM_LLaDACausalLM
from .sparse_dllm.dream_wrapper import Sparse_dLLM_DreamCausalLM
from .sparse_dllm.dream_wrapper_instruct import Sparse_dLLM_DreamCausalLMInstruct
```

## Evaluation

Copy the directory `Sparse-dLLM/myeval/` to your OpenCompass directory and then you can try the following evaluations.

### Performance Evaluation

Go to your OpenCompass directory and run performance evaluation:

```
opencompass run.py myeval/eval_performance/eval_sparse_dllm_***.py
```

Replace `***` with the corresponding model name (e.g., `dream_base`, `dream_chat`, `llada_chat`, `llada_1.5`).

### Speed Evaluation

Go to your OpenCompass directory  and run the corresponding script. For example:

```
bash myeval/eval_speed/eval_speed_dream_example.sh
bash myeval/eval_speed/eval_speed_llada_example.sh
```

Or run the Python code directly (with parameters):

```
python myeval/eval_speed/dream_sparse_dllm.py --model_path <MODEL_PATH> --model_type <MODEL_TYPE> --data_path <DATA_PATH> --data_type <DATA_TYPE> --output_dir <OUTPUT_DIR> --kernel_size 3 --keep_ratio 0.5 --block_length 32 --apply_chat_template True
```

See codes for more details.

## Results

<p align="center">
<img src="./img/LLaDA.png" width="750"/>
<p>

<p align="center">
<img src="./img/Dream.png" width="750"/>
<p>

<p align="center">
<img src="./img/long-context.png" width="500"/>
<p>

## Citation

```
@article{song2025sparse,
  title={Sparse-dllm: Accelerating diffusion llms with dynamic cache eviction},
  author={Song, Yuerong and Liu, Xiaoran and Li, Ruixiao and Liu, Zhigeng and Huang, Zengfeng and Guo, Qipeng and He, Ziwei and Qiu, Xipeng},
  journal={arXiv preprint arXiv:2508.02558},
  year={2025}
}
```
