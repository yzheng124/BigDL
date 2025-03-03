# Baichuan2
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Baichuan2 models. For illustration purposes, we utilize the [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat) as a reference Baichuan model.

## 0. Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Baichuan model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install bigdl-llm[all] # install bigdl-llm with 'all' option
pip install transformers_stream_generator  # additional package required for Baichuan-13B-Chat to conduct generation
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Baichuan2 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'baichuan-inc/Baichuan2-13B-Chat'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Baichuan model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```powershell
python ./generate.py 
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set BigDL-LLM env variables
source bigdl-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 2.3 Sample Output
#### [baichuan-inc/Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<human>AI是什么？ <bot>
-------------------- Output --------------------
<human>AI是什么？ <bot>人工智能（AI）是指由计算机系统或其他数字设备模拟、扩展和增强人类智能的科学和技术。它涉及到多个领域，如机器学习、计算机视觉、
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<human>解释一下“温故而知新” <bot>
-------------------- Output --------------------
<human>解释一下“温故而知新” <bot>这句话出自《论语·为政》篇，意思是通过回顾过去的事情来获取新的理解和认识。简单来说就是：温习学过的知识，可以从中
```
