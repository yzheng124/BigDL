#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import intel_extension_for_pytorch as ipex
import time
import argparse

from bigdl.llm.transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer

# you could tune the prompt based on your own model,
FLAN_T5_PROMPT_FORMAT = "<|User|>:{prompt}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for flan-t5 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="google/flan-t5-xxl",
                        help='The huggingface repo id for the flan-t5 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Translate to German: My name is Arthur",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format.
    # "wo" module is not converted due to some issues of T5 model 
    # (https://github.com/huggingface/transformers/issues/20287),
    # "lm_head" module is not converted to generate outputs with better quality
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path,
                                                  load_in_4bit=True,
                                                  optimize_model=False,
                                                  trust_remote_code=True,
                                                  use_cache=True,
                                                  modules_to_not_convert=["wo", "lm_head"])
    model = model.to('xpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)

    # Generate predicted tokens
    with torch.inference_mode():
        prompt = FLAN_T5_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
        # ipex model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)

        # start inference
        st = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with BigDL-LLM INT4 optimizations
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        end = time.time()
        output = output.cpu()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        output_str = output_str.split("<eoa>")[0]
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
