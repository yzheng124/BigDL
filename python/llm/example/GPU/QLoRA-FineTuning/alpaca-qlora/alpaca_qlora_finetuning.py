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
# Some parts of this file is adapted from
# https://github.com/tloen/alpaca-lora/blob/main/finetune.py
#
# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
import accelerate

from transformers import LlamaTokenizer
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from utils.prompter import Prompter

import intel_extension_for_pytorch as ipex
from bigdl.llm.transformers import AutoModelForCausalLM
# import them from bigdl.llm.transformers.qlora to get a BigDL-LLM compatible Peft model
from bigdl.llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training,\
    LoraConfig
from bigdl.llm.utils.common import invalidInputError


def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return int(default)

def _get_trainer_cls(training_mode):
    if training_mode == "relora":
        from bigdl.llm.transformers.relora import ReLoRATrainer
        return ReLoRATrainer
    return transformers.Trainer
 
local_rank = get_int_from_env(["LOCAL_RANK","MPI_LOCALRANKID"], "0")
world_size = get_int_from_env(["WORLD_SIZE","PMI_SIZE"], "1")
port = get_int_from_env(["MASTER_PORT"], 29500)
os.environ["LOCAL_RANK"] = str(local_rank)
os.environ["WORLD_SIZE"] = str(world_size)
os.environ["RANK"] = str(local_rank)
os.environ["MASTER_PORT"] = str(port)

def train(
    # model/data params
    base_model: str = "meta-llama/Llama-2-7b-hf",  # the only required argument, default to be "meta-llama/Llama-2-7b-hf"
    saved_low_bit_model: str = None,  # optional, the path to the saved model with bigdl-llm low-bit optimization
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./bigdl-qlora-alpaca",
    # training hyperparams
    bf16: bool = True,  # default to bf16
    batch_size: int = 128,
    micro_batch_size: int = 2,  # default to be 2, limited by GPU memory
    num_epochs: int = 3,
    learning_rate: float = 3e-5,  # default to be 3e-5 to avoid divergence
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj"
    ],  # according to the QLoRA paper (https://arxiv.org/pdf/2305.14314.pdf), it's suggested to fine tune all linear layers
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    gradient_checkpointing: bool = False,
    deepspeed: str = None,
    training_mode: str = "qlora",
    # relora params, relora_steps should > 0 if the training mode is `relora`,
    # Implements the ReLoRA training procedure from https://arxiv.org/abs/2307.05695,
    # minus the initial full fine-tune.
    relora_steps: int = 300,         # Number of steps per ReLoRA restart
    relora_warmup_steps: int = 10,   # Number of per-restart warmup steps
    relora_cpu_offload: bool = True, # True to perform lora weight merges on cpu during restarts, for modest gpu memory savings
):
    invalidInputError(training_mode in ["qlora", "qalora", "lora", "relora"],
                      "Only qlora / qalora / lora / relora are supported for training_mode now.")
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
            f"training_mode: {training_mode}\n"
            f"relora_steps: {relora_steps}\n"
            f"relora_warmup_steps: {relora_warmup_steps}\n"
            f"relora_cpu_offload: {relora_cpu_offload}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    if training_mode == "relora":
        assert(relora_steps > 0), "The relora_steps should > 0 if the training_mode is relora."

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if saved_low_bit_model is not None:
        # Load the low bit optimized model if provide the saved path
        model = AutoModelForCausalLM.load_low_bit(
            saved_low_bit_model,
            optimize_model=False,
            torch_dtype=torch.bfloat16,
            modules_to_not_convert=["lm_head"],
        )
    else:
        # According to the QLoRA paper, using "nf4" could yield better model quality than "int4"
        # Default 4-bit format for qa-lora is sym_int4
        if training_mode == "qalora":
            low_bit_format = "sym_int4"
        elif training_mode == "lora":
            low_bit_format = "bf16"
        else:
            low_bit_format = "nf4"
        # Load the base model from a directory or the HF Hub to 4-bit format
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_low_bit=low_bit_format,
            optimize_model=False,
            torch_dtype=torch.bfloat16,
            # device_map=device_map,
            modules_to_not_convert=["lm_head"],
        )
    print(f"Model loaded on rank {os.environ.get('LOCAL_RANK')}")
    model = model.to(f'xpu:{os.environ.get("LOCAL_RANK", 0)}')
    print(f"Model moved to rank {os.environ.get('LOCAL_RANK')}")

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    print(f"Tokenizer loaded on rank {os.environ.get('LOCAL_RANK')}")

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    print(model)

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    # Prepare a BigDL-LLM compatible Peft model
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        training_mode=training_mode,
    )
    print(f"Lora Config: {config}")
    model = get_peft_model(model, config)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    # Unused
    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    trainer_cls = _get_trainer_cls(training_mode=training_mode)
    extra_args = {}
    if training_mode == "relora":
        extra_args["base_model"] = base_model
        extra_args["relora_steps"] = relora_steps
        extra_args["relora_warmup_steps"] = relora_warmup_steps
        extra_args["relora_cpu_offload"] = relora_cpu_offload
        extra_args["resume_from_checkpoint"] = resume_from_checkpoint

    trainer = trainer_cls(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        **extra_args,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # warmup_ratio=0.03,
            # warmup_steps=100,
            max_grad_norm=0.3,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type="constant" if training_mode=="qalora" else "cosine",
            bf16=True,  # ensure training more stable
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=100 if val_set_size > 0 else None,
            save_steps=100,
            output_dir=output_dir,
            save_total_limit=100 if training_mode != "relora" else 4, # relora will save the whole model, here we use 4 to save the disk space.
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
            gradient_checkpointing=gradient_checkpointing,
            ddp_backend="ccl",
            deepspeed=deepspeed,
            save_safetensors=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
