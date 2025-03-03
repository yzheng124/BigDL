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
# https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/models/llama/modeling_llama.py
# which is licensed under Apache License 2.0:
#
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import torch
import importlib
import torch.nn as nn
from typing import Optional, Tuple, Union, List
import math
import torch.nn.functional as F
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from bigdl.llm.transformers.models.utils import is_enough_kv_cache_room_4_31, apply_rotary_pos_emb
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb_no_cache_xpu
from transformers.modeling_outputs import BaseModelOutputWithPast
from bigdl.llm.transformers.low_bit_linear import SYM_INT4
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from bigdl.llm.utils.common import invalidInputError


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states
    go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads,
                                                           n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

KV_CACHE_ALLOC_BLOCK_LENGTH = 256


_ipex_version = None


def get_ipex_version():

    global _ipex_version
    if _ipex_version is not None:
        return _ipex_version

    import intel_extension_for_pytorch as ipex
    _ipex_version = ipex.__version__
    return _ipex_version


def llama_rms_norm_forward(self, hidden_states):
    if hidden_states.device.type == "xpu" and not (self.training and hidden_states.requires_grad):
        import linear_q4_0
        result = linear_q4_0.fused_rms_norm(hidden_states,
                                            [self.weight.size(0)],
                                            self.weight,
                                            None,
                                            self.variance_epsilon)
        # if nelement == 0, means fused norm failed, go back to python implement.
        if result.nelement != 0:
            return result
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


def llama_mlp_forward(
    self,
    x: torch.Tensor,
) -> torch.Tensor:
    x_2d = x.view(-1, x.shape[-1])
    if x_2d.shape[0] == 1 and x.device.type == 'xpu' \
            and self.gate_proj.qtype == ggml_tensor_qtype["sym_int4"] \
            and not (self.training and x.requires_grad):
        import linear_q4_0
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        return self.down_proj(linear_q4_0.mlp_forward_q4_0_xpu(
            x_2d, self.gate_proj.weight.data, self.up_proj.weight.data,
            x_2d.shape[0], x_2d.shape[1], self.gate_proj.out_len,
        ))
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def should_use_fuse_rope(self, query_states, position_ids):
    use_fuse_rope = query_states.device.type == "xpu"
    use_fuse_rope = use_fuse_rope and not (self.training and query_states.requires_grad)
    use_fuse_rope = use_fuse_rope and self.config.rope_scaling is None
    use_fuse_rope = use_fuse_rope and position_ids is not None
    return use_fuse_rope


def llama_attention_forward_4_31(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype
    if not self.training and not hidden_states.requires_grad:
        fsdp_flag = use_flash_attention(hidden_states)
    else:
        fsdp_flag = False
    if fsdp_flag:
        attention_dtype = torch.float16  # use fp16 for flash attention
    else:
        attention_dtype = original_dtype

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_31(past_key_value)
    is_q4_0 = self.q_proj.qtype == SYM_INT4
    no_tp = not self.config.pretraining_tp > 1
    decoding_fast_path = (no_tp and is_q4_0 and use_fuse_rope and
                          enough_kv_room and bsz * q_len == 1)

    # single batch decoding fast path
    # forward_qkv takes will perform QKV projection, rotary position embedding
    # and save the key/value states to cache, then return query states and the
    # extended key/value cache
    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        kv_seq_len = past_key_value[0].shape[-2]
        cache_k = past_key_value[0]
        cache_v = past_key_value[1]
        import linear_q4_0
        query_states, key_states, value_states = linear_q4_0.forward_qkv(hidden_states,
                                                                         self.q_proj.weight,
                                                                         self.k_proj.weight,
                                                                         self.v_proj.weight,
                                                                         position_ids,
                                                                         cache_k, cache_v,
                                                                         self.q_proj.weight.qtype,
                                                                         kv_seq_len,
                                                                         self.head_dim)
        kv_seq_len += 1

    else:
        if self.config.pretraining_tp > 1:
            key_value_slicing = ((self.num_key_value_heads * self.head_dim) //
                                 self.config.pretraining_tp)
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim)
                                                    // self.config.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i])
                            for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i])
                          for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i])
                            for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len,
                                         self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len,
                                     self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len,
                                         self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        if use_fuse_rope:
            query_states, key_states = apply_rotary_pos_emb_no_cache_xpu(query_states,
                                                                         key_states,
                                                                         position_ids,
                                                                         "llama")
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                            cos, sin, position_ids, "llama")

        if past_key_value is not None:
            # reuse k, v, self_attention
            cache_k = past_key_value[0]
            cache_v = past_key_value[1]
            if not enough_kv_room:
                # allocate new
                new_cache_k, new_cache_v = extend_kv_cache(bsz,
                                                           self.num_key_value_heads,  # Support GQA
                                                           self.head_dim,
                                                           cache_k.size(2),
                                                           kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                           dtype=cache_k.dtype,
                                                           device=device)
                new_cache_k[:] = cache_k
                new_cache_v[:] = cache_v
                cache_k = new_cache_k
                cache_v = new_cache_v

            key_states, value_states = append_kv_cache(cache_k, cache_v, key_states, value_states)

        elif use_cache:
            max_cache_length = kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH
            new_key_states, new_value_states = init_kv_cache(bsz,
                                                             self.num_key_value_heads,
                                                             self.head_dim,
                                                             kv_seq_len,
                                                             max_cache_length,
                                                             dtype=key_states.dtype,
                                                             device=device)
            new_key_states[:] = key_states
            new_value_states[:] = value_states
            key_states = new_key_states
            value_states = new_value_states

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups).to(device,
                                                                     dtype=attention_dtype)
    value_states = repeat_kv(value_states, self.num_key_value_groups).to(device,
                                                                         dtype=attention_dtype)

    if fsdp_flag:
        attn_output = F.scaled_dot_product_attention(query_states.to(dtype=attention_dtype),
                                                     key_states,
                                                     value_states,
                                                     is_causal=True)
        attn_weights = None
    elif use_esimd_sdp(q_len, self.head_dim, query_states):
        import linear_fp16_esimd
        attn_output = linear_fp16_esimd.sdp_forward(query_states,
                                                    key_states.contiguous(),
                                                    value_states.contiguous())
        attn_output = attn_output.view(query_states.shape)
        attn_weights = None
    else:
        # otherwise, use native attention
        attn_output, attn_weights = native_sdp(query_states, key_states, value_states,
                                               attention_mask,
                                               bsz, q_len, kv_seq_len,
                                               self.head_dim, self.num_heads)

    attn_output_size = (bsz, self.num_heads, q_len, self.head_dim)
    if attn_output.size() != attn_output_size:
        invalidInputError(False,
                          f"`attn_output` should be of size {attn_output_size},"
                          f" but is {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp,
                                                 dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i])
                           for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output.to(original_dtype), attn_weights, past_key_value


def llama_attention_selective_batching_forward_4_31(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # Minimize this value to reduce memory allocation.
    KV_CACHE_ALLOC_BLOCK_LENGTH = 64
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device
    # for flash attention
    original_dtype = hidden_states.dtype
    # TODO: consider this later - flash attention
    # if not self.training and not hidden_states.requires_grad:
    #     fsdp_flag = use_flash_attention(hidden_states)
    # else:
    #     fsdp_flag = False
    # if fsdp_flag and q_len > 1:
    #     attention_dtype = torch.float16  # use fp16 for flash attention
    # else:
    #     attention_dtype = original_dtype

    attention_dtype = original_dtype

    # TODO: decoding fast path
    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = past_key_value is not None and is_enough_kv_cache_room_4_31(past_key_value[0])
    is_q4_0 = self.q_proj.qtype == SYM_INT4
    no_tp = not self.config.pretraining_tp > 1
    decoding_fast_path = (no_tp and is_q4_0 and use_fuse_rope and
                          bsz * q_len == 1)

    updated_past_key_values = []
    # single batch decoding fast path
    # forward_qkv takes will perform QKV projection, rotary position embedding
    # and save the key/value states to cache, then return query states and the
    # extended key/value cache
    if decoding_fast_path:
        past_k = past_key_value[0][0]
        past_v = past_key_value[0][1]
        kv_seq_len = past_k.shape[-2]
        if not enough_kv_room:
            new_cache_k, new_cache_v = extend_kv_cache(1,
                                                       self.num_key_value_heads,  # Support GQA
                                                       self.head_dim,
                                                       kv_seq_len,
                                                       kv_seq_len +
                                                       KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                       dtype=past_k.dtype,
                                                       device=device)
            new_cache_k[:] = past_k
            new_cache_v[:] = past_v
            past_k = new_cache_k
            past_v = new_cache_v
        hidden_states = hidden_states.view(1, -1)
        import linear_q4_0
        query_states, key_states, value_states = linear_q4_0.forward_qkv(hidden_states,
                                                                         self.q_proj.weight,
                                                                         self.k_proj.weight,
                                                                         self.v_proj.weight,
                                                                         position_ids,
                                                                         past_k, past_v,
                                                                         self.q_proj.weight.qtype,
                                                                         kv_seq_len,
                                                                         self.head_dim)
        kv_seq_len += 1
    else:
        if self.config.pretraining_tp > 1:
            invalidInputError(False, f"vLLM: config.pretraining_tp > 1 not supported yet")
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len,
                                         self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len,
                                     self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len,
                                         self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += max(kv_pair[0].shape[-2] for kv_pair in past_key_value)

        if use_fuse_rope:
            query_states, key_states = apply_rotary_pos_emb_no_cache_xpu(query_states,
                                                                         key_states,
                                                                         position_ids,
                                                                         "llama")
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                            cos, sin, position_ids, "llama")

        if past_key_value is not None:
            batched_attention_output = []
            # print(f"type of attention_mask is {type(attention_mask)}")
            for batch in range(bsz):
                enough_kv_room = is_enough_kv_cache_room_4_31(past_key_value[batch])
                past_k, past_v = past_key_value[batch]
                current_kv_len = past_k.shape[-2] + 1
                if not enough_kv_room:
                    # allocate new
                    new_cache_k, new_cache_v = extend_kv_cache(1,
                                                               self.num_key_value_heads,
                                                               self.head_dim,
                                                               past_k.size(2),
                                                               current_kv_len +
                                                               KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                               dtype=past_k.dtype,
                                                               device=device)
                    new_cache_k[:] = past_k
                    new_cache_v[:] = past_v
                    past_k = new_cache_k
                    past_v = new_cache_v

                current_key_states = key_states[batch: batch + 1, :, :, :]
                current_value_states = value_states[batch: batch + 1, :, :, :]
                current_key_states, current_value_states = append_kv_cache(past_k,
                                                                           past_v,
                                                                           current_key_states,
                                                                           current_value_states)
                updated_past_key_values.append((current_key_states, current_value_states))

                current_key_states = repeat_kv(current_key_states, self.num_key_value_groups)
                current_value_states = repeat_kv(current_value_states, self.num_key_value_groups)

                current_query_states = query_states[batch: batch + 1, :, :, :]
                attn_output, attn_weights = native_sdp(current_query_states,
                                                       current_key_states,
                                                       current_value_states,
                                                       attention_mask[batch],
                                                       1,
                                                       1,
                                                       current_kv_len,
                                                       self.head_dim,
                                                       self.num_heads)
                if attn_output.size() != (1, self.num_heads, 1, self.head_dim):
                    invalidInputError(False,
                                      f"`attn_output` should be of size "
                                      f"{(1, self.num_heads, 1, self.head_dim)}, but is"
                                      f" {attn_output.size()}")
                batched_attention_output.append(attn_output)
            # For loop ends
            # TODO: handle attention_weights later
            attn_output = torch.concat(batched_attention_output, dim=0)
            batched_attention_output.clear()
            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                invalidInputError(False,
                                  f"`attn_output` should be of size "
                                  f"{(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                                  f" {attn_output.size()}")
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, updated_past_key_values

    # Assume always use_cache
    # prefill or decoding fast path
    for batch in range(bsz):
        updated_past_key_values.append((key_states[batch: batch + 1, :, :, :],
                                        value_states[batch: batch+1, :, :, :]))

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups).to(device,
                                                                     dtype=attention_dtype)
    value_states = repeat_kv(value_states, self.num_key_value_groups).to(device,
                                                                         dtype=attention_dtype)
    # Can also happens for decoding fast path
    if isinstance(attention_mask, list):
        # For decoding fast path
        attention_mask = attention_mask[0]
    attn_output, attn_weights = native_sdp(query_states,
                                           key_states,
                                           value_states,
                                           attention_mask,
                                           bsz,
                                           q_len,
                                           kv_seq_len,
                                           self.head_dim,
                                           self.num_heads)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        invalidInputError(False,
                          f"`attn_output` should be of size "
                          f"{(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                          f" {attn_output.size()}")
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)
    return attn_output.to(original_dtype), attn_weights, updated_past_key_values


def use_flash_attention(query):
    bsz, q_len, _ = query.size()
    # check whether ipex flash attention can be used
    if bsz > 1:
        # only use flash attention for batch_size = 1 now
        # as flash attention doesn't support attn_mask in ipex 2.1,
        # so it will cause output error for padded batch input
        return False
    if q_len == 1:
        # now only use flash attention for first token
        # as it seems have no performance benifit for rest token now
        return False
    if query.device.type != "xpu":
        # ipex flash attention only support for xpu
        return False
    ipex_version = get_ipex_version()
    if ipex_version <= "2.0.110+xpu":
        # ipex flash attention is supported from ipex 2.1
        return False
    if not torch.xpu.has_xetla():
        # ipex flash attention is only supported for xetla
        # may update this later
        return False
    if query.dtype not in [torch.float32, torch.float16]:
        # only use flash attention for fp32/fp16 input
        return False
    return True


def use_esimd_sdp(q_len, head_dim, query_states):
    if head_dim != 128:
        # esimd_sdp only support head_dim = 128 now
        return False
    elif q_len != 1:
        # esimd_sdp only support rest token now
        return False
    elif query_states.device.type != "xpu":
        # esimd_sdp only support GPU now
        return False
    elif query_states.dtype != torch.float16:
        # esimd_sdp only has optimization for FP16 now
        return False
    else:
        device_name = torch.xpu.get_device_name(query_states.device.index)
        if device_name.startswith("Intel(R) Arc(TM) A") or \
                device_name.startswith("Intel(R) Data Center GPU Flex"):
            import linear_fp16_esimd
            if hasattr(linear_fp16_esimd, "sdp_forward"):
                return True
            else:
                return False
        else:
            return False


def native_sdp(query, key, value, attention_mask,
               bsz, q_len, kv_seq_len, head_dim, num_heads):
    attn_weights = torch.matmul(query,
                                key.transpose(2, 3)) / math.sqrt(head_dim)

    attn_weights_size = (bsz, num_heads, q_len, kv_seq_len)
    if attn_weights.size() != attn_weights_size:
        invalidInputError(False,
                          f"Attention weights should be of size {attn_weights_size}, "
                          f"but is {attn_weights.size()}")

    if attention_mask is not None:
        attn_mask_size = (bsz, 1, q_len, kv_seq_len)
        if attention_mask.size() != attn_mask_size:
            invalidInputError(False,
                              f"Attention mask should be of size {attn_mask_size}, "
                              f"but is {attention_mask.size()}")
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1,
                                         dtype=torch.float32).to(value.dtype)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


def llama_model_selective_batching_forward_4_31(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    if output_attentions is not None:
        output_attentions = output_attentions
    else:
        output_attentions = self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        invalidInputError(False,
                          "You cannot specify both decoder_input_ids"
                          " and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        invalidInputError(False,
                          "You have to specify either "
                          "decoder_input_ids or decoder_inputs_embeds")

    # seq_length_with_past = seq_length
    past_key_values_length = 0

    # The original position_ids in the format of [1, 1]
    # However, this only applies when kv_len is the same for all the sequences
    # We should set it to format of [batch, position_id]
    # TODO: validate correctness
    device = input_ids.device if input_ids is not None else inputs_embeds.device
    if position_ids is None:
        invalidInputError("vLLM: position_ids should never be None")
    else:
        # print(f"Original position_ids is {position_ids}")
        position_ids = position_ids.view(-1, seq_length)
        # print(f"after position_ids is {position_ids}")
    # if past_key_values is None:
    #     # For prefill
    #     position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
    #     position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    # else:
    #     past_key_values_length = []
    #     for sequence_kv in past_key_values[0]:
    #         key = sequence_kv[0]
    #         past_key_values_length.append(key.shape[-2])
    #     position_ids = torch.tensor(past_key_values_length, dtype=torch.long, device=device)
    #     position_ids = position_ids.unsqueeze(0).view(-1, 1)

    if past_key_values is not None:
        # past_key_values in the format of num_layers x num_seqs x 2
        # TODO: this may be incorrect
        past_key_values_length = past_key_values[0][0][0].shape[2]
        # seq_length_with_past = seq_length_with_past + past_key_values_length

    # if position_ids is None:
    #     device = input_ids.device if input_ids is not None else inputs_embeds.device
    #     # [start, end)
    #     position_ids = torch.arange(
    #         past_key_values_length, seq_length +
    #         past_key_values_length, dtype=torch.long, device=device
    #     )
    #     position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    # else:
    #     position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # embed positions
    if attention_mask is None:
        invalidInputError(False, "attention_mask should never be None")
    # print(f"attention_mask before expanding: {attention_mask}")
    if past_key_values is None:
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
    else:
        i = 0
        for attn_mask in attention_mask:
            past_key_value_length = past_key_values[0][i][0].shape[2]
            new_mask = self._prepare_decoder_attention_mask(
                attn_mask, (1, seq_length), inputs_embeds, past_key_value_length
            )
            attention_mask[i] = new_mask
            i += 1

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        invalidInputError(False, "gradient_checkpointing is not supported")

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, output_attentions, None)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                attention_mask,
                position_ids,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)  # noqa
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
