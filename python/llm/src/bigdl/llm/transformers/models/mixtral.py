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
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py

# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

""" PyTorch Mixtral model."""
import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from bigdl.llm.ggml.quantize import ggml_tensor_qtype
from bigdl.llm.utils.common import invalidInputError
from bigdl.llm.transformers.models.utils import init_kv_cache, extend_kv_cache, append_kv_cache
from bigdl.llm.transformers.models.utils import apply_rotary_pos_emb,\
    apply_rotary_pos_emb_no_cache_xpu, is_enough_kv_cache_room_4_36
from bigdl.llm.transformers.models.mistral import should_use_fuse_rope, use_decoding_fast_path


KV_CACHE_ALLOC_BLOCK_LENGTH = 256


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads,
                                                           n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def mixtral_moeblock_forward(self,
                             hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    bs = hidden_states.shape[0]
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    routing_weights = routing_weights.to(hidden_states.dtype)

    if bs > 1:
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts,
                                                  num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state,
                                                 routing_weights[top_x_list, idx_list, None])

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    else:
        selected_experts = selected_experts[0].cpu().tolist()
        for idx in range(self.top_k):
            exp_id = selected_experts[idx]
            expert_layer = self.experts[exp_id]
            weight = routing_weights[:, idx]
            if idx == 0:
                final_hidden_states = expert_layer(hidden_states, weight)
            else:
                final_hidden_states = final_hidden_states + expert_layer(hidden_states, weight)

    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def mixtral_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor]=None,
    position_ids: Optional[torch.LongTensor]=None,
    past_key_value: Optional[Tuple[torch.Tensor]]=None,
    output_attentions: bool=False,
    use_cache: bool=False,
    padding_mask: Optional[torch.Tensor]=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    device = hidden_states.device

    use_fuse_rope = should_use_fuse_rope(self, hidden_states, position_ids)
    enough_kv_room = is_enough_kv_cache_room_4_36(past_key_value, self.layer_idx)
    decoding_fast_path = use_decoding_fast_path(self.q_proj.qtype,
                                                use_fuse_rope,
                                                enough_kv_room,
                                                bsz * q_len)

    if decoding_fast_path:
        hidden_states = hidden_states.view(1, -1)
        cache_k = past_key_value.key_cache[self.layer_idx]
        cache_v = past_key_value.value_cache[self.layer_idx]
        kv_seq_len = cache_k.shape[-2]
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
        # update past_key_value's seem_tokens and kv caches.
        if self.layer_idx == 0:
            past_key_value.seen_tokens = kv_seq_len
        past_key_value.key_cache[self.layer_idx] = key_states
        past_key_value.value_cache[self.layer_idx] = value_states

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len,
                                     self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len,
                                         self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                invalidInputError(False,
                                  "The cache structure has changed since version v4.36. "
                                  f"If you are using {self.__class__.__name__} for "
                                  "auto-regressive decodingwith k/v caching, please make sure "
                                  "to initialize the attention class with a layer index.")
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if use_fuse_rope:
            query_states, key_states = apply_rotary_pos_emb_no_cache_xpu(query_states,
                                                                         key_states,
                                                                         position_ids,
                                                                         "mixtral")
        else:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                            cos, sin, position_ids, "mixtral")

        if past_key_value is not None:
            # update the number of seen tokens
            if self.layer_idx == 0:
                past_key_value.seen_tokens += key_states.shape[-2]

            # reuse k, v, self_attention
            # update `past_key_value` with `key_states` and `value_states` for layer `layer_idx`
            if len(past_key_value.key_cache) <= self.layer_idx:
                past_key_value.key_cache.append(key_states)
                past_key_value.value_cache.append(value_states)
            else:
                cache_k = past_key_value.key_cache[self.layer_idx]
                cache_v = past_key_value.value_cache[self.layer_idx]

                if not enough_kv_room:
                    # allocate new
                    new_c_k, new_c_v = extend_kv_cache(bsz,
                                                       self.num_key_value_heads,  # Support GQA
                                                       self.head_dim,
                                                       cache_k.size(2),
                                                       kv_seq_len + KV_CACHE_ALLOC_BLOCK_LENGTH,
                                                       dtype=cache_k.dtype,
                                                       device=device)

                    new_c_k[:] = cache_k
                    new_c_v[:] = cache_v
                    cache_k = new_c_k
                    cache_v = new_c_v

                key_states, value_states = append_kv_cache(cache_k,
                                                           cache_v,
                                                           key_states,
                                                           value_states)

                # update past_key_value
                past_key_value.key_cache[self.layer_idx] = key_states
                past_key_value.value_cache[self.layer_idx] = value_states

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(
        query_states,
        key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        invalidInputError(
            False,
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)},"
            f" but is {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            invalidInputError(
                False,
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)},"
                f" but is {attention_mask.size()}"
            )

        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.\
        softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        invalidInputError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)},"
            f" but is {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def mixtral_mlp_forward(
    self,
    x: torch.Tensor,
    routing_weights
) -> torch.Tensor:
    if x.shape[0] == 1 and x.device.type == 'xpu' \
            and self.w1.qtype == ggml_tensor_qtype["sym_int4"] \
            and not (self.training and x.requires_grad):
        import linear_q4_0
        return self.w2(linear_q4_0.mlp_forward_q4_0_xpu(
            x, self.w1.weight.data, self.w3.weight.data,
            x.shape[0], x.shape[1], self.w1.out_len,
        )) * routing_weights
    else:
        current_hidden_states = self.act_fn(self.w1(x)) * self.w3(x)
        current_hidden_states = self.w2(current_hidden_states)
        return routing_weights * current_hidden_states
