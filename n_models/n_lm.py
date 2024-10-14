"""
Universal language model, which accepts as its core a Transformer or a Mamba.

The Transformer is implemented in PyTorch and supports FlashAttention-2/
For Mamba, you have the choice : use mamba.py's pure PyTorch implementation (cf mamba/mamba.py) or use the CUDA implementation.
"""

from typing import Union, List
import inspect
import math

import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.transformer import Transformer, TransformerConfig
from models.mamba.mamba2 import Mamba2, Mamba2Config
from models.mamba.mamba import Mamba, MambaConfig

class LM(nn.Module):
    def __init__(self, model_config: Union[TransformerConfig, MambaConfig, Mamba2Config], vocab_size: int, rng: torch.Generator = None):
        super().__init__()

        self.config = model_config
        self.vocab_size = vocab_size
        self.rng = rng

        self.embedding = nn.Embedding(self.vocab_size, self.config.d_model)
        self.embedding.NORMALIZE = 1
        
        if isinstance(self.config, TransformerConfig):
            self.core = Transformer(self.config)
        elif isinstance(self.config, MambaConfig):
            self.core = Mamba(self.config)
        elif isinstance(self.config, Mamba2Config):
            self.core = Mamba2(self.config)
        else:
            raise NotImplementedError

        self.lm_head = nn.Linear(self.config.d_model, self.vocab_size, bias=False)
        self.lm_head.NORMALIZE = 1
        #self.embedding.weight = self.lm_head.weight

        if rng is None:
            rng = torch.Generator()

        if self.config.mup and isinstance(self.config, TransformerConfig):
            for pn, p in self.named_parameters():
                if any(pn.endswith(w) for w in ['sa.c_attn.weight', 'sa.c_proj.weight', 'mlp.fc_1.weight', 'mlp.fc_2.weight', 'mlp.fc_3.weight']):
                    std = self.config.base_std

                    if any(pn.endswith(w) for w in ['sa.c_proj.weight', 'mlp.fc_2.weight']):
                        std = std / math.sqrt(2 * self.config.n_layers)
                    
                    torch.nn.init.normal_(p, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult), generator=self.rng)

                    #if pn.endswith('sa.c_attn.weight'):
                    #    torch.nn.init.zeros_(p[self.config.d_model:]) # init query proj to 0

                elif pn == "embedding.weight":
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std, generator=self.rng)
                elif pn == "core.PE.weight":
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std, generator=self.rng)
                else:
                    # here, we only have biases and rotary_emb.freqs
                    assert p.dim() == 1, f"a 2d param ({pn}) has not been filtered out for init. please check."

                    if "bias" in pn:
                        torch.nn.init.zeros_(p)

        elif self.config.mup and isinstance(self.config, MambaConfig):
            for pn, p in self.named_parameters():
                if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.x_delta_proj.weight', 'mixer.dt_proj.weight', 'mixer.out_proj.weight', 'mixer.x_proj.weight']):
                    std = self.config.base_std

                    if 'mixer.out_proj.weight' in pn:
                        std = std / math.sqrt(2 * self.config.n_layers)

                    if 'mixer.dt_proj.weight' in pn:
                        std = self.config.dt_rank**-0.5 * self.config.dt_scale

                    torch.nn.init.normal_(p, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult), generator=self.rng)
                elif 'mixer.x_BC_proj.weight' in pn:
                    torch.nn.init.zeros_(p[self.config.dt_rank:])
                elif 'mixer.conv1d.weight' in pn:
                    torch.nn.init.zeros_(p)
                elif pn == "embedding.weight":
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std, generator=self.rng)
                elif pn == "lm_head.weight":
                    torch.nn.init.zeros_(p)
                elif any(pn.endswith(w) for w in ['mixer.A_log', 'mixer.D']):
                    pass
                else:
                    # here, we only have biases
                    assert p.dim() == 1, f"a 2d param ({pn}) has not been filtered out for init. please check."

                    if ("in_proj.bias" in pn) or ("out_proj.bias" in pn):
                        torch.nn.init.zeros_(p)
            
        elif self.config.mup and isinstance(self.config, Mamba2Config):
            for pn, p in self.named_parameters():
                
                if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.out_proj.weight']):
                    std = self.config.base_std

                    if 'mixer.out_proj.weight' in pn:
                        std = std / math.sqrt(2 * self.config.n_layers)

                    torch.nn.init.normal_(p, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult), generator=self.rng)
                
                elif 'mixer.conv1d.weight' in pn:
                    torch.nn.init.zeros_(p)
                
                elif pn == "embedding.weight":
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std, generator=self.rng)
                
                elif pn == "lm_head.weight":
                    torch.nn.init.zeros_(p)
                
                elif any(pn.endswith(w) for w in ['mixer.A_log', 'mixer.D', 'mixer.dt_bias']):
                    pass
                else:
                    # here, we only have biases
                    assert p.dim() == 1, f"a 2d param ({pn}) has not been filtered out for init. please check."

                    if ("in_proj.bias" in pn) or ("out_proj.bias" in pn):
                        torch.nn.init.zeros_(p)

        else: # transformer and mamba
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith('fc_2.weight') or pn.endswith('c_proj.weight') or pn.endswith('mixer.out_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std/math.sqrt(2 * self.config.n_layers))#, generator=self.rng)

    def forward(self, tokens, targets=None, caches=None, seq_pos=0):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)
        if caches is None:
            x = self.core(x)
        else:
            x, caches = self.core(x, caches, seq_pos)

        if self.config.mup:
            x = x / self.config.mup_width_mult

        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return loss
        else:
            return logits, caches
    
    @torch.no_grad()
    def norm_weights(self):
        for module in self.modules():
            if hasattr(module, 'NORMALIZE'):
                F.normalize(module.weight)
        
    def generate(self, prompt, num_tokens: int, sample: bool = True, top_k: int = None, temperature: float = 1.0):
        # prompt : (B, L)

        # generation : (B, l)

        # L>>l

        if top_k is not None:
            top_k = min(top_k, self.vocab_size)
        
        input_device = prompt.device
        prompt = prompt.to(self.embedding.weight.device)

        self.eval()
        generated = prompt.clone()

        with torch.no_grad():
            for _ in range(num_tokens):
                logits = self.forward(generated) # (B, L, vocab_size)
                next_token_logits = logits[:, -1]

                if sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)
                    
                    if top_k is not None:
                        values, _ = torch.topk(probs, k=top_k) # (B, k) ordered from lowest to biggest
                        probs[probs < values[:, -1, None]] = 0 # zero-out all probs except the k first
                        probs = probs / probs.sum(axis=1, keepdims=True)

                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        
        self.train()
        
        return generated.to(input_device)[:, -num_tokens:]
    
    def generate4(self, prompts, num_tokens: List[int], sample: bool = True, top_k: int = None, temperature: float = 1.0):
        # prompts : list of B x (L) tensors

        B = len(prompts)
        min_len = min(prompt.size(0) for prompt in prompts)
        max_len = max(prompt.size(0) for prompt in prompts)

        max_num_tokens = max(num_tokens)

        max_len_generation = max([len(p) + nt for (p, nt) in zip(prompts, num_tokens)]) # max timestep that wil be reached during generation

        assert not isinstance(self.core, Mamba), "Mamba1 doesn't support decoding with the generate4 function."
        if isinstance(self.core, Mamba2):
            assert self.config.use_mem_eff_path, "Mamba2 should use the mem_eff_path when decoding with the generate4 function"
            assert min_len >= self.config.d_conv
            
        if top_k is not None:
            top_k = min(top_k, self.vocab_size)

        input_device = prompts[0].device
        model_device = self.embedding.weight.device
        
        self.eval()
        
        padded_prompts = [F.pad(prompt, (0, max_len-prompt.size(0))) for prompt in prompts]
        padded_prompts = torch.stack(padded_prompts) # to : model_device?
        
        batched_generated = torch.zeros(B, max_len+max_num_tokens, dtype=torch.long, device=model_device)
        batched_generated[:, :max_len] = padded_prompts
        
        prompt_lengths = torch.tensor([p.size(0) for p in prompts], device=input_device)
        position_ids = torch.arange(max_len+max_num_tokens, device=input_device).unsqueeze(0).expand(B, -1)
        active_mask = position_ids < prompt_lengths.unsqueeze(1)
        active_mask = active_mask.to(model_device)

        # caches is a list of cache, one per layer
        # cache is composed of : - if Mamba(2) layer : the hidden state, and the last d_conv-1 inputs (see more in mamba_lm.py)
        #                        - if attention layer : the KV cache, ie 2 tensors of shape (B, num_kv_heads, L, head_dim)
        caches = [layer.get_empty_cache(min_len) for layer in self.core.layers]

        with torch.no_grad():
            # process prompt in one go
            logits, caches = self.forward(batched_generated[:, :min_len], caches) # (B, L, vocab_size)
            next_token_logits = logits[:, -1] # (B, vocab_size)

            for t in range(min_len, max_len_generation):
                if sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)

                    if top_k is not None:
                        values, _ = torch.topk(probs, k=top_k) # (B, k) ordered from lowest to biggest
                        probs[probs < values[:, -1, None]] = 0 # zero-out all probs except the k first
                        probs = probs / probs.sum(axis=1, keepdims=True)

                    next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True) # (B, 1)

                # here, choose if modify batched_generated[:, t] with next_token or leave it as is

                update_mask = ~active_mask[:, t]
                batched_generated[:, t] = torch.where(update_mask, next_token.squeeze(1), batched_generated[:, t])

                next_token_logits, caches = self.forward(batched_generated[:, [t]], caches, seq_pos=t) # (B, 1, vocab_size), caches
                next_token_logits = next_token_logits.squeeze(1) # (B, vocab_size)

        self.train()

        generated = [seq[prompts[i].size(0):prompts[i].size(0) + nt].to(input_device) for i, (seq, nt) in enumerate(zip(batched_generated, num_tokens))]
        return generated
        
    def setup_generation(self, sample: bool = False, top_k: int = None, temperature: float = 1.0):
        warning_length = [False]
        
        def generate(prompts, num_tokens):
            if isinstance(num_tokens, int):
                num_tokens= [num_tokens] * len(prompts)

            if max(p.size(0) + nt for (p, nt) in zip(prompts, num_tokens)) > self.config.max_len and not warning_length[0]:
                print(f"Warning: going over the context length seen during training. ({max(p.size(0) + nt for (p, nt) in zip(prompts, num_tokens))})")
                #warning_length[0] = True
            
            generation = self.generate4(prompts, num_tokens=num_tokens, sample=sample, top_k=top_k, temperature=temperature)

            return generation
        
        return generate
    
    # obsolete
    def generate3(self, prompt, num_tokens: int, sample: bool = True, top_k: int = None, temperature: float = 1.0):
        # prompt : (1, L)

        assert not isinstance(self.core, Mamba), "Mamba1 doesn't support decoding with the generate3 function."
        if isinstance(self.core, Mamba2):
            assert self.config.use_mem_eff_path, "Mamba2 should use the mem_eff_path when decoding with the generate3 function"
            assert prompt.size(1) >= self.config.d_conv

        if top_k is not None:
            top_k = min(top_k, self.vocab_size)

        input_device = prompt.device
        model_device = self.embedding.weight.device

        prompt = prompt.to(model_device)

        self.eval()

        len_prompt = prompt.size(1)
        generated = prompt.clone()

        # caches is a list of cache, one per layer
        # cache is composed of : - if Mamba(2) layer : the hidden state, and the last d_conv-1 inputs (see more in mamba_lm.py)
        #                        - if attention layer : the KV cache, ie 2 tensors of shape (B, num_kv_heads, L, head_dim)
        caches = [layer.get_empty_cache(prompt.size(0)) for layer in self.core.layers]

        with torch.no_grad():
            # process prompt in one go
            logits, caches = self.forward(prompt, caches) # (B, L, vocab_size)
            next_token_logits = logits[:, -1] # (B, vocab_size)

            for t in range(num_tokens):
                if sample:
                    probs = F.softmax(next_token_logits / temperature, dim=-1)

                    if top_k is not None:
                        values, _ = torch.topk(probs, k=top_k) # (B, k) ordered from lowest to biggest
                        probs[probs < values[:, -1, None]] = 0 # zero-out all probs except the k first
                        probs = probs / probs.sum(axis=1, keepdims=True)

                    next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
                else:
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True) # (B, 1)

                generated = torch.cat([generated, next_token], dim=1)
                next_token_logits, caches = self.forward(generated[:, [len_prompt+t]], caches, seq_pos=len_prompt+t) # (B, 1, vocab_size), caches
                next_token_logits = next_token_logits.squeeze(1) # (B, vocab_size)

        self.train()
        return generated.to(input_device)[:, -num_tokens:]
    
    # do not use for setups with large prompt and small gen
    def generate2(self, prompt, num_tokens: int, sample: bool = True, top_k: int = 40, temperature: float = 1.0):
        # prompt : (B, L)

        top_k = min(top_k, self.vocab_size)

        self.eval()

        # caches is a list of cache, one per layer
        # cache is composed of : - if Mamba(2) layer : the hidden state, and the last d_conv-1 inputs (see more in mamba_lm.py)
        #                        - if attention layer : the KV cache, ie 2 tensors of shape (B, num_kv_heads, L, head_dim)
        caches = [layer.get_empty_cache(prompt.size(0)) for layer in self.core.layers]

        for i in range(prompt.size(1) + num_tokens - 1):
            with torch.no_grad():
                # forward the new output, get new cache
                next_token_logits, caches = self.forward(prompt[:, [i]], caches, seq_pos=i) # (B, 1, vocab_size), caches
                next_token_logits = next_token_logits.squeeze(1) # (B, vocab_size)

            # sample (no sampling when the prompt is being processed)
            if i+1 >= prompt.size(1):
                probs = F.softmax(next_token_logits / temperature, dim=-1) # (B, vocab_size)

                if top_k is not None:
                    values, _ = torch.topk(probs, k=top_k) # (B, k) ordered from lowest to biggest
                    probs[probs < values[:, -1, None]] = 0 # zero-out all probs except the k first
                    probs = probs / probs.sum(axis=1, keepdims=True)

                if sample:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1) # (B)
                else:
                    next_token = torch.argmax(probs, dim=-1) # (B)

                prompt = torch.cat([prompt, next_token.unsqueeze(1)], dim=1)
        
        self.train()

        return prompt[:, -num_tokens:]
    
    # non-muP init
    # taken from llama2.c
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)#, generator=self.rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std)#, generator=self.rng)

    # adapted from llama2.c, with muP
    def configure_optimizers(self, optimizer, weight_decay, learning_rate, betas, device_type, beta3=None, alpha=None, T_ab3=None):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        if self.config.mup and isinstance(self.config, TransformerConfig):
            mup_params_keys = set([pn for pn in param_dict.keys() if any(pn.endswith(w) for w in ['sa.c_attn.weight', 'sa.c_proj.weight', 'mlp.fc_1.weight', 'mlp.fc_2.weight', 'mlp.fc_3.weight'])])
            dim2_params_keys = set([pn for pn in param_dict.keys() if param_dict[pn].dim() >= 2])

            assert dim2_params_keys.difference(mup_params_keys) == ({'embedding.weight'} if self.config.pos_emb == "rope" else {'embedding.weight', 'core.PE.weight'})
            assert mup_params_keys.difference(dim2_params_keys) == set()

            dim2_params_keys = dim2_params_keys.difference(mup_params_keys) # only biases and embd left

            mup_parameters = [p for n, p in param_dict.items() if n in mup_params_keys]
            decay_params = [p for n, p in param_dict.items() if n in dim2_params_keys]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

            optim_groups = [
                {'params': mup_parameters, 'weight_decay': weight_decay * self.config.mup_width_mult, 'lr': learning_rate / self.config.mup_width_mult},
                {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
            ]

        elif self.config.mup and isinstance(self.config, MambaConfig):
            mup_params_keys = set([pn for pn in param_dict.keys() if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.x_delta_proj.weight', 'mixer.dt_proj.weight', 'mixer.out_proj.weight', 'mixer.x_proj.weight'])])
            
            dim2_params_keys = set([pn for pn in param_dict.keys() if param_dict[pn].dim() >= 2])
            dim2_params_keys = dim2_params_keys.difference(mup_params_keys)

            mup_parameters = [p for n, p in param_dict.items() if n in mup_params_keys]
            decay_params = [p for n, p in param_dict.items() if n in dim2_params_keys]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # biases and D

            optim_groups = [
                {'params': mup_parameters, 'weight_decay': weight_decay * self.config.mup_width_mult, 'lr': learning_rate / self.config.mup_width_mult},
                {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
            ]

        elif self.config.mup and isinstance(self.config, Mamba2Config):
            mup_params_keys = set([pn for pn in param_dict.keys() if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.out_proj.weight'])])
            
            dim2_params_keys = set([pn for pn in param_dict.keys() if param_dict[pn].dim() >= 2])
            dim2_params_keys = dim2_params_keys.difference(mup_params_keys)

            mup_parameters = [p for n, p in param_dict.items() if n in mup_params_keys]
            decay_params = [p for n, p in param_dict.items() if n in dim2_params_keys]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # biases and D and A

            optim_groups = [
                {'params': mup_parameters, 'weight_decay': weight_decay * self.config.mup_width_mult, 'lr': learning_rate / self.config.mup_width_mult},
                {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
            ]

        else:
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] # weights in matrices and embeddings
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # biases and norms

            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate},
                {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate}
            ]

        if optimizer == "AdamW":
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            optimizer = torch.optim.AdamW(optim_groups, betas=betas, fused=use_fused)
        else:
            raise NotImplementedError

        return optimizer

def load_model(load_dir, vocab_size, device="cuda"):
    config_dir = os.path.join(load_dir, 'config.json')
    checkpoint_dir = os.path.join(load_dir, 'model.pth')

    config_json = json.load(open(config_dir))
    architecture = config_json['architecture']
    del config_json['architecture']

    if architecture == "Transformer":
        config = TransformerConfig(**config_json)
    elif architecture == "Mamba":
        config = MambaConfig(**config_json)
    elif architecture == "Mamba2":
        config = Mamba2Config(**config_json)
    else:
        raise NotImplementedError

    model = LM(config, vocab_size=vocab_size).to(device)
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model
