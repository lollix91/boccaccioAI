import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import BoccaccioConfig
from src.model.layers import RMSNorm, SwiGLU
from src.model.attention import GroupedQueryAttention


class BoccaccioBlock(nn.Module):

    def __init__(self, config: BoccaccioConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class BoccaccioModel(nn.Module):

    def __init__(self, config: BoccaccioConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [BoccaccioBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self._init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def _init_weights(self) -> None:
        std = self.config.initializer_range
        residual_std = std / math.sqrt(2 * self.config.num_layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                name = None
                for n, m in self.named_modules():
                    if m is module:
                        name = n
                        break
                if name is not None and (
                    name.endswith(".o_proj") or name.endswith(".down_proj")
                ):
                    nn.init.normal_(module.weight, mean=0.0, std=residual_std)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)


class BoccaccioForCausalLM(nn.Module):

    def __init__(self, config: BoccaccioConfig) -> None:
        super().__init__()
        self.config = config
        self.model = BoccaccioModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        hidden_states = self.model(input_ids, position_ids, attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
