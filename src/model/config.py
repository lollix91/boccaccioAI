from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class BoccaccioConfig:
    vocab_size: int = 32000
    hidden_size: int = 1536
    num_layers: int = 26
    num_attention_heads: int = 12
    num_kv_heads: int = 4
    intermediate_size: int = 4096
    max_position_embeddings: int = 2048
    activation_function: str = "swiglu"
    positional_embedding: str = "rope"
    rope_theta: float = 10000.0
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    embedding_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        self.validate()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def validate(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )

    @classmethod
    def from_yaml(cls, path: str, variant: str = "model") -> BoccaccioConfig:
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        section = data[variant]
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in section.items() if k in fields}
        return cls(**kwargs)

    @classmethod
    def nano(cls) -> BoccaccioConfig:
        return cls(
            hidden_size=256,
            num_layers=4,
            num_attention_heads=8,
            num_kv_heads=2,
            intermediate_size=704,
            max_position_embeddings=512,
        )
