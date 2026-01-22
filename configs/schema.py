from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, ConfigDict, field_validator


class HubConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    push_to_hub: bool = False
    repo_name: Optional[str] = None


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["decoder"] = "decoder"
    attn: Literal["sdpa", "eager", "flash_attn"] = "sdpa"

    checkpoint: Optional[str] = None
    world_size: int = 1
    force: bool = False
    model_info: bool = True

    training_type: Literal["pre-train", "instruction-tuning"] = "pre-train"

    epoch: int = 1
    lr: float = 1e-4
    eval_steps: int = 10
    eval_sample: int = 1
    training_steps: int 
    
    @field_validator("world_size", "epoch", "eval_steps", "eval_sample")
    @classmethod
    def _positive_ints(cls, v: int, info):
        if v <= 0:
            raise ValueError(f"train.{info.field_name} must be > 0")
        return v

    @field_validator("lr")
    @classmethod
    def _lr_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("train.lr must be > 0")
        return v


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 1
    hf_tokenizer: str
    pretraining_hf_data: Optional[str] = None
    it_hf_data: Optional[str] = None

    text_column_name: str = "text"
    token_path: str

    tokens_chunks_size: int
    test_split: float = 0.0
    token_dtype: str = "np.uint16"

    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 0
    pin_memory: bool = True

    max_seq_len: int = 512
    pad_token: int = 0

    @field_validator("tokens_chunks_size", "max_seq_len")
    @classmethod
    def _positive_ints(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"data.{info.field_name} must be > 0")
        return v

    @field_validator("num_workers", "pad_token")
    @classmethod
    def _nonneg_ints(cls, v: int, info) -> int:
        if v < 0:
            raise ValueError(f"data.{info.field_name} must be >= 0")
        return v

    @field_validator("test_split")
    @classmethod
    def _test_split_range(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError("data.test_split must be in [0.0, 1.0)")
        return v

    @field_validator("token_dtype", mode="before")
    @classmethod
    def _normalize_token_dtype(cls, v):
        # Accept: np.uint16, "np.uint16", "uint16", etc.
        if v is None:
            return "np.uint16"
        s = str(v).strip()
        # normalize common variants
        if s in {"uint16", "np.uint16", "numpy.uint16"}:
            return "np.uint16"
        if s in {"uint32", "np.uint32", "numpy.uint32"}:
            return "np.uint32"
        if s in {"int32", "np.int32", "numpy.int32"}:
            return "np.int32"
        return s


class ModelSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attention_bias: bool = False
    attention_dropout: float = 0.0

    bos_token_id: int
    eos_token_id: int

    intermediate_size: int
    hidden_act: str = "silu"

    head_dim: int
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int

    vocab_size: int
    max_position_embeddings: int

    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_scaling: Optional[float] = None
    use_cache: bool = True
    attn_type: Literal["eager", "sdpa", "flash_attn", "flash_attention", "xformers"] = "eager"
    torch_dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"

    @field_validator(
        "intermediate_size",
        "head_dim",
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
    )
    @classmethod
    def _positive_ints(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"model.{info.field_name} must be > 0")
        return v

    @field_validator("attention_dropout")
    @classmethod
    def _dropout_range(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError("model.attention_dropout must be in [0.0, 1.0)")
        return v


# -------------------------
# Root
# -------------------------

class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hub: HubConfig
    train: TrainConfig
    data: DataConfig
    model: ModelSection
