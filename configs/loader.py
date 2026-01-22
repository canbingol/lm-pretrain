import yaml
import argparse
import yaml
from dataclasses import dataclass

from configs.schema import AppConfig

def get_config():
    parser = argparse.ArgumentParser()
    # model choosing
    parser.add_argument("--config", type=str, help="Path to YAML config file")

    args = parser.parse_args()
    path = args.config

    with open(path) as stream:
        yaml_configs = yaml.safe_load(stream=stream)

    cfg = AppConfig(**yaml_configs)
    return cfg


def set_config(model_config, model_section):

    model_config.attention_bias = model_section.attention_bias
    model_config.attention_dropout = model_section.attention_dropout
    model_config.bos_token_id = model_section.bos_token_id
    model_config.eos_token_id = model_section.eos_token_id

    model_config.intermediate_size = model_section.intermediate_size
    model_config.hidden_act = model_section.hidden_act

    model_config.head_dim = model_section.head_dim
    model_config.hidden_size = model_section.hidden_size
    model_config.num_attention_heads = model_section.num_attention_heads
    model_config.num_hidden_layers = model_section.num_hidden_layers
    model_config.num_key_value_heads = model_section.num_key_value_heads
    model_config.vocab_size = model_section.vocab_size
    model_config.max_position_embeddings = model_section.max_position_embeddings

    model_config.rms_norm_eps = model_section.rms_norm_eps
    model_config.rope_scaling = model_section.rope_scaling
    model_config.rope_theta = model_section.rope_theta

    model_config.use_cache = model_section.use_cache
    model_config.attn_type = model_section.attn_type
    model_config.torch_dtype = model_section.torch_dtype



