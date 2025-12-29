# Decoder Only Models
info



## Config Parameters
| Parameters | Value | Description |
|----------|-------|----------|
| config_name | decoder_only | Specifies the name of the model configuration|
| attention_bias | False | Controls whether a bias term is added to the attention linear projections|
| attention_dropout | 0.0 |Dropout probability applied to the attention weights.|
| bos_token_id | 151643 | Token ID used to represent the beginning of a sequence. |
| eos_token_id | 151645 | Token ID used to represent the end of a sequence. |
| head_dim | 128 | Dimensionality of each attention head. |
| hidden_act | silu | Activation function used in the feed-forward network. |
| hidden_size | 512 | Dimensionality of the model’s hidden representations. |
| initializer_range | 0.02 | Standard deviation used for initializing model weights. |
| intermediate_size | 3072 | Dimensionality of the intermediate feed-forward layer. |
| max_position_embeddings | 40960 | Maximum sequence length supported by the model. |
| max_window_layers | 28 | Number of layers that apply windowed attention when enabled. |
| model_type | decoder_only | Specifies the architectural type of the model. |
| num_attention_heads | 16 | Number of attention heads in each attention layer. |
| num_hidden_layers | 1 | Number of transformer layers in the model. |
| num_key_value_heads | 8 | Number of key/value heads used for grouped or multi-query attention. |
| rms_norm_eps | 1e-6 | Epsilon value used for numerical stability in RMS normalization. |
| rope_scaling | None | Configuration for scaling Rotary Positional Embeddings, if used. |
| rope_theta | 1000000 | Base value controlling the frequency scale of Rotary Positional Embeddings. |
| sliding_window | None | Window size used for sliding-window attention, if enabled. |
| tie_word_embeddings | True | Whether input and output word embeddings share the same weights. |
| torch_dtype | bfloat16 | Data type used for model parameters and computation. |
| use_cache | True | Whether to cache past key/value states for faster autoregressive decoding. |
| use_sliding_window | False | Whether sliding-window attention is enabled. |
| vocab_size | 50176 | Size of the model vocabulary. |

