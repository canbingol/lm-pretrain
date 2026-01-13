<p align="center">
  <img src="assets/logo.svg" alt="LM-Pretrain Logo" width="600"/>
</p>

# LM-Pretrain: Pretraining and Instruction Tuning for Language Models

---

## Project Status

This project is currently under **active development** and should be considered **experimental**.
The goal is to develop a fully modular and extensible framework for **pretraining**, **instruction tuning**, and **RLHF** on large language models, enabling easy architectural customization.

I'm actively working on improving the codebase and would appreciate any feedback or suggestions.

Feel free to reach out:

- **Email**: [canobingol2@gmail.com](mailto:canobingol2@gmail.com)
- **LinkedIn**: [linkedin.com/in/canbingöl](https://www.linkedin.com/in/canbing%C3%B6l)

---

> [!WARNING]
> In this repository, several components are hard-coded.
> Make sure to review and adjust them before adapting the code to your own setup.

## Features

- **Decoder-only Transformer**: Custom implementation with Grouped Query Attention (GQA), RoPE embeddings, and RMSNorm
- **Multiple Attention Backends**: FlashAttention, PyTorch SDPA, and eager attention
- **Distributed Training**: Full DDP support with torchrun
- **Two Training Modes**: Pretraining and instruction tuning pipelines
- **Learning Rate Scheduling**: Warmup + cosine annealing scheduler
- **Checkpointing**: Automatic best model saving based on validation loss
- **Text Generation**: Sampling with temperature and top-k support

## Installation

```bash
git clone https://github.com/canbingol/lm-pretrain.git
cd lm-pretrain
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- Hugging Face Transformers
- Hugging Face Datasets
- (Optional) flash-attn for FlashAttention support

## Project Structure

```
lm-pretrain/
├── main.py              # Main entry point for training and inference
├── data_prepare.py      # Dataset preparation for pretraining and IT
├── inference.py         # Text generation utilities
├── utils.py             # Utility functions and dataclasses
├── run.sh               # Training launcher script
├── setup_config.yaml    # Example configuration file
├── models/
│   └── decoder_model.py # Decoder-only transformer implementation
├── train/
│   ├── trainer.py       # Training loop with DDP
│   └── loss_func.py     # Loss calculation and evaluation
└── data/
    └── pretrain/        # Tokenized binary files directory
```

## Model Architecture

The decoder-only model includes:

| Component | Description |
|-----------|-------------|
| **Attention** | Grouped Query Attention (GQA) with configurable head counts |
| **Position Encoding** | Rotary Position Embeddings (RoPE) |
| **Normalization** | RMSNorm (pre-norm architecture) |
| **Activation** | SiLU (SwiGLU-style MLP) |
| **Attention Types** | `flash_attn`, `sdpa`, `eager` |

### Default Model Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `hidden_size` | 512 | Hidden state dimensionality |
| `intermediate_size` | 3072 | MLP intermediate size |
| `num_hidden_layers` | 1 | Number of transformer layers |
| `num_attention_heads` | 16 | Number of attention heads |
| `num_key_value_heads` | 8 | Number of KV heads (for GQA) |
| `head_dim` | 128 | Dimension per attention head |
| `vocab_size` | 50176 | Vocabulary size |
| `max_position_embeddings` | 40960 | Maximum sequence length |
| `torch_dtype` | bfloat16 | Model precision |
| `attn_type` | flash_attn | Attention implementation |

## Configuration

All training parameters are specified in a YAML configuration file:

### Example Configuration (`setup_config.yaml`)

```yaml
inference:
  inference: False
  max_new_tokens: 64
  prompt: "merhaba"
  checkpoint: None

hub:
  push_to_hub: True
  repo_name: canbingol/my-awesome-model

train:
  model: "decoder"
  world_size: 1
  force: False
  training_type: pre-train    # Options: "pre-train" or "instruction-tuning"
  epoch: 5
  lr: 1e-4
  eval_steps: 10
  eval_sample: 1

data:
  hf_tokenizer: "vngrs-ai/Kumru-2B"
  pretraining_hf_data: "canbingol/vngrs-web-corpus-500k"
  it_hf_data: "canbingol/turkish_instructions"
  token_path: "./data/pretrain/default_tokenizer"
  tokens_chunks_size: 25000000
  test_split: 0.05
  token_dtype: np.uint16
  vocab_size: 50176
  batch_size: 1
  max_seq_len: 512
  shuffle: False
  drop_last: True
  num_workers: 0
  pin_memory: True
```

### Hugging Face Hub Integration

To upload checkpoints and final models to Hugging Face Hub:

- Set `push_to_hub=True`.
- Provide a valid `repo_name` (e.g., `canbingol/my-awesome-model`).
- Authenticate via CLI:
  ```bash
  huggingface-cli login
  ```
### Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_type` | `str` | Training mode: `pre-train` or `instruction-tuning` |
| `model` | `str` | Model architecture (currently: `decoder`) |
| `world_size` | `int` | Number of GPUs for distributed training |
| `epoch` | `int` | Number of training epochs |
| `lr` | `float` | Learning rate |
| `eval_steps` | `int` | Evaluation interval (in steps) |
| `eval_sample` | `int` | Number of batches for evaluation |
| `hf_tokenizer` | `str` | Hugging Face tokenizer path |
| `pretraining_hf_data` | `str` | HF dataset for pretraining |
| `it_hf_data` | `str` | HF dataset for instruction tuning |
| `token_path` | `str` | Directory for tokenized binary files |
| `vocab_size` | `int` | Vocabulary size |
| `batch_size` | `int` | Training batch size |
| `max_seq_len` | `int` | Maximum sequence length |
| `force` | `bool` | Force training from scratch (ignore checkpoints) |
| `push_to_hub`  | bool | Whether to push the model and tokenizer to the Hugging Face Hub. |
| `repo_name`    | str  | Hugging Face repository name (e.g., `canbingol/my-awesome-model`). |

## Usage

### Training

```bash
cd lm-pretrain
sh run.sh setup_config.yaml
```

This will:
1. Parse the YAML configuration
2. Initialize distributed training with `torchrun`
3. Load or create tokenized data
4. Train the model with periodic evaluation
5. Save the best checkpoint based on validation loss

### Training Types

The training type is determined by `training_type` in the config:

- **`pre-train`**: Uses the pretraining pipeline which processes plain text into token sequences for next-token prediction.
- **`instruction-tuning`**: Uses the instruction tuning pipeline which constructs instruction-response pairs with target masking.

### Inference

Set `inference: True` in the config and provide a checkpoint path:

```yaml
inference:
  inference: True
  max_new_tokens: 64
  prompt: "Your prompt here"
  checkpoint: "output/decoder_pre-train_60M/decoder_only_best_model.pt"
```

Then run the same command:

```bash
sh run.sh setup_config.yaml
```

## Data Preparation

### Pre-Training Data

For pretraining, you need a Hugging Face dataset with a `text` column:

| text |
|------|
| "örnek cümle..." |
| ... |

Tokenized data is automatically generated and cached as `.bin` files:

```python
from transformers import AutoTokenizer
from data_prepare import create_tokens_file

tokenizer = AutoTokenizer.from_pretrained("vngrs-ai/Kumru-2B")

create_tokens_file(
    hf_dataset="canbingol/vngrs-web-corpus-500k",
    hf_tokenizer="vngrs-ai/Kumru-2B",
    base_dir="./data/pretrain"
)
```

#### Data Directory Structure

```
data/
└── pretrain/
    └── <tokenizer_name>/
        ├── train_00.bin
        ├── train_01.bin
        ├── ...
        └── validation_00.bin
```

### Instruction Tuning Data

For instruction tuning, the dataset should have three columns:

| question | input | answer |
|----------|-------|--------|
| "Bir e-postayı özetle." | "E-posta içeriği..." | "Kısa özet:" |

The data is formatted using Kumru-2B's chat template:

```
<|start_header_id|>user<|end_header_id|>
{question}
<|start_header_id|>input<|end_header_id|>
{input}
<|start_header_id|>assistant<|end_header_id|>
{answer}
```

> [!NOTE]
> Only the assistant response portion contributes to the loss during instruction tuning (target masking is applied).

## Output Directory Structure

```
output/
└── decoder_pre-train_60.00M/
    ├── decoder_only_best_model.pt   # Best checkpoint (lowest val loss)
    ├── train_loss.txt               # Training loss per step
    └── validation_loss.txt          # Validation loss per eval step
```

If `force: True`, a unique directory with timestamp is created.

## Training Details

### Optimizer and Scheduler

- **Optimizer**: AdamW with weight decay 0.1
- **Warmup**: Linear warmup for 100 steps
- **Decay**: Cosine annealing to minimum LR (3e-5)

### Distributed Training

The framework uses PyTorch DDP (DistributedDataParallel):

- Backend: `nccl` (multi-GPU) or `gloo` (single GPU)
- Launcher: `torchrun --standalone --nproc_per_node=N`

### Checkpointing

Checkpoints include:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Training/validation loss
- Epoch and step information

Training automatically resumes from checkpoint unless `force: True`.

## Text Generation

The model supports autoregressive generation with KV caching:

```python
from models.decoder_model import DecoderCausalLM, ModelConfig
from transformers import AutoTokenizer
import torch

# Load model and tokenizer
config = ModelConfig()
model = DecoderCausalLM(config)
model.load_state_dict(torch.load("checkpoint.pt")["model_state_dict"])
tokenizer = AutoTokenizer.from_pretrained("vngrs-ai/Kumru-2B")

# Generate
prompt = tokenizer.encode("Merhaba", return_tensors="pt")
output = model.generate(prompt, max_new_tokens=64, temperature=0.7, top_k=50)
print(tokenizer.decode(output[0]))
```

Generation parameters:
- `max_new_tokens`: Maximum tokens to generate
- `temperature`: Sampling temperature (higher = more random)
- `top_k`: Top-k sampling (limits vocabulary)
- `use_cache`: Enable KV caching for faster generation

## Future Work

- **Refactoring**: Eliminate hard-coded logic and improve maintainability
- **RLHF Integration**: Add reinforcement learning from human feedback
- **Multi-GPU Scaling**: Improve scaling efficiency
- **Additional Models**: Support for more architectures
- **Documentation**: Expand with tutorials and examples

The long-term goal is to create a **stable, modular, and extensible framework** for building, fine-tuning, and aligning large language models.

## License

This project is open source. Please check the repository for license details.
