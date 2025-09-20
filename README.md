# LM-Pretrain

This repository was created for the experimental pretraining of SOTA language models.  
Currently supported architectures: **DeepSeek**, **Qwen**, **LLaMA** ve **Gemma**.

---

## Project Status

This project is currently under **active development** and should be considered **experimental**.  
The aim is to build a **fully modular and extensible pretraining framework** for LLMs that supports easy architecture customization.

I’m actively working on improving the codebase and would appreciate any feedback or suggestions.

Feel free to reach out:

- **Email**: [canobingol2@gmail.com](mailto:canobingol2@gmail.com)
- **LinkedIn**: [linkedin.com/in/canbingöl](https://www.linkedin.com/in/canbing%C3%B6l)

---

## Model Design Notes

- Some models in this repository are implemented based on their original repositories. Others are adapted from Hugging Face’s `transformers` library and simplified.
- The configuration files in `config/` or passed via CLI are **not exact replicas** of official SOTA model configs.
- Tokenizers are trained from scratch using `train_tokenizer.py`.
- You can modify the model configuration (number of layers, hidden size, vocab size, etc.) via CLI or config files.

## Installation

```bash
pip install -r requirements.txt

```

## Sample Usage

```bash
cd lm-pretrain
python main.py   --model gemma2    --hf-data savasy/ttc4900   --vocab-size 5_000   --batch-size 10   --model-info   --epoch 2   --training-steps 400   --eval-steps 20 --eval-sample 20
```

## Output Directory Structure

When training starts, an output directory is created at `output/<model_name>` containing the following files:

```
output/
├── gemma/
│   ├── gemma.tokenizer.model     # SentencePiece tokenizer model
│   ├── gemma.tokenizer.vocab     # Tokenizer vocabulary file
│   ├── gemma2_best_model.pt      # Best checkpoint (lowest validation loss)
│   ├── sample.txt                # Human-readable example tokenizations
│   └── ttc4900.txt               # Text dump used for tokenizer training
├── qwen3/
├── qwen3_force/
```

> If the `--force` flag is used, a new directory like `output/<model_name>_force/` is created.  
> In this case, **all assets including tokenizer and checkpoints will be regenerated**, and no previously saved files will be reused.

## Command-Line Arguments

| Argument             | Description                                                        | Default          |
| -------------------- | ------------------------------------------------------------------ | ---------------- |
| `--model`            | Model architecture to use (`qwen3`, `deepseek`, `gemma`, `llama2`) | **Required**     |
| `--epoch`            | Number of training epochs                                          | `1`              |
| `--training-steps`   | Maximum training steps (if `None`, computed automatically)         | `None`           |
| `--eval-steps`       | Evaluation interval (in steps)                                     | `200`            |
| `--eval-sample`      | Number of samples used for each evaluation run                     | `200`            |
| `--vocab-size`       | Vocabulary size for tokenizer and model                            | `10000`          |
| `--batch-size`       | Batch size for training and evaluation                             | `1`              |
| `--lr`               | Learning rate                                                      | `1e-3`           |
| `--text-column_name` | Name of the text column in the HF dataset                          | `"text"`         |
| `--hf-data`          | Name or path of Hugging Face dataset                               | `None`           |
| `--device`           | Device to run training/inference (`cuda`, `cpu`, etc.)             | `"cuda"`         |
| `--prompt`           | Prompt used for inference                                          | `"Merhaba ben "` |
| `--max-new-tokens`   | Max tokens to generate during inference                            | `256`            |
| `--train-tokenizer`  | Train a new tokenizer from scratch                                 | `False`          |
| `--model-info`       | Print model configuration before training                          | `False`          |
| `--inference`        | Run in inference mode (skip training)                              | `False`          |
| `--force`            | Force training from scratch (do not load checkpoint)               | `False`          |

> **Note:** The `--force` flag disables checkpoint loading and triggers a full reset of the training environment.  
> A new output directory will be created, and all assets — including tokenizer, model config, and checkpoints — will be regenerated from scratch.

> **Note:** The inference functionality (`--inference`) has not been fully tested and may not produce stable outputs in its current state.

> **Note:** The `--train-tokenizer` flag forces tokenizer training even if a tokenizer already exists.  
> The newly trained tokenizer will overwrite any existing tokenizer files in the specified output directory.

## Data Notes

Currently, only Hugging Face datasets with a structure similar to the example below are supported:

| category | text             |
| -------- | ---------------- |
| siyaset  | "örnek cümle..." |
| ...      | ...              |

The dataset must contain a `"text"` field, which will be used for tokenization and training.

### Tokenizer Training

If `--train-tokenizer` is enabled:

- A plain `.txt` file will be generated using the specified Hugging Face dataset.
- This text file will be used to train a SentencePiece tokenizer from scratch.
- Tokenizer training uses the column name defined by `--text-column_name` (default: `"text"`).
- Tokenizer input text is lowercased and ensured to end with a period.

Tokenizer data preparation is handled by:

```python
prepare_tokenizer_data(hf_data_name, text_column_name, output_path)
```

### Pretraining Dataset Construction

The pretraining dataset is prepared using the same HF dataset (not the `.txt` file).  
The text is tokenized using the trained tokenizer and split into chunks.

Each chunk is:

- `context_len` tokens long (default: 512)
- Preceded by a BOS token
- Paired with a right-shifted target sequence

Pretraining dataset preparation is handled by:

```python
prepare_train_data(hf_data_name, tokenizer, output_path, batch_size)
```

During dataset preparation, a sample.txt file is generated under output_path/, containing examples of input and target token IDs and their decoded text.
