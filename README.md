<p align="center">
  <img src="assets/logo.svg" alt="LM-Pretrain Logo" width="600"/>
</p>

# LM-Pretrain: Pretraining and Instruction Tuning for Language Models


---

## Project Status

This project is currently under **active development** and should be considered **experimental**.
The goal is to develop a fully modular and extensible framework for **pretraining**, **instruction tuning**, and **RLHF** on large language models, enabling easy architectural customization.

I’m actively working on improving the codebase and would appreciate any feedback or suggestions.

Feel free to reach out:

- **Email**: [canobingol2@gmail.com](mailto:canobingol2@gmail.com)
- **LinkedIn**: [linkedin.com/in/canbingöl](https://www.linkedin.com/in/canbing%C3%B6l)

---

> [!WARNING]
> In this repository, several components are hard-coded.
> Make sure to review and adjust them before adapting the code to your own setup.


## Model Design Notes

- The configuration files in `config/` or passed via CLI are **not exact replicas** of official SOTA model configs.
- Hugging Face models are used for the tokenizer (Also you can train your tokenizer with train_tokenizer.py).
- You can modify the model configuration (number of layers, hidden size, vocab size, etc.) via CLI or config files.

## Installation

```bash
git clone https://github.com/canbingol/lm-pretrain.git
cd lm-pretrain
pip install -r requirements.txt

```

## Pre-Training Data Directory Structure

For efficiency, pretraining tokens are precomputed using the tokenizer and saved as binary files.

```
data/
├── pretrain/
│   ├── tokenizer_name/
│   |   ├── train_00.bin
|   |   |── train_01.bin
|   |   |── ....
|   |   |── validation_00.bin
|   |   |── ....
```


## Output Directory Structure

When training starts, an output directory is created at `output/<model_name>` containing the following files:

```
output/
├── decoder_pretrain_60M/
│   ├── decoder_only_best_model.pt      # Best checkpoint (lowest validation loss)
│   ├── train_loss.txt                  # Train loss (each step)
│   ├── validaiton_loss.txt             # Train loss (each step)



```

> [!WARNING]
> Pipeline no longer train a tokenizer; instead, it using a pretrained tokenizer from Hugging Face.

> If the `--force` flag is used, a new directory like `output/<model_name>_force_current_time/` is created.
> In this case, **all assets including checkpoints will be regenerated**, and no previously saved files will be reused.

## Arguments

### Yaml File Structure
```
config_yamls/
├── inference_config.yaml
├── train_config.yaml
```

> [!NOTE]
> Both `--config` and `--training-type` are provided inside `run.sh`, while all other arguments should be defined in the corresponding YAML configuration files.



| Argument | Type | Description | Default |
|---------|------|-------------|---------|
| `--config` | `str` | Path to the YAML configuration file | `None` |
| `--training-type` | `str` | Training mode to run (no validation enforced in code) | `None` |
| `--model` | `str` | Model architecture to use (choices: `decoder`) | `None` |
| `--epoch` | `int` | Number of training epochs | `None` |
| `--world-size` | `int` | Number of processes (GPUs) used in distributed training | `None` |
| `--world_size` | `int` | **Duplicate argument** (underscore version of `--world-size`) | `None` |
| `--training-steps` | `int` | Maximum number of training steps | `None` |
| `--eval-steps` | `int` | Evaluation interval (in steps) | `None` |
| `--eval-sample` | `int` | Number of samples used during each evaluation phase | `None` |
| `--vocab-size` | `int` | Vocabulary size used by tokenizer and model | `None` |
| `--batch-size` | `int` | Batch size for training and evaluation | `None` |
| `--lr` | `float` | Learning rate | `None` |
| `--max-seq-len` | `int` | Maximum sequence length per input | `None` |
| `--text-column-name` | `str` | Column name containing the input text in the dataset | `None` |
| `--pre-training-hf-data` | `str` | Hugging Face dataset name or path used for pretraining | `None` |
| `--it-hf-data` | `str` | Hugging Face dataset name or path used for instruction tuning | `None` |
| `--hf-tokenizer` | `str` | Path or name of the Hugging Face tokenizer to load | `None` |
| `--saved-token-path` | `str` | Directory to store processed tokenized binary files | `None` |
| `--checkpoint` | `str` | Path to the model checkpoint file | `None` |
| `--max-new-tokens` | `int` | Maximum number of tokens to generate during inference | `None` |
| `--prompt` | `str` | Input prompt text used during inference | `None` |
| `--model-info` | `flag` | Print model configuration before training starts | `False` |
| `--train-tokenizer` | `flag` | Train a new tokenizer from scratch | `False` |
| `--inference` | `flag` | Run inference mode (skip training) | `False` |
| `--force` | `flag` | Force training from scratch (ignore checkpoints) | `False` |
| `--shuffle` | `flag` | Shuffle dataset during loading | `False` |
| `--drop-last` | `flag` | Drop the last incomplete batch (**always True due to code**) | `True` |
| `--num-workers` | `int` | Number of DataLoader workers | `0` |
| `--pin-memory` | `flag` | Enable pinned memory (**always True due to code**) | `True` |
| `--single-file` | `str` | Path to a single dataset file (debug/manual runs) | `None` |



## Sample Usage

```bash
cd lm-pretrain
sh run.sh pre-train
```

```bash
cd lm-pretrain
sh run.sh instruction-tuning
```

```bash
cd lm-pretrain
sh run.sh inference
```

> [!TIP]
> **Accepted `type` values:**
> `pretrain` | `pre_train` | `pre-train` |
> `instruction-tuning` | `instruction_tuning` | `it` | `inference`

### Choosing Training Type

The training type is determined by the `--training-type` argument provided in `run.sh`.
Depending on the selected type, the repository automatically adjusts how the dataset is prepared:

- **`pre-train`** → Uses the pretraining data pipeline (`prepare_train_data`), which processes plain text datasets into token sequences.
- **`instruction-tuning`** → Uses the instruction tuning data pipeline (`prepare_it_data`), which constructs instruction–response pairs and applies target masking.

Apart from the dataset preparation logic, **the rest of the training pipeline remains identical** —
including model forward passes, loss computation, optimization, and checkpoint saving.


## Data Notes

Currently, the repository supports two dataset types:

### Pretraining Datasets
Pretraining requires Hugging Face datasets that follow a simple text-based structure, for example:

| text             |
| ---------------- |
| "örnek cümle..." |
| ...              |

The dataset **must contain a `"text"` field**, which is used for tokenization and language model training.

---

### Instruction Tuning Datasets
For instruction tuning (IT), the dataset should contain **three columns** representing the supervised fine-tuning format:

| talimat (instruction) | giriş (input) | çıktı (output) |
| ---------------------- | -------------- | --------------- |
| "Bir e-postayı özetle." | "E-posta içeriği..." | "Kısa özet:" |

The `"talimat"`, `"giriş"`, and `"çıktı"` fields are used to construct instruction–response pairs during supervised fine-tuning.

---

> [!NOTE]
> Both dataset types must be accessible via Hugging Face Datasets or stored locally in the same column format.

### Tokenizer Training
> [!WARNING]
> In this version, the `--train-tokenizer` flag is not supported.
> To enable it, you need to modify line **109** in `main.py`, along with a few additional adjustments in related components.


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

Instead of tokenizing the dataset during every training run,
you can **precompute and store tokenized data as `.bin` files**, which is the recommended approach.
This significantly reduces preprocessing time and avoids redundant tokenization overhead.

Example usage:

```python
from transformers import AutoTokenizer
from data_prepare import create_tokens_file

tokenizer_path = "vngrs-ai/Kumru-2B"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

pre_train_dataset = "canbingol/vngrs-web-corpus-200k"

create_tokens_file(
    hf_dataset=pre_train_dataset,
    tokenizer=tokenizer,
    base_dir="./data/pretrain",
    tokenizer_name="kumru_tokenizer"
)
```
> [!NOTE]
> - During training, the script automatically checks the `saved_token_path` specified in the YAML configuration file.
>   If tokenized files already exist in that directory, they are loaded directly.
>   Otherwise, the code automatically generates and saves new tokenized `.bin` files.
> - Update the `pre_train_dataset` variable in **`data_prepare.py`** to use your own Hugging Face dataset.
> - The generated `.bin` files are stored under
>   `data/pretrain/<tokenizer_name>/` (for example, `data/pretrain/kumru_tokenizer/`).
> - To change the output directory or tokenizer name, modify the `base_dir` or `tokenizer_name` arguments in the `create_tokens_file()` function.

## Future Work

The upcoming development roadmap focuses on improving code stability, modularity, and extensibility across all components of the repository. Planned enhancements include:

- **Refactoring the codebase** to eliminate hard-coded logic and improve overall maintainability.
- **Stabilizing the training pipelines** for both pretraining and instruction tuning to ensure reproducibility and efficiency.
- **Optimizing model implementations** for better computational performance and cleaner architectural design.
- **Integrating RLHF (Reinforcement Learning from Human Feedback)** as an additional training stage for alignment.
- **Expanding documentation**, including in-depth explanations of the training process, configuration options, and architectural components.

The long-term goal is to turn this project into a **fully stable, modular, and extensible framework** for building, fine-tuning, and aligning large language models.
