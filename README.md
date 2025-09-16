# Lm-pretrain

**WORK IN PROGRESS**

---

## Sample Usage

```
pip install -r requirements.txt
```

```bash
python main.py \
  --model qwen3 \
  --hf-data savasy/ttc4900 \
  --vocab-size 5_000 \
  --batch-size 20 \
  --model-info \
  --epoch 2 \
  --max-train-step 150 \
  --max-eval-step 50
```
