from transformers import AutoTokenizer
from models.decoder_model import DecoderCausalLM

model_path = "canbingol/exp6_flash_attn_1epoch_lr1e4_500k_vngr_corpus_10layers"

model = DecoderCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

input_ids = tokenizer.encode("selam canım", return_tensors="pt")

out_tokens = model.generate(input_ids)
generated_text = tokenizer.decode(out_tokens.flatten())

print(generated_text)