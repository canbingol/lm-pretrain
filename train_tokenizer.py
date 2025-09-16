import os
import sentencepiece as spm

class TrainTokenizer:
    def __init__(self, vocab_size, data_path,OUTPUT_PATH):
        model_prefix = f"{OUTPUT_PATH}/{OUTPUT_PATH}.tokenizer"
        spm.SentencePieceTrainer.train(
            input=data_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size, 
            model_type="bpe",   
            character_coverage=1.0,
            pad_id=0,
            unk_id=1, 
            bos_id=2,
            eos_id=3,
            byte_fallback=True
        )

        self.model_path = model_prefix + ".model"
        self.vocab_path = model_prefix + ".vocab"

    def get_model_path(self):
        return self.model_path
    def get_vocab_path(self):
        return self.vocab_path