from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
import pyarrow


def read_dataset(split="train"):
    return pd.read_parquet(fr"D:\Programming\Python\Tokenization\Data\{split}-00000-of-00001.parquet")


if __name__ == "__main__":
    # Define special tokens
    unk_token = "[UNK]"
    pad_token = "[PAD]"
    mask_token = "[MASK]"
    sos_token = "<sos>"
    eos_token = "<eos>"
    special_tokens = [unk_token, pad_token, sos_token, eos_token]

    en_tokenizer = Tokenizer(BPE(unk_token=unk_token))
    en_tokenizer.pre_tokenizer = Whitespace()
    en_trainer = BpeTrainer(special_tokens=special_tokens)
    en_tokenizer.train_from_iterator(read_dataset('train')['en'].tolist(), trainer=en_trainer)

    en_tokenizer.save("bpe.json")