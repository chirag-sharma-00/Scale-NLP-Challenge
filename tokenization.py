import sys
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

def train_tokenizer(vocab_path, token_file):
    tokenizer = BertWordPieceTokenizer(clean_text=False, handle_chinese_chars=False, strip_accents=False, lowercase=False)
    tokenizer.train(files=vocab_path, vocab_size=5000, special_tokens=["[UNK]", "[BOS]", "[EOS]", "[CLS]", "[SEP]", "[PAD]"])
    vocab = tokenizer.get_vocab()
    with open(token_file, 'w') as tokenfile:
        for v in vocab.keys():
            tokenfile.write(v + "\n")
    tokenfile.close()
    return tokenizer

def main():
    train_tokenizer("vocab.txt", "tokens.txt")

if __name__ == "__main__":
    main()
