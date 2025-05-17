import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def remove_repetitive_symbols(text):
    text = re.sub(r'[!?.]{2,}', '', text)
    text = re.sub(r'"{2,}', '', text)
    return text

def bpe_tokenization (text):
    return tokenizer.tokenize(text)

# Test
# if __name__ == "__main__":
#     sample = 'goo very light lang..... ganda naman pero mas maganda ako!!!! hahahahahabababababababababababahahahahahahahahaha'
#     print("Original:", sample)
#     print("Cleaned:", remove_repetitive_symbols(sample))
#     print("Tokens:", process_data(sample))