import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("jcblaise/roberta-tagalog-base")

def remove_repetitive_symbols(text):
    text = re.sub(r'[!?.]{2,}', '', text)
    text = re.sub(r'"{2,}', '', text)
    return text

def is_gibberish (text):

    return False

def remove_gibberish (text):
    text_list = text.split(" ")
    processed_list = []

    for i in text_list:
        if is_gibberish(i) == False:
            processed_list.append(i)
    return ' '.join(processed_list)

def is_noninformative_review (text):

    return False

def bpe_tokenization (text):
    return tokenizer.tokenize(text)

def process_data (text):
    text = remove_repetitive_symbols(text)
    
    if is_gibberish(text) or is_noninformative_review(text):
        return None
    
    tokens = bpe_tokenization(text)
    return tokens