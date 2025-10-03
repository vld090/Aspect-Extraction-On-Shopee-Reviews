import re

def remove_symbols(text):
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return text

def word_tokens(text):
    text = re.split(r"\s", text)
    return text

# Test
# if __name__ == "__main__":
#     sample = 'goo very light lang..... ganda naman pero mas maganda ako!!!! hahahahahabababababababababababahahahahahahahahaha'
#     print("Original:", sample)
#     print("Cleaned:", remove_repetitive_symbols(sample))
#     print("Tokens:", process_data(sample))