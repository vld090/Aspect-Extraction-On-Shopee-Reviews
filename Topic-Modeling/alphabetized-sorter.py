##### Don't mind this. I just use this to sort the stopwords-new.txt in alphabetical order because
# I add new stopwords from time-to-time.

def sort_and_deduplicate_words(input_file, output_file=None):
    # Read the words from the file
    with open(input_file, 'r', encoding='utf-8') as file:
        words = file.read().splitlines()

    # Remove duplicates by converting to a set, then sort alphabetically
    unique_sorted_words = sorted(set(words))

    # Determine output file
    if output_file is None:
        output_file = input_file  # Overwrite original file

    # Write the sorted, unique words back to the file
    with open(output_file, 'w', encoding='utf-8') as file:
        for word in unique_sorted_words:
            file.write(word + '\n')

    print(f"Processed {len(words)} words, reduced to {len(unique_sorted_words)} unique words.")
    print(f"Output written to: {output_file}")

# Example usage:
sort_and_deduplicate_words('stopwords-new.txt')  # Replace 'words.txt' with your actual file name
