import csv
import json
from collections import defaultdict
import os

# --- Configuration ---
# Updated to match your exact filename
INPUT_CSV = '001-050 _ Thesis Annotation Sheet - Trial.csv'
OUTPUT_JSONL = 'absa_finetuning_data.jsonl'
INSTRUCTION = (
    "You are an expert in Aspect-Based Sentiment Analysis. Your task is to identify "
    "explicit and implicit aspects from the given user review. "
    "Explicit aspects are terms mentioned directly in the text. "
    "Implicit aspects are categories that are implied but not explicitly stated. "
    "Extract all explicit aspect terms and all implicit aspect categories. "
    "Present the output in a JSON format with two keys: 'explicit_aspects' and 'implicit_aspects'."
)

def extract_aspects_from_review(review_data):
    """Extracts explicit and implicit aspects from a list of token annotations."""
    explicit_aspects = []
    implicit_aspects = set()
    current_explicit_aspect = ""

    for token, expl_bio, _, impl_aspect_cat in review_data:
        # Extract implicit aspects (unique categories per review)
        if impl_aspect_cat and impl_aspect_cat != 'O':
            implicit_aspects.add(impl_aspect_cat)

        # Extract explicit aspects based on BIO tagging
        if expl_bio == 'B':
            if current_explicit_aspect:
                explicit_aspects.append(current_explicit_aspect.strip())
            current_explicit_aspect = token
        elif expl_bio == 'I' and current_explicit_aspect:
            current_explicit_aspect += " " + token
        elif expl_bio == 'O':
            if current_explicit_aspect:
                explicit_aspects.append(current_explicit_aspect.strip())
                current_explicit_aspect = ""
                
    # Add the last aspect if it exists
    if current_explicit_aspect:
        explicit_aspects.append(current_explicit_aspect.strip())

    return list(set(explicit_aspects)), sorted(list(implicit_aspects))

def main():
    """Main function to process the CSV and generate JSONL."""
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Error: The file '{INPUT_CSV}' was not found in this directory.")
        return
        
    reviews = defaultdict(list)
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            review_id = row.get('Review #')
            token = row.get('Token')
            final_expl = row.get('Final Tag (For Explicit Aspects)', '')
            final_impl = row.get('Final Tag (For Implicit Aspects)', '')

            if not review_id or not token:
                continue

            expl_bio, expl_aspect_cat = (final_expl.split('-', 1) + [''])[:2] if '-' in final_expl else (final_expl, '')
            impl_aspect_cat = final_impl.split('-', 1)[-1] if '-' in final_impl else final_impl

            reviews[review_id].append((token, expl_bio.strip(), expl_aspect_cat.strip(), impl_aspect_cat.strip()))

    count = 0
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for _, review_data in reviews.items():
            review_text = " ".join([item[0] for item in review_data])
            explicit, implicit = extract_aspects_from_review(review_data)
            
            if not explicit and not implicit:
                continue

            output_data = {"explicit_aspects": explicit, "implicit_aspects": implicit}
            json_record = { "instruction": INSTRUCTION, "input": review_text, "output": json.dumps(output_data, indent=2) }
            f.write(json.dumps(json_record) + "\n")
            count += 1
            
    print(f"✅ Successfully created '{OUTPUT_JSONL}' with {count} records.")

if __name__ == '__main__':
    main()