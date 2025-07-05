#
# prepare_data.py
#
# Now splits our annotated CSV into separate training and testing files (80/20 split).
#

import csv
import json
import random
from collections import defaultdict

# --- CONFIG ---
INPUT_CSV = '001-050 _ Thesis Annotation Sheet - Trial.csv'
TRAIN_FILE_OUT = 'training_data.jsonl'
TEST_FILE_OUT = 'testing_data.jsonl'
TRAIN_SPLIT_RATIO = 0.8 # 80% for training, 20% for testing.

def create_training_and_testing_data():
    """
    Processes the CSV and splits the data into two separate JSONL files.
    """
    print(f"Reading from {INPUT_CSV}...")
    reviews = defaultdict(list)
    with open(INPUT_CSV, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Review #'):
                reviews[row['Review #']].append(row)

    # Convert defaultdict to a list and shuffle it to ensure random distribution.
    review_items = list(reviews.items())
    random.shuffle(review_items)

    # Calculate the split index.
    split_idx = int(len(review_items) * TRAIN_SPLIT_RATIO)
    train_reviews = review_items[:split_idx]
    test_reviews = review_items[split_idx:]

    print(f"Total reviews: {len(review_items)}. Splitting into {len(train_reviews)} training and {len(test_reviews)} testing examples.")

    # Process both training and testing sets.
    _write_jsonl_file(train_reviews, TRAIN_FILE_OUT)
    _write_jsonl_file(test_reviews, TEST_FILE_OUT)

    print(f"\nDone! Data successfully split into {TRAIN_FILE_OUT} and {TEST_FILE_OUT}.")

def _write_jsonl_file(review_set, output_filename):
    """Helper function to process and write a set of reviews to a JSONL file."""
    with open(output_filename, 'w', encoding='utf-8') as f:
        for review_id, rows in review_set:
            full_review_text = " ".join([row['Token'] for row in rows])

            explicit_aspects = []
            implicit_aspects = set()

            for row in rows:
                if row['BIO Tag (For Explicit Aspects)'] in ('B', 'I'):
                    tag = f"{row['BIO Tag (For Explicit Aspects)']} - {row['Aspect Tag (For Explicit Aspects)']}"
                    explicit_aspects.append({"token": row['Token'], "tag": tag})

                if row['Final Tag (For Implicit Aspects)']:
                    implicit_tag = row['Final Tag (For Implicit Aspects)'].strip().split(' - ')[-1]
                    if implicit_tag:
                        implicit_aspects.add(implicit_tag)

            instruction = (
                "Analyze the following customer review to identify all explicit and implicit "
                "product or service aspects. Return the answer in JSON format with two keys: "
                "'explicit_aspects' and 'implicit_aspects'.\n\n"
                f"Review: \"{full_review_text}\""
            )

            response_json = {
                "explicit_aspects": explicit_aspects,
                "implicit_aspects": sorted(list(implicit_aspects))
            }

            record = {
                "instruction": instruction,
                "response": json.dumps(response_json, indent=2)
            }
            f.write(json.dumps(record) + '\n')

if __name__ == '__main__':
    create_training_and_testing_data()