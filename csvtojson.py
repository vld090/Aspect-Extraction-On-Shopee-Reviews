import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    # Dictionary to store grouped reviews
    reviews_dict = {}
    
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader) # Skip header row
        
        for row in csv_reader:
            review_no, review_text, extraction = row
            
            if review_no == "Review no." or review_no == "review no.":
                continue

            if review_no not in reviews_dict:
                reviews_dict[review_no] = {
                    "review_no": review_no,
                    "review": review_text,
                    "extractions": []
                }
            
            # Add extraction with tokens
            reviews_dict[review_no]["extractions"].append({
                "phrase": extraction,
                "tokens_normalized": extraction.split()
            })
    
    # Convert dictionary to list format
    final_output = list(reviews_dict.values())
    
    # Write to JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(final_output, json_file, indent=2, ensure_ascii=False)

# Example usage
csv_file_path = 'product_valid.csv'  # Replace with your input CSV file path
json_file_path = 'ground_truth.json'  # Replace with your desired output JSON file path
csv_to_json(csv_file_path, json_file_path)