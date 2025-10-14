import csv
import json
import pandas as pd

def csv_to_json(csv_file_path, json_file_path):
    # Dictionary to store grouped reviews
    reviews_dict = {}
    
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        # next(csv_reader) # Skip header row
        
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
# csv_file_path = 'aspect_extraction/service-examples.csv'  # Replace with your input CSV file path
# json_file_path = 'aspect_extraction/service-examples.json'  # Replace with your desired output JSON file path
# csv_to_json(csv_file_path, json_file_path)

def convert_xlsx_to_csv(xlsx_file: str, csv_file: str):
    """Convert Excel file to CSV"""
    df = pd.read_excel(xlsx_file)
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"Converted {xlsx_file} to {csv_file}")

for i in range(1, 84):
    xlsx_path = f'annotations/BTK_annotations/specific_aspect_identification/product_identify/{i}.xlsx'
    csv_path = f'annotations/BTK_annotations/specific_aspect_identification/product_identify/{i}.csv'
    convert_xlsx_to_csv(xlsx_path, csv_path)