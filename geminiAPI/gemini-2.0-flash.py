# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import pathlib

load_dotenv()

def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    #retrieve file paths
    # fp_keywords = pathlib.Path('../keywords.pdf')
    fp_codebook = pathlib.Path('../GeneralCodebook.pdf')
    fp_train = pathlib.Path('../multi-gen-examples.csv')
    # fp_test = pathlib.Path('../test.csv')
    fp_test = pathlib.Path('../review-only-test.csv')

    model = "gemini-2.0-flash"
    
    # Read file contents
    with open(fp_codebook, 'rb') as f:
        codebook_data = f.read()
    # with open(fp_keywords, 'rb') as f:
    #     keywords_data = f.read()
    with open(fp_train, 'r', encoding='utf-8') as f:
        train_data = f.read()
    with open(fp_test, 'r', encoding='utf-8') as f:
        test_data = f.read()
    
    # Create parts list
    parts = []
    
    # Add the main prompt text
    parts.append(types.Part(text="You are an expert ABSA (Aspect-Based Sentiment Analysis) annotator specializing in multilingual "
    "and code-switched data. Your task is to annotate aspects in Taglish reviews. " \
    
        "Input Files: " \
        "multi-gen-examples.csv: Contains annotated Taglish reviews (use as reference examples) " \
        "review-only-test.csv: Contains new Taglish reviews to annotate " \
        "GeneralCodebook.pdf: Contains aspect definitions " \
        "Step-by-step process: " \
        "1. Analyze multi-gen-examples.csv to understand the annotation patterns and consistency. " \
        "2. Study GeneralCodebook.pdf to learn Aspect definitions and categories. " \
        "3. Annotate each review in review-only-test.csv. " \
        "4. Identify the General Aspect. If general aspect is found tag with 1, else 0" \
        "5. After completing the review annotations in review-only-test.csv, include an explanation justifying each tagging decision." \
        "Output format: CSV with exactly these columns only: " \
        "1. Review, " \
        "2. product, " \
        "3. delivery, " \
        "4. price, " \
        "5. service, " \
        "6. explanation" \
        "Output only the newly annotated data, review-only-test.csv, in CSV format, no extra text." \
        "Rules: " \
        "1. Each review can only have a maximum of 4 aspect tag " \
        "2. Aspect categories: Use only the aspects defined in GeneralCodebook.pdf " \
        "3. Default to 0 for reviews with no aspect/s found " \
        "4. Context is key: Always prioritize the meaning and context of the sentence over mere keyword matching. " \
        "5. Use Keywords only as guide: Keywords are guides, not hard rules. Do not assign an aspect solely because a keyword appears if context suggests otherwise " \
        "Strict Requirements: " \
        "1. Output ONLY the CSV data with headers. " \
        "3. Maintain consistent annotation standards throughout. " \
        "4. Process ALL reviews in review-only-test.csv." \
        
    ))
    
    # Add PDF files as inline data
    parts.append(types.Part(
        inline_data=types.Blob(
            mime_type="application/pdf",
            data=base64.b64encode(codebook_data).decode('utf-8')
        )
    ))
    
    # parts.append(types.Part(
    #     inline_data=types.Blob(
    #         mime_type="application/pdf",
    #         data=base64.b64encode(keywords_data).decode('utf-8')
    #     )
    # ))
    
    # Add CSV files as text
    parts.append(types.Part(text=train_data))
    parts.append(types.Part(text=test_data))
    
    contents = [
        types.Content(
            role="user",
            parts=parts,
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    # Collect the full response
    full_response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        full_response += chunk.text
        print(chunk.text, end="")
    
    # Save the response to a CSV file
    with open('annotated_test_data.csv', 'w', encoding='utf-8') as f:
        f.write(full_response)
    
    print(f"\n\nResponse saved to 'annotated_test_data.csv'")

if __name__ == "__main__":
    generate()
