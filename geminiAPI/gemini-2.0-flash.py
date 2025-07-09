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
    # fp_codebook = pathlib.Path('../codebook.pdf')
    # fp_keywords = pathlib.Path('../keywords.pdf')
    fp_codebook = pathlib.Path('../Codebook.pdf')
    fp_train = pathlib.Path('../few-shot-examples.csv')
    fp_test = pathlib.Path('../test.csv')

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
    "and code-switched data. Your task is to annotate explicit aspects in Taglish reviews using the BIO tagging scheme. " \
        "Input Files: " \
        "few-shot-examples.csv: Contains annotated Taglish reviews (use as reference examples) " \
        "test.csv: Contains new Taglish reviews to annotate " \
        "Codebook.pdf: Contains annotation rules, aspect definitions, and keywords " \
        "Step-by-step process: " \
        "1. Analyze few-shot-examples.csv to understand the annotation patterns and consistency. " \
        "2. Study Codebook.pdf to learn Aspect definitions and categories, BIO tagging rules, Keywords for explicit aspects, and Annotation guidelines. " \
        "3. For each review in test.csv, annotate token by token. " \
        "Output format: CSV with exactly these columns: review_no,token,bio_tag,general_aspect " \
        "Output the annotated data in CSV format, no explanations or extra text." \
        "Rules: " \
        "1. Only annotate EXPLICIT aspects (clearly mentioned in text) " \
        "2. Aspect categories: Use only the general categories defined in Codebook.pdf " \
        "3. Default to 'O' and blank aspect when no aspects is found " \
        "4. Context is key: Keywords are guides, but context and definitions determine the final aspect. " \
        "5. Aspects can be in the form of phrases. " \
        "Strict Requirements: " \
        "1. Output ONLY the CSV data with headers. " \
        "2. No explanations, commentary, or extra text. " \
        "3. Maintain consistent annotation standards throughout. " \
        "4. Process ALL reviews in test.csv." \
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
    with open('annotated_explicit_test_data.csv', 'w', encoding='utf-8') as f:
        f.write(full_response)
    
    print(f"\n\nResponse saved to 'annotated_explicit_test_data.csv'")

if __name__ == "__main__":
    generate()
