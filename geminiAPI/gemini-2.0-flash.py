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
    fp_codebook = pathlib.Path('../codebook.pdf')
    fp_keywords = pathlib.Path('../keywords.pdf')
    fp_train = pathlib.Path('../train-data.csv')
    fp_test = pathlib.Path('../test-data.csv')

    model = "gemini-2.0-flash"
    
    # Read file contents
    with open(fp_codebook, 'rb') as f:
        codebook_data = f.read()
    with open(fp_keywords, 'rb') as f:
        keywords_data = f.read()
    with open(fp_train, 'r', encoding='utf-8') as f:
        train_data = f.read()
    with open(fp_test, 'r', encoding='utf-8') as f:
        test_data = f.read()
    
    # Create parts list
    parts = []
    
    # Add the main prompt text
    parts.append(types.Part(text="You are an expert in ABSA for multilingual and code-switched data. " \
        "Given the attached files: train-data.csv (annotated Taglish reviews), keywords.pdf (list of keywords for explicit aspects) "
        "and codebook.pdf (aspect definitions), annotate the new Taglish reviews in test-data.csv in the same format as " \
        "train-data.csv: review no., token, BIO Tag (Explicit), Aspect Tag (Explicit), " \
        "Final Tag (Explicit), General Aspect, Specific Aspect Category (Explicit), " \
        "Aspect Tag (Implicit), Final Tag (Implicit), General Aspect, " \
        "Specific Aspect Category (Implicit). Output only the annotations in CSV format, " \
        "no explanations or extra text." \
        "Rules: " \
        "1. Follow the definitions in codebook.pdf strictly when identifying and categorizing aspects. " \
        "2. The goal is to produce high-quality, consistent annotations to train an ABSA model. " \
        "3. If unsure about an aspect, default to 'O' for BIO tagging and leave aspect categories blank. " \
        "4. a review can be both explicit and implicit, but each token can only be either explicit or implicit and have one aspect tag. " \
        "5. Use keywords.pdf only as a guide; always rely on context and the definitions in codebook.pdf to determine the aspect."
    ))
    
    # Add PDF files as inline data
    parts.append(types.Part(
        inline_data=types.Blob(
            mime_type="application/pdf",
            data=base64.b64encode(codebook_data).decode('utf-8')
        )
    ))
    
    parts.append(types.Part(
        inline_data=types.Blob(
            mime_type="application/pdf",
            data=base64.b64encode(keywords_data).decode('utf-8')
        )
    ))
    
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
