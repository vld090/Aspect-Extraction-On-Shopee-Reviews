import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pathlib import Path
from .prompts import get_prompt, TaskType

load_dotenv()

def generate(task_type: TaskType, example_file=None, test_file=None, output_file=None):
    """Generate content using Gemini API"""
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # File paths setup
    base_path = Path(__file__).parent.parent
    fp_keywords = base_path / 'TMKeywords.pdf'
    fp_codebook = base_path / 'GeneralCodebook.pdf'

    # Use provided files or defaults
    fp_example = example_file if example_file else (base_path / 'multi-gen-examples.csv')
    fp_test = test_file if test_file else (base_path / 'review-only-test.csv')
    output = output_file if output_file else 'test_result.csv'

    # Read files
    with open(fp_codebook, 'rb') as f:
        codebook_data = f.read()
    with open(fp_keywords, 'rb') as f:
        keywords_data = f.read()
    with open(fp_example, 'r', encoding='utf-8') as f:
        example_data = f.read()
    with open(fp_test, 'r', encoding='utf-8') as f:
        test_data = f.read()
    
    # Create parts list
    parts = [
        types.Part(text=get_prompt(task_type)),
        types.Part(
            inline_data=types.Blob(
                mime_type="application/pdf",
                data=base64.b64encode(codebook_data).decode('utf-8')
            )
        ),
        types.Part(
            inline_data=types.Blob(
                mime_type="application/pdf",
                data=base64.b64encode(keywords_data).decode('utf-8')
            )
        ),
        types.Part(text=example_data),
        types.Part(text=test_data)
    ]

    # Generate content
    model = "gemini-2.0-flash"
    contents = [types.Content(role="user", parts=parts)]
    config = types.GenerateContentConfig(response_mime_type="text/plain")

    full_response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        full_response += chunk.text
        print(chunk.text, end="")
    
    # Save response
    with open(output, 'w', encoding='utf-8') as f:
        f.write(full_response)
    
    print(f"\n\nResponse saved to '{output}'")

if __name__ == "__main__":
    generate("identification")  # or generate("extraction")
