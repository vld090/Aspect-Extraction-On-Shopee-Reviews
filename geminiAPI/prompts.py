from enum import Enum

class TaskType(Enum):
    IDENTIFICATION = "identification"
    EXTRACTION = "extraction"

def get_prompt(task_type: TaskType) -> str:
    """Get the appropriate prompt based on task type"""
    if task_type == TaskType.IDENTIFICATION:
        return """You are an expert ABSA (Aspect-Based Sentiment Analysis) annotator specializing in multilingual 
and code-switched data. Your task is to annotate aspects in Taglish reviews.

Input Files:
multi-gen-examples.csv: Contains annotated Taglish reviews (use as reference examples)
review-only-test.csv: Contains new Taglish reviews to annotate
GeneralCodebook.pdf: Contains aspect definitions
TMKeywords.pdf: Contains keywords for explicit aspects only

Step-by-step process:
1. Analyze multi-gen-examples.csv to understand the annotation patterns and consistency.
2. Study GeneralCodebook.pdf to learn Aspect definitions and categories.
3. Annotate each review in review-only-test.csv.
4. Identify the General Aspect. If general aspect is found tag with 1, else 0

Output format: CSV with exactly these columns only:
1. Review,
2. product,
3. delivery,
4. price,
5. service
Output only the newly annotated data in CSV format, no extra text.

Rules:
1. Each review can only have a maximum of 4 aspect tag
2. Aspect categories: Use only the aspects defined in GeneralCodebook.pdf
3. Default to 0 for reviews with no aspect/s found
4. Context is key: Always prioritize the meaning and context of the sentence over mere keyword matching.
5. Use Keywords only as guide: Keywords are guides, not hard rules. Do not assign an aspect solely because a keyword appears if context suggests otherwise

Strict Requirements: 
1. Output ONLY the CSV data in review-only-test.csv. 
2. Do not include the data from multi-gen-examples.csv
3. Maintain consistent annotation standards throughout.
4. Process ALL reviews in review-only-test.csv.
"""

    elif task_type == TaskType.EXTRACTION:
        return """You are an expert ABSA annotator. Your task is to perform Aspect Phrase Extraction on Taglish reviews.

Input Files:
product-examples.json: Contains annotated reviews
product-test.csv: Contains new reviews to process
GeneralCodebook.pdf: Contains aspect definitions
TMKeywords.pdf: Contains keywords for explicit aspects only

Additional examples:
{
    "review_no": "41",
    "review": "good masikip na kontingbut good quality maganda ang fit sa paa",
    "extractions": [
      {
        "phrase": "masikip na kontingbut",
        "tokens_normalized": [
          "masikip",
          "na",
          "kontingbut"
        ]
      },
      {
        "phrase": "good quality",
        "tokens_normalized": [
          "good",
          "quality"
        ]
      },
      {
        "phrase": "maganda ang fit sa paa",
        "tokens_normalized": [
          "maganda",
          "ang",
          "fit",
          "sa",
          "paa"
        ]
      }
    ]
  },
  {
    "review_no": "42",
    "review": "akala ko maganda sana sa shop nlng ako na una ko inorderan bumili di mn lng nag iiba ng kulay pag nasa outdoor  anti rad lng to tapos malaki pa",
    "extractions": [
      {
        "phrase": "di mn lng nag iiba ng kulay pag nasa outdoor anti rad",
        "tokens_normalized": [
          "di",
          "mn",
          "lng",
          "nag",
          "iiba",
          "ng",
          "kulay",
          "pag",
          "nasa",
          "outdoor",
          "anti",
          "rad"
        ]
      }
    ]
  },
  {
    "review_no": "43",
    "review": "salamat po seller good quality pa din pangalawang beses ko na itong bili sa item na to hindi ito akin pinabili lang nakakabadtrip lang po ang ninja van bukod sa sakto sa time frame talaga na delay pa ang delivery nila akoy naiinis dahil hindi ko napansin na sila pala",
    "extractions": [
      {
        "phrase": "good quality",
        "tokens_normalized": [
          "good",
          "quality"
        ]
      }
    ]
  }

Step-by-step process:
1. For each review in product-test.csv, analyze the product-examples.json for annotation style, then study the Codebook and Keywords.
2. Identify and extract **ALL** phrases/keywords (including context modifiers) that refer to **PRODUCT aspects**. This includes both **explicit** and **implicit** aspects.
3. For each extracted phrase, provide a detailed explanation of the aspect category it belongs to (referencing the Codebook) and whether the extraction is **Explicit** (directly named) or **Implicit** (inferred).
4. Maintain the same JSON structure as shown in the examples above.

Output format: JSON with exactly the same format as the examples above
1. review no.,
2. review,
3. extractions: list of objects with keys "phrase" and "tokens_normalized"

Rules:
1. Extract the exact text span with context
2. Avoid Concatenated Words
3. Each review can have multiple extracted phrases
4. Context is key
5. Use Keywords only as guide

Output only the JSON data."""
    else:
        raise ValueError(f"Unknown task type: {task_type}")
