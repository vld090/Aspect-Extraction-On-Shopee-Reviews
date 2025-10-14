from enum import Enum

class TaskType(Enum):
    IDENTIFY_GENERAL = "identify_general"
    IDENTIFY_PRODUCT_SPECIFIC = "identify_product_specific"
    IDENTIFY_DELIVERY_SPECIFIC = "identify_delivery_specific"
    IDENTIFY_PRICE_SPECIFIC = "identify_price_specific"
    IDENTIFY_SERVICE_SPECIFIC = "identify_service_specific"
    EXTRACT_PRODUCT = "extract_product"
    EXTRACT_DELIVERY = "extract_delivery"
    EXTRACT_PRICE = "extract_price"
    EXTRACT_SERVICE = "extract_service"

def get_prompt(task_type: TaskType) -> str:
    """Get the appropriate prompt based on task type and carefully read and analyze the instructions"""
    if task_type == TaskType.IDENTIFY_GENERAL:
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
    elif task_type == TaskType.IDENTIFY_PRODUCT_SPECIFIC:
        return """You are an expert ABSA (Aspect-Based Sentiment Analysis) annotator specializing in multilingual
        and code-switched data. All the reviews you receive are tagged as **PRODUCT** reviews. 
        Your task is to annotate product **SPECIFIC** aspects in Taglish reviews.

              Input Files:
              example file: Contains annotated Taglish reviews (use as reference examples)
              test file: Contains new Taglish reviews to annotate
              CompleteCodebook.pdf: Contains aspect definitions
              TMKeywords.pdf: Contains keywords for explicit aspects only sorted by general aspect

              Step-by-step process:
              1. Analyze example file to understand the annotation patterns and consistency.
              2. Study CommpleteCodebook.pdf to learn Aspect definitions and categories.
              3. Annotate each review in test file.
              4. Identify the Specific Product Aspect. If the specific product aspect is found tag with 1, else 0
              5. Review newly annotated data and ensure all reviews have proper tags.

              Output format: CSV with exactly these columns only:
              1. Review,
              2. color,
              3. condition,
              4. correctness,
              5. durability,
              6. effectiveness,
              7. functionality,
              8. material,
              9. sensory,
              10. measurement,
              11. general
              Output only the newly annotated data in CSV format, no extra text.

              Rules:
              1. Each review can only have a maximum of 10 product specific aspect tag
              2. Aspect categories: Use only the aspects listed above and defined in CompleteCodebook.pdf under the Product Aspect section
              3. Default to 0 for reviews with no aspect/s found
              4. Context is key: Always prioritize the meaning and context of the sentence over mere keyword matching.
              5. Use Keywords only as guide: Keywords are guides, not hard rules. Do not assign an aspect solely because a keyword appears if context suggests otherwise

              Strict Requirements: 
              1. Output **ONLY** the CSV data in  the test csv. 
              2. **DO NOT INCLUDE** the data from examples csv.
              3. Maintain consistent annotation standards throughout.
              4. **Process ALL reviews** in test csv.
              5. DO NOT enclose output in markdowns (```csv```)
        """
    elif task_type == TaskType.IDENTIFY_DELIVERY_SPECIFIC:
        return """You are an expert ABSA (Aspect-Based Sentiment Analysis) annotator specializing in multilingual
        and code-switched data. All the reviews you receive are tagged as **DELIVERY** reviews. 
        Your task is to annotate delivery **SPECIFIC** aspects in Taglish reviews.

              Input Files:
              delSP-example file: Contains annotated Taglish reviews (use as reference examples)
              delSP-test file: Contains new Taglish reviews to annotate
              CompleteCodebook.pdf: Contains aspect definitions
              TMKeywords.pdf: Contains keywords for explicit aspects only sorted by general aspect

              Step-by-step process:
              1. Analyze delSP-example file to understand the annotation patterns and consistency.
              2. Study CommpleteCodebook.pdf to learn Aspect definitions and categories.
              3. Annotate each review in delSP-test file.
              4. Identify the Specific Delivery Aspect. If the specific delivery aspect is found tag with 1, else 0
              5. Review newly annotated data and ensure all reviews have proper tags.

              Output format: CSV with exactly these columns only:
              1. Review,
              2. condition,
              3. correctness,
              4. timeliness,
              5. general
              Output only the newly annotated data in CSV format, no extra text.

              Rules:
              1. Each review can only have a maximum of 4 delivery specific aspect tag
              2. Aspect categories: Use only the aspects listed above and defined in CompleteCodebook.pdf under the Delivery Aspect section
              3. Default to 0 for reviews with no aspect/s found
              4. Context is key: Always prioritize the meaning and context of the sentence over mere keyword matching.
              5. Use Keywords only as guide: Keywords are guides, not hard rules. Do not assign an aspect solely because a keyword appears if context suggests otherwise

              Strict Requirements: 
              1. Output **ONLY** the CSV data in  the delSP-test file csv. 
              2. **DO NOT INCLUDE** the data from delSP-examples in delSP-predicted csv.
              3. Maintain consistent annotation standards throughout.
              4. Process ALL reviews in delSP-test file csv.
              5. DO NOT enclose output in markdowns (```csv```)
        """
    elif task_type == TaskType.IDENTIFY_PRICE_SPECIFIC:
        return """You are an expert ABSA (Aspect-Based Sentiment Analysis) annotator specializing in multilingual
        and code-switched data. All the reviews you receive are tagged as **PRICE** reviews. 
        Your task is to annotate price **SPECIFIC** aspects in Taglish reviews.

              
              Input Files:
              example file: Contains annotated Taglish reviews (use as reference examples)
              test file: Contains new Taglish reviews to annotate
              CompleteCodebook.pdf: Contains aspect definitions
              TMKeywords.pdf: Contains keywords for explicit aspects only sorted by general aspect

              Step-by-step process:
              1. Analyze example file to understand the annotation patterns and consistency.
              2. Study CommpleteCodebook.pdf to learn Aspect definitions and categories.
              3. Annotate each review in test csv file.
              4. Identify the Specific Price Aspect. If the specific price aspect is found tag with 1, else 0

              Output format: CSV with exactly these columns only:
              1. Review,
              2. affordability,
              3. value_for_money,
              4. general
              Output only the newly annotated data in CSV format, no extra text.

              Rules:
              1. Each review can only have a maximum of 3 price specific aspect tag
              2. Aspect categories: Use only the aspects listed above and defined in CompleteCodebook.pdf under the Price Aspect section
              3. Default to 0 for reviews with no aspect/s found
              4. Context is key: Always prioritize the meaning and context of the sentence over mere keyword matching.
              5. Use Keywords only as guide: Keywords are guides, not hard rules. Do not assign an aspect solely because a keyword appears if context suggests otherwise

              Strict Requirements: 
              1. Output **ONLY** the CSV data in  the test csv. 
              2. **DO NOT INCLUDE** the data from examples csv.
              3. Maintain consistent annotation standards throughout.
              4. Process ALL reviews in test csv.
              5. DO NOT enclose output in markdowns (```csv```)
        """
    elif task_type == TaskType.IDENTIFY_SERVICE_SPECIFIC:
        return """You are an expert ABSA (Aspect-Based Sentiment Analysis) annotator specializing in multilingual
        and code-switched data. All the reviews you receive are tagged as **SERVICE** reviews. 
        Your task is to annotate service **SPECIFIC** aspects in Taglish reviews.

              Input Files:
              serSP-example file: Contains annotated Taglish reviews (use as reference examples)
              serSP-test file: Contains new Taglish reviews to annotate
              CompleteCodebook.pdf: Contains aspect definitions
              TMKeywords.pdf: Contains keywords for explicit aspects only sorted by general aspect

              Step-by-step process:
              1. Read and Analyze serSP-example file to understand the annotation patterns and consistency.
              2. Study CommpleteCodebook.pdf to learn Aspect definitions and categories.
              3. Annotate each review in serSP-test csv file.
              4. Identify the Specific Service Aspect. If the specific service aspect is found tag with 1, else 0.
              5. Review newly annotated data and ensure all reviews have proper tags.

              Output format: CSV with exactly these columns only:
              1. Review,
              2. handling,
              3. responsiveness,
              4. trustworthiness,
              5. general
              Output only the newly annotated data in CSV format, no extra text.

              Rules:
              1. Each review can only have a maximum of 4 service specific aspect tag
              2. Aspect categories: Use only the aspects listed above and defined in CompleteCodebook.pdf under the Service Aspect section
              3. Default to 0 for reviews with no aspect/s found
              4. Context is key: Always prioritize the meaning and context of the sentence over mere keyword matching.
              5. Use Keywords only as guide: Keywords are guides, not hard rules. Do not assign an aspect solely because a keyword appears if context suggests otherwise

              Strict Requirements: 
              1. Output **ONLY** the CSV data in the serSP-test csv. 
              2. DO NOT INCLUDE the data from serSP-examples csv in the output.
              3. Maintain consistent annotation standards throughout.
              4. Process ALL reviews in test csv.
              5. DO NOT enclose output in markdowns (```csv```)
        """
    
    elif task_type == TaskType.EXTRACT_PRODUCT:
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
        6. DO NOT enclose output in markdowns (```json```)

        Output only the JSON data."""
    
    elif task_type == TaskType.EXTRACT_DELIVERY:
        return """
        You are an expert ABSA annotator. Your task is to perform Aspect Phrase Extraction on Taglish reviews.

        Input Files:
        delivery-examples.json: Contains annotated reviews
        delivery-test.csv: Contains new reviews to process
        GeneralCodebook.pdf: Contains aspect definitions
        TMKeywords.pdf: Contains keywords for explicit aspects only

        Additional examples:
        {
          "review_no": "39",
          "review": "subrang tagal e ship nag order ako ng 9 9 ngayon ko palang na receive 924 mag 1 month bago ko marecive ang item ang pangit ng service di responsible si seller",
          "extractions": [
            {
              "phrase": "subrang tagal e ship",
              "tokens_normalized": [
                "subrang",
                "tagal",
                "e",
                "ship"
              ]
            },
            {
              "phrase": "nag order ako ng 9 9 ngayon ko palang na receive 924 mag 1 month bago ko marecive ang item",
              "tokens_normalized": [
                "nag",
                "order",
                "ako",
                "ng",
                "9",
                "9",
                "ngayon",
                "ko",
                "palang",
                "na",
                "receive",
                "924",
                "mag",
                "1",
                "month",
                "bago",
                "ko",
                "marecive",
                "ang",
                "item"
              ]
            }
          ]
        }
        {
          "review_no": "41",
          "review": "late dumating ung parcel and sira ung producthnd gmaganaon lng tpos wla na",
          "extractions": [
            {
              "phrase": "late dumating ung parcel",
              "tokens_normalized": [
                "late",
                "dumating",
                "ung",
                "parcel"
              ]
            }
          ]
        }

        Step-by-step process:
        1. For each review in delivery-test.csv, analyze the delivery-examples.json for annotation style, then study the Codebook and Keywords.
        2. Identify and extract **ALL** phrases/keywords (including context modifiers) that refer to **DELIVERY aspects**. This includes both **explicit** and **implicit** aspects.
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
        6. DO NOT enclose output in markdowns (```json```)

        Output only the JSON data.
        """
    
    elif task_type == TaskType.EXTRACT_PRICE:
        return """
        You are an expert ABSA annotator. Your task is to perform Aspect Phrase Extraction on Taglish reviews.

        Input Files:
        price-examples.json: Contains annotated reviews
        price-test.csv: Contains new reviews to process
        GeneralCodebook.pdf: Contains aspect definitions
        TMKeywords.pdf: Contains keywords for explicit aspects only

        Additional examples:
          {
            "review_no": "16",
            "review": "510 ang mahal ng payong nd nmn ganun ka quality sira pa sabi ko seller if ayaw bilik pera ko change items nd na nag reply",
            "extractions": [
              {
                "phrase": "ang mahal ng payong",
                "tokens_normalized": [
                  "ang",
                  "mahal",
                  "ng",
                  "payong"
                ]
              }
            ]
          }
          
        Step-by-step process:
        1. For each review in price-test.csv, analyze the price-examples.json for annotation style, then study the Codebook and Keywords.
        2. Identify and extract **ALL** phrases/keywords (including context modifiers) that refer to **PRICE aspects**. This includes both **explicit** and **implicit** aspects.
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
        6. DO NOT enclose output in markdowns (```json```)

        Output only the JSON data.
        """
    
    elif task_type == TaskType.EXTRACT_SERVICE:
        return """
        You are an expert ABSA annotator. Your task is to perform Aspect Phrase Extraction on Taglish reviews.

        Input Files:
        service-examples.json: Contains annotated reviews
        service-test.csv: Contains new reviews to process
        GeneralCodebook.pdf: Contains aspect definitions
        TMKeywords.pdf: Contains keywords for explicit aspects only

        Additional examples:
        {
          "review_no": "58",
          "review": "sayang ang pera sayang ang peranagmessage ako ky seller n make sure n wlang deffect ang item bgo i ship pero no response hndi rin nagsend ng picture bgo i ship pagdating ng item di gumagananagmessage ako para sbihin n deffective ang item pero no response",
          "extractions": [
            {
              "phrase": "no response",
              "tokens_normalized": [
                "no",
                "response"
              ]
            },
            {
              "phrase": "hndi rin nagsend ng picture bgo i ship",
              "tokens_normalized": [
                "hndi",
                "rin",
                "nagsend",
                "ng",
                "picture",
                "bgo",
                "i",
                "ship"
              ]
            },
            {
              "phrase": "no response",
              "tokens_normalized": [
                "no",
                "response"
              ]
            }
          ]
        },
        {
          "review_no": "59",
          "review": "i dont know what to say pero disappointed ako although magaganda yung nakuha ng mga kasama ko sa bahay yung nakuha ko eto gamepad at sampayan wala ring kwenta yung customer service walang reply at all ang tagal din ng deliver",
          "extractions": [
            {
              "phrase": "wala ring kwenta yung customer service",
              "tokens_normalized": [
                "wala",
                "ring",
                "kwenta",
                "yung",
                "customer",
                "service"
              ]
            },
            {
              "phrase": "walang reply at all",
              "tokens_normalized": [
                "walang",
                "reply",
                "at",
                "all"
              ]
            }
          ]
        },
        {
          "review_no": "81",
          "review": "1 earbuds is not working di gumagana ang isang earbuds sayang maganda sana next time po paki double check kung working ang product na binibenta nyo bago i ship sa buyer",
          "extractions": [
            {
              "phrase": "sana next time po paki double check kung working ang product na binibenta nyo bago i ship sa buyer",
              "tokens_normalized": [
                "sana",
                "next",
                "time",
                "po",
                "paki",
                "double",
                "check",
                "kung",
                "working",
                "ang",
                "product",
                "na",
                "binibenta",
                "nyo",
                "bago",
                "i",
                "ship",
                "sa",
                "buyer"
              ]
            }
          ]
        }

        Step-by-step process:
        1. For each review in service-test.csv, analyze the service-examples.json for annotation style, then study the Codebook and Keywords.
        2. Identify and extract **ALL** phrases/keywords (including context modifiers) that refer to **SERVICE aspects**. This includes both **explicit** and **implicit** aspects.
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

        STRICT INSTRUCTION:
        - Process ONLY the test reviews provided at the end
        - Output ONLY new annotations, never repeat examples
        - DO NOT enclose output in markdowns (```json```)

        Output only the JSON data.
        """
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")

