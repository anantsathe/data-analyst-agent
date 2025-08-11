#!/usr/bin/env python3
"""
Local testing script for the Data Analyst Agent
"""

import requests
import json
import os

def test_wikipedia_question():
    """Test the Wikipedia question example"""
    
    # Create a test question file
    question_content = """Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.
"""
    
    # Write to temporary file
    with open('test_question.txt', 'w') as f:
        f.write(question_content)
    
    # Test the endpoint
    url = "http://localhost:8000/api/"
    
    try:
        with open('test_question.txt', 'rb') as f:
            files = {'files': ('questions.txt', f, 'text/plain')}
            response = requests.post(url, files=files, timeout=180)  # 3 minute timeout
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Wikipedia test successful!")
            print("Response:")
            print(json.dumps(result, indent=2))
            
            # Validate response structure
            if isinstance(result, list) and len(result) == 4:
                print("✅ Response structure is correct")
                print(f"Answer 1 (number): {result[0]}")
                print(f"Answer 2 (film): {result[1]}")
                print(f"Answer 3 (correlation): {result[2]}")
                print(f"Answer 4 (plot): {result[3][:50]}..." if len(str(result[3])) > 50 else result[3])
            else:
                print("❌ Response structure is incorrect")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
    finally:
        # Clean up
        if os.path.exists('test_question.txt'):
            os.remove('test_question.txt')

def test_court_question():
    """Test the Indian High Court question example"""
    
    question_content = """The Indian high court judgement dataset contains judgements from the Indian High Courts, downloaded from [ecourts website](https://judgments.ecourts.gov.in/). It contains judgments of 25 high courts, along with raw metadata (as .json) and structured metadata (as .parquet).

- 25 high courts
- ~16M judgments
- ~1TB of data

Structure of the data in the bucket:

- `data/pdf/year=2025/court=xyz/bench=xyz/judgment1.pdf,judgment2.pdf`
- `metadata/json/year=2025/court=xyz/bench=xyz/judgment1.json,judgment2.json`
- `metadata/parquet/year=2025/court=xyz/bench=xyz/metadata.parquet`
- `metadata/tar/year=2025/court=xyz/bench=xyz/metadata.tar.gz`
- `data/tar/year=2025/court=xyz/bench=xyz/pdfs.tar`

This DuckDB query counts the number of decisions in the dataset.

```sql
INSTALL httpfs; LOAD httpfs;
INSTALL parquet; LOAD parquet;

SELECT COUNT(*) FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1');
```

Here are the columns in the data:

| Column                 | Type    | Description                    |
| ---------------------- | ------- | ------------------------------ |
| `court_code`           | VARCHAR | Court identifier (e.g., 33~10) |
| `title`                | VARCHAR | Case title and parties         |
| `description`          | VARCHAR | Case description               |
| `judge`                | VARCHAR | Presiding judge(s)             |
| `pdf_link`             | VARCHAR | Link to judgment PDF           |
| `cnr`                  | VARCHAR | Case Number Register           |
| `date_of_registration` | VARCHAR | Registration date              |
| `decision_date`        | DATE    | Date of judgment               |
| `disposal_nature`      | VARCHAR | Case outcome                   |
| `court`                | VARCHAR | Court name                     |
| `raw_html`             | VARCHAR | Original HTML content          |
| `bench`                | VARCHAR | Bench identifier               |
| `year`                 | BIGINT  | Year partition                 |

Here is a sample row:

```json
{
  "court_code": "33~10",
  "title": "CRL MP(MD)/4399/2023 of Vinoth Vs The Inspector of Police",
  "description": "No.4399 of 2023 BEFORE THE MADURAI BENCH OF MADRAS HIGH COURT ( Criminal Jurisdiction ) Thursday, ...",
  "judge": "HONOURABLE  MR JUSTICE G.K. ILANTHIRAIYAN",
  "pdf_link": "court/cnrorders/mdubench/orders/HCMD010287762023_1_2023-03-16.pdf",
  "cnr": "HCMD010287762023",
  "date_of_registration": "14-03-2023",
  "decision_date": "2023-03-16",
  "disposal_nature": "DISMISSED",
  "court": "33_10",
  "raw_html": "<button type='button' role='link'..",
  "bench": "mdubench",
  "year": 2023
}
```

Answer the following questions and respond with a JSON object containing the answer.

```json
{
  "Which high court disposed the most cases from 2019 - 2022?": "...",
  "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "...",
  "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": "data:image/webp:base64,..."
}
```
"""
    
    # Write to temporary file
    with open('test_question2.txt', 'w') as f:
        f.write(question_content)
    
    # Test the endpoint
    url = "http://localhost:8000/api/"
    
    try:
        with open('test_question2.txt', 'rb') as f:
            files = {'files': ('questions.txt', f, 'text/plain')}
            response = requests.post(url, files=files, timeout=180)  # 3 minute timeout
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Court data test successful!")
            print("Response:")
            print(json.dumps(result, indent=2))
            
            # Validate response structure
            if isinstance(result, dict) and len(result) == 3:
                print("✅ Response structure is correct")
                for key, value in result.items():
                    if "base64" in str(value):
                        print(f"{key}: {str(value)[:50]}...")
                    else:
                        print(f"{key}: {value}")
            else:
                print("❌ Response structure is incorrect")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
    finally:
        # Clean up
        if os.path.exists('test_question2.txt'):
            os.remove('test_question2.txt')

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ Health check successful!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed with status {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    print("🧪 Starting Data Analyst Agent Tests")
    print("=" * 50)
    
    # Check if server is running
    print("\n1. Testing health check...")
    test_health_check()
    
    print("\n2. Testing Wikipedia analysis...")
    test_wikipedia_question()
    
    print("\n3. Testing Court data analysis...")
    test_court_question()
    
    print("\n" + "=" * 50)
    print("🏁 Testing complete!")
    
    print("\n📝 To run the server locally:")
    print("1. Set your CLAUDE_API_KEY environment variable")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    print("4. Run this test script: python test_local.py")