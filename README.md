---
title: Data Analyst Agent
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# Data Analyst Agent

An API-driven data analyst powered by LLMs that can source, clean, analyze, and visualize any data.

---

##  Overview

- This project exposes a single API endpoint—e.g., `https://app.example.com/api/`—to handle data analysis tasks via POST requests.
- You submit a `questions.txt` file containing your analysis task along with optional data attachments (`.csv`, `.json`, images, etc.).
- The agent processes your request using LLM-generated Python code and returns results—numeric answers, JSON objects, or visualizations—within 3 minutes.

---

##  API Usage

Send requests like this:

```bash
curl "https://app.example.com/api/" \
  -F "questions.txt=@question.txt" \
  -F "image.png=@image.png" \
  -F "data.csv=@data.csv"
questions.txt: Always required, contains the instructions or questions.

Optional attachments: Any number of files such as .csv, .json, or image files.

Responses must be delivered within 3 minutes in the requested format.

Example Tasks
Example 1 (question-1.txt):
Scrape the list of highest-grossing films from Wikipedia and answer:

Number of movies that made ≥ $2 bn before 2000.

Earliest movie with > $1.5 bn revenue.

Correlation between “Rank” and “Peak”.

Scatterplot with a dotted red regression line (base64 PNG, <100 kB).

Sample Response:


[1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0KG..."]
Example 2 (question-2.txt):
Analyze an Indian High Court judgments dataset (~16M records, 1 TB), including:

Which high court disposed the most cases from 2019–2022?

Regression slope of registration-to-decision delay by year for court 33_10.

Scatterplot of year vs. delay days (base64 under 100k characters).

Expected JSON Response:


{
  "Which high court disposed the most cases from 2019 - 2022?": "...",
  "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": "...",
  "Plot the year and # of days of delay ...": "data:image/webp;base64,..."
}
Evaluation Workflow (using promptfoo)

description: "TDS Data Analyst Agent – generic eval (20-point rubric)"

providers:
  - id: https
    config:
      url: https://app.example.com/api/
      method: POST
      body: file://question.txt
      transformResponse: json

assert:
  - type: is-json
    value: {type: array, minItems: 4, maxItems: 4}
    weight: 0
  - type: python
    weight: 4
    value: |
      import json
      print(json.loads(output)[0] == 1)
  - type: python
    weight: 4
    value: |
      import json, re
      print(bool(re.search(r'titanic', json.loads(output)[1], re.I)))
  - type: python
    weight: 4
    value: |
      import json
      print(abs(float(json.loads(output)[2]) - 0.485782) <= 0.001)
  - type: llm-rubric
    provider: openai:gpt-4.1-nano
    weight: 8
    preprocess: |
      import json
      data = json.loads(output)
      context['plot'] = data[3]
    rubricPrompt: |
      [
        { "role": "system", "content": "Grade the scatterplot. Award score 1 only if ALL are true: (a) it's a scatterplot of Rank vs Peak; (b) red dotted regression line; (c) axes visible & labelled; (d) file size < 100 kB. Otherwise score 0." },
        { "role": "user", "content": [{ "type": "image_url", "image_url": { "url": "{{plot}}" } }, { "type": "text", "text": "Here is the original task:\n\n{{vars.question}}\n\nReview the image and JSON above." }] }
      ]
    threshold: 0.99

tests:
  - description: "Data analysis"
Your final grade equals the score returned by promptfoo—no scaling, no normalization.

Project Structure

data-analyst-agent/
├── api/
│   └── main.py
├── requirements.txt
├── .env
├── claude_client.py
├── code_executor.py
└── vercel.json
main.py
Built with FastAPI to handle file uploads and return JSON.

Implements a self-correction loop: it calls Claude to generate Python code, executes it (via code_executor.py), and retries up to 3 times if errors occur.

Ensures modular execution, temporary file handling, and robust error reporting.

Features & Guidelines
LLM-powered code generation: Uses Claude to generate python code based only on prompt and context.

Strict format control: Responses inside final_answer, with no extra text.

Data cleaning: Handles currency symbols, commas, non-numeric strings via regex and safe conversions.

Visual output constraints: Base64 images in specified formats, size-limited (<100 kB or characters).

Retry logic: 3 attempts for robust execution; detailed errors returned after failures.

Full dependency isolation: Imports controlled libraries only; runs in temp directories per request.

Installation & Deployment
Clone this repo.

Copy .env.example to .env and supply Claude / LLM credentials.

Install dependencies:


pip install -r requirements.txt
Deploy using Vercel (supported via vercel.json) or host anywhere compatible with FastAPI.

Contributing
Enhancements welcome! Suggestions include:

Adding more robust data sources or cleaning routines.

Supporting additional plot formats or output types.

Integrating alternative LLMs or execution sandboxes.

Contact & Credits

Developed with the vision of making intelligent, LLM-powered data analysis accessible and evaluable through a simple API interface.
