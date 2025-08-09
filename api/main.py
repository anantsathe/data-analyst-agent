# main.py
import os
import tempfile
import json
import traceback
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from typing import List, Optional
import asyncio

from claude_client import query_claude
from code_executor import execute_user_code
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Data Analyst Agent", version="1.0.0")

# Add CORS middleware for Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """
You are an expert data analyst agent capable of web scraping, data analysis, and visualization. You must write executable Python code to answer questions.

Available libraries: pandas, numpy, matplotlib, seaborn, plotly, requests, BeautifulSoup, duckdb, json, base64, io, re, datetime, time

CRITICAL REQUIREMENTS:
1. Generate ONLY valid Python code inside ```python ``` markdown blocks
2. NO explanations or text outside code blocks
3. Handle errors gracefully with try-except blocks
4. For plots: save as PNG, convert to base64, format as "data:image/png;base64,..."
5. Keep base64 images under 100KB
6. Assign final result to variable `final_answer`
7. For JSON responses, ensure proper formatting
8. For web scraping, use requests + BeautifulSoup
9. For data analysis, use pandas/duckdb as appropriate
10. Always include proper imports at the top
11. IMPORTANT: Clean data thoroughly - remove currency symbols, commas, handle non-numeric strings
12. Use regex and pandas methods to clean text data before converting to numbers

DATA CLEANING GUIDELINES:
- Remove $ signs, commas, and other currency formatting
- Extract only numeric parts from strings
- Handle footnote references and citations in scraped data
- Use pd.to_numeric() with errors='coerce' for safe conversion
- Always validate data types before mathematical operations

RESPONSE FORMATS:
- Single values: final_answer = value
- JSON arrays: final_answer = [item1, item2, item3, item4]
- JSON objects: final_answer = {"key1": "value1", "key2": "value2"}
- Base64 images: final_answer = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."

For plots:
```python
import matplotlib.pyplot as plt
import base64
import io

# Your plotting code here
plt.figure(figsize=(8, 6))
# ... plot creation ...

# Save plot to base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
buffer.seek(0)
image_base64 = base64.b64encode(buffer.getvalue()).decode()
plt.close()
final_answer = f"data:image/png;base64,{image_base64}"
```

Remember: NO text outside code blocks. Only executable Python code.
"""

def get_python_code_from_claude(question_text: str, file_contexts: dict, error_context: str = "") -> str:
    """Get Python code from Claude for data analysis tasks."""
    
    file_info = ""
    if file_contexts:
        file_info = "\n\nAvailable files:\n"
        for filename, filepath in file_contexts.items():
            if filename.endswith('.csv'):
                file_info += f"- {filename}: CSV file at '{filepath}'\n"
            elif filename.endswith('.json'):
                file_info += f"- {filename}: JSON file at '{filepath}'\n"
            elif filename.endswith('.txt'):
                file_info += f"- {filename}: Text file at '{filepath}'\n"
            else:
                file_info += f"- {filename}: File at '{filepath}'\n"
    
    error_info = ""
    if error_context:
        error_info = f"\n\nPREVIOUS ERROR TO FIX:\n{error_context}\n"
    
    user_content = f"""Question: {question_text}{file_info}{error_info}

Generate Python code to answer this question. Use file_contexts dictionary to access file paths."""
    
    messages = [
        {"role": "user", "content": user_content}
    ]
    
    return query_claude(messages, system_message=SYSTEM_PROMPT)

def extract_python_code(claude_response: str) -> str:
    """Extract Python code from Claude's response."""
    import re
    
    # Look for code blocks
    code_blocks = re.findall(r'```python\s*\n(.*?)\n```', claude_response, re.DOTALL)
    
    if code_blocks:
        return code_blocks[0].strip()
    
    # If no code blocks found, look for any Python-like content
    lines = claude_response.split('\n')
    python_lines = []
    in_code = False
    
    for line in lines:
        if any(keyword in line for keyword in ['import ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'final_answer']):
            in_code = True
        
        if in_code:
            python_lines.append(line)
    
    if python_lines:
        return '\n'.join(python_lines)
    
    # Last resort: return the whole response
    return claude_response

@app.post("/api/")
async def handle_request(
    questions: UploadFile = File(...),
    files: List[UploadFile] = File(default=[])
):
    """Main API endpoint for data analysis requests."""
    
    temp_dir = None
    try:
        # Read the question from the uploaded file
        question_content = await questions.read()
        question_text = question_content.decode("utf-8").strip()
        
        # Create temporary directory for files
        temp_dir = tempfile.mkdtemp()
        file_contexts = {}
        
        # Save all uploaded files
        for file in files:
            if file.filename:  # Skip empty file uploads
                filepath = os.path.join(temp_dir, file.filename)
                with open(filepath, "wb") as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                file_contexts[file.filename] = filepath
        
        # Initialize execution context
        globals_dict = {
            "file_contexts": file_contexts,
            "temp_dir": temp_dir
        }
        
        # Self-correction loop
        max_attempts = 3
        attempt = 0
        success = False
        last_error = ""
        
        while attempt < max_attempts and not success:
            attempt += 1
            
            try:
                # Get code from Claude
                claude_response = get_python_code_from_claude(
                    question_text, 
                    file_contexts, 
                    last_error if attempt > 1 else ""
                )
                
                # Extract Python code
                python_code = extract_python_code(claude_response)
                
                if not python_code:
                    raise ValueError("No valid Python code found in Claude's response")
                
                # Execute the code
                result = execute_user_code(python_code, globals_dict.copy())
                
                if result["success"]:
                    success = True
                    final_answer = result["globals"].get("final_answer", "No answer generated")
                    
                    # Ensure JSON response format
                    if isinstance(final_answer, (dict, list)):
                        return JSONResponse(content=final_answer)
                    else:
                        return JSONResponse(content={"result": final_answer})
                else:
                    last_error = result["error"]
                    if attempt == max_attempts:
                        return JSONResponse(
                            status_code=500,
                            content={
                                "error": "Code execution failed after maximum attempts",
                                "details": last_error,
                                "code": python_code
                            }
                        )
                    
            except Exception as e:
                last_error = f"Exception in attempt {attempt}: {str(e)}\n{traceback.format_exc()}"
                if attempt == max_attempts:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": "Failed after maximum attempts",
                            "details": last_error
                        }
                    )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Request processing failed",
                "details": str(e)
            }
        )
    
    finally:
        # Cleanup temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "Data Analyst Agent is running", "version": "1.0.0"}

@app.get("/health")
async def health():
    """Health check for deployment."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)