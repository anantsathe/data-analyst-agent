# claude_client.py
import os
import requests
import json

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.getenv("CLAUDE_API_KEY")
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"

def query_claude(messages: list[dict], model: str = ANTHROPIC_MODEL, system_message: str = "You are a helpful assistant.") -> str:
    """
    Query Claude API with messages and system prompt.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model to use (defaults to ANTHROPIC_MODEL)
        system_message: System prompt (defaults to generic assistant)
    
    Returns:
        str: Claude's response text
    """
    
    if not ANTHROPIC_API_KEY:
        raise ValueError("CLAUDE_API_KEY environment variable not set")
    
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json"
    }
    
    # Filter out any system messages from the messages array
    # (they should be in the system parameter instead)
    filtered_messages = []
    for msg in messages:
        if msg.get("role") != "system":
            filtered_messages.append(msg)
        else:
            # If there's a system message in the array, use it as the system prompt
            system_message = msg["content"]
    
    payload = {
        "model": model,
        "system": system_message,
        "messages": filtered_messages,
        "max_tokens": 4096,  # Increased for longer code responses
        "temperature": 0.1   # Lower temperature for more consistent code generation
    }
    
    try:
        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result["content"][0]["text"]
        
    except requests.exceptions.HTTPError as e:
        error_details = ""
        try:
            error_details = response.json()
        except:
            error_details = response.text
        
        raise RuntimeError(f"Claude API call failed: {e}\nResponse: {error_details}")
    
    except requests.exceptions.Timeout:
        raise RuntimeError("Claude API call timed out")
    
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Claude API request failed: {e}")
    
    except Exception as e:
        raise RuntimeError(f"Unexpected error in Claude API call: {e}")
