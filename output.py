import re
import requests

def generate_response(prompt,context,url):
    """
    Generate a response from the model based on the given prompt.
    
    Args:
        prompt (str): The input prompt for the model.
        model: The language model to generate responses.
        tokenizer: The tokenizer to encode and decode text.
    
    Returns:
        str: The generated response from the model.
    """

    payload = {
    "prompt": prompt,
    "retrieved_docs":context
    }
    decoded= requests.post(url, json=payload)

    print(decoded)
    try:
        decoded = decoded.json()  # Try to parse JSON
    except ValueError:
        decoded = decoded.text  # If it's not JSON, treat it as plain text
    
    match = re.search(r"(?i)Answer\s*:\s*(.*)", decoded, re.DOTALL)
    if match:
        return match.group(1).strip()
    return decoded.strip()