import re

def generate_response(prompt, model, tokenizer):
    """
    Generate a response from the model based on the given prompt.
    
    Args:
        prompt (str): The input prompt for the model.
        model: The language model to generate responses.
        tokenizer: The tokenizer to encode and decode text.
    
    Returns:
        str: The generated response from the model.
    """
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
    
    # Decode the output
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    match = re.search(r"(?i)Answer\s*:\s*(.*)", decoded, re.DOTALL)
    if match:
        return match.group(1).strip()
    return decoded.strip()