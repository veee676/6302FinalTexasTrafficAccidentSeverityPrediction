import ollama

def generate_explanation(input_data, severity, injuries):
    prompt = f'''
    You are a traffic safety expert.

    A crash occurred with:
    {input_data}

    Model predictions:
    - Crash Severity: {severity}
    - Injury Count: {injuries}

    Explain why this crash likely resulted in this severity.
    '''

    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response['message']['content']
