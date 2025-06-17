import openai

# Replace with your OpenAI API key
openai.api_key = 'your-api-key'

def chat_with_gpt(prompt):
    # Call the OpenAI API to interact with GPT-4 model
    response = openai.Completion.create(
        model="gpt-4",  # Specify the GPT-4 model
        prompt=prompt,
        max_tokens=150,  # Maximum number of tokens in the response
        temperature=0.7,  # Controls randomness of the output (higher = more random)
        top_p=1.0,  # Controls the cumulative probability for top-p sampling
        frequency_penalty=0.0,  # Controls repetition of words (higher = less repetition)
        presence_penalty=0.0  # Controls whether new topics are introduced
    )
    return response.choices[0].text.strip()

# Example usage
prompt = "Hello, GPT-4! How are you today?"
response = chat_with_gpt(prompt)
print(response)
