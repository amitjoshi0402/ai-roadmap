import os
from dotenv import load_dotenv
from openai import OpenAI

def main():
    load_dotenv()  # loads OPENAI_API_KEY from .env into environment

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = "Say hello in one sentence and give one practical tip for learning AI engineering."

    # Responses API (recommended by the SDK)
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )

    # The SDK provides a convenient output_text for plain text
    print(resp.output_text)

if __name__ == "__main__":
    main()