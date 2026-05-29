import os
from google import genai

api_key = os.getenv("GEMINI_API_KEY")
print("API key loaded:", bool(api_key))

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explain how AI works in one short sentence."
)

print("\nGemini response:")
print(response.text)