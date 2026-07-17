from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

response = client.models.generate_content(
    model="models/gemini-3.5-flash",
    contents="Reply with only: API is working"
)

print(response.text)