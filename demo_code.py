from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

req1 = "The UAV shall charge to 50%] in less than 3 hours."
req2 = "The UAV shall fully charge in less than 3 hours."

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a requirements engineer. Detect if there is a conflict between Req 1 and Req 2. Answer ONLY with 'Conflict' or 'Neutral'."},
        {"role": "user", "content": f"Req 1: {req1}\nReq 2: {req2}"}
    ]
)

print("zero-shot response:", response.choices[0].message.content)