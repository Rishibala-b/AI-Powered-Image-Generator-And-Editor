import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
client = genai.Client(api_key=key)

for name in ["gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash", "gemini-3.5-flash-lite"]:
    try:
        r = client.models.generate_content(model=name, contents="Say hi")
        print("OK  ", name, "->", r.text.strip()[:40])
    except Exception as e:
        print("FAIL", name, "->", str(e)[:400])