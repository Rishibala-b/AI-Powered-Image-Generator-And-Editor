"""Gemini prompt helper (uses the current `google-genai` SDK).

Why this was rewritten
----------------------
* `gemini-1.5-flash` was shut down by Google on 29 Sep 2025 -> every call
  returned 404 NOT_FOUND.
* New AI Studio keys start with "AQ." (authorization keys).  Old
  `langchain-google-genai==0.0.11` / `google-generativeai` do not send them
  correctly; the maintained `google-genai` SDK does.

You can test the connection on its own, without Streamlit:

    python prompt_generator.py "a cat sitting on the moon"
"""
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

# Tried in order.  Override the first one with GEMINI_MODEL in your .env.
# "gemini-flash-latest" is an alias Google keeps pointing at a live Flash model,
# so it survives future model retirements.
DEFAULT_MODELS = ["gemini-3.8-flash", "gemini-flash-latest"]

GENERATE_TEMPLATE = """You are a professional image prompt engineer.

Convert the user's idea into a detailed prompt for an AI image-generation model.

Include:
- Subject and visual details
- Lighting and composition
- Colors and atmosphere
- Artistic style and quality

Do not wrap the response in quotes.
Keep it under 60 words (the image model ignores anything longer).
Return only the final image prompt, in one paragraph.

User idea: {input}

Detailed image prompt:"""

# InstructPix2Pix expects a short *edit instruction*, not a scene description.
EDIT_TEMPLATE = """You write instructions for an AI image-editing model
(InstructPix2Pix). Rewrite the user's idea as ONE short, clear edit
instruction in the imperative form, for example "make it snowy" or
"turn the car red". Maximum 15 words.

Do not wrap the response in quotes. Return only the instruction.

User idea: {input}

Edit instruction:"""


def explain_error(exc: Exception) -> str:
    """Turn an SDK exception into a message a human can act on."""
    code = getattr(exc, "code", None)
    text = str(exc)
    if code == 404:
        return ("Model not found (404). The model name is retired or misspelled. "
                "Set GEMINI_MODEL in .env to a current model, e.g. gemini-3.8-flash.")
    if code in (401, 403) or "API key not valid" in text or "UNAUTHENTICATED" in text:
        return (f"Authentication failed ({code}). Check GEMINI_API_KEY in .env: "
                "create a fresh key in Google AI Studio, paste it without quotes "
                "or spaces, and make sure `pip install -U google-genai` is done.")
    if code == 429:
        return ("Quota / rate limit reached (429). Wait a minute, or enable billing "
                "for the project in Google AI Studio.")
    if code == 400:
        return f"Bad request (400): {text}"
    if code is not None and code >= 500:
        return f"Gemini server error ({code}). Try again shortly."
    return f"{type(exc).__name__}: {text}"


class PromptGenerator:
    def __init__(self, temp: float = 0.8, model: str | None = None):
        # GOOGLE_API_KEY wins if both are set (same rule as the Google SDK).
        self.api_key = (
            os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        ).strip().strip('"').strip("'")
        self.temp = temp
        first = model or os.getenv("GEMINI_MODEL", "").strip()
        self.models = ([first] if first else []) + [
            m for m in DEFAULT_MODELS if m != first
        ]
        self.model_used: str | None = None
        self.last_error: str | None = None
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    def get_response(self, user_idea: str, mode: str = "generate") -> str | None:
        """Return the enhanced prompt, or None (reason stored in .last_error).

        mode="generate" -> detailed text-to-image prompt
        mode="edit"     -> short instruction for the image-to-image model
        """
        self.last_error = None
        if self.client is None:
            self.last_error = ("No API key found. Add GEMINI_API_KEY=... "
                               "to the .env file next to app.py.")
            return None

        template = EDIT_TEMPLATE if mode == "edit" else GENERATE_TEMPLATE
        prompt = template.format(input=user_idea)

        for model in self.models:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=self.temp),
                )
                text = (response.text or "").strip().strip('"')
                if text:
                    self.model_used = model
                    return text
                self.last_error = f"{model} returned an empty answer."
            except errors.APIError as e:
                self.last_error = f"[{model}] {explain_error(e)}"
                if e.code != 404:      # only a missing model is worth retrying
                    break
            except Exception as e:     # network down, DNS, proxy, ...
                self.last_error = f"[{model}] {explain_error(e)}"
                break
        print("GEMINI ERROR:", self.last_error, file=sys.stderr)
        return None


if __name__ == "__main__":
    idea = " ".join(sys.argv[1:]) or "a cat sitting on the moon"
    gen = PromptGenerator()
    result = gen.get_response(idea)
    if result:
        print(f"OK (model: {gen.model_used})\n\n{result}")
    else:
        print(f"FAILED\n{gen.last_error}")
