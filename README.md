# Gemini-to-Image

Type an idea -> Gemini writes a detailed image prompt -> Stable Diffusion XL
(text-to-image) or InstructPix2Pix (edit an uploaded image) draws it.
Streamlit front end.

## Setup
```
python -m venv .venv          # Python 3.10 or 3.11 recommended
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env        # then paste your key into .env
```
Get a key at https://aistudio.google.com/apikey. New keys start with `AQ.` -
that is normal and works with the `google-genai` SDK used here.

## Run
```
streamlit run app.py
```
Use **Test Gemini connection** in the sidebar to check the key/model before
generating. You can also test without Streamlit: `python prompt_generator.py "a cat"`.

## Troubleshooting
| Message | Fix |
|---|---|
| `Model not found (404)` | Model retired. Set `GEMINI_MODEL=gemini-2.5-flash` in `.env`. |
| `Authentication failed (401/403)` | New key from AI Studio, no quotes/spaces, `pip install -U google-genai`. |
| `Quota ... (429)` | Wait, or enable billing on the project. |
| `GPU ran out of memory` | Add `LOW_VRAM=1` to `.env`. |
| Very slow, no NVIDIA GPU | It runs on CPU (needs ~16 GB RAM, minutes per image). |
| Models re-download | Set `HF_HOME` in `.env` to your existing cache folder. |

If Gemini fails, the app still works: it uses your original text as the prompt
and shows the exact reason.

## Credits
Google Gemini, Hugging Face (SDXL, LCM-LoRA, Papercut LoRA, InstructPix2Pix), Streamlit. MIT License.
